"""Run one real-weight DP2 + EP arm and save per-rank evidence as JSON."""
import argparse
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import time

import nixl_ep
import torch
import torch.distributed as dist
import vllm
from vllm import EngineArgs, SamplingParams
from vllm.distributed import get_ep_group
from vllm.forward_context import override_forward_context
from vllm.inputs import tokens_input
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu import model_runner, ubatch_utils
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

MODELS_DIR = Path(os.environ.get("DBO_MODELS_DIR", "/workspace/models"))
MODELS = {
    "gdn": str(MODELS_DIR / "Qwen3-Next-80B-A3B-Instruct-int4-AutoRound"),
    "mamba2": str(MODELS_DIR / "granite-4.0-h-tiny"),
}
GRAPH_SIZES = [1, 2, 4, 8, 16, 32]


def descriptor(desc):
    return {
        "mode": desc.cg_mode.name,
        "padded_tokens": desc.num_tokens,
        "padded_requests": desc.num_reqs,
        "uniform_tokens": desc.uniform_token_count,
        "ubatches": desc.num_ubatches,
    }


def installed_version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def logprobs(values):
    return [
        {str(token): info.logprob for token, info in row.items()} if row else None
        for row in values
    ]


def begin_serial_microbatches(runner, model, model_inputs, ubatch_state, for_capture=False):
    """Use the same slices and metadata on one stream for a diagnostic control."""
    def finish():
        outputs = []
        with runner.sm_control:
            for item, context in zip(ubatch_state.slices, ubatch_state.forward_contexts):
                inputs = ubatch_utils.slice_model_inputs(model_inputs, item.token_slice)
                with override_forward_context(context):
                    outputs.append(model(**inputs))
        return ubatch_utils.merge_ubatch_outputs(outputs)

    return finish


class Trace:
    def __init__(self, rank, filename):
        self.rank = rank
        self.filename = filename
        self.case = None
        self.row = None
        self.capture_calls = Counter()
        self.startup_calls = Counter()
        self.query_rank_mask = None
        self.suppress_split = False
        self.keys = {}
        self.results = {"rank": rank, "cases": {}}

    def save(self):
        temp = self.filename.with_suffix(".tmp")
        temp.write_text(json.dumps(self.results, separators=(",", ":"), allow_nan=False))
        temp.replace(self.filename)

    def install(self):
        execute = model_runner.GPUModelRunner.execute_model
        dispatch = model_runner.dispatch_cg_and_sync_dp
        prepare = ubatch_utils.UBatchRunner.prepare
        replay = ModelCudaGraphManager.run_fullgraph

        @wraps(execute)
        def record_execute(runner, scheduled, *args, **kwargs):
            if self.case is None:
                return execute(runner, scheduled, *args, **kwargs)
            row = {
                "dummy": kwargs.get("dummy_run", False),
                "scheduled_tokens": scheduled.total_num_scheduled_tokens,
                "checkpoint": scheduled.has_mamba_prefill_checkpoint,
                "step": self.case["step_count"],
                "nixl_ep_calls": {},
            }
            self.case["steps"].append(row)
            self.row = row
            try:
                return execute(runner, scheduled, *args, **kwargs)
            finally:
                self.row = None

        @wraps(dispatch)
        def record_dispatch(manager, num_reqs, num_tokens, *args, **kwargs):
            if self.suppress_split:
                kwargs["allow_ubatching"] = False
            desc, sync = dispatch(manager, num_reqs, num_tokens, *args, **kwargs)
            if self.row is not None:
                self.row.update(descriptor(desc))
                self.row["real_tokens"] = 0 if self.row["dummy"] else num_tokens
                self.row["allow_ubatching"] = kwargs.get("allow_ubatching", False)
            return desc, sync

        @wraps(prepare)
        def record_prepare(runner, inputs, *args, **kwargs):
            state = prepare(runner, inputs, *args, **kwargs)
            if self.row is None:
                return state
            self.row["slices"] = [
                {
                    "token_range": [item.token_slice.start, item.token_slice.stop],
                    "request_range": [
                        item.request_slice.start,
                        item.request_slice.stop,
                    ],
                    "valid_tokens": max(
                        0, min(inputs.num_tokens, item.token_slice.stop)
                        - item.token_slice.start,
                    ),
                }
                for item in state.slices
            ]
            self.record_requests(inputs)
            return state

        @wraps(replay)
        def record_replay(manager, desc):
            value = replay(manager, desc)
            if self.row is not None:
                self.row["replay"] = descriptor(desc)
            return value

        make_inputs = model_runner.GPUModelRunner.prepare_inputs

        @wraps(make_inputs)
        def record_prepare_inputs(runner, *args, **kwargs):
            inputs = make_inputs(runner, *args, **kwargs)
            if self.row is not None:
                self.record_requests(inputs)
            return inputs

        model_runner.GPUModelRunner.execute_model = record_execute
        model_runner.dispatch_cg_and_sync_dp = record_dispatch
        model_runner.GPUModelRunner.prepare_inputs = record_prepare_inputs
        ubatch_utils.UBatchRunner.prepare = record_prepare
        ModelCudaGraphManager.run_fullgraph = record_replay
        for name in ("dispatch", "combine"):
            self.install_nixl_ep(name)

    def install_nixl_ep(self, name):
        method = getattr(nixl_ep.Buffer, name)

        @wraps(method)
        def call(*args, **kwargs):
            if self.row is not None:
                counts = self.row["nixl_ep_calls"]
                counts[name] = counts.get(name, 0) + 1
            elif self.case is None:
                self.startup_calls[name] += 1
                if torch.cuda.is_current_stream_capturing():
                    self.capture_calls[name] += 1
            return method(*args, **kwargs)

        setattr(nixl_ep.Buffer, name, call)

    def record_requests(self, inputs):
        if "requests" in self.row:
            return
        self.row["requests"] = [
            {
                "id": self.keys.get(req_id, req_id),
                "q": int(inputs.num_scheduled_tokens[index]),
                "query_start": int(inputs.query_start_loc_np[index]),
                "context": int(inputs.num_computed_tokens_np[index]),
                "prefilling": bool(inputs.is_prefilling_np[index]),
            }
            for index, req_id in enumerate(inputs.req_ids)
        ]

    @contextmanager
    def in_case(self, name):
        self.case = self.results["cases"][name] = {
            "step_count": 0, "steps": [], "inputs": {}, "outputs": {},
        }
        print(f"CASE_START rank={self.rank} {name}", flush=True)
        start = time.monotonic()
        try:
            yield
            if self.query_rank_mask is not None:
                status = self.query_rank_mask().tolist()
                self.case["rank_mask"] = status
                assert status == [0, 0], (name, status)
        finally:
            case, self.case = self.case, None
            case["wall_seconds"] = round(time.monotonic() - start, 3)
            self.save()
            shapes = Counter(
                f"{row.get('mode', 'NONE')}/{row.get('ubatches', 0)}"
                for row in case["steps"]
            )
            print(
                f"CASE_END rank={self.rank} {name} outputs={len(case['outputs'])} "
                f"shapes={dict(shapes)} seconds={case['wall_seconds']}",
                flush=True,
            )


class Suite:
    def __init__(self, engine, trace):
        self.engine = engine
        self.trace = trace
        self.rank = trace.rank

    def submit(self, label, prompt, *, n=4, force=True, score=False, hit=False):
        seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")
        if score:
            force = False
        forced = random.Random(seed ^ 7281).sample(range(400, 8500), n) if force else None
        params = SamplingParams(
            temperature=0,
            max_tokens=n,
            ignore_eos=True,
            detokenize=False,
            logprobs=5,
            prompt_logprobs=1 if score else None,
            skip_reading_prefix_cache=not hit,
            trace_decode_token_ids=forced,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        digest = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        self.trace.case["inputs"][label] = {
            "prompt_size": len(prompt), "prompt_sha256": digest,
            "forced": forced, "max_tokens": n, "allow_prefix_hit": hit,
        }
        internal_key = self.engine.add_request(label, tokens_input(prompt), params)
        self.trace.keys[internal_key] = label

    def step(self):
        case = self.trace.case
        case["step_count"] += 1
        for output in self.engine.step():
            req_id = self.trace.keys.get(output.request_id, output.request_id)
            assert output.finished, output
            assert req_id not in case["outputs"], req_id
            result, = output.outputs
            inputs = case["inputs"][req_id]
            tokens = list(result.token_ids)
            assert len(tokens) == inputs["max_tokens"], (req_id, len(tokens), inputs)
            assert result.finish_reason == "length", (req_id, result.finish_reason)
            assert inputs["forced"] is None or tokens == inputs["forced"], (req_id, tokens)
            case["outputs"][req_id] = {
                "tokens": tokens,
                "sample_logprobs": logprobs(result.logprobs),
                "prompt_logprobs": logprobs(output.prompt_logprobs or []),
                "prompt_token_ids": output.prompt_token_ids if output.prompt_logprobs else [],
                "cached_tokens": output.num_cached_tokens,
            }

    def advance(self):
        assert self.engine.has_unfinished_requests()
        self.step()

    def drain(self):
        for _ in range(32):
            if not self.engine.has_unfinished_requests():
                assert self.trace.case["inputs"].keys() == self.trace.case["outputs"].keys()
                return
            self.step()
        raise RuntimeError("Requests did not finish within 32 steps")

    def prompt(self, name, size):
        return make_prompt(f"{name}-{self.rank}", size)

    def one(self, name, sizes, *, force=True, score=True):
        with self.trace.in_case(name):
            size = sizes[self.rank]
            if size:
                self.submit(name, self.prompt(name, size), force=force, score=score)
            self.drain()

    def mixed(self, length):
        with self.trace.in_case("mixed"):
            for index in range(3):
                label = f"decode-{index}"
                self.submit(label, self.prompt(f"mixed-{label}", 9 + index), n=5)
            self.advance()
            self.submit("prefill", self.prompt("mixed-prefill", length), score=True)
            self.drain()

    def decode(self, sizes, *, greedy=False, default_thresholds=False):
        prefix = "default_decode" if default_thresholds else "decode"
        name = f"{prefix}_{sizes[0]}_{sizes[1]}"
        if greedy:
            name += "_greedy"
        with self.trace.in_case(name):
            for index in range(sizes[self.rank]):
                label = f"req-{index}"
                self.submit(label, self.prompt(f"{name}-{label}", 4), force=not greedy)
            self.drain()

    def align_prefix(self, block_size):
        assert self.engine.reset_prefix_cache()
        common = self.prompt("prefix-common", block_size)
        warm = common + self.prompt("prefix-warm-tail", 17)
        hit = common + self.prompt("prefix-hit-tail", 59)
        for name, prompt, allow_hit, count in (
            ("prefix_warm", warm, False, 1),
            ("prefix_hit", hit, True, 4),
            ("prefix_cold", hit, False, 4),
        ):
            with self.trace.in_case(name):
                self.submit("target", prompt, hit=allow_hit, n=count)
                self.drain()

    def checkpoint(self, block_size):
        assert self.engine.reset_prefix_cache()
        name = "checkpoint_veto"
        with self.trace.in_case(name):
            if self.rank == 0:
                prefix = self.prompt("checkpoint-prefix", block_size)
                self.submit("target", prefix + self.prompt(name, 89), score=True)
            else:
                self.submit("target", self.prompt(name, 173), score=True)
            self.drain()
        with self.trace.in_case("checkpoint_prefix"):
            if self.rank == 0:
                self.submit(
                    "target", prefix + self.prompt("checkpoint-hit-tail", 63), hit=True
                )
            else:
                self.submit("target", self.prompt("checkpoint-peer", 173))
            self.drain()

    def run(self, args, block_size, parallel):
        long_cases = args.max_batched_tokens >= block_size + 89
        cross_lengths = (min(701, block_size - 7),) * 2 if long_cases else (101, 117)
        self.one("cross_prefill", cross_lengths)
        if args.smoke:
            self.decode((5, 8))
            return
        self.one("cross_prefill_forced", (127, 149), score=False)
        self.one("cross_prefill_greedy", (137, 163), force=False)
        self.one("prefill_padding_rank0", (39, 111))
        self.mixed(min(603, block_size - 11) if long_cases else 109)
        for sizes in ((1, 1), (2, 2), (3, 4), (3, 8), (4, 8), (5, 8),
                      (7, 8), (8, 8), (16, 16), (31, 32), (32, 32)):
            self.decode(sizes)
        self.decode((4, 8), greedy=True)
        self.one("empty_rank0", (0, 97))
        self.one("empty_rank1", (97, 0))
        with self.trace.in_case("both_empty"):
            assert not self.engine.has_unfinished_requests()
            self.step()
        if not long_cases:
            return
        if args.cache == "align":
            self.align_prefix(block_size)
            self.checkpoint(block_size)
        parallel.dbo_prefill_token_threshold = 512
        parallel.dbo_decode_token_threshold = 32
        self.one("default_prefill_511_512", (511, 512), score=False)
        self.one("default_prefill_512_512", (512, 512), score=False)
        self.decode((31, 32), default_thresholds=True)
        self.decode((32, 32), default_thresholds=True)


def make_prompt(label, size):
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    generator = random.Random(seed)
    return [generator.randrange(1024, 20000) for _ in range(size)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--cache", choices=("none", "align"), required=True)
    parser.add_argument("--dbo", choices=("on", "off"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--serial-split", action="store_true")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--repeat-suite", action="store_true")
    controls.add_argument("--unsplit-control", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    if args.serial_split:
        if args.dbo != "on":
            parser.error("--serial-split requires --dbo on")
        ubatch_utils.UBatchRunner.begin_capturable_run = begin_serial_microbatches
    if args.unsplit_control and args.dbo != "on":
        parser.error("--unsplit-control requires --dbo on")
    trace = Trace(rank, args.output / f"rank{rank}.json")
    trace.results["serial_split"] = args.serial_split
    trace.install()
    options = dict(
        model=MODELS[args.model], dtype="bfloat16",
        skip_tokenizer_init=True, tensor_parallel_size=1, data_parallel_size=2,
        distributed_executor_backend="external_launcher",
        enable_expert_parallel=True, all2all_backend="nixl_ep",
        max_model_len=4096, max_num_seqs=32,
        max_num_batched_tokens=args.max_batched_tokens,
        kv_cache_memory_bytes=3 * 1024**3, enable_extensible_kv_cache=False,
        block_size=256 if args.model == "gdn" else 512,
        enable_prefix_caching=args.cache == "align", mamba_cache_mode=args.cache,
        enable_dbo=args.dbo == "on", async_scheduling=False,
        dbo_prefill_token_threshold=32, dbo_decode_token_threshold=2,
        enforce_eager=False, disable_log_stats=True, enable_trace_replay=True, seed=2701,
        compilation_config={
            "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
            "decode_cudagraph_capture_request_sizes": GRAPH_SIZES,
        },
    )
    print(f"ENGINE_START rank={rank} options={options}", flush=True)
    trace.results["options"] = options.copy()
    trace.save()
    engine = LLMEngine.from_engine_args(EngineArgs(**options))
    core = engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.model_runner
    mamba_groups = [
        {
            "type": group.kv_cache_spec.mamba_type.name,
            "block_size": group.kv_cache_spec.block_size,
            "checkpoints": group.kv_cache_spec.num_prefill_checkpoint_blocks,
            "checkpoint_alignment": group.kv_cache_spec.prefill_checkpoint_alignment,
            "layers": len(group.layer_names),
        }
        for group in runner.kv_cache_config.kv_cache_groups
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]
    cfg = engine.vllm_config
    manager = get_ep_group().device_communicator.all2all_manager
    trace.query_rank_mask = manager.query_active_mask
    block_size = cfg.cache_config.block_size
    trace.results["runtime"] = {
        "source": str(vllm.__file__), "vllm": vllm.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(), "nixl": metadata.version("nixl"),
        "nixl_cu13": metadata.version("nixl-cu13"),
        "nixl_ep_module": nixl_ep.Config.__module__,
        "nixl_device_mode_env": os.environ.get("NIXL_EP_DEVICE_MODE"),
        "ucx_tls_env": os.environ.get("UCX_TLS"),
        "initial_rank_mask": trace.query_rank_mask().tolist(),
        "flashinfer_cubin": installed_version("flashinfer-cubin"),
        "flashinfer_jit_cache": installed_version("flashinfer-jit-cache"),
        "manager": type(manager).__name__, "cache_mode": cfg.cache_config.mamba_cache_mode,
        "block_size": block_size, "mamba_groups": mamba_groups,
        "scheduler_mamba_internal_checkpoints": core.scheduler.mamba_has_prefill_checkpoint_blocks,
        "scheduler_hash_block_size": core.scheduler.hash_block_size,
        "scheduler_partial_cache_hit": core.scheduler.mamba_partial_cache_hit,
        "effective_cg_mode": cfg.compilation_config.cudagraph_mode.name,
        "graphs": [
            descriptor(desc) for desc in runner.cudagraph_manager.graphs
        ] if runner.cudagraph_manager is not None else [],
        "startup_nixl_ep_calls": dict(trace.startup_calls),
        "capture_nixl_ep_calls": dict(trace.capture_calls),
    }
    trace.save()
    print(f"ENGINE_READY rank={rank} runtime={trace.results['runtime']}", flush=True)
    assert mamba_groups and (runner.ubatch_runner is not None) == options["enable_dbo"]
    assert cfg.cache_config.mamba_cache_mode == args.cache
    assert trace.results["runtime"]["initial_rank_mask"] == [0, 0]
    suite = Suite(engine, trace)
    suite.run(args, block_size, cfg.parallel_config)
    trace.results["complete"] = True
    trace.save()
    if args.repeat_suite or args.unsplit_control:
        repeat_name = "same-process-unsplit" if args.unsplit_control else "same-process-repeat"
        repeat_dir = args.output / repeat_name
        repeat_dir.mkdir(exist_ok=True)
        trace.filename = repeat_dir / f"rank{rank}.json"
        trace.results = {**trace.results, "cases": {}, "complete": False, "control": repeat_name}
        trace.keys.clear()
        trace.suppress_split = args.unsplit_control
        assert engine.reset_prefix_cache()
        cfg.parallel_config.dbo_prefill_token_threshold = options["dbo_prefill_token_threshold"]
        cfg.parallel_config.dbo_decode_token_threshold = options["dbo_decode_token_threshold"]
        suite.run(args, block_size, cfg.parallel_config)
        trace.results["complete"] = True
        trace.save()
    dist.barrier(group=engine.dp_group)
    engine.engine_core.shutdown()
    if dist.is_initialized():
        dist.destroy_process_group()
    print(f"ARM_COMPLETE rank={rank}", flush=True)


if __name__ == "__main__":
    main()
