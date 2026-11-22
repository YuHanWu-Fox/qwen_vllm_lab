"""Measure real-weight DP2 + EP DBO workloads through offline LLMEngine.step()."""

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import wraps
from importlib import metadata
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
import torch.distributed as dist

import vllm
from vllm import EngineArgs, SamplingParams
from vllm.distributed import get_ep_group, get_world_group
from vllm.inputs import tokens_input
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager


MODELS_DIR = Path(os.environ.get("DBO_MODELS_DIR", "/workspace/models"))
MODELS = {
    "gdn": str(MODELS_DIR / "Qwen3-Next-80B-A3B-Instruct-int4-AutoRound"),
    "mamba2": str(MODELS_DIR / "granite-4.0-h-tiny"),
}
BACKENDS = {
    "ht": "deepep_high_throughput",
    "ll": "deepep_low_latency",
    "nixl": "nixl_ep",
}


@dataclass(frozen=True)
class Wave:
    step: int
    requests: int
    prompt_len: int
    output_len: int


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    threshold: int
    waves: tuple[Wave, ...]


def prefill_cases(totals, threshold):
    for total in totals:
        if total % 256 == 0 and total > 256:
            width = total // 256
            yield Case(f"prefill_{width}x256", "prefill", threshold, (
                Wave(0, width, 256, 1),
            ))
        yield Case(f"prefill_1x{total}", "prefill", threshold, (Wave(0, 1, total, 1),))


def cases_for(args):
    if args.suite == "large":
        totals = (
            list(map(int, args.prefill_totals.split(",")))
            if args.prefill_totals else [4096, 8192, 16384, 32768, 32786]
        )
        return list(prefill_cases(
            [n for n in totals if n <= args.max_batched_tokens], args.prefill_threshold,
        ))
    if args.suite == "probe":
        return list(prefill_cases([args.probe_tokens], args.prefill_threshold))
    totals = sorted({
        n for n in (256, 512, 1024, 2048, 4096, args.max_batched_tokens)
        if n <= args.max_batched_tokens
    })
    cases = list(prefill_cases(totals, args.prefill_threshold))
    if args.model == "gdn" and args.max_model_len > 32786:
        for n in (16384, 32786):
            cases.append(Case(
                f"prefill_chunked_1x{n}", "prefill", args.prefill_threshold,
                (Wave(0, 1, n, 1),),
            ))
    cases += [
        Case(f"decode_{count}x32", "decode", args.prefill_threshold, (
            Wave(0, count, 32, 25),
        ))
        for count in (16, 32, 64)
    ]
    cases += [
        Case(
            f"mixed_32decode_2x{prompt_len}",
            "mixed",
            args.prefill_threshold,
            (Wave(0, 32, 32, 25),)
            + tuple(Wave(step, 2, prompt_len, 8) for step in (2, 6, 10, 14)),
        )
        for prompt_len in (320, 704)
    ]
    return cases


def version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


class Coverage:
    """Record runtime dispatch and graph replay in a separate, untimed pass."""

    def __init__(self):
        self.rows = None
        self.phase = None

    def install(self):
        execute = model_runner.GPUModelRunner.execute_model
        dispatch = model_runner.dispatch_cg_and_sync_dp
        replay = ModelCudaGraphManager.run_fullgraph

        @wraps(execute)
        def capture_scheduler(runner, scheduled, *args, **kwargs):
            self.checkpoint = bool(scheduled.has_mamba_prefill_checkpoint)
            self.dummy = bool(kwargs.get("dummy_run", False))
            return execute(runner, scheduled, *args, **kwargs)

        @wraps(dispatch)
        def capture_dispatch(manager, num_reqs, num_tokens, *args, **kwargs):
            desc, dp_state = dispatch(manager, num_reqs, num_tokens, *args, **kwargs)
            if self.rows is not None and desc.num_tokens:
                self.rows.append({
                    "phase": self.phase,
                    "requests": num_reqs,
                    "tokens": 0 if self.dummy else num_tokens,
                    "padded_tokens": desc.num_tokens,
                    "ubatches": desc.num_ubatches,
                    "mode": desc.cg_mode.name,
                    "replay": False,
                    "checkpoint": self.checkpoint,
                    "dummy": self.dummy,
                })
            return desc, dp_state

        @wraps(replay)
        def capture_replay(manager, desc):
            result = replay(manager, desc)
            if self.rows:
                self.rows[-1]["replay"] = True
            return result

        model_runner.GPUModelRunner.execute_model = capture_scheduler
        model_runner.dispatch_cg_and_sync_dp = capture_dispatch
        ModelCudaGraphManager.run_fullgraph = capture_replay


def build_requests(case, rank, trial):
    # An identical token stream for each DBO configuration at a given rank and trial.
    rng = np.random.default_rng([2701 + rank, trial])
    waves = {}
    for wave in case.waves:
        waves[wave.step] = [
            (
                tokens_input(rng.integers(1024, 20000, wave.prompt_len).tolist()),
                wave.output_len,
            )
            for _ in range(wave.requests)
        ]
    return waves


def run_case(engine, cpu_group, case, rank, trial, coverage=None):
    waves = build_requests(case, rank, trial)
    requested = {}

    def submit(step, arrival_ns):
        for index, (prompt, output_len) in enumerate(waves[step]):
            request_id = f"{case.name}-{trial}-{step}-{index}"
            options = SamplingParams(
                temperature=0.0,
                max_tokens=output_len,
                ignore_eos=True,
                detokenize=False,
                skip_reading_prefix_cache=True,
                output_kind=RequestOutputKind.DELTA,
            )
            engine.add_request(request_id, prompt, options)
            requested[request_id] = {
                "wave": step, "arrival_ns": arrival_ns, "expected": output_len,
                "produced": 0, "first_ns": None, "last_ns": None, "finished": False,
            }

    submit(0, None)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)
    start = perf_counter_ns()
    for request in requested.values():
        request["arrival_ns"] = start
    step = 0
    step_ms = []
    while engine.has_unfinished_requests():
        if step in waves and step > 0:
            submit(step, perf_counter_ns())
        if coverage is not None:
            if case.kind == "prefill" or step == 0 or step in waves:
                coverage.phase = "prefill" if case.kind != "mixed" else "mixed_prefill"
            else:
                coverage.phase = "decode"
        step_start = perf_counter_ns()
        outputs = engine.step()
        now = perf_counter_ns()
        step_ms.append((now - step_start) / 1e6)
        for output in outputs:
            request = requested[output.request_id]
            completion, = output.outputs
            if completion.token_ids:
                if request["first_ns"] is None:
                    request["first_ns"] = now
                request["last_ns"] = now
                request["produced"] += len(completion.token_ids)
            if output.finished:
                assert completion.finish_reason == "length", completion.finish_reason
                request["finished"] = True
        step += 1
        assert step < 256, (case.name, "excessive step count")

    torch.cuda.synchronize()
    stop = perf_counter_ns()
    assert len(requested) == sum(wave.requests for wave in case.waves), case.name
    for request in requested.values():
        assert request["finished"] and request["produced"] == request["expected"], request
    return {
        "start_ns": start,
        "stop_ns": stop,
        "step_ms": step_ms,
        "requests": list(requested.values()),
    }


def arm_options(args):
    return dict(
        model=MODELS[args.model],
        dtype="bfloat16",
        skip_tokenizer_init=True,
        tensor_parallel_size=1,
        data_parallel_size=2,
        distributed_executor_backend="external_launcher",
        enable_expert_parallel=True,
        all2all_backend=BACKENDS[args.backend],
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        kv_cache_memory_bytes=args.kv_gib * 1024**3,
        enable_extensible_kv_cache=False,
        block_size=256 if args.model == "gdn" else 512,
        enable_prefix_caching=args.cache == "align",
        mamba_cache_mode=args.cache,
        enable_dbo=args.dbo == "on",
        async_scheduling=False,
        dbo_prefill_token_threshold=args.prefill_threshold,
        dbo_decode_token_threshold=32,
        enforce_eager=args.suite != "regular" or args.backend == "ht",
        disable_log_stats=True,
        seed=2701,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "decode_cudagraph_capture_request_sizes": [16, 32, 64],
        } if args.suite == "regular" and args.backend != "ht" else {"mode": 0, "cudagraph_mode": "NONE"},
    )


def runtime_config(engine):
    core = engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.model_runner
    cfg = engine.vllm_config
    return {
        "source": str(vllm.__file__),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "deep_ep": version("deep_ep"),
        "nixl": version("nixl"),
        "flashinfer_cubin": version("flashinfer-cubin"),
        "flashinfer_jit_cache": version("flashinfer-jit-cache"),
        "all2all_manager": type(
            get_ep_group().device_communicator.all2all_manager
        ).__name__,
        "block_size": cfg.cache_config.block_size,
        "cache_mode": cfg.cache_config.mamba_cache_mode,
        "mode": cfg.compilation_config.cudagraph_mode.name,
        "has_ubatch_runner": runner.ubatch_runner is not None,
        "mamba_groups": [
            {
                "type": group.kv_cache_spec.mamba_type.name,
                "block_size": group.kv_cache_spec.block_size,
                "checkpoints": group.kv_cache_spec.num_prefill_checkpoint_blocks,
                "layers": len(group.layer_names),
            }
            for group in runner.kv_cache_config.kv_cache_groups
            if isinstance(group.kv_cache_spec, MambaSpec)
        ],
    }


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--cache", choices=("none", "align"), required=True)
    parser.add_argument("--dbo", choices=("off", "on"), required=True)
    parser.add_argument("--suite", choices=("large", "regular", "probe"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batched-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--prefill-threshold", type=int, required=True)
    parser.add_argument("--max-seqs", type=int, default=128)
    parser.add_argument("--kv-gib", type=int, default=6)
    parser.add_argument("--probe-tokens", type=int, default=8192)
    parser.add_argument("--prefill-totals", help="comma-separated tokens/rank for large suite")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--cases", help="restrict to these comma-separated case names")
    args = parser.parse_args()
    full_repetitions = args.suite == "large" or (
        args.model == "mamba2" and args.backend == "ll" and args.cache == "none"
    )
    if args.warmup is None:
        args.warmup = 6 if full_repetitions else 3
    if args.trials is None:
        args.trials = 16 if full_repetitions else 8
    if args.warmup < 0 or args.trials < 1:
        parser.error("warmup must be nonnegative and trials must be positive")
    rank = int(os.environ["RANK"])
    cases = cases_for(args)
    if args.cases:
        selected = set(args.cases.split(","))
        cases = [case for case in cases if case.name in selected]
        assert {case.name for case in cases} == selected, selected
    path = args.output / f"rank{rank}.json"
    options = arm_options(args)
    results = {
        "rank": rank, "options": options, "complete": False, "cases": {},
        "suite": args.suite, "warmup": args.warmup, "trials": args.trials,
    }
    save(path, results)
    print(f"ENGINE_START rank={rank} {options}", flush=True)
    engine = LLMEngine.from_engine_args(EngineArgs(**options))
    cpu_group = get_world_group().cpu_group
    runtime = runtime_config(engine)
    results["runtime"] = runtime
    save(path, results)
    assert runtime["mamba_groups"] and runtime["has_ubatch_runner"] == options["enable_dbo"]
    assert runtime["cache_mode"] == args.cache
    print(f"ENGINE_READY rank={rank} {runtime}", flush=True)
    for case in cases:
        engine.vllm_config.parallel_config.dbo_prefill_token_threshold = case.threshold
        print(f"CASE_START rank={rank} {case.name}", flush=True)
        row = {
            "kind": case.kind, "threshold": case.threshold, "trials": [],
            "prompt_tokens": sum(w.requests * w.prompt_len for w in case.waves),
            "output_tokens": sum(w.requests * w.output_len for w in case.waves),
        }
        results["cases"][case.name] = row
        for trial in range(args.warmup + args.trials):
            stats = run_case(engine, cpu_group, case, rank, trial)
            if trial >= args.warmup:
                row["trials"].append(stats)
        save(path, results)
        time_ms = np.median([
            (trial["stop_ns"] - trial["start_ns"]) / 1e6 for trial in row["trials"]
        ])
        print(f"CASE_END rank={rank} {case.name} median_ms={time_ms:.3f}", flush=True)

    coverage = Coverage()
    coverage.install()
    for case in cases:
        coverage.rows = []
        engine.vllm_config.parallel_config.dbo_prefill_token_threshold = case.threshold
        run_case(engine, cpu_group, case, rank, args.warmup, coverage)
        results["cases"][case.name]["coverage"] = coverage.rows
        shapes = Counter(f"{r['mode']}/{r['ubatches']}" for r in coverage.rows)
        print(f"COVERAGE rank={rank} {case.name} {dict(shapes)}", flush=True)
    results["complete"] = True
    save(path, results)
    dist.barrier(group=cpu_group)
    engine.engine_core.shutdown()
    if dist.is_initialized():
        dist.destroy_process_group()
    print(f"ARM_COMPLETE rank={rank}", flush=True)


if __name__ == "__main__":
    main()
