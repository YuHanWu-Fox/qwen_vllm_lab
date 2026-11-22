"""Measure and trace Granite DeepEP HT with a fixed batch and microbatch plan."""

import argparse
from functools import wraps
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import deep_ep
import numpy as np
import torch
import torch.distributed as dist
from torch.cuda import profiler

from vllm import EngineArgs
from vllm.distributed import get_world_group
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.worker.gpu import ubatch_utils
from vllm.v1.worker.ubatching import UBatchContext

RUNS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUNS / "dbo-perf-h100-base-214be07d6"))
sys.path.insert(0, str(RUNS / "deepep-h100-base-214be07d6"))

from bench_dbo import Coverage, arm_options, prefill_cases, run_case, runtime_config
from validate_deepep import begin_serial_microbatches


def add_range(owner, name, label):
    original = getattr(owner, name)

    @wraps(original)
    def profiled(*args, **kwargs):
        with torch.cuda.nvtx.range(label):
            return original(*args, **kwargs)

    setattr(owner, name, profiled)


def trace_case(engine, cpu_group, case, rank, trial, coverage):
    original = engine.step

    def step():
        with torch.cuda.nvtx.range(f"step/{rank}/{case.name}/{trial}"):
            return original()

    engine.step = step
    try:
        return run_case(engine, cpu_group, case, rank, trial, coverage)
    finally:
        engine.step = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", choices=("none", "align"), required=True)
    parser.add_argument("--variant", choices=("off", "on", "serial"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--only-8192", action="store_true")
    parser.add_argument("--gil-ms", type=float)
    parser.add_argument("--cpus", help="comma-separated base CPU numbers, one per rank")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    if args.gil_ms is not None:
        sys.setswitchinterval(args.gil_ms / 1000)
    if args.cpus:
        start = int(args.cpus.split(",")[rank])
        os.sched_setaffinity(0, set(range(start, start + 8)))
    if args.variant == "serial":
        ubatch_utils.UBatchRunner.begin_capturable_run = begin_serial_microbatches

    settings = SimpleNamespace(
        model="mamba2", backend="ht", cache=args.cache,
        dbo="off" if args.variant == "off" else "on",
        suite="large", max_model_len=33792, max_seqs=128, max_batched_tokens=32786,
        kv_gib=6, prefill_threshold=8192,
    )
    options = arm_options(settings)
    engine = LLMEngine.from_engine_args(EngineArgs(**options))
    cpu_group = get_world_group().cpu_group
    wanted = {"prefill_32x256"}
    if not args.only_8192:
        wanted.update(("prefill_16x256", "prefill_128x256"))
        if args.cache == "align":
            wanted.add("prefill_1x8192")
    cases = [case for case in prefill_cases((4096, 8192, 32768), 8192)
             if case.name in wanted]
    data = {
        "rank": rank, "options": options, "variant": args.variant,
        "runtime": runtime_config(engine), "switch_interval_ms": sys.getswitchinterval()*1000,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "cases": {},
    }

    for case in cases:
        rows = []
        for trial in range(14):
            trial_data = run_case(engine, cpu_group, case, rank, trial)
            assert len(trial_data["step_ms"]) == 1, (case, trial_data["step_ms"])
            if trial >= 6:
                rows.append(trial_data)
        data["cases"][case.name] = {"timed": rows}
        p50 = np.median([row["step_ms"][0] for row in rows])
        print(f"TIMED rank={rank} {case.name} P50={p50:.3f} ms", flush=True)

    coverage = Coverage()
    coverage.install()
    if args.trace:
        add_range(deep_ep.Buffer, "dispatch", "DeepEP/dispatch")
        add_range(deep_ep.Buffer, "get_dispatch_layout", "DeepEP/layout")
        add_range(deep_ep.Buffer, "combine", "DeepEP/combine")
        add_range(UBatchContext, "_cpu_yield", "DBO/cpu_yield")
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        profiler.start()
        dist.barrier(group=cpu_group)

    for case in cases:
        coverage.rows = []
        observed = []
        for trial in (14, 15, 16):
            r = trace_case(engine, cpu_group, case, rank, trial, coverage) if args.trace else (
                run_case(engine, cpu_group, case, rank, trial, coverage)
            )
            observed.append(r["step_ms"][0])
        assert len(coverage.rows) == 3, (case, coverage.rows)
        data["cases"][case.name]["traced_step_ms"] = observed
        data["cases"][case.name]["coverage"] = coverage.rows
        print(f"PATH rank={rank} {case.name} {coverage.rows[0]}", flush=True)

    dist.barrier(group=cpu_group)
    if args.trace:
        profiler.stop()
    path = args.output / f"rank{rank}.json"
    path.write_text(json.dumps(data, indent=2, allow_nan=False))
    dist.barrier(group=cpu_group)
    engine.engine_core.shutdown()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
