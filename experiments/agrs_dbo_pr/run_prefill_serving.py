#!/usr/bin/env python3
"""Measure prefill DBO with the ordinary decode CUDA graphs still enabled."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import httpx

MODEL_NAME = "deepseek-v2-lite-chat"
PREEMPTIONS = re.compile(r"^vllm:num_preemptions_total\{[^}]*\}\s+([0-9.eE+-]+)$", re.M)
METRICS = (
    "request_throughput",
    "total_token_throughput",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_e2el_ms",
    "p99_e2el_ms",
)


def wait_for_health(client, server, url, startup_log):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Server exited ({server.returncode}): {startup_log}")
        try:
            if client.get(f"{url}/health").status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Server did not start: {startup_log}")


def preemptions(client, url):
    response = client.get(f"{url}/metrics", timeout=10)
    response.raise_for_status()
    counts = PREEMPTIONS.findall(response.text)
    if not counts:
        raise RuntimeError("No preemption counters on /metrics")
    return sum(map(float, counts))


def stop_server(server):
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=5)


def run_arm(args, arm, env):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    url = f"http://127.0.0.1:{port}"
    cli = [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    command = cli + [
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        port,
        "--served-model-name",
        MODEL_NAME,
        "--api-server-count",
        "1",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "2",
        "--enable-expert-parallel",
        "--all2all-backend",
        "allgather_reducescatter",
        "--moe-backend",
        "triton",
        "--attention-backend",
        "TRITON_MLA",
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        str(args.max_batched),
        "--no-enable-prefix-caching",
        "--no-enable-extensible-kv-cache",
        "--kv-cache-memory-bytes",
        str(3 << 30),
        "--shutdown-timeout",
        "0",
        "--dbo-prefill-token-threshold",
        str(args.prefill_threshold),
        "--dbo-decode-token-threshold",
        "8192",
        "--cudagraph-metrics",
        "--compilation-config",
        json.dumps(
            {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "decode_cudagraph_capture_request_sizes": [1, 2, 4, 8, 16, 32, 64],
            }
        ),
    ]
    if arm == "on":
        command.append("--enable-dbo")
    if args.probe:
        env = {**env, "AGRS_DBO_BATCH_PROBE": "1", "VLLM_GPU_SYNC_CHECK": "error"}
    startup_log = args.output / f"{arm}.server.log"
    (args.output / f"{arm}.serve-command.json").write_text(json.dumps(command) + "\n")
    print(f"Starting {arm}: {startup_log}", flush=True)
    with startup_log.open("w") as log:
        server = subprocess.Popen(
            command,
            cwd=args.repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        with httpx.Client(trust_env=False, timeout=2.0) as client:
            wait_for_health(client, server, url, startup_log)
            print(f"{arm}: ready", flush=True)
            for concurrency in args.concurrency:
                for repeat in range(args.repeats):
                    name = f"{arm}.c{concurrency}.r{repeat + 1}"
                    bench = cli + [
                        "bench",
                        "serve",
                        "--backend",
                        "vllm",
                        "--base-url",
                        url,
                        "--model",
                        str(args.model),
                        "--served-model-name",
                        MODEL_NAME,
                        "--dataset-name",
                        "random",
                        "--trust-remote-code",
                        "--random-input-len",
                        str(args.input_len),
                        "--random-output-len",
                        str(args.output_len),
                        "--random-range-ratio",
                        "0.0",
                        "--ignore-eos",
                        "--temperature",
                        "0",
                        "--num-prompts",
                        str(args.requests),
                        "--seed",
                        str(42 + repeat),
                        "--request-rate",
                        "inf",
                        "--max-concurrency",
                        str(concurrency),
                        "--num-warmups",
                        str(concurrency),
                        "--disable-tqdm",
                        "--percentile-metrics",
                        "ttft,tpot,itl,e2el",
                        "--save-result",
                        "--save-detailed",
                        "--result-dir",
                        str(args.output),
                        "--result-filename",
                        f"{name}.json",
                    ]
                    count_before = preemptions(client, url)
                    log_path = args.output / f"{name}.bench.log"
                    with log_path.open("w") as log:
                        log.write(json.dumps(bench) + "\n")
                        log.flush()
                        subprocess.run(
                            bench,
                            cwd=args.repo,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=300,
                        )
                    result = json.loads((args.output / f"{name}.json").read_text())
                    preempted = preemptions(client, url) - count_before
                    if (
                        result["completed"] != args.requests
                        or result["failed"]
                        or set(result["output_lens"]) != {args.output_len}
                        or min(result["input_lens"]) < args.input_len * 0.99
                        or max(result["input_lens"]) > args.input_len * 1.01
                    ):
                        raise RuntimeError(
                            f"Incomplete or mismatched benchmark: {name}"
                        )
                    yield {
                        "arm": arm,
                        "concurrency": concurrency,
                        "repeat": repeat + 1,
                        "seed": 42 + repeat,
                        "completed": result["completed"],
                        "failed": result["failed"],
                        "total_input_tokens": result["total_input_tokens"],
                        "min_input_tokens": min(result["input_lens"]),
                        "max_input_tokens": max(result["input_lens"]),
                        "total_output_tokens": result["total_output_tokens"],
                        "preemptions_including_warmup": preempted,
                        **{key: result[key] for key in METRICS},
                    }
            # Let the existing vLLM stats logger flush the final graph rows.
            time.sleep(2.5)
    finally:
        stop_server(server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms", nargs="+", choices=["off", "on"], default=["off", "on"]
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=8)
    parser.add_argument("--max-batched", type=int, default=8192)
    parser.add_argument("--prefill-threshold", type=int, default=512)
    parser.add_argument(
        "--probe", action="store_true", help="For the separate batch probe patch only."
    )
    args = parser.parse_args()
    args.repo, args.model, args.output = (
        p.resolve() for p in (args.repo, args.model, args.output)
    )
    args.output.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(args.repo), os.getenv("PYTHONPATH")))
        ),
        "HF_HUB_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_USE_NCCL_SYMM_MEM": "0",
        "OMP_NUM_THREADS": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_LOG_STATS_INTERVAL": "2",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    rows = args.output / "trials.jsonl"
    with rows.open("w") as out:
        for arm in args.arms:
            for trial in run_arm(args, arm, env):
                line = json.dumps(trial)
                print(line, file=out, flush=True)
                print(line, flush=True)


if __name__ == "__main__":
    main()
