"""Verify the paired observations and summarize latency, throughput and runtime paths."""

import argparse
from collections import Counter
import csv
import json
import re
from pathlib import Path

import numpy as np


def percentiles(samples):
    if not samples:
        return {"p50": None, "p95": None}
    p50, p95 = np.percentile(samples, [50, 95])
    return {"p50": float(p50), "p95": float(p95)}


def load_arm(path):
    files = [path / f"rank{rank}.json" for rank in range(2)]
    if not all(file.exists() for file in files):
        return None
    ranks = [json.loads(file.read_text()) for file in files]
    if not all(rank["complete"] for rank in ranks):
        return None
    assert ranks[0]["options"] == ranks[1]["options"], path
    assert ranks[0]["cases"].keys() == ranks[1]["cases"].keys(), path
    return ranks


def step_coverage(case, case_pair, width, output_tokens):
    expected = node_prompt_tokens(case_pair) + output_tokens - width
    rank_steps = [row["coverage"] for row in case_pair]
    for steps in rank_steps:
        assert sum(row["tokens"] for row in steps) == expected, (case, steps)
    counts = Counter()
    signatures = []
    for batch in zip(*rank_steps, strict=True):
        counts["steps"] += 1
        n0, n1 = batch
        assert n0["ubatches"] == n1["ubatches"], (case, batch)
        assert n0["mode"] == n1["mode"], (case, batch)
        assert n0["replay"] == n1["replay"], (case, batch)
        assert n0["replay"] == (n0["mode"] == "FULL"), (case, batch)
        counts["split_steps"] += n0["ubatches"] > 1
        counts["checkpoint_steps"] += any(row["checkpoint"] for row in batch)
        counts["full_steps"] += n0["replay"]
        counts["split_full_steps"] += n0["replay"] and n0["ubatches"] > 1
        counts["dummy_rank_steps"] += sum(row["dummy"] for row in batch)
        counts["pad_tokens"] += sum(row["padded_tokens"] - row["tokens"] for row in batch)
        signatures.append({
            "real_tokens": [row["tokens"] for row in batch],
            "padded_tokens": [row["padded_tokens"] for row in batch],
            "num_requests": [row["requests"] for row in batch],
            "ubatches": n0["ubatches"],
            "mode": n0["mode"],
            "checkpoint": [row["checkpoint"] for row in batch],
        })
    return dict(counts), signatures


def node_prompt_tokens(pair):
    assert pair[0]["prompt_tokens"] == pair[1]["prompt_tokens"]
    return pair[0]["prompt_tokens"]


def summarize_case(name, pair):
    assert pair[0]["kind"] == pair[1]["kind"]
    assert pair[0]["output_tokens"] == pair[1]["output_tokens"]
    kind = pair[0]["kind"]
    prompt_tokens = node_prompt_tokens(pair)
    output_tokens = pair[0]["output_tokens"]
    wall_ms, all_first_ms, ttft_ms, late_ttft_ms, tpot_ms, throughput = (
        [], [], [], [], [], [],
    )
    widths = set()
    for left, right in zip(pair[0]["trials"], pair[1]["trials"], strict=True):
        start = min(left["start_ns"], right["start_ns"])
        stop = max(left["stop_ns"], right["stop_ns"])
        wall_ms.append((stop - start) / 1e6)
        reqs = left["requests"] + right["requests"]
        assert len(left["requests"]) == len(right["requests"]), name
        assert all(req["finished"] and req["expected"] == req["produced"] for req in reqs)
        assert sum(req["produced"] for req in reqs) == output_tokens * 2, name
        widths.add(len(left["requests"]))
        first = max(req["first_ns"] for req in reqs if req["wave"] == 0)
        last = max(req["last_ns"] for req in reqs)
        all_first_ms.append((first - start) / 1e6)
        if kind == "prefill":
            throughput.append(prompt_tokens * 2e9 / (first - start))
        elif kind == "decode":
            first_observed = min(req["first_ns"] for req in reqs)
            decode_tokens = output_tokens * 2 - len(reqs)
            throughput.append(decode_tokens * 1e9 / (last - first_observed))
        else:
            throughput.append(output_tokens * 2e9 / (last - start))
        for req in reqs:
            observed_ttft = (req["first_ns"] - req["arrival_ns"]) / 1e6
            ttft_ms.append(observed_ttft)
            if req["wave"]:
                late_ttft_ms.append(observed_ttft)
            if req["produced"] > 1:
                tpot_ms.append(
                    (req["last_ns"] - req["first_ns"]) / (req["produced"] - 1) / 1e6
                )
    width, = widths
    coverage, schedule = step_coverage(name, pair, width, output_tokens)
    return dict(
        kind=kind,
        num_trials=len(wall_ms),
        requests_per_rank=width,
        prompt_tokens_per_rank=prompt_tokens,
        output_tokens_per_rank=output_tokens,
        wall_ms=percentiles(wall_ms),
        all_first_ms=percentiles(all_first_ms),
        ttft_ms=percentiles(ttft_ms),
        late_ttft_ms=percentiles(late_ttft_ms),
        tpot_ms=percentiles(tpot_ms),
        throughput=percentiles(throughput),
        throughput_p05=float(np.percentile(throughput, 5)),
        coverage=coverage,
        schedule=schedule,
    )


def summarize_pair(key, paths, arms):
    off, on = arms
    base_options = off[0]["options"].copy()
    dbo_options = on[0]["options"].copy()
    assert not base_options.pop("enable_dbo") and dbo_options.pop("enable_dbo"), key
    assert base_options == dbo_options, key
    assert off[0]["warmup"] == on[0]["warmup"], key
    assert off[0]["trials"] == on[0]["trials"], key
    assert off[0]["cases"].keys() == on[0]["cases"].keys(), key
    values = {}
    for case in off[0]["cases"]:
        pair = [summarize_case(case, [rank["cases"][case] for rank in arm]) for arm in arms]
        assert pair[0]["coverage"]["split_steps"] == 0, key
        for field in ("kind", "num_trials", "prompt_tokens_per_rank", "output_tokens_per_rank"):
            assert pair[0][field] == pair[1][field], (key, case, field)
        off_metrics, on_metrics = pair
        deltas = {}
        for metric in ("all_first_ms", "ttft_ms", "late_ttft_ms", "tpot_ms", "wall_ms", "throughput"):
            for q in ("p50", "p95"):
                a, b = (arm[metric][q] for arm in pair)
                if a:
                    deltas[f"{metric}_{q}_pct"] = (b / a - 1) * 100
        values[case] = {"off": off_metrics, "on": on_metrics, "deltas": deltas}
    return {
        "arms": [path.name for path in paths],
        "options": base_options,
        "runtime": [arm[0]["runtime"] for arm in arms],
        "warmup": off[0]["warmup"],
        "trials": off[0]["trials"],
        "cases": values,
    }


def as_csv_rows(results):
    for key, group in results.items():
        for name, pair in group["cases"].items():
            a, b = pair["off"], pair["on"]
            row = {
                "group": key,
                "case": name,
                "kind": a["kind"],
                "scheduler_budget_per_rank": group["options"]["max_num_batched_tokens"],
                "prefill_threshold": group["options"]["dbo_prefill_token_threshold"],
                "prompt_tokens_per_rank": a["prompt_tokens_per_rank"],
                "requests_per_rank": a["requests_per_rank"],
                "trials": a["num_trials"],
            }
            for arm, data in (("off", a), ("on", b)):
                for metric in ("all_first_ms", "ttft_ms", "late_ttft_ms", "tpot_ms", "wall_ms", "throughput"):
                    for q in ("p50", "p95"):
                        val = data[metric][q]
                        row[f"{arm}_{metric}_{q}"] = round(val, 3) if val is not None else None
                for count in ("steps", "split_steps", "checkpoint_steps", "full_steps", "split_full_steps", "pad_tokens"):
                    row[f"{arm}_{count}"] = data["coverage"][count]
            for metric in ("all_first_ms", "ttft_ms", "late_ttft_ms", "tpot_ms", "wall_ms", "throughput"):
                for q in ("p50", "p95"):
                    val = pair["deltas"].get(f"{metric}_{q}_pct")
                    row[f"delta_{metric}_{q}_pct"] = round(val, 2) if val is not None else None
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    results = {}
    for off in sorted((args.directory / "arms").glob("*-off*")):
        group, tag = re.split("-off", off.name, maxsplit=1)
        on = off.with_name(f"{group}-on{tag}")
        arms = [load_arm(path) for path in (off, on)]
        if all(arms):
            results[group + tag] = summarize_pair(group + tag, [off, on], arms)
    if not results:
        raise SystemExit("No completed pairs")
    (args.directory / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    rows = list(as_csv_rows(results))
    with (args.directory / "case-results.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(results)} completed pairs, {len(rows)} paired workloads")


if __name__ == "__main__":
    main()
