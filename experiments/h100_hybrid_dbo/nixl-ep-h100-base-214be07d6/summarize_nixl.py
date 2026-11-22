"""Compare two NIXL-EP arms without comparing different autoregressive histories."""
import argparse
from collections import Counter
import csv
import json
from math import isfinite
from pathlib import Path


def read_pair(path):
    return [json.loads((path / f"rank{rank}.json").read_text()) for rank in (0, 1)]


def percentile(values, quantile):
    if not values:
        return None
    return sorted(values)[round((len(values) - 1) * quantile)]


def stats(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values), "equal": sum(delta == 0 for delta in values),
        "p50": percentile(values, .50), "p95": percentile(values, .95),
        "p99": percentile(values, .99), "max": max(values),
        "over_0.05": sum(delta > .05 for delta in values),
        "over_0.10": sum(delta > .10 for delta in values),
        "over_0.25": sum(delta > .25 for delta in values),
        "over_1.00": sum(delta > 1 for delta in values),
    }


def greedy_margin(row):
    ordered = sorted(row.values(), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) >= 2 else None


def compare_request(reference, candidate, structural):
    errors = []
    sample_deltas = []
    prompt_deltas = []
    divergent = None
    ref_tokens = reference["tokens"]
    got_tokens = candidate["tokens"]
    if structural and reference["cached_tokens"] != candidate["cached_tokens"]:
        errors.append("A/B cached token counts differ")
    if len(ref_tokens) != len(got_tokens):
        errors.append("output lengths differ")
    ref_sample = reference["sample_logprobs"]
    got_sample = candidate["sample_logprobs"]
    if len(ref_sample) != len(ref_tokens) or len(got_sample) != len(got_tokens):
        errors.append("missing sample logprobs")
    for index, (want, got, ref, result) in enumerate(
        zip(ref_tokens, got_tokens, ref_sample, got_sample)
    ):
        if not ref or not result or str(want) not in ref or str(got) not in result:
            errors.append(f"missing selected sample logprob at position {index}")
            break
        if not all(isfinite(v) for v in (*ref.values(), *result.values())):
            errors.append(f"non-finite sample logprob at position {index}")
            break
        if want != got:
            divergent = {
                "position": index, "off_token": want, "on_token": got,
                "off_top2_margin": greedy_margin(ref),
                "on_top2_margin": greedy_margin(result),
                "in_each_others_top5": str(want) in result and str(got) in ref,
                "off_top5": ref, "on_top5": result,
            }
            if str(want) in result:
                sample_deltas.append(abs(ref[str(want)] - result[str(want)]))
            break
        sample_deltas.append(abs(ref[str(want)] - result[str(want)]))

    ref_prompt = reference["prompt_logprobs"]
    got_prompt = candidate["prompt_logprobs"]
    prompt_ids = reference.get("prompt_token_ids") or candidate.get("prompt_token_ids")
    if len(ref_prompt) != len(got_prompt):
        errors.append("prompt logprob lengths differ")
    if ref_prompt and (not prompt_ids or len(ref_prompt) != len(prompt_ids)):
        errors.append("missing prompt token IDs")
    for index, (want, got, token) in enumerate(zip(ref_prompt, got_prompt, prompt_ids or [])):
        if index == 0 and want is None and got is None:
            continue
        key = str(token)
        if not want or not got or key not in want or key not in got:
            errors.append(f"missing prompt logprob at position {index}")
            break
        if not isfinite(want[key]) or not isfinite(got[key]):
            errors.append(f"non-finite prompt logprob at position {index}")
            break
        prompt_deltas.append(abs(want[key] - got[key]))
    return sample_deltas, prompt_deltas, divergent, errors


def forward_rows(case):
    return [row for row in case["steps"] if "mode" in row]


def summarize_paths(cases):
    shapes = Counter()
    paths = Counter()
    numbers = Counter()
    for case in cases:
        for row in forward_rows(case):
            mode = row["mode"]
            count = row["ubatches"]
            shapes[f"{mode}/{count}"] += 1
            numbers["runtime_replay"] += "replay" in row
            numbers["runtime_microbatch_replay"] += (
                "replay" in row and count == 2 and not row["dummy"]
            )
            numbers["dummy_forward"] += row["dummy"]
            numbers["checkpoint_veto"] += row["checkpoint"]
            slices = row.get("slices", [])
            if slices:
                numbers["empty_second"] += slices[1]["valid_tokens"] == 0
                numbers["one_valid_in_second"] += slices[1]["valid_tokens"] == 1
                split = slices[0]["token_range"][1]
                requests = row.get("requests", [])
                crosses = [
                    r for r in requests
                    if r["prefilling"] and r["query_start"] < split < r["query_start"] + r["q"]
                ]
                numbers["split_inside_prefill"] += bool(crosses)
                numbers["mixed_split_inside_prefill"] += bool(crosses) and any(
                    not r["prefilling"] for r in requests
                )
            paths.update(row["nixl_ep_calls"])
    return {"shapes": dict(shapes), "coverage": dict(numbers), "nixl_ep_calls": dict(paths)}


def review_case(name, off, on, block, runtime):
    errors = []
    ref_rows = [forward_rows(case) for case in off]
    rows = [forward_rows(case) for case in on]
    flat = [row for rank in rows for row in rank]
    evidence = summarize_paths(on)
    covered = evidence["coverage"]
    is_cg = runtime["effective_cg_mode"] == "FULL_DECODE_ONLY"
    if any(case.get("rank_mask") != [0, 0] for case in (*off, *on)):
        errors.append("EP rank masked or mask unreadable at end of case")

    if any(row["ubatches"] != 1 for rank in ref_rows for row in rank):
        errors.append("DBO-off issued a split")
    for left, right in (ref_rows, rows):
        if len(left) != len(right):
            errors.append("DP ranks entered different numbers of forwards")
        for first, second in zip(left, right):
            if any(first[key] != second[key] for key in ("step", "mode", "ubatches")):
                errors.append("DP ranks disagreed on the step, execution mode, or split count")
            if [p["token_range"] for p in first.get("slices", [])] != [
                p["token_range"] for p in second.get("slices", [])
            ]:
                errors.append("DP ranks disagreed on the microbatch token ranges")
        for row in (*left, *right):
            if row["mode"] == "FULL" and not row.get("replay"):
                errors.append("FULL descriptor did not replay a graph")
            if row["mode"] == "NONE":
                calls = row["nixl_ep_calls"]
                suffix = row["ubatches"]
                if not any(name.endswith("dispatch") and n >= suffix for name, n in calls.items()):
                    errors.append("no NIXL-EP dispatch observed on an eager forward")
                if not any(name.endswith("combine") and n >= suffix for name, n in calls.items()):
                    errors.append("no NIXL-EP combine observed on an eager forward")
    if name.startswith("cross_prefill") and not covered.get("split_inside_prefill"):
        errors.append("prefill did not cross the microbatch boundary")
    if name == "mixed" and not covered.get("mixed_split_inside_prefill"):
        errors.append("mixed batch did not split inside the prefill")
    if name == "prefill_padding_rank0":
        if not any(
            row["mode"] == "NONE" and len(row.get("slices", [])) == 2
            and row["slices"][1]["valid_tokens"] == 0 and not row["dummy"]
            for row in rows[0]
        ):
            errors.append("rank 0 did not execute an eager all-padding second microbatch")
    if name in ("empty_rank0", "empty_rank1", "both_empty"):
        empty_ranks = (0, 1) if name == "both_empty" else (int(name[-1]),)
        if any(on[rank]["outputs"] for rank in empty_ranks):
            errors.append("an empty rank emitted outputs")
        if any(row["ubatches"] != 1 for row in flat):
            errors.append("empty rank did not veto splitting")
        if any(not any(row["dummy"] for row in rows[rank]) for rank in empty_ranks):
            errors.append("no dummy forward observed on an empty rank")
    if name in ("prefix_hit", "checkpoint_prefix"):
        for rank in (0, 1) if name == "prefix_hit" else (0,):
            for case in (off[rank], on[rank]):
                if case["outputs"]["target"]["cached_tokens"] != block:
                    errors.append(f"rank {rank} expected {block} cache-hit tokens")
    if name == "prefix_cold" and any(
        case["outputs"]["target"]["cached_tokens"] for case in (*off, *on)
    ):
        errors.append("skip_reading_prefix_cache still produced a hit")
    if name == "checkpoint_veto" and runtime["scheduler_mamba_internal_checkpoints"]:
        veto_steps = [row["step"] for row in rows[0] if row["checkpoint"]]
        if not veto_steps:
            errors.append("no internal checkpoint scheduled")
        if any(
            row["ubatches"] != 1 for row in flat if row["step"] in veto_steps
        ):
            errors.append("checkpoint did not veto microbatching on both ranks")
        if any(
            not any(row["step"] == step and row["real_tokens"] >= 32 for row in rows[1])
            for step in veto_steps
        ):
            errors.append("peer workload was below prefill threshold at checkpoint veto")

    if name.startswith(("decode_", "default_decode_")):
        decode_rows = [
            row for row in flat if row.get("requests")
            and not any(req["prefilling"] for req in row["requests"])
        ]
        if not decode_rows:
            errors.append("no uniform decode forward")
        unsplit = name in ("decode_1_1", "default_decode_31_32")
        if not all(row["ubatches"] == (1 if unsplit else 2) for row in decode_rows):
            errors.append("unexpected decode microbatch count")
        if is_cg and not name.startswith("default_"):
            expected_mode = "NONE" if name == "decode_3_8" else "FULL"
            if not all(
                row["mode"] == expected_mode
                and (expected_mode != "FULL" or "replay" in row)
                for row in decode_rows
            ):
                errors.append(f"decode did not execute expected {expected_mode} path")
            for test_name, expected_valid in (("decode_4_8", 0), ("decode_5_8", 1)):
                if name == test_name and not any(
                    row.get("replay") and row.get("slices")
                    and row["slices"][1]["valid_tokens"] == expected_valid
                    for row in rows[0]
                ):
                    errors.append(f"FULL second microbatch lacked the {expected_valid}-token case")

    if name.startswith("default_prefill_"):
        expected = 1 if "511_512" in name else 2
        prefill = [row for row in flat if row.get("requests")
                   and any(req["prefilling"] for req in row["requests"])]
        if not prefill or not all(row["ubatches"] == expected for row in prefill):
            errors.append("prefill boundary at the default threshold was not observed")
    return evidence, list(dict.fromkeys(errors))


def compare_arms(off, on):
    all_errors = []
    variants = []
    sample_total, prompt_total = [], []
    result = {"cases": {}, "runtime": on[0]["runtime"]}
    for left, right in zip(off, on):
        config = {**left["options"], "enable_dbo": True}
        if config != right["options"]:
            all_errors.append("A/B launch configurations differ beyond enable_dbo")
        if not left.get("complete") or not right.get("complete"):
            all_errors.append("an A/B arm did not finish")
    if not all(rank["runtime"]["manager"] == "NixlEPAll2AllManager" for rank in (*off, *on)):
        all_errors.append("NIXL-EP all-to-all manager was not active on both ranks")
    if not all(
        rank["runtime"]["effective_cg_mode"] == "FULL_DECODE_ONLY"
        and any(graph["ubatches"] == 2 for graph in rank["runtime"]["graphs"])
        and rank["runtime"]["capture_nixl_ep_calls"].get("dispatch", 0)
        and rank["runtime"]["capture_nixl_ep_calls"].get("combine", 0)
        for rank in on
    ):
        all_errors.append("no NIXL-EP microbatch graph capture on both ranks")
    case_names = on[0]["cases"]
    if any(case_names.keys() != rank["cases"].keys() for rank in (*off, *on)):
        all_errors.append("A/B ran different sets of cases")
    for name in case_names:
        case_off = [rank["cases"][name] for rank in off]
        case_on = [rank["cases"][name] for rank in on]
        runtime = on[0]["runtime"]
        path, errors = review_case(name, case_off, case_on, runtime["block_size"], runtime)
        scores, prompts, differences = [], [], []
        request_count = 0
        greedy_count = 0
        for rank, (left, right) in enumerate(zip(case_off, case_on)):
            if left["inputs"] != right["inputs"]:
                errors.append(f"rank {rank} inputs differ")
            if left["outputs"].keys() != right["outputs"].keys():
                errors.append(f"rank {rank} completed requests differ")
            for label, expected in left["outputs"].items():
                request_count += 1
                greedy_count += left["inputs"][label]["forced"] is None
                actual = right["outputs"][label]
                sample, prompt, mismatch, issues = compare_request(expected, actual, True)
                scores.extend(sample)
                prompts.extend(prompt)
                errors.extend(f"rank {rank} {label}: {issue}" for issue in issues)
                if mismatch:
                    differences.append({"rank": rank, "request": label, **mismatch})
        all_errors.extend(f"{name}: {error}" for error in errors)
        variants.extend({"case": name, **difference} for difference in differences)
        sample_total.extend(scores)
        prompt_total.extend(prompts)
        result["cases"][name] = {
            "requests_per_arm": request_count, "greedy_requests": greedy_count,
            "greedy_differences": differences, "sample_logprob": stats(scores),
            "prompt_logprob": stats(prompts), "dbo_on": path,
            "dbo_off": summarize_paths(case_off), "coverage_errors": errors,
        }
    if "prefix_hit" in case_names:
        checks = {}
        for label, arm in (("off", off), ("on", on)):
            scores = []
            for rank in arm:
                hit_output = rank["cases"]["prefix_hit"]["outputs"]["target"]
                cold_output = rank["cases"]["prefix_cold"]["outputs"]["target"]
                sample, _, mismatch, errors = compare_request(cold_output, hit_output, False)
                scores.extend(sample)
                if mismatch or errors:
                    all_errors.append(f"{label}: cached and uncached check diverged structurally")
            checks[label] = stats(scores)
        result["prefix_hit_vs_cold"] = checks
    result.update(
        requests_per_arm=sum(row["requests_per_arm"] for row in result["cases"].values()),
        greedy_requests=sum(row["greedy_requests"] for row in result["cases"].values()),
        greedy_differing_requests=len(variants),
        greedy_differences_outside_other_top5=sum(
            not event["in_each_others_top5"] for event in variants
        ),
        sample_logprob=stats(sample_total),
        prompt_logprob=stats(prompt_total),
        coverage_errors=all_errors,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    result = {}
    for name in ("mamba2-nixl-none", "mamba2-nixl-align", "gdn-nixl-none", "gdn-nixl-align"):
        root = args.run_root / "arms"
        off_path, on_path = root / f"{name}-off", root / f"{name}-on"
        try:
            off, on = read_pair(off_path), read_pair(on_path)
        except FileNotFoundError:
            continue
        if not all(rank.get("complete") for rank in (*off, *on)):
            continue
        result[name] = compare_arms(off, on)
    with (args.run_root / "summary.json").open("w") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (args.run_root / "case-results.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(("configuration", "case", "requests_per_arm", "greedy_mismatches",
                         "sample_delta_p99", "sample_delta_max", "prompt_delta_p99",
                         "prompt_delta_max", "on_paths", "coverage_errors"))
        for name, arm in result.items():
            for case, row in arm["cases"].items():
                writer.writerow((
                    name, case, row["requests_per_arm"], len(row["greedy_differences"]),
                    row["sample_logprob"].get("p99"), row["sample_logprob"].get("max"),
                    row["prompt_logprob"].get("p99"), row["prompt_logprob"].get("max"),
                    row["dbo_on"]["shapes"], row["coverage_errors"],
                ))
            print(name, "requests/arm", arm["requests_per_arm"],
                  "greedy differing", arm["greedy_differing_requests"],
                  "sample", arm["sample_logprob"], "prompt", arm["prompt_logprob"],
                  "coverage errors", arm["coverage_errors"])


if __name__ == "__main__":
    main()
