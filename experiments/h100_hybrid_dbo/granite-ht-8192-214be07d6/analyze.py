"""Summarize host ranges, CUDA kernel work and GPU gaps for traced engine steps."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3
from statistics import median

ROOT = Path(__file__).parent
MILLISECOND = 1_000_000


def interval_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def elapsed_ms(intervals):
    return sum(end - start for start, end in intervals) / MILLISECOND


def kernel_family(name):
    if "deep_ep::" in name:
        if "::notify_dispatch" in name or "::cached_notify_" in name:
            return "deep_ep_notify"
        if "::dispatch<" in name:
            return "deep_ep_dispatch"
        if "::combine<" in name:
            return "deep_ep_combine"
        return "deep_ep_layout"
    if name == "fused_moe_kernel":
        return "moe_gemm"
    if "flash" in name.lower() and ("attn" in name.lower() or "attention" in name.lower()):
        return "attention"
    if any(s in name for s in ("_causal_conv", "_chunk", "_ssm", "_state_passing")):
        return "mamba"
    return "other"


def profile(sqlite):
    db = sqlite3.connect(f"file:{sqlite}?mode=ro", uri=True)
    steps = db.execute("""
        select start, end, globalTid, text
        from NVTX_EVENTS where text like 'step/%'
        order by start
    """).fetchall()
    assert steps, sqlite
    result = defaultdict(list)
    for start, end, tid, label in steps:
        _, rank, case, trial = label.split("/")
        rank = int(rank)
        pid = tid >> 24
        bounds = (rank, start, end)
        kernels = db.execute("""
            select k.start, k.end, k.streamId, n.value, s.value, k.correlationId
            from CUPTI_ACTIVITY_KIND_KERNEL k
            left join StringIds n on n.id=k.demangledName
            left join StringIds s on s.id=k.shortName
            where deviceId=? and k.start>=? and k.end<=?
            order by k.start
        """, bounds).fetchall()
        assert kernels, (sqlite, label)
        memory_ops = []
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            memory_ops += db.execute(
                f"select start, end from {table} where deviceId=? and start>=? and end<=?",
                bounds,
            ).fetchall()
        spans = interval_union([(a, b) for a, b, *_ in kernels] + memory_ops)
        ordered = [(start, start), *spans, (end, end)]
        gaps = [(a[1], b[0]) for a,b in zip(ordered, ordered[1:]) if a[1] < b[0]]
        ranges = db.execute("""
            select start, end, text from NVTX_EVENTS
            where (globalTid >> 24)=? and start>=? and end<=?
              and (text like 'DeepEP/%' or text like 'DBO/%')
        """, (pid, start, end)).fetchall()
        calls = db.execute("""
            select r.start, r.end, name.value, r.correlationId
            from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds name on name.id = r.nameId
            where (r.globalTid >> 24)=? and r.start>=? and r.end<=?
        """, (pid, start, end)).fetchall()
        launch = {}
        for a,b,n,c in calls:
            if "Launch" in n:
                launch[c] = min(launch.get(c, a), a)
        host_gaps = []
        following = {a: launch[c] for a, _, _, _, _, c in kernels if c in launch}
        for a, b in gaps:
            if b in following:
                cpu_at = following[b]
                if a < cpu_at:
                    host_gaps.append((a, min(cpu_at, b)))
        first_layout = min(
            a for a, _, _, name, _, _ in kernels
            if "deep_ep::" in name and "get_dispatch_layout" in name
        )
        last_combine = max(
            b for _, b, _, name, _, _ in kernels
            if "::intranode::combine<" in name
        )
        gaps_inside_moe = [
            (max(a, first_layout), min(b, last_combine))
            for a, b in host_gaps if a < last_combine and b > first_layout
        ]
        activity = defaultdict(list)
        name_times = defaultdict(list)
        for a,b,_,name,short,c in kernels:
            group = kernel_family(name if "deep_ep::" in name else (short or name))
            activity[group].append((a,b))
            name_times[short or name].append((a,b))
        ranges_by_name = defaultdict(list)
        for a,b,n in ranges:
            ranges_by_name[n].append((a,b))
        calls_by_name = defaultdict(list)
        for a,b,n,_ in calls:
            calls_by_name[n].append((a,b))
        comm = [p for g, spans_g in activity.items() if g.startswith("deep_ep_") for p in spans_g]
        comp = [p for g, spans_g in activity.items() if not g.startswith("deep_ep_") for p in spans_g]
        overlap_ms = (
            elapsed_ms(interval_union(comm)) + elapsed_ms(interval_union(comp))
            - elapsed_ms(interval_union(comm + comp))
        )
        result[case].append({
            "rank": rank, "trial": int(trial), "step_ms": (end - start) / MILLISECOND,
            "gpu_busy_ms": elapsed_ms(spans),
            "gpu_idle_ms": elapsed_ms(gaps),
            "gpu_idle_before_host_launch_ms": elapsed_ms(host_gaps),
            "step_to_first_layout_ms": (first_layout - start) / MILLISECOND,
            "gpu_idle_before_host_launch_inside_moe_ms": elapsed_ms(gaps_inside_moe),
            "gpu_largest_idle_ms": max(b-a for a,b in gaps) / MILLISECOND,
            "gpu_compute_comm_overlap_ms": overlap_ms,
            "kernel_groups": {g: [len(activities), elapsed_ms(activities)] for g, activities in activity.items()},
            "kernels": {g: [len(activities), elapsed_ms(activities)] for g, activities in name_times.items()},
            "nvtx": {g: [len(acts), elapsed_ms(acts)] for g, acts in ranges_by_name.items()},
            "cuda_api": {g: [len(acts), elapsed_ms(acts)] for g, acts in calls_by_name.items()},
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostic-summary.json")
    parser.add_argument("--phase-csv", type=Path)
    args = parser.parse_args()
    records = {}
    phase_rows = []
    for arm in sorted(ROOT.iterdir()):
        paths = [arm / f"rank{i}.json" for i in range(2)]
        if not all(p.is_file() for p in paths):
            continue
        a,b = [json.loads(p.read_text()) for p in paths]
        print(arm.name)
        rows = {}
        for name in a["cases"]:
            trials = zip(a["cases"][name]["timed"], b["cases"][name]["timed"], strict=True)
            last_first = []
            for u,v in trials:
                start = min(u["start_ns"], v["start_ns"])
                finish = max(r["first_ns"] for x in (u, v) for r in x["requests"])
                last_first.append((finish - start) / MILLISECOND)
            coverage = a["cases"][name]["coverage"][0]
            print(f"  {name:<18} TTFT_P50={median(last_first):.2f} ms "
                  f"ubatches={coverage['ubatches']} checkpoint={coverage['checkpoint']}")
            rows[name] = {"ttft_ms": last_first, "schedule": coverage}
        if (arm / "trace.sqlite").exists():
            for case, samples in profile(arm / "trace.sqlite").items():
                assert len(samples) == 6, (arm, case, len(samples))
                rows[case]["trace"] = samples
                if case != "prefill_32x256":
                    continue
                metrics = (
                    "step_to_first_layout_ms",
                    "gpu_idle_before_host_launch_inside_moe_ms",
                )
                phase_rows.append([arm.name, *(
                    round(median(sample[k] for sample in samples), 3) for k in metrics
                )])
                print("  trace", " ".join(
                    f"{metric}={median(s[metric] for s in samples):.1f}"
                    for metric in ("step_ms", "gpu_busy_ms", "gpu_idle_ms",
                                   "gpu_idle_before_host_launch_ms", "gpu_compute_comm_overlap_ms")
                ))
                for key in ("kernel_groups", "nvtx"):
                    data = defaultdict(list)
                    for sample in samples:
                        for name, pair in sample[key].items():
                            data[name].append(pair)
                    print(" ", key, "; ".join(
                        f"{name}: {median(p[0] for p in pairs):.0f} calls / "
                        f"{median(p[1] for p in pairs):.1f} ms"
                        for name, pairs in data.items()
                    ))
        records[arm.name] = rows
    args.output.write_text(json.dumps(records, indent=2))
    if args.phase_csv is not None:
        with args.phase_csv.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "arm", "step_to_first_layout_p50_ms",
                "gpu_idle_before_host_launch_inside_moe_p50_ms",
            ])
            writer.writerows(phase_rows)


if __name__ == "__main__":
    main()
