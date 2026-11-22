"""Compare native overlap, an optional repeat, and a same-split serial control."""
import argparse
import json
from pathlib import Path

from summarize_nixl import compare_request, read_pair, stats


def paired_scores(left_arm, right_arm):
    assert all(rank.get("complete") for rank in (*left_arm, *right_arm))
    total_sample, total_prompt = [], []
    errors, greedy_mismatches = [], []
    cases = {}
    for left, right in zip(left_arm, right_arm):
        shared = left["cases"].keys() & right["cases"].keys()
        for name in sorted(shared):
            expected, actual = left["cases"][name], right["cases"][name]
            if expected["inputs"] != actual["inputs"]:
                errors.append(f"rank {left['rank']} {name}: inputs changed")
                continue
            case = cases.setdefault(name, {"samples": [], "prompts": [], "mismatches": 0})
            for request, out in expected["outputs"].items():
                sample, prompt, mismatch, issues = compare_request(
                    out, actual["outputs"][request], True
                )
                total_sample.extend(sample)
                total_prompt.extend(prompt)
                case["samples"].extend(sample)
                case["prompts"].extend(prompt)
                if mismatch:
                    case["mismatches"] += 1
                    greedy_mismatches.append(
                        {"case": name, "rank": left["rank"], "request": request, **mismatch}
                    )
                errors.extend(f"rank {left['rank']} {name} {request}: {issue}" for issue in issues)
    return {
        "sample": stats(total_sample), "prompt": stats(total_prompt),
        "greedy_mismatches": greedy_mismatches, "errors": errors,
        "cases": {
            name: {"sample": stats(case["samples"]), "prompt": stats(case["prompts"]),
                   "mismatches": case["mismatches"]}
            for name, case in cases.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = paired_scores(read_pair(args.left), read_pair(args.right))
    result["left"] = str(args.left)
    result["right"] = str(args.right)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("sample", result["sample"], "prompt", result["prompt"],
          "greedy mismatches", len(result["greedy_mismatches"]), "errors", result["errors"])


if __name__ == "__main__":
    main()
