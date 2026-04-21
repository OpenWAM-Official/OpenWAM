"""Summarize per-episode step-count records written by the step-analysis runner.

Input: a JSONL file with one record per episode:
    {"task": "...", "mode": "demo_clean", "episode": 0,
     "steps": 123, "success": true, "step_lim": 400}

Output:
    - CSV with one row per (task, mode, success-bucket): count, min/mean/median/max steps
    - Optional markdown table summary
    - A brief per-mode aggregate printed to stdout
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

BUCKETS: List[Tuple[str, object]] = [
    ("success", True),
    ("failure", False),
    ("all", None),
]


def load_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] {path}:{lineno} skipped: {exc}", file=sys.stderr)
    return records


def fmt_stats(steps: List[int]) -> Dict[str, float]:
    if not steps:
        return {"count": 0, "min": 0, "mean": 0.0, "median": 0.0, "max": 0}
    return {
        "count": len(steps),
        "min": min(steps),
        "mean": round(statistics.fmean(steps), 1),
        "median": statistics.median(steps),
        "max": max(steps),
    }


def summarize(records: List[dict]):
    # keyed by (task, mode, bucket)
    grouped: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    step_lims: Dict[Tuple[str, str], int] = {}

    for r in records:
        task = r.get("task", "unknown")
        mode = r.get("mode", "unknown")
        success = bool(r.get("success", False))
        steps = int(r.get("steps", 0))
        step_lim = int(r.get("step_lim", -1))
        if step_lim > 0:
            step_lims[(task, mode)] = step_lim

        for bucket_name, wanted in BUCKETS:
            if wanted is None or wanted == success:
                grouped[(task, mode, bucket_name)].append(steps)

    rows = []
    tasks = sorted({k[0] for k in grouped})
    modes = sorted({k[1] for k in grouped})
    for task in tasks:
        for mode in modes:
            row = {"task": task, "mode": mode}
            row["step_lim"] = step_lims.get((task, mode), "")
            for bucket_name, _ in BUCKETS:
                st = fmt_stats(grouped.get((task, mode, bucket_name), []))
                for k, v in st.items():
                    row[f"{bucket_name}_{k}"] = v
            rows.append(row)

    return rows, tasks, modes


def write_csv(rows: List[dict], path: Path) -> None:
    import csv
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: List[dict], path: Path) -> None:
    if not rows:
        path.write_text("# No records\n", encoding="utf-8")
        return
    cols = [
        ("task", "task"),
        ("mode", "mode"),
        ("step_lim", "step_lim"),
        ("success_count", "s_n"),
        ("success_mean", "s_mean"),
        ("success_median", "s_med"),
        ("success_min", "s_min"),
        ("success_max", "s_max"),
        ("failure_count", "f_n"),
        ("failure_mean", "f_mean"),
        ("failure_median", "f_med"),
        ("failure_min", "f_min"),
        ("failure_max", "f_max"),
    ]
    lines = ["| " + " | ".join(label for _, label in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_stdout_summary(records: List[dict]) -> None:
    by_mode: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_mode[r.get("mode", "unknown")].append(r)

    print(f"\n=== Episodes: {len(records)} ===")
    for mode, rs in sorted(by_mode.items()):
        succ = [r for r in rs if r.get("success")]
        fail = [r for r in rs if not r.get("success")]
        sr = (len(succ) / len(rs) * 100) if rs else 0.0
        print(f"\n[{mode}]  n={len(rs)}  success={len(succ)} ({sr:.1f}%)  failure={len(fail)}")

        succ_steps = [int(r["steps"]) for r in succ if "steps" in r]
        fail_steps = [int(r["steps"]) for r in fail if "steps" in r]
        if succ_steps:
            s = fmt_stats(succ_steps)
            print(f"  success steps:  mean={s['mean']}  median={s['median']}  min={s['min']}  max={s['max']}")
        if fail_steps:
            s = fmt_stats(fail_steps)
            print(f"  failure steps:  mean={s['mean']}  median={s['median']}  min={s['min']}  max={s['max']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, help="path to steps.jsonl")
    ap.add_argument("-o", "--output", type=Path, default=None, help="CSV output path")
    ap.add_argument("--markdown", type=Path, default=None, help="markdown table output path")
    args = ap.parse_args()

    if not args.jsonl.is_file():
        print(f"[ERROR] {args.jsonl} not found", file=sys.stderr)
        return 1

    records = load_records(args.jsonl)
    if not records:
        print("[WARN] no records parsed", file=sys.stderr)
        return 0

    rows, _, _ = summarize(records)

    if args.output:
        write_csv(rows, args.output)
        print(f"[INFO] CSV → {args.output}")
    if args.markdown:
        write_markdown(rows, args.markdown)
        print(f"[INFO] Markdown → {args.markdown}")

    print_stdout_summary(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
