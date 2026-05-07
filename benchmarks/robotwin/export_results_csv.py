#!/usr/bin/env python3
"""Export RoboTwin eval results from a log directory into a CSV file.

Supports the log layout produced by:
  - benchmarks/robotwin/dlc_parallel_eval.sh
  - benchmarks/robotwin/parallel_eval.sh (best effort when summary.tsv is absent)

Primary data source is ``summary.tsv``. For each task log, this script also
parses the last ``Success rate`` line and exposes it as a numeric CSV column.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SUCCESS_RATE_PATTERNS = (
    re.compile(
        r"success rate:\s*\d+/\d+\s*=>\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        re.IGNORECASE,
    ),
    re.compile(r"success rate[^%]*?([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export RoboTwin evaluation logs to a CSV summary."
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="DLC/shared evaluation log directory, e.g. .../openwam_all_dlc_run123",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <log_dir>/results.csv)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any task log is missing or its success rate cannot be parsed.",
    )
    return parser.parse_args()


def load_run_env(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.is_file():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def parse_success_rate(log_path: Path) -> Optional[float]:
    if not log_path.is_file():
        return None
    last_match: Optional[float] = None
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            clean_line = strip_ansi(line)
            for pattern in SUCCESS_RATE_PATTERNS:
                match = pattern.search(clean_line)
                if match:
                    last_match = float(match.group(1))
    except OSError:
        return None
    return last_match


def iter_summary_rows(summary_path: Path) -> Iterable[Dict[str, str]]:
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield {k: (v or "").strip() for k, v in row.items()}


def infer_rows_without_summary(log_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for log_path in sorted(log_dir.rglob("*.log")):
        stem = log_path.stem
        if stem.endswith("_demo_clean"):
            task = stem[: -len("_demo_clean")]
            mode = "demo_clean"
        elif stem.endswith("_demo_randomized"):
            task = stem[: -len("_demo_randomized")]
            mode = "demo_randomized"
        else:
            continue
        node = ""
        worker = ""
        try:
            rel_parts = log_path.relative_to(log_dir).parts
        except ValueError:
            rel_parts = log_path.parts
        for part in rel_parts:
            if part.startswith("node"):
                node = part[len("node") :]
            elif part.startswith("worker"):
                worker = part[len("worker") :]
        rows.append(
            {
                "task": task,
                "mode": mode,
                "node": node,
                "worker": worker,
                "status": "",
                "exit_code": "",
                "log": str(log_path),
            }
        )
    return rows


def expected_job_keys(run_env: Dict[str, str]) -> Optional[Set[Tuple[str, str]]]:
    tasks = [task for task in run_env.get("tasks", "").split() if task]
    requested_mode = run_env.get("mode", "").strip()
    if not tasks or not requested_mode:
        return None
    if requested_mode == "all":
        modes = ("demo_clean", "demo_randomized")
    else:
        modes = (requested_mode,)
    return {(task, mode) for task in tasks for mode in modes}


def format_job_keys(keys: Sequence[Tuple[str, str]], limit: int = 8) -> str:
    items = [f"{task}:{mode}" for task, mode in keys[:limit]]
    if len(keys) > limit:
        items.append(f"... (+{len(keys) - limit} more)")
    return ", ".join(items)


def main() -> int:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        print(f"[ERROR] log_dir not found: {log_dir}", file=sys.stderr)
        return 1

    output_csv = args.output.resolve() if args.output else log_dir / "results.csv"
    summary_path = log_dir / "summary.tsv"
    run_env = load_run_env(log_dir / "run.env")

    if summary_path.is_file():
        raw_rows = list(iter_summary_rows(summary_path))
    else:
        raw_rows = infer_rows_without_summary(log_dir)
        if not raw_rows:
            print(
                f"[ERROR] Neither {summary_path} nor task logs under {log_dir} were found.",
                file=sys.stderr,
            )
            return 1

    failures: List[str] = []
    exported_rows: List[Dict[str, str]] = []
    row_key_counts = Counter(
        (row.get("task", ""), row.get("mode", ""))
        for row in raw_rows
        if row.get("task") and row.get("mode")
    )

    duplicate_keys = sorted(key for key, count in row_key_counts.items() if count > 1)
    if duplicate_keys:
        failures.append(
            "duplicate task/mode rows: "
            + format_job_keys(duplicate_keys)
        )

    expected_keys = expected_job_keys(run_env)
    if expected_keys is not None:
        seen_keys = set(row_key_counts)
        missing_keys = sorted(expected_keys - seen_keys)
        extra_keys = sorted(seen_keys - expected_keys)
        if len(raw_rows) != len(expected_keys):
            failures.append(
                f"summary row count mismatch: expected {len(expected_keys)}, got {len(raw_rows)}"
            )
        if missing_keys:
            failures.append(
                f"missing task/mode rows: {format_job_keys(missing_keys)}"
            )
        if extra_keys:
            failures.append(
                f"unexpected task/mode rows: {format_job_keys(extra_keys)}"
            )
    else:
        expected_total_jobs = run_env.get("total_jobs", "").strip()
        if expected_total_jobs.isdigit() and len(raw_rows) != int(expected_total_jobs):
            failures.append(
                f"summary row count mismatch: expected {expected_total_jobs}, got {len(raw_rows)}"
            )

    for row in raw_rows:
        log_path = Path(row.get("log", ""))
        if not log_path.is_absolute():
            log_path = (log_dir / log_path).resolve()
        success_rate = parse_success_rate(log_path)

        if not log_path.is_file():
            failures.append(f"missing log: {log_path}")
        elif success_rate is None:
            failures.append(f"missing success rate: {log_path}")

        exported_rows.append(
            {
                "run_id": run_env.get("run_id", ""),
                "policy_name": run_env.get("policy_name", ""),
                "requested_mode": run_env.get("mode", ""),
                "task": row.get("task", ""),
                "mode": row.get("mode", ""),
                "node": row.get("node", ""),
                "worker": row.get("worker", ""),
                "status": row.get("status", ""),
                "exit_code": row.get("exit_code", ""),
                "success_rate": "" if success_rate is None else f"{success_rate:.6f}",
                "log_path": str(log_path),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "policy_name",
                "requested_mode",
                "task",
                "mode",
                "node",
                "worker",
                "status",
                "exit_code",
                "success_rate",
                "log_path",
            ],
        )
        writer.writeheader()
        writer.writerows(exported_rows)

    ok_rows = sum(1 for row in exported_rows if row["status"] in ("", "ok"))
    parsed_rows = sum(1 for row in exported_rows if row["success_rate"] != "")
    print(f"[INFO] wrote CSV: {output_csv}")
    print(f"[INFO] rows={len(exported_rows)} parsed_success_rate={parsed_rows} ok_or_unknown={ok_rows}")

    if failures:
        print("[WARN] encountered issues while parsing:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        if args.strict:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
