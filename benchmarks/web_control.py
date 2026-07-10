#!/usr/bin/env python3
"""Unified real-time web console for benchmark log directories.

The console is intentionally dependency-free so it can run inside bare DLC or
cluster job containers:

    python benchmarks/web_control.py /path/to/log_dir --benchmark robotwin --host 0.0.0.0

It exposes a stable HTTP/API surface backed by small benchmark adapters. The
RoboTwin adapter understands the layout produced by ``dlc_parallel_eval.sh``;
the generic adapter works as a live tail viewer for arbitrary log directories.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import shutil
import sys
import time
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SUCCESS_RATE_PATTERNS = (
    re.compile(
        r"success rate:\s*\d+/\d+\s*=>\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        re.IGNORECASE,
    ),
    re.compile(r"success rate[^%]*?([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE),
)
TASK_LOG_RE = re.compile(r"(?P<task>.+)_(?P<mode>demo_clean|demo_randomized)\.log$")
CLAIMED_SUFFIX_RE = re.compile(r"\.node(?P<node>\d+)\.worker(?P<worker>\d+)$")
LOG_FILE_PATTERNS = ("*.log", "*.out", "*.err", "stdout*", "stderr*")
FINAL_JOB_STATUSES = {"ok", "failed", "done"}


def parse_positive_int(value: str, *, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    number = max(minimum, number)
    if maximum is not None:
        number = min(number, maximum)
    return number


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 1:
        return ""
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def mode_list(requested_mode: str) -> tuple[str, ...]:
    return ("demo_clean", "demo_randomized") if requested_mode == "all" else ((requested_mode,) if requested_mode else ())


def expected_job_keys(run_env: dict[str, str]) -> set[tuple[str, str]]:
    tasks = [task for task in run_env.get("tasks", "").split() if task]
    return {(task, mode) for task in tasks for mode in mode_list(run_env.get("mode", "").strip())}


def unique_job_key(item: dict[str, Any]) -> tuple[str, str] | None:
    task = str(item.get("task", ""))
    mode = str(item.get("mode", ""))
    return (task, mode) if task and mode else None


def numeric_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def safe_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: safe_cell(row.get(name, "")) for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def html_response_bytes(text: str) -> bytes:
    body = html.escape(text)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Benchmark Control Error</title>"
        "<style>body{font-family:sans-serif;margin:2rem;background:#f1eee7;color:#17211e}"
        "pre{white-space:pre-wrap;background:#fffaf0;border:1px solid #ddd;padding:1rem;border-radius:.75rem}</style>"
        f"</head><body><h1>Benchmark Control Error</h1><pre>{body}</pre></body></html>"
    ).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a real-time web control console for a benchmark log directory."
    )
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Benchmark log directory, e.g. .../openwam_all_dlc_run123",
    )
    parser.add_argument(
        "--benchmark",
        choices=("auto", "robotwin", "generic"),
        default="auto",
        help="Benchmark adapter to use. Default: auto",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host. Use 0.0.0.0 when viewing from another machine. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port. Default: 8765",
    )
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=200_000,
        help="Bytes returned for the first log tail request. Default: 200000",
    )
    parser.add_argument(
        "--state-tail-bytes",
        type=int,
        default=256_000,
        help="Bytes scanned per task log for success-rate parsing. Default: 256000",
    )
    parser.add_argument(
        "--max-logs",
        type=int,
        default=2_000,
        help="Maximum log-like files shown in the log browser. Default: 2000",
    )
    parser.add_argument(
        "--max-task-log-bytes",
        type=int,
        default=4_000_000,
        help="Maximum bytes per task log scanned for error snippets and timeline data. Default: 4000000",
    )
    parser.add_argument(
        "--max-error-snippets",
        type=int,
        default=200,
        help="Maximum failed-task error snippets retained in state. Default: 200",
    )
    parser.add_argument(
        "--refresh-sec",
        type=float,
        default=2.0,
        help="Browser polling interval in seconds. Default: 2.0",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the console URL in the local browser after startup.",
    )
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def utc_iso(ts: float | None = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def read_tail(path: Path, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
        start = max(0, size - max_bytes)
        with path.open("rb") as f:
            f.seek(start)
            data = f.read(max_bytes)
    except OSError:
        return ""
    return strip_ansi(data.decode("utf-8", errors="replace"))


def latest_line(path: Path, max_bytes: int = 16_384) -> str:
    text = read_tail(path, max_bytes)
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line[-500:]
    return ""


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def read_nonempty_lines(path: Path, limit: int | None = None) -> list[str]:
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                value = line.strip()
                if not value:
                    continue
                lines.append(value)
                if limit is not None and len(lines) >= limit:
                    break
    except OSError:
        return []
    return lines


def parse_task_mode_line(line: str) -> tuple[str, str] | None:
    task, sep, mode = line.strip().partition("|")
    if not sep or not task or not mode:
        return None
    return task, mode


def parse_key_values(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.is_file():
        return data
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    except OSError:
        return {}
    return data


def parse_success_rate_from_text(text: str) -> float | None:
    last_match: float | None = None
    for line in strip_ansi(text).splitlines():
        for pattern in SUCCESS_RATE_PATTERNS:
            match = pattern.search(line)
            if match:
                try:
                    last_match = float(match.group(1))
                except ValueError:
                    pass
    return last_match


def parse_success_counts_from_text(text: str) -> tuple[int, int] | None:
    last_counts: tuple[int, int] | None = None
    count_re = re.compile(r"success rate:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    for line in strip_ansi(text).splitlines():
        match = count_re.search(line)
        if not match:
            continue
        try:
            last_counts = (int(match.group(1)), int(match.group(2)))
        except ValueError:
            continue
    return last_counts


# RoboTwin prints ``step: N / M`` per step and ``Success!`` / ``Fail!`` per
# episode. A ``Fail!`` whose last step reached ``N >= M`` was truncated at the
# step limit (out of steps) rather than the model reaching a terminal state —
# a non-model cause of a low success rate. This is computed over the full log
# (not the tail) so it is only wired into the on-demand CSV export, never the
# polled live state where a tail window would undercount it.
STEP_PROGRESS_RE = re.compile(r"step:\s*(\d+)\s*/\s*(\d+)")
EPISODE_VERDICT_RE = re.compile(r"\b(Success|Fail)!")


def count_step_limit_hits(text: str) -> int:
    hits = 0
    last_step: tuple[int, int] | None = None
    for line in strip_ansi(text).splitlines():
        step_match = STEP_PROGRESS_RE.search(line)
        if step_match:
            last_step = (int(step_match.group(1)), int(step_match.group(2)))
            continue
        verdict = EPISODE_VERDICT_RE.search(line)
        if verdict:
            if (
                verdict.group(1) == "Fail"
                and last_step is not None
                and last_step[1] > 0
                and last_step[0] >= last_step[1]
            ):
                hits += 1
            last_step = None
    return hits


def safe_relative(root: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def normalize_log_ref(root: Path, value: str) -> tuple[str, Path | None]:
    """Return a display path plus an absolute path when the file is under root."""
    if not value:
        return "", None
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve()
    except OSError:
        return value, None
    rel = safe_relative(root, resolved)
    if rel is not None:
        return rel, resolved
    return value, None


def resolve_requested_file(root: Path, requested: str) -> Path:
    if not requested:
        raise ValueError("missing file query parameter")
    decoded = unquote(requested)
    candidate = (root / decoded).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("file must be inside log_dir") from exc
    if not candidate.is_file():
        raise FileNotFoundError(decoded)
    return candidate


def file_meta(root: Path, path: Path) -> dict[str, Any] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    rel = safe_relative(root, path)
    if rel is None:
        return None
    task_info = task_mode_from_log_name(path.name)
    category = categorize_log(root, path)
    return {
        "rel": rel,
        "name": path.name,
        "category": category,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "mtime_iso": utc_iso(stat.st_mtime),
        "latest": latest_line(path),
        "task": task_info[0] if task_info else "",
        "mode": task_info[1] if task_info else "",
    }


def task_mode_from_log_name(name: str) -> tuple[str, str] | None:
    match = TASK_LOG_RE.match(name)
    if not match:
        return None
    return match.group("task"), match.group("mode")


def categorize_log(root: Path, path: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    if "servers" in parts:
        return "server"
    if path.name == "worker.log":
        return "worker"
    if task_mode_from_log_name(path.name):
        return "task"
    return "log"


def find_task_log(root: Path, task: str, mode: str, node: str = "", worker: str = "") -> str:
    if not task or not mode:
        return ""
    filename = f"{task}_{mode}.log"
    candidates: list[Path] = []
    if node != "" and worker != "":
        candidates.append(root / f"node{node}" / f"worker{worker}" / filename)
    if worker != "":
        candidates.append(root / f"worker{worker}" / filename)
    candidates.append(root / filename)
    for candidate in candidates:
        if candidate.is_file():
            return safe_relative(root, candidate) or ""
    try:
        matches = sorted(root.rglob(filename), key=path_mtime, reverse=True)
    except OSError:
        return ""
    for match in matches:
        if match.is_file():
            return safe_relative(root, match) or ""
    return ""


def task_error_snippet(path: Path, max_bytes: int) -> str:
    if not path.is_file():
        return ""
    text = read_tail(path, max_bytes)
    lines = text.splitlines()
    if not lines:
        return ""
    markers = (
        "traceback",
        "error",
        "exception",
        "failed",
        "runtimeerror",
        "valueerror",
        "assertionerror",
        "cuda out of memory",
        "segmentation fault",
    )
    start = max(0, len(lines) - 25)
    for idx in range(len(lines) - 1, -1, -1):
        lower = lines[idx].lower()
        if any(marker in lower for marker in markers):
            start = max(0, idx - 6)
            break
    snippet = "\n".join(lines[start:])
    return snippet[-5000:]


def task_timeline_from_log(path: Path, max_bytes: int) -> dict[str, Any]:
    if not path.is_file():
        return {}
    stat = path.stat()
    text = read_tail(path, max_bytes)
    counts = parse_success_counts_from_text(text)
    rate = parse_success_rate_from_text(text)
    success = counts[0] if counts else None
    episodes = counts[1] if counts else None
    return {
        "mtime": stat.st_mtime,
        "mtime_iso": utc_iso(stat.st_mtime),
        "size": stat.st_size,
        "latest": latest_line(path),
        "success_rate": rate,
        "success": success,
        "episodes": episodes,
    }


class BenchmarkConsoleAdapter:
    """Base adapter for benchmark-specific state and CSV generation."""

    benchmark = "generic"
    title = "Benchmark Control"
    csv_filename = "benchmark_results.csv"

    def __init__(
        self,
        root: Path,
        *,
        max_logs: int,
        state_tail_bytes: int,
        max_task_log_bytes: int,
        max_error_snippets: int,
    ):
        self.root = root.resolve()
        self.max_logs = max_logs
        self.state_tail_bytes = state_tail_bytes
        self.max_task_log_bytes = max_task_log_bytes
        self.max_error_snippets = max_error_snippets

    def build_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def build_results_rows(self) -> list[dict[str, Any]]:
        return []

    def build_custom_metrics(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Return optional card-shaped metrics for benchmark-specific dashboards."""
        return []

    def csv_fieldnames(self, rows: list[dict[str, Any]]) -> list[str]:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        return fields or ["benchmark", "log_path"]

    @staticmethod
    def _base_progress(total: int, completed: int, running: int, pending: int, failed: int = 0) -> dict[str, Any]:
        denominator = max(total, completed + running + pending, 1)
        return {
            "total": total,
            "completed": completed,
            "ok": max(0, completed - failed),
            "failed": failed,
            "done": 0,
            "running": running,
            "pending": pending,
            "percent": min(100.0, 100.0 * completed / denominator),
            "eta_sec": None,
            "eta": "",
            "longest_running_sec": None,
            "longest_running": "",
        }

    def _collect_logs(self) -> list[dict[str, Any]]:
        paths: list[Path] = []
        try:
            seen: set[Path] = set()
            for pattern in LOG_FILE_PATTERNS:
                for path in self.root.rglob(pattern):
                    if path.is_file() and path not in seen:
                        paths.append(path)
                        seen.add(path)
        except OSError:
            return []

        paths.sort(key=path_mtime, reverse=True)
        logs: list[dict[str, Any]] = []
        for path in paths[: self.max_logs]:
            meta = file_meta(self.root, path)
            if meta:
                logs.append(meta)
        return logs


class GenericLogAdapter(BenchmarkConsoleAdapter):
    benchmark = "generic"
    title = "Benchmark Control"
    csv_filename = "benchmark_logs.csv"

    def build_state(self) -> dict[str, Any]:
        logs = self._collect_logs()
        newest_mtime = max([float(item.get("mtime", 0.0)) for item in logs], default=path_mtime(self.root))
        log_jobs = [
            {
                "task": item.get("name", ""),
                "mode": item.get("category", "log"),
                "node": "",
                "worker": "",
                "status": "log",
                "exit_code": "",
                "log": item.get("rel", ""),
                "success_rate": item.get("success_rate"),
                "success": item.get("success"),
                "episodes": item.get("episodes"),
                "latest": item.get("latest", ""),
                "size": item.get("size", 0),
                "mtime": item.get("mtime", 0.0),
                "mtime_iso": item.get("mtime_iso", ""),
                "started_at": None,
                "started_at_iso": "",
                "duration_sec": None,
                "duration": "",
                "source": "generic-log",
            }
            for item in logs
        ]
        state = {
            "benchmark": self.benchmark,
            "title": self.title,
            "now": utc_iso(),
            "log_dir": str(self.root),
            "exists": self.root.is_dir(),
            "run_env": {
                "run_id": self.root.name,
                "policy_name": "",
                "mode": "generic",
                "total_jobs": str(len(logs)),
            },
            "summary": {
                "path": "",
                "exists": False,
                "row_count": 0,
                "ok": 0,
                "failed": 0,
                "duplicates": [],
            },
            "queue": {
                "pending_count": 0,
                "claimed_count": 0,
                "running_count": 0,
                "pending": [],
                "claimed": [],
                "running": [],
                "ready": False,
                "done_nodes": [],
            },
            "progress": self._base_progress(total=len(logs), completed=len(logs), running=0, pending=0),
            "rates": {
                "weighted_success": 0,
                "weighted_total": 0,
                "weighted_success_rate": None,
                "mean_task_success_rate": None,
                "parsed_task_count": 0,
            },
            "validation": {
                "ok": True,
                "issue_count": 0,
                "issues": [],
                "expected_total": len(logs),
                "observed_total": len(logs),
            },
            "failures": [],
            "jobs": log_jobs,
            "nodes": [],
            "logs": logs,
            "last_update": utc_iso(newest_mtime) if newest_mtime > 0 else "",
        }
        state["run"] = self._run_summary(state)
        state["metrics"] = self._metrics_summary(state)
        state["custom_metrics"] = self.build_custom_metrics(state)
        return state

    def build_results_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "benchmark": self.benchmark,
                "log_path": item.get("rel", ""),
                "category": item.get("category", ""),
                "size": item.get("size", 0),
                "mtime_iso": item.get("mtime_iso", ""),
                "latest": item.get("latest", ""),
            }
            for item in self._collect_logs()
        ]

    @staticmethod
    def _run_summary(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": state["run_env"].get("run_id", ""),
            "policy": "",
            "mode": "generic",
            "total": state["progress"].get("total", 0),
            "log_dir": state.get("log_dir", ""),
            "last_update": state.get("last_update", ""),
            "metadata": state.get("run_env", {}),
        }

    @staticmethod
    def _metrics_summary(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "success_rate": None,
            "weighted_success": 0,
            "weighted_total": 0,
            "parsed_task_count": 0,
        }


class SnapshotBuilder(BenchmarkConsoleAdapter):
    benchmark = "robotwin"
    title = "RoboTwin Control"
    csv_filename = "robotwin_results.csv"
    def build(self) -> dict[str, Any]:
        run_env = parse_key_values(self.root / "run.env")
        summary = self._collect_summary()
        fallback = self._collect_fallback_worker_results(set(summary["all_keys"]))
        fallback_completed_keys = {
            key for row in fallback if row.get("status") in FINAL_JOB_STATUSES for key in [unique_job_key(row)] if key
        }
        completed_keys = set(summary["completed_keys"]) | fallback_completed_keys
        queue = self._collect_queue(completed_keys)
        logs = self._collect_logs()
        nodes = self._collect_nodes(queue, summary)
        jobs = self._merge_jobs(summary["rows"], queue, fallback)
        expected_keys = expected_job_keys(run_env)
        validation = self._validate(run_env, expected_keys, jobs, summary)
        total = self._infer_total(run_env, expected_keys, summary, queue, jobs)
        progress = self._progress(total, jobs)
        failures = self._collect_failures(jobs)
        rates = self._success_aggregate(jobs)

        newest_mtime = max(
            [path_mtime(self.root / "summary.tsv"), path_mtime(self.root / "run.env")]
            + [float(item.get("mtime", 0.0)) for item in logs]
            + [float(item.get("mtime", 0.0)) for item in jobs],
            default=0.0,
        )

        state = {
            "benchmark": self.benchmark,
            "title": self.title,
            "now": utc_iso(),
            "log_dir": str(self.root),
            "exists": self.root.is_dir(),
            "run_env": run_env,
            "summary": {
                "path": "summary.tsv",
                "exists": (self.root / "summary.tsv").is_file(),
                "row_count": len(summary["rows"]),
                "ok": summary["ok"],
                "failed": summary["failed"],
                "duplicates": summary["duplicates"],
            },
            "queue": queue,
            "progress": progress,
            "rates": rates,
            "validation": validation,
            "failures": failures,
            "jobs": jobs,
            "nodes": nodes,
            "logs": logs,
            "last_update": utc_iso(newest_mtime) if newest_mtime > 0 else "",
        }
        state["run"] = self._run_summary(state)
        state["metrics"] = self._metrics_summary(state)
        state["custom_metrics"] = self.build_custom_metrics(state)
        return state

    def build_state(self) -> dict[str, Any]:
        return self.build()

    def build_results_rows(self) -> list[dict[str, Any]]:
        snapshot = self.build()
        run_env = snapshot["run_env"]
        rows = []
        for job in snapshot["jobs"]:
            full_log = self._full_log_stats_for_job(job)
            rows.append(
                {
                    "run_id": run_env.get("run_id", ""),
                    "policy_name": run_env.get("policy_name", ""),
                    "requested_mode": run_env.get("mode", ""),
                    "task": job.get("task", ""),
                    "mode": job.get("mode", ""),
                    "node": job.get("node", ""),
                    "worker": job.get("worker", ""),
                    "status": job.get("status", ""),
                    "exit_code": job.get("exit_code", ""),
                    "success_rate": "" if job.get("success_rate") is None else f"{float(job['success_rate']):.6f}",
                    "success": "" if full_log["success"] is None else full_log["success"],
                    "episodes": "" if full_log["episodes"] is None else full_log["episodes"],
                    "step_limit_hits": "" if full_log["step_limit_hits"] is None else full_log["step_limit_hits"],
                    "duration_sec": "" if job.get("duration_sec") is None else f"{float(job['duration_sec']):.3f}",
                    "duration": job.get("duration", ""),
                    "log_path": job.get("log", ""),
                    "source": job.get("source", ""),
                }
            )
        return rows

    def _full_log_stats_for_job(self, job: dict[str, Any]) -> dict[str, int | None]:
        """Full-log ``success``/``episodes``/``step_limit_hits`` for a job.

        Reads the log once (not the tail window used for the polled live
        state) so all three columns come from the same pass — pairing
        ``episodes`` from a tail-windowed read with a full-log
        ``step_limit_hits`` could otherwise report a stale/blank episode
        count alongside a real hit count when the last ``success rate: X / Y``
        line falls outside the tail (e.g. behind a long traceback).

        ``Path(root) / log`` collapses to ``log`` when it is already absolute
        (as summary.tsv records it), so both relative and absolute refs work.
        """
        log_ref = job.get("log", "")
        if not log_ref:
            return {"success": None, "episodes": None, "step_limit_hits": None}
        log_path = self.root / log_ref
        if not log_path.is_file():
            return {"success": None, "episodes": None, "step_limit_hits": None}
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {"success": None, "episodes": None, "step_limit_hits": None}
        counts = parse_success_counts_from_text(text)
        return {
            "success": counts[0] if counts else None,
            "episodes": counts[1] if counts else None,
            "step_limit_hits": count_step_limit_hits(text),
        }

    def csv_fieldnames(self, rows: list[dict[str, Any]]) -> list[str]:
        return [
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
            "success",
            "episodes",
            "step_limit_hits",
            "duration_sec",
            "duration",
            "log_path",
            "source",
        ]

    @staticmethod
    def _run_summary(state: dict[str, Any]) -> dict[str, Any]:
        env = state.get("run_env", {})
        progress = state.get("progress", {})
        return {
            "id": env.get("run_id", ""),
            "policy": env.get("policy_name", ""),
            "mode": env.get("mode", ""),
            "total": progress.get("total", 0),
            "log_dir": state.get("log_dir", ""),
            "last_update": state.get("last_update", ""),
            "metadata": env,
        }

    @staticmethod
    def _metrics_summary(state: dict[str, Any]) -> dict[str, Any]:
        rates = state.get("rates", {})
        return {
            "success_rate": rates.get("weighted_success_rate") or rates.get("mean_task_success_rate"),
            "weighted_success": rates.get("weighted_success", 0),
            "weighted_total": rates.get("weighted_total", 0),
            "parsed_task_count": rates.get("parsed_task_count", 0),
        }

    def build_custom_metrics(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        return self._mode_success_metrics(state.get("jobs", []))

    @classmethod
    def _mode_success_metrics(cls, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        targets = (
            ("demo_clean", "demo_clean Success"),
            ("demo_randomized", "demo_randomized Success"),
        )
        stats: dict[str, dict[str, Any]] = {
            mode: {"success": 0, "episodes": 0, "rates": []}
            for mode, _label in targets
        }
        for job in jobs:
            mode = str(job.get("mode", ""))
            if mode not in stats or job.get("status") not in FINAL_JOB_STATUSES:
                continue
            success = cls._optional_int(job.get("success"))
            episodes = cls._optional_int(job.get("episodes"))
            if success is not None and episodes is not None and episodes > 0:
                stats[mode]["success"] += success
                stats[mode]["episodes"] += episodes
            rate = cls._optional_float(job.get("success_rate"))
            if rate is not None:
                stats[mode]["rates"].append(rate)

        metrics: list[dict[str, Any]] = []
        for mode, label in targets:
            mode_stats = stats[mode]
            weighted_success = int(mode_stats["success"])
            weighted_total = int(mode_stats["episodes"])
            rates = mode_stats["rates"]
            if weighted_total > 0:
                raw_value = 100.0 * weighted_success / weighted_total
                source = "weighted_success"
                description = f"{weighted_success}/{weighted_total} successful episodes for {mode}."
            elif rates:
                raw_value = sum(rates) / len(rates)
                source = "mean_success_rate"
                description = f"Mean success rate over {len(rates)} completed {mode} task(s)."
            else:
                raw_value = None
                source = "none"
                description = f"No completed {mode} tasks with parsed success data yet."
            metrics.append(
                {
                    "id": f"robotwin_{mode}_success_rate",
                    "label": label,
                    "value": "-" if raw_value is None else f"{raw_value:.2f}%",
                    "raw_value": raw_value,
                    "unit": "%",
                    "kind": "percent",
                    "class": cls._success_metric_class(raw_value),
                    "description": description,
                    "source": source,
                    "weighted_success": weighted_success,
                    "weighted_total": weighted_total,
                    "parsed_task_count": len(rates),
                }
            )
        return metrics

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _success_metric_class(value: float | None) -> str:
        if value is None:
            return ""
        if value >= 80:
            return "success"
        if value >= 50:
            return "warning"
        return "failed"

    def _collect_summary(self) -> dict[str, Any]:
        path = self.root / "summary.tsv"
        rows: list[dict[str, Any]] = []
        all_keys: set[tuple[str, str]] = set()
        completed_keys: set[tuple[str, str]] = set()
        seen_counts: dict[tuple[str, str], int] = {}
        duplicates: list[str] = []
        ok = 0
        failed = 0
        if not path.is_file():
            return {
                "rows": rows,
                "all_keys": all_keys,
                "completed_keys": completed_keys,
                "ok": ok,
                "failed": failed,
                "duplicates": duplicates,
            }

        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for raw in reader:
                    row = {str(k): (v or "").strip() for k, v in raw.items() if k is not None}
                    task = row.get("task", "")
                    mode = row.get("mode", "")
                    status = row.get("status", "") or "done"
                    log_rel, log_abs = normalize_log_ref(self.root, row.get("log", ""))
                    timeline = task_timeline_from_log(log_abs, self.state_tail_bytes) if log_abs else {}
                    started_at = self._job_started_at(task, mode, row.get("node", ""), row.get("worker", ""))
                    finished_at = timeline.get("mtime") or path_mtime(path)
                    item = {
                        "task": task,
                        "mode": mode,
                        "node": row.get("node", ""),
                        "worker": row.get("worker", ""),
                        "status": status,
                        "exit_code": row.get("exit_code", ""),
                        "log": log_rel,
                        "success_rate": timeline.get("success_rate"),
                        "success": timeline.get("success"),
                        "episodes": timeline.get("episodes"),
                        "latest": timeline.get("latest", ""),
                        "size": timeline.get("size", 0),
                        "mtime": finished_at,
                        "mtime_iso": utc_iso(finished_at) if finished_at else "",
                        "started_at": started_at,
                        "started_at_iso": utc_iso(started_at) if started_at else "",
                        "duration_sec": (finished_at - started_at) if started_at and finished_at >= started_at else None,
                        "duration": format_duration((finished_at - started_at) if started_at and finished_at >= started_at else None),
                        "source": "summary",
                    }
                    rows.append(item)
                    key = unique_job_key(item)
                    if key:
                        all_keys.add(key)
                        seen_counts[key] = seen_counts.get(key, 0) + 1
                        if status in FINAL_JOB_STATUSES:
                            completed_keys.add(key)
                    if status == "ok":
                        ok += 1
                    elif status == "failed":
                        failed += 1
        except OSError:
            return {
                "rows": rows,
                "all_keys": all_keys,
                "completed_keys": completed_keys,
                "ok": ok,
                "failed": failed,
                "duplicates": duplicates,
            }
        duplicates = [f"{task}:{mode} x{count}" for (task, mode), count in seen_counts.items() if count > 1]
        return {
            "rows": rows,
            "all_keys": all_keys,
            "completed_keys": completed_keys,
            "ok": ok,
            "failed": failed,
            "duplicates": sorted(duplicates),
        }

    def _collect_queue(self, completed_keys: set[tuple[str, str]]) -> dict[str, Any]:
        pending_dir = self.root / "queue" / "pending"
        claimed_dir = self.root / "queue" / "claimed"

        if pending_dir.is_dir() or claimed_dir.is_dir():
            pending = [
                self._job_file_item(path, "pending")
                for path in sorted(pending_dir.glob("*.job"))
                if path.is_file()
            ]
            claimed = [
                self._job_file_item(path, "claimed")
                for path in sorted(claimed_dir.glob("*.job*"))
                if path.is_file()
            ]
        else:
            pending = self._legacy_queue_items(completed_keys)
            claimed = []

        running = [
            item
            for item in claimed
            if unique_job_key(item) not in completed_keys
        ]
        pending_open = [
            item
            for item in pending
            if unique_job_key(item) not in completed_keys
        ]
        return {
            "pending_count": len(pending_open),
            "claimed_count": len(claimed),
            "running_count": len(running),
            "pending": pending_open,
            "claimed": claimed,
            "running": running,
            "ready": (self.root / ".queue_ready").is_file(),
            "done_nodes": sorted(
                path.name
                for path in self.root.glob(".node*_done")
                if path.is_file()
            ),
        }

    def _legacy_queue_items(self, completed_keys: set[tuple[str, str]]) -> list[dict[str, Any]]:
        queue_file = self.root / ".queue.txt"
        if not queue_file.is_file():
            return []
        items: list[dict[str, Any]] = []
        for idx, line in enumerate(read_nonempty_lines(queue_file)):
            parsed = parse_task_mode_line(line)
            if parsed is None:
                continue
            task, mode = parsed
            if (task, mode) in completed_keys:
                continue
            items.append(
                {
                    "task": task,
                    "mode": mode,
                    "status": "pending",
                    "node": "",
                    "worker": "",
                    "job_file": ".queue.txt",
                    "mtime": path_mtime(queue_file),
                    "order": idx,
                }
            )
        return items

    def _job_file_item(self, path: Path, status: str) -> dict[str, Any]:
        data = parse_key_values(path)
        rel = safe_relative(self.root, path) or path.name
        match = CLAIMED_SUFFIX_RE.search(path.name)
        node = match.group("node") if match else ""
        worker = match.group("worker") if match else ""
        return {
            "task": data.get("task", ""),
            "mode": data.get("mode", ""),
            "status": status,
            "node": node,
            "worker": worker,
            "job_file": rel,
            "mtime": path_mtime(path),
        }

    def _collect_fallback_worker_results(self, known_keys: set[tuple[str, str]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            worker_dirs = [path for path in self.root.rglob("worker*") if path.is_dir()]
        except OSError:
            return rows
        for worker_dir in sorted(worker_dirs):
            node = ""
            worker = worker_dir.name.removeprefix("worker") if worker_dir.name.startswith("worker") else ""
            try:
                parts = worker_dir.relative_to(self.root).parts
            except ValueError:
                parts = worker_dir.parts
            for part in parts:
                if part.startswith("node"):
                    node = part.removeprefix("node")
                    break
            rows.extend(self._worker_result_rows(worker_dir, "finished.txt", "ok", node, worker, known_keys))
            rows.extend(self._worker_result_rows(worker_dir, "failed.txt", "failed", node, worker, known_keys))
        return rows

    def _worker_result_rows(
        self,
        worker_dir: Path,
        file_name: str,
        status: str,
        node: str,
        worker: str,
        known_keys: set[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        marker = worker_dir / file_name
        for line in read_nonempty_lines(marker):
            parsed = parse_task_mode_line(line)
            if parsed is None or parsed in known_keys:
                continue
            task, mode = parsed
            log_rel = find_task_log(self.root, task, mode, node, worker)
            log_abs = (self.root / log_rel).resolve() if log_rel else None
            timeline = task_timeline_from_log(log_abs, self.state_tail_bytes) if log_abs else {}
            finished_at = timeline.get("mtime") or path_mtime(marker)
            started_at = self._job_started_at(task, mode, node, worker)
            rows.append(
                {
                    "task": task,
                    "mode": mode,
                    "node": node,
                    "worker": worker,
                    "status": status,
                    "exit_code": "" if status == "ok" else "unknown",
                    "log": log_rel,
                    "success_rate": timeline.get("success_rate"),
                    "success": timeline.get("success"),
                    "episodes": timeline.get("episodes"),
                    "latest": timeline.get("latest", ""),
                    "size": timeline.get("size", 0),
                    "mtime": finished_at,
                    "mtime_iso": utc_iso(finished_at) if finished_at else "",
                    "started_at": started_at,
                    "started_at_iso": utc_iso(started_at) if started_at else "",
                    "duration_sec": (finished_at - started_at) if started_at and finished_at >= started_at else None,
                    "duration": format_duration((finished_at - started_at) if started_at and finished_at >= started_at else None),
                    "source": file_name,
                }
            )
            known_keys.add(parsed)
        return rows

    def _collect_logs(self) -> list[dict[str, Any]]:
        paths: list[Path] = []
        try:
            seen: set[Path] = set()
            for pattern in LOG_FILE_PATTERNS:
                for path in self.root.rglob(pattern):
                    if path.is_file() and path not in seen:
                        paths.append(path)
                        seen.add(path)
        except OSError:
            return []

        paths.sort(key=path_mtime, reverse=True)
        logs: list[dict[str, Any]] = []
        for path in paths[: self.max_logs]:
            meta = file_meta(self.root, path)
            if meta is None:
                continue
            if meta["category"] == "task":
                timeline = task_timeline_from_log(path, self.state_tail_bytes)
                meta.update(
                    {
                        "success_rate": timeline.get("success_rate"),
                        "success": timeline.get("success"),
                        "episodes": timeline.get("episodes"),
                    }
                )
            logs.append(meta)
        return logs

    def _collect_nodes(self, queue: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
        discovered: set[str] = set()
        try:
            for path in self.root.glob("node*"):
                if path.is_dir():
                    discovered.add(path.name.removeprefix("node"))
        except OSError:
            pass
        for name in queue.get("done_nodes", []):
            match = re.match(r"\.node(\d+)_done$", name)
            if match:
                discovered.add(match.group(1))
        for row in summary.get("rows", []):
            node = str(row.get("node", ""))
            if node:
                discovered.add(node)
        for item in queue.get("claimed", []):
            node = str(item.get("node", ""))
            if node:
                discovered.add(node)

        nodes: list[dict[str, Any]] = []
        for rank in sorted(discovered, key=numeric_sort_key):
            node_dir = self.root / f"node{rank}"
            server_logs = []
            for log_path in sorted((node_dir / "servers").glob("*.log")) if node_dir.is_dir() else []:
                meta = file_meta(self.root, log_path)
                if meta:
                    server_logs.append(meta)

            workers = []
            worker_dirs = []
            if node_dir.is_dir():
                worker_dirs = [path for path in sorted(node_dir.glob("worker*")) if path.is_dir()]
            known_workers = {path.name.removeprefix("worker") for path in worker_dirs}
            for row in summary.get("rows", []):
                if str(row.get("node", "")) == rank and str(row.get("worker", "")):
                    known_workers.add(str(row.get("worker", "")))
            for item in queue.get("claimed", []):
                if str(item.get("node", "")) == rank and str(item.get("worker", "")):
                    known_workers.add(str(item.get("worker", "")))

            for worker in sorted(known_workers, key=numeric_sort_key):
                worker_dir = node_dir / f"worker{worker}"
                worker_log = worker_dir / "worker.log"
                worker_rel = safe_relative(self.root, worker_log) if worker_log.is_file() else ""
                running = sum(
                    1
                    for item in queue.get("running", [])
                    if str(item.get("node", "")) == rank and str(item.get("worker", "")) == worker
                )
                workers.append(
                    {
                        "worker": worker,
                        "log": worker_rel or "",
                        "latest": latest_line(worker_log) if worker_log.is_file() else "",
                        "finished": count_lines(worker_dir / "finished.txt"),
                        "failed": count_lines(worker_dir / "failed.txt"),
                        "running": running,
                    }
                )

            nodes.append(
                {
                    "rank": rank,
                    "done": (self.root / f".node{rank}_done").is_file(),
                    "servers": server_logs,
                    "workers": workers,
                }
            )
        return nodes

    def _merge_jobs(
        self,
        summary_rows: list[dict[str, Any]],
        queue: dict[str, Any],
        fallback_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in summary_rows:
            jobs.append(row)
            key = unique_job_key(row)
            if key:
                seen.add(key)
        for row in fallback_rows:
            key = unique_job_key(row)
            if key and key not in seen:
                jobs.append(row)
                seen.add(key)
        for item in queue["running"]:
            key = unique_job_key(item)
            if key and key not in seen:
                log_rel = self._guess_task_log(item)
                log_abs = (self.root / log_rel).resolve() if log_rel else None
                timeline = task_timeline_from_log(log_abs, self.state_tail_bytes) if log_abs else {}
                started_at = item.get("mtime") or None
                jobs.append(
                    {
                        "task": item.get("task", ""),
                        "mode": item.get("mode", ""),
                        "node": item.get("node", ""),
                        "worker": item.get("worker", ""),
                        "status": "running",
                        "exit_code": "",
                        "log": log_rel,
                        "success_rate": timeline.get("success_rate"),
                        "success": timeline.get("success"),
                        "episodes": timeline.get("episodes"),
                        "latest": timeline.get("latest", ""),
                        "size": timeline.get("size", 0),
                        "mtime": timeline.get("mtime") or item.get("mtime", 0.0),
                        "mtime_iso": timeline.get("mtime_iso", ""),
                        "started_at": started_at,
                        "started_at_iso": utc_iso(started_at) if started_at else "",
                        "duration_sec": (time.time() - started_at) if started_at else None,
                        "duration": format_duration((time.time() - started_at) if started_at else None),
                        "source": "queue/claimed",
                    }
                )
                seen.add(key)
        for item in queue["pending"]:
            key = unique_job_key(item)
            if key and key not in seen:
                jobs.append(
                    {
                        "task": item.get("task", ""),
                        "mode": item.get("mode", ""),
                        "node": "",
                        "worker": "",
                        "status": "pending",
                        "exit_code": "",
                        "log": "",
                        "success_rate": None,
                        "success": None,
                        "episodes": None,
                        "latest": "",
                        "size": 0,
                        "mtime": item.get("mtime", 0.0),
                        "mtime_iso": utc_iso(item.get("mtime", 0.0)) if item.get("mtime") else "",
                        "started_at": None,
                        "started_at_iso": "",
                        "duration_sec": None,
                        "duration": "",
                        "source": "queue/pending",
                    }
                )
                seen.add(key)

        priority = {"failed": 0, "running": 1, "pending": 2, "ok": 3, "done": 4}
        jobs.sort(
            key=lambda item: (
                priority.get(str(item.get("status", "")), 9),
                str(item.get("task", "")),
                str(item.get("mode", "")),
            )
        )
        return jobs

    def _guess_task_log(self, item: dict[str, Any]) -> str:
        return find_task_log(
            self.root,
            str(item.get("task", "")),
            str(item.get("mode", "")),
            str(item.get("node", "")),
            str(item.get("worker", "")),
        )

    def _job_started_at(self, task: str, mode: str, node: str, worker: str) -> float | None:
        if not task or not mode:
            return None
        if node != "" and worker != "":
            claimed_dir = self.root / "queue" / "claimed"
            for path in claimed_dir.glob(f"*_{task}_{mode}.job.node{node}.worker{worker}"):
                return path_mtime(path)
            worker_log = self.root / f"node{node}" / f"worker{worker}" / "worker.log"
            if worker_log.is_file():
                return path_mtime(worker_log)
        return None

    def _validate(
        self,
        run_env: dict[str, str],
        expected_keys: set[tuple[str, str]],
        jobs: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        issues: list[dict[str, str]] = []
        seen_counts: dict[tuple[str, str], int] = {}
        for job in jobs:
            key = unique_job_key(job)
            if key:
                seen_counts[key] = seen_counts.get(key, 0) + 1
        seen_keys = set(seen_counts)
        if expected_keys:
            missing = sorted(expected_keys - seen_keys)
            extra = sorted(seen_keys - expected_keys)
            if missing:
                issues.append({"level": "warn", "message": f"missing jobs: {self._format_keys(missing)}"})
            if extra:
                issues.append({"level": "warn", "message": f"unexpected jobs: {self._format_keys(extra)}"})
            expected_total = len(expected_keys)
        else:
            expected_total_str = run_env.get("total_jobs", "")
            expected_total = int(expected_total_str) if expected_total_str.isdigit() else 0
        if expected_total and len(jobs) != expected_total:
            issues.append({"level": "warn", "message": f"job count mismatch: expected {expected_total}, observed {len(jobs)}"})
        duplicates = [key for key, count in seen_counts.items() if count > 1]
        if duplicates:
            issues.append({"level": "error", "message": f"duplicate jobs: {self._format_keys(sorted(duplicates))}"})
        for item in summary.get("duplicates", []):
            issues.append({"level": "error", "message": f"duplicate summary row: {item}"})
        stale_claims = [
            item for item in jobs
            if item.get("status") == "running"
            and item.get("started_at")
            and time.time() - float(item["started_at"]) > 24 * 3600
        ]
        if stale_claims:
            issues.append({"level": "warn", "message": f"{len(stale_claims)} running job(s) have been claimed for more than 24h"})
        return {
            "ok": not any(issue["level"] == "error" for issue in issues),
            "issue_count": len(issues),
            "issues": issues,
            "expected_total": expected_total,
            "observed_total": len(jobs),
        }

    def _collect_failures(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        failures = []
        for job in jobs:
            if job.get("status") != "failed":
                continue
            log_ref = str(job.get("log", ""))
            log_rel, log_abs = normalize_log_ref(self.root, log_ref)
            failures.append(
                {
                    "task": job.get("task", ""),
                    "mode": job.get("mode", ""),
                    "node": job.get("node", ""),
                    "worker": job.get("worker", ""),
                    "exit_code": job.get("exit_code", ""),
                    "log": log_rel,
                    "snippet": task_error_snippet(log_abs, self.max_task_log_bytes) if log_abs else "",
                }
            )
            if len(failures) >= self.max_error_snippets:
                break
        return failures

    @staticmethod
    def _success_aggregate(jobs: list[dict[str, Any]]) -> dict[str, Any]:
        weighted_success = 0
        weighted_total = 0
        rates: list[float] = []
        for job in jobs:
            if job.get("status") not in FINAL_JOB_STATUSES:
                continue
            success = job.get("success")
            episodes = job.get("episodes")
            if success is not None and episodes:
                weighted_success += int(success)
                weighted_total += int(episodes)
            if job.get("success_rate") is not None:
                rates.append(float(job["success_rate"]))
        return {
            "weighted_success": weighted_success,
            "weighted_total": weighted_total,
            "weighted_success_rate": (100.0 * weighted_success / weighted_total) if weighted_total else None,
            "mean_task_success_rate": (sum(rates) / len(rates)) if rates else None,
            "parsed_task_count": len(rates),
        }

    @staticmethod
    def _infer_total(
        run_env: dict[str, str],
        expected_keys: set[tuple[str, str]],
        summary: dict[str, Any],
        queue: dict[str, Any],
        jobs: list[dict[str, Any]],
    ) -> int:
        total_jobs = run_env.get("total_jobs", "")
        if total_jobs.isdigit():
            return int(total_jobs)
        if expected_keys:
            return len(expected_keys)
        return max(len(jobs), len(summary["rows"]) + queue["running_count"] + queue["pending_count"])

    @staticmethod
    def _progress(total: int, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        ok = sum(1 for item in jobs if item.get("status") == "ok")
        failed = sum(1 for item in jobs if item.get("status") == "failed")
        done = sum(1 for item in jobs if item.get("status") == "done")
        completed = ok + failed + done
        running = sum(1 for item in jobs if item.get("status") == "running")
        pending = sum(1 for item in jobs if item.get("status") == "pending")
        denominator = max(total, completed + running + pending, 1)
        percent = min(100.0, 100.0 * completed / denominator)
        eta_sec = None
        running_started = [float(item["started_at"]) for item in jobs if item.get("status") == "running" and item.get("started_at")]
        completed_durations = [
            float(item["duration_sec"])
            for item in jobs
            if item.get("duration_sec") and float(item["duration_sec"]) >= 1
        ]
        if pending and completed_durations:
            eta_sec = (sum(completed_durations) / len(completed_durations)) * pending
        return {
            "total": total,
            "completed": completed,
            "ok": ok,
            "failed": failed,
            "done": done,
            "running": running,
            "pending": pending,
            "percent": percent,
            "eta_sec": eta_sec,
            "eta": format_duration(eta_sec),
            "longest_running_sec": (time.time() - min(running_started)) if running_started else None,
            "longest_running": format_duration((time.time() - min(running_started)) if running_started else None),
        }

    @staticmethod
    def _format_keys(keys: list[tuple[str, str]], limit: int = 8) -> str:
        labels = [f"{task}:{mode}" for task, mode in keys[:limit]]
        if len(keys) > limit:
            labels.append(f"... (+{len(keys) - limit} more)")
        return ", ".join(labels)

def json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def detect_benchmark(root: Path) -> str:
    if (root / "run.env").is_file() or (root / "summary.tsv").is_file() or (root / "queue").is_dir():
        return "robotwin"
    try:
        if any(root.glob("node*/worker*")):
            return "robotwin"
    except OSError:
        pass
    return "generic"


def create_adapter(
    root: Path,
    benchmark: str,
    *,
    max_logs: int,
    state_tail_bytes: int,
    max_task_log_bytes: int,
    max_error_snippets: int,
) -> BenchmarkConsoleAdapter:
    selected = detect_benchmark(root) if benchmark == "auto" else benchmark
    adapter_cls: type[BenchmarkConsoleAdapter]
    if selected == "robotwin":
        adapter_cls = SnapshotBuilder
    elif selected == "generic":
        adapter_cls = GenericLogAdapter
    else:
        raise ValueError(f"unsupported benchmark adapter: {benchmark}")
    return adapter_cls(
        root,
        max_logs=max_logs,
        state_tail_bytes=state_tail_bytes,
        max_task_log_bytes=max_task_log_bytes,
        max_error_snippets=max_error_snippets,
    )


def build_handler(
    root: Path,
    *,
    benchmark: str = "auto",
    tail_bytes: int,
    state_tail_bytes: int,
    max_logs: int,
    max_task_log_bytes: int,
    max_error_snippets: int,
    refresh_sec: float,
) -> type[BaseHTTPRequestHandler]:
    root = root.resolve()

    def adapter() -> BenchmarkConsoleAdapter:
        return create_adapter(
            root,
            benchmark,
            max_logs=max_logs,
            state_tail_bytes=state_tail_bytes,
            max_task_log_bytes=max_task_log_bytes,
            max_error_snippets=max_error_snippets,
        )

    class ConsoleHandler(BaseHTTPRequestHandler):
        server_version = "BenchmarkWebControl/1.0"

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html()
                elif parsed.path == "/api/state":
                    self._send_state()
                elif parsed.path == "/api/tail":
                    self._send_tail(parsed.query)
                elif parsed.path == "/api/results.csv":
                    self._send_results_csv()
                elif parsed.path == "/raw":
                    self._send_raw(parsed.query)
                else:
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except BrokenPipeError:
                return
            except Exception as exc:  # Keep the web console alive on bad files.
                if urlparse(self.path).path == "/":
                    self._send_bytes(
                        html_response_bytes(f"{type(exc).__name__}: {exc}"),
                        "text/html; charset=utf-8",
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                else:
                    self._send_json(
                        {"error": type(exc).__name__, "message": str(exc)},
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    )

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def _send_bytes(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send_bytes(json_bytes(data), "application/json; charset=utf-8", status)

        def _send_html(self) -> None:
            payload = INDEX_HTML.replace("__REFRESH_SEC__", repr(refresh_sec)).encode("utf-8")
            self._send_bytes(payload, "text/html; charset=utf-8")

        def _send_state(self) -> None:
            snapshot = adapter().build_state()
            self._send_json(snapshot)

        def _send_results_csv(self) -> None:
            current_adapter = adapter()
            rows = current_adapter.build_results_rows()
            fieldnames = current_adapter.csv_fieldnames(rows)
            self._send_bytes(
                csv_bytes(rows, fieldnames),
                "text/csv; charset=utf-8",
                headers={"Content-Disposition": f"attachment; filename={current_adapter.csv_filename}"},
            )

        def _send_tail(self, query: str) -> None:
            params = parse_qs(query)
            file_param = params.get("file", [""])[0]
            offset_param = params.get("offset", ["-1"])[0]
            max_bytes_param = params.get("max_bytes", [str(tail_bytes)])[0]

            try:
                offset = int(offset_param)
            except ValueError:
                offset = -1
            max_bytes = parse_positive_int(
                max_bytes_param,
                default=tail_bytes,
                minimum=1,
                maximum=5_000_000,
            )

            try:
                path = resolve_requested_file(root, file_param)
            except FileNotFoundError as exc:
                self._send_json({"error": f"file not found: {exc}"}, HTTPStatus.NOT_FOUND)
                return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return

            stat = path.stat()
            size = stat.st_size
            rotated = offset > size
            if offset < 0 or rotated:
                start = max(0, size - max_bytes)
                truncated = start > 0
            else:
                start = offset
                truncated = False
                if size - start > max_bytes:
                    start = max(0, size - max_bytes)
                    truncated = True

            with path.open("rb") as f:
                f.seek(start)
                raw = f.read(max_bytes)
            text = strip_ansi(raw.decode("utf-8", errors="replace"))
            self._send_json(
                {
                    "file": safe_relative(root, path) or path.name,
                    "offset": start,
                    "next_offset": start + len(raw),
                    "size": size,
                    "data": text,
                    "truncated": truncated,
                    "rotated": rotated,
                    "mtime": stat.st_mtime,
                    "mtime_iso": utc_iso(stat.st_mtime),
                }
            )

        def _send_raw(self, query: str) -> None:
            params = parse_qs(query)
            file_param = params.get("file", [""])[0]
            try:
                path = resolve_requested_file(root, file_param)
            except FileNotFoundError as exc:
                self._send_json({"error": f"file not found: {exc}"}, HTTPStatus.NOT_FOUND)
                return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Disposition",
                f"inline; filename={quote(path.name)}",
            )
            self.end_headers()
            with path.open("rb") as f:
                shutil.copyfileobj(f, self.wfile)

    return ConsoleHandler


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark Control</title>
  <style>
    :root {
      --bg: #ece7dc;
      --bg-deep: #d9ded6;
      --panel: rgba(255, 252, 244, 0.90);
      --panel-solid: #fffaf0;
      --ink: #141d1a;
      --muted: #68736c;
      --line: rgba(20, 29, 26, 0.13);
      --line-strong: rgba(20, 29, 26, 0.24);
      --teal: #0d746c;
      --teal-soft: #d7f0eb;
      --amber: #b96f22;
      --amber-soft: #f5dfbf;
      --red: #b42318;
      --red-soft: #f8d6d2;
      --green: #207a43;
      --green-soft: #dff0df;
      --blue: #2d5e99;
      --blue-soft: #dce8f6;
      --terminal: #0d1512;
      --terminal-ink: #d7f6e8;
      --shadow: 0 24px 70px rgba(20, 29, 26, 0.15);
      --shadow-soft: 0 10px 30px rgba(20, 29, 26, 0.08);
      --mono: "JetBrains Mono", "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      --sans: "Aptos", "Trebuchet MS", "Gill Sans", sans-serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at 8% 6%, rgba(13, 116, 108, 0.22), transparent 26rem),
        radial-gradient(circle at 86% 14%, rgba(185, 111, 34, 0.18), transparent 24rem),
        linear-gradient(135deg, var(--bg) 0%, var(--bg-deep) 100%);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .32;
      background-image:
        linear-gradient(rgba(20, 29, 26, .04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(20, 29, 26, .04) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(to bottom, #000 0%, transparent 80%);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 20;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 1rem;
      align-items: center;
      padding: 1rem clamp(1rem, 2.5vw, 2rem);
      border-bottom: 1px solid var(--line);
      background: rgba(236, 231, 220, 0.82);
      backdrop-filter: blur(18px);
    }
    .brand {
      display: flex;
      align-items: center;
      min-width: 0;
      gap: .85rem;
    }
    .mark {
      width: 2.65rem;
      height: 2.65rem;
      flex: 0 0 auto;
      border-radius: 1rem;
      background:
        linear-gradient(135deg, rgba(13, 116, 108, 1), rgba(27, 51, 46, 1));
      box-shadow: var(--shadow-soft);
      position: relative;
      overflow: hidden;
    }
    .mark::after {
      content: "";
      position: absolute;
      inset: 24% -30% auto 28%;
      height: 58%;
      transform: rotate(-18deg);
      background: rgba(255,255,255,.28);
    }
    h1 {
      margin: 0;
      font-size: clamp(1.55rem, 2.5vw, 2.8rem);
      line-height: .95;
      letter-spacing: -0.055em;
    }
    .subline {
      margin-top: .42rem;
      color: var(--muted);
      font-family: var(--mono);
      font-size: .78rem;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 72vw;
    }
    .toolbar {
      display: flex;
      gap: .55rem;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    button, .pill-link {
      border: 1px solid var(--line);
      background: rgba(255, 250, 240, .88);
      color: var(--ink);
      border-radius: 999px;
      padding: .58rem .86rem;
      font: 800 .82rem var(--sans);
      cursor: pointer;
      box-shadow: var(--shadow-soft);
      text-decoration: none;
      transition: transform .16s ease, border-color .16s ease, background .16s ease;
    }
    button:hover, .pill-link:hover { transform: translateY(-1px); border-color: var(--line-strong); }
    input, select {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: .58rem .82rem;
      background: rgba(255, 250, 240, 0.88);
      color: var(--ink);
      font: 700 .82rem var(--sans);
      outline: none;
      min-height: 2.25rem;
    }
    label { display: inline-flex; gap: .35rem; align-items: center; }
    main {
      position: relative;
      z-index: 1;
      padding: 1.1rem clamp(1rem, 2.5vw, 2rem) 2rem;
      display: grid;
      gap: 1rem;
    }
    .banner {
      display: none;
      border: 1px solid rgba(180, 35, 24, .28);
      color: var(--red);
      background: rgba(248, 214, 210, .78);
      border-radius: 1rem;
      padding: .8rem 1rem;
      font-weight: 800;
      box-shadow: var(--shadow-soft);
    }
    .banner.show { display: block; }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 1rem;
      align-items: end;
      padding: 1.25rem;
      border: 1px solid var(--line);
      border-radius: 1.35rem;
      background:
        linear-gradient(135deg, rgba(255, 250, 240, .92), rgba(236, 241, 232, .76));
      box-shadow: var(--shadow);
      overflow: hidden;
      position: relative;
    }
    .hero::after {
      content: "";
      position: absolute;
      right: -5rem;
      top: -5rem;
      width: 17rem;
      height: 17rem;
      border-radius: 50%;
      border: 1.6rem solid rgba(13, 116, 108, .08);
    }
    .hero-title {
      margin: 0 0 .4rem;
      font-size: clamp(1.4rem, 2.2vw, 2.45rem);
      letter-spacing: -0.045em;
    }
    .hero-copy {
      margin: 0;
      color: var(--muted);
      font: 700 .84rem/1.5 var(--mono);
      word-break: break-word;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: .45rem;
      width: fit-content;
      border-radius: 999px;
      padding: .42rem .72rem;
      color: #073f3a;
      background: var(--teal-soft);
      font: 900 .75rem var(--sans);
      letter-spacing: .08em;
      text-transform: uppercase;
      border: 1px solid rgba(13, 116, 108, .22);
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(8, minmax(7.5rem, 1fr));
      gap: .85rem;
    }
    .card, .panel {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 1.15rem;
      box-shadow: var(--shadow-soft);
    }
    .card {
      min-height: 7rem;
      padding: .95rem;
      position: relative;
      overflow: hidden;
      animation: rise .32s ease both;
    }
    .card::after {
      content: "";
      position: absolute;
      inset: auto -24% -46% 20%;
      height: 5.4rem;
      transform: rotate(-8deg);
      background: rgba(13, 116, 108, 0.10);
    }
    .card.success::after { background: rgba(32, 122, 67, 0.14); }
    .card.warning::after { background: rgba(185, 111, 34, 0.16); }
    .card.failed::after { background: rgba(180, 35, 24, 0.14); }
    .card.running::after { background: rgba(185, 111, 34, 0.16); }
    .card.progress::after { background: rgba(45, 94, 153, 0.14); }
    .label {
      text-transform: uppercase;
      letter-spacing: .1em;
      color: var(--muted);
      font-size: .66rem;
      font-weight: 900;
    }
    .value {
      margin-top: .36rem;
      font: 950 clamp(1.55rem, 2.3vw, 2.15rem)/1 var(--sans);
      letter-spacing: -0.052em;
      position: relative;
      z-index: 1;
    }
    .progress-wrap {
      height: .72rem;
      background: rgba(20, 29, 26, 0.08);
      border-radius: 999px;
      overflow: hidden;
      margin-top: .85rem;
      position: relative;
      z-index: 1;
    }
    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, var(--teal), #56aa9f);
      width: 0%;
      border-radius: inherit;
      transition: width .35s ease;
    }
    .masonry {
      position: relative;
      min-height: 1px;
      transition: height .2s ease;
    }
    .masonry > .panel {
      position: absolute;
      width: calc((100% - 1rem) / 2);
      transition: transform .22s ease, width .22s ease;
    }
    .masonry > .panel.wide { width: 100%; }
    .panel { overflow: hidden; }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: .8rem;
      padding: .92rem 1rem;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.34);
    }
    .panel-title {
      margin: 0;
      font-size: 1rem;
      letter-spacing: -0.02em;
    }
    .panel-body { padding: .9rem 1rem 1rem; }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: .65rem;
    }
    .meta {
      border: 1px solid var(--line);
      border-radius: .9rem;
      padding: .68rem .78rem;
      background: rgba(255, 255, 255, 0.42);
      min-width: 0;
    }
    .meta b {
      display: block;
      font-size: .66rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .09em;
      margin-bottom: .28rem;
    }
    .meta span {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font: 800 .82rem var(--mono);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: .82rem;
    }
    th, td {
      padding: .62rem .55rem;
      border-bottom: 1px solid rgba(20, 29, 26, 0.10);
      text-align: left;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: var(--panel-solid);
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .08em;
      font-size: .65rem;
    }
    tbody tr:hover { background: rgba(13, 116, 108, 0.065); }
    .table-wrap {
      max-height: 31rem;
      overflow: auto;
    }
    .status {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: .2rem .53rem;
      font: 900 .68rem var(--sans);
      text-transform: uppercase;
      letter-spacing: .055em;
      white-space: nowrap;
    }
    .status.ok { color: var(--green); background: var(--green-soft); }
    .status.failed { color: var(--red); background: var(--red-soft); }
    .status.running { color: #87500b; background: var(--amber-soft); }
    .status.pending { color: var(--blue); background: var(--blue-soft); }
    .status.done { color: var(--muted); background: rgba(20, 29, 26, .08); }
    .status.task { color: var(--teal); background: var(--teal-soft); }
    .status.worker { color: #87500b; background: var(--amber-soft); }
    .status.server { color: var(--blue); background: var(--blue-soft); }
    .status.log { color: var(--muted); background: rgba(20, 29, 26, .08); }
    .status.warn { color: #87500b; background: var(--amber-soft); }
    .status.error { color: var(--red); background: var(--red-soft); }
    .status.info { color: var(--blue); background: var(--blue-soft); }
    .mono { font-family: var(--mono); }
    .muted { color: var(--muted); }
    .log-row { cursor: pointer; }
    .log-row.active { background: var(--teal-soft); }
    .tail-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto auto;
      gap: .5rem;
      align-items: center;
    }
    .tail-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font: 900 .82rem var(--mono);
    }
    pre {
      margin: 0;
      padding: 1rem;
      min-height: 26rem;
      max-height: 44rem;
      overflow: auto;
      background:
        linear-gradient(180deg, rgba(255,255,255,.025), transparent 12rem),
        var(--terminal);
      color: var(--terminal-ink);
      border-radius: 0 0 1.15rem 1.15rem;
      font: 12px/1.58 var(--mono);
      white-space: pre-wrap;
      word-break: break-word;
    }
    .node-grid, .health-list, .failure-list { display: grid; gap: .68rem; }
    .node, .issue, .failure {
      border: 1px solid var(--line);
      border-radius: .92rem;
      padding: .72rem .78rem;
      background: rgba(255, 255, 255, .38);
    }
    .node h3 { margin: 0 0 .48rem; font-size: .9rem; }
    .small-list { display: grid; gap: .3rem; color: var(--muted); font-size: .78rem; }
    .small-list button { padding: .2rem .48rem; font-size: .7rem; box-shadow: none; margin-left: .35rem; }
    .empty {
      padding: 1.2rem;
      color: var(--muted);
      text-align: center;
      border: 1px dashed var(--line-strong);
      border-radius: .95rem;
      background: rgba(255,255,255,.25);
    }
    .issue { display: flex; align-items: center; gap: .6rem; justify-content: space-between; }
    .failure-head { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; margin-bottom: .45rem; }
    .snippet {
      margin-top: .45rem;
      padding: .72rem;
      max-height: 16rem;
      overflow: auto;
      border-radius: .7rem;
      background: var(--terminal);
      color: var(--terminal-ink);
      font: 12px/1.45 var(--mono);
      white-space: pre-wrap;
    }
    .compact { font-size: .76rem; }
    @keyframes rise {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 1220px) {
      .cards { grid-template-columns: repeat(4, minmax(8rem, 1fr)); }
      .masonry > .panel {
        position: static;
        width: 100%;
        transform: none !important;
        margin-bottom: 1rem;
      }
      .masonry { height: auto !important; }
    }
    @media (max-width: 780px) {
      header, .hero { grid-template-columns: 1fr; align-items: start; }
      .toolbar { justify-content: flex-start; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .meta-grid { grid-template-columns: 1fr; }
      .tail-head { grid-template-columns: 1fr; }
      .subline { max-width: 92vw; }
      main { padding-inline: .75rem; }
      th, td { min-width: 7rem; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="mark" aria-hidden="true"></div>
      <div>
        <h1 id="pageTitle">Benchmark Control</h1>
        <div id="logDir" class="subline">Loading...</div>
      </div>
    </div>
    <div class="toolbar">
      <span id="heartbeat" class="muted mono">--</span>
      <a class="pill-link" href="/api/results.csv">CSV</a>
      <button id="refreshBtn">Refresh</button>
      <label class="muted"><input id="autoRefresh" type="checkbox" checked> auto</label>
    </div>
  </header>

  <main>
    <div id="statusBanner" class="banner"></div>

    <section class="hero">
      <div>
        <span id="benchmarkBadge" class="badge">benchmark</span>
        <h2 class="hero-title">Live run visibility without benchmark-specific UI forks.</h2>
        <p id="heroCopy" class="hero-copy">Waiting for adapter state...</p>
      </div>
      <div class="toolbar">
        <span id="lastUpdate" class="muted mono"></span>
      </div>
    </section>

    <section class="cards" id="cards"></section>

    <section class="panel">
      <div class="panel-head">
        <h2 class="panel-title">Run Metadata</h2>
        <span id="metaSummary" class="muted mono"></span>
      </div>
      <div class="panel-body">
        <div id="metaGrid" class="meta-grid"></div>
      </div>
    </section>

    <section class="masonry" id="masonry">
      <div class="panel wide" data-masonry-priority="0">
        <div class="panel-head">
          <h2 class="panel-title">Work Items</h2>
          <div class="toolbar">
            <select id="jobStatusFilter">
              <option value="">all statuses</option>
              <option value="failed">failed</option>
              <option value="running">running</option>
              <option value="pending">pending</option>
              <option value="ok">ok</option>
              <option value="done">done</option>
              <option value="log">log</option>
            </select>
            <input id="jobFilter" placeholder="filter task/mode/path">
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>Item</th>
                <th>Mode</th>
                <th>Node</th>
                <th>Worker</th>
                <th>Success</th>
                <th>Episodes</th>
                <th>Duration</th>
                <th>Log</th>
              </tr>
            </thead>
            <tbody id="jobsBody"></tbody>
          </table>
        </div>
      </div>

      <div class="panel" data-masonry-priority="1">
        <div class="panel-head">
          <h2 class="panel-title">Run Health</h2>
          <span id="healthSummary" class="muted mono"></span>
        </div>
        <div class="panel-body">
          <div id="healthBody" class="health-list"></div>
        </div>
      </div>

      <div class="panel" data-masonry-priority="2">
        <div class="panel-head">
          <h2 class="panel-title">Failure Snippets</h2>
          <span id="failureSummary" class="muted mono"></span>
        </div>
        <div class="panel-body">
          <div id="failuresBody" class="failure-list"></div>
        </div>
      </div>

      <div class="panel" data-masonry-priority="3">
        <div class="panel-head">
          <h2 class="panel-title">Nodes</h2>
        </div>
        <div class="panel-body">
          <div id="nodes" class="node-grid"></div>
        </div>
      </div>

      <div class="panel" data-masonry-priority="4">
        <div class="panel-head">
          <h2 class="panel-title">Log Browser</h2>
          <div class="toolbar">
            <select id="logCategoryFilter">
              <option value="">all logs</option>
              <option value="task">task</option>
              <option value="worker">worker</option>
              <option value="server">server</option>
              <option value="log">other</option>
            </select>
            <input id="logFilter" placeholder="filter path/text">
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Kind</th>
                <th>Path</th>
                <th>Size</th>
                <th>Updated</th>
                <th>Latest line</th>
              </tr>
            </thead>
            <tbody id="logsBody"></tbody>
          </table>
        </div>
      </div>

      <div class="panel" data-masonry-priority="5">
        <div class="panel-head tail-head">
          <div id="tailTitle" class="tail-title">Select a log to tail</div>
          <select id="tailBytes" title="Tail request size">
            <option value="200000">200 KB</option>
            <option value="1000000">1 MB</option>
            <option value="5000000">5 MB</option>
          </select>
          <label class="muted"><input id="followTail" type="checkbox" checked> follow</label>
          <a id="rawLink" class="pill-link" href="#" target="_blank" rel="noopener">raw</a>
        </div>
        <pre id="tailOutput"></pre>
      </div>
    </section>
  </main>

  <script>
    const REFRESH_SEC = __REFRESH_SEC__;
    let state = null;
    let selectedFile = "";
    let tailOffset = -1;
    let tailTimer = null;

    const $ = (id) => document.getElementById(id);

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[ch]));
    }

    function showBanner(message) {
      const banner = $("statusBanner");
      if (!message) {
        banner.textContent = "";
        banner.classList.remove("show");
        return;
      }
      banner.textContent = message;
      banner.classList.add("show");
    }

    function fmtBytes(bytes) {
      bytes = Number(bytes || 0);
      const units = ["B", "KB", "MB", "GB", "TB"];
      let i = 0;
      while (bytes >= 1024 && i < units.length - 1) {
        bytes /= 1024;
        i += 1;
      }
      return `${bytes.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
    }

    function fmtTime(ts) {
      if (!ts) return "";
      const date = new Date(ts * 1000);
      return date.toLocaleString();
    }

    function fmtPercent(value) {
      return value == null ? "-" : `${Number(value).toFixed(2)}%`;
    }

    function fmtEpisodes(success, episodes) {
      return episodes == null ? "-" : `${success ?? "?"}/${episodes}`;
    }

    function safeClassNames(value) {
      return String(value || "").split(/\s+/)
        .map((part) => part.toLowerCase().replace(/[^a-z0-9_-]/g, "-"))
        .filter(Boolean)
        .join(" ");
    }

    function card(label, value, cls = "", title = "") {
      const extra = safeClassNames(cls);
      const titleAttr = title ? ` title="${esc(title)}"` : "";
      return `<div class="card${extra ? ` ${esc(extra)}` : ""}"${titleAttr}><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`;
    }

    function metricValue(metric) {
      if (metric.value != null && metric.value !== "") return metric.value;
      if (metric.raw_value == null || metric.raw_value === "") return "-";
      if (metric.kind === "percent") return fmtPercent(metric.raw_value);
      return `${metric.raw_value}${metric.unit || ""}`;
    }

    function statusPill(status) {
      const s = status || "done";
      const cls = String(s).toLowerCase().replace(/[^a-z0-9_-]/g, "-");
      return `<span class="status ${esc(cls)}">${esc(s)}</span>`;
    }

    function logButton(rel, label = "tail") {
      if (!rel) return `<span class="muted">-</span>`;
      return `<button data-log="${esc(rel)}">${esc(label)}</button>`;
    }

    function renderCards(data) {
      const progress = data.progress || {};
      const rates = data.rates || {};
      const percent = Number(progress.percent || 0);
      const success = rates.weighted_success_rate ?? rates.mean_task_success_rate;
      const customCards = (data.custom_metrics || []).map((metric) =>
        card(metric.label || metric.id || "Metric", metricValue(metric), metric.class || "", metric.description || "")
      );
      $("cards").innerHTML = [
        card("Total", progress.total ?? 0),
        card("Complete", progress.completed ?? 0),
        card("Success", fmtPercent(success)),
        card("Failed", progress.failed ?? 0, "failed"),
        card("Running", progress.running ?? 0, "running"),
        card("Pending", progress.pending ?? 0),
        card("ETA", progress.eta || "-"),
        `<div class="card progress"><div class="label">Progress</div><div class="value">${percent.toFixed(1)}%</div><div class="progress-wrap"><div class="progress-bar" style="width:${Math.min(100, percent)}%"></div></div></div>`,
        ...customCards
      ].join("");
    }

    function renderMeta(data) {
      const env = data.run_env || {};
      const progress = data.progress || {};
      const rates = data.rates || {};
      const items = [
        ["benchmark", data.benchmark || "generic"],
        ["run_id", env.run_id || data.run?.id || "-"],
        ["policy", env.policy_name || data.run?.policy || "-"],
        ["mode", env.mode || data.run?.mode || "-"],
        ["total", env.total_jobs || progress.total || "-"],
        ["nodes", env.nnodes || (data.nodes || []).length || "-"],
        ["workers/node", env.num_workers_per_node || "-"],
        ["queue_ready", data.queue?.ready ? "yes" : "no"],
        ["eta", progress.eta || "-"],
        ["longest_running", progress.longest_running || "-"],
        ["parsed_rates", rates.parsed_task_count ?? 0],
        ["weighted_success", rates.weighted_total ? `${rates.weighted_success}/${rates.weighted_total}` : "-"],
        ["ckpt_dir", env.ckpt_dir || "-"],
      ];
      $("metaGrid").innerHTML = items.map(([k, v]) => `<div class="meta"><b>${esc(k)}</b><span title="${esc(v)}">${esc(v)}</span></div>`).join("");
      $("lastUpdate").textContent = data.last_update ? `last update ${data.last_update}` : "";
      $("metaSummary").textContent = `${(data.logs || []).length} log(s)`;
    }

    function renderHero(data) {
      const title = data.title || "Benchmark Control";
      const benchmark = data.benchmark || "generic";
      const progress = data.progress || {};
      $("pageTitle").textContent = title;
      document.title = title;
      $("benchmarkBadge").textContent = benchmark;
      $("heroCopy").textContent = `${progress.completed ?? 0}/${progress.total ?? 0} complete, ${progress.running ?? 0} running, ${progress.failed ?? 0} failed. Root: ${data.log_dir || "-"}`;
    }

    function renderHealth(data) {
      const validation = data.validation || {};
      const issues = validation.issues || [];
      $("healthSummary").textContent = issues.length ? `${issues.length} issue(s)` : "ok";
      if (!issues.length) {
        $("healthBody").innerHTML = `<div class="empty">No consistency issues detected.</div>`;
        return;
      }
      $("healthBody").innerHTML = issues.map((issue) => `
        <div class="issue">
          <span>${statusPill(issue.level || "info")}</span>
          <span class="mono compact">${esc(issue.message || "")}</span>
        </div>`).join("");
    }

    function renderFailures(data) {
      const failures = data.failures || [];
      $("failureSummary").textContent = failures.length ? `${failures.length} failed` : "none";
      if (!failures.length) {
        $("failuresBody").innerHTML = `<div class="empty">No failed task rows.</div>`;
        return;
      }
      $("failuresBody").innerHTML = failures.map((failure) => `
        <div class="failure">
          <div class="failure-head">
            ${statusPill("failed")}
            <span class="mono">${esc(failure.task)} ${esc(failure.mode)}</span>
            <span class="muted compact">node=${esc(failure.node || "-")} worker=${esc(failure.worker || "-")} exit=${esc(failure.exit_code || "-")}</span>
            ${logButton(failure.log)}
          </div>
          <div class="snippet">${esc(failure.snippet || "No error snippet found in the scanned tail.")}</div>
        </div>`).join("");
      bindLogButtons($("failuresBody"));
    }

    function renderJobs(data) {
      const text = $("jobFilter").value.trim().toLowerCase();
      const statusFilter = $("jobStatusFilter").value;
      const rows = (data.jobs || []).filter((job) => {
        const hay = `${job.task} ${job.mode} ${job.status} ${job.node} ${job.worker} ${job.log}`.toLowerCase();
        return (!text || hay.includes(text)) && (!statusFilter || job.status === statusFilter);
      });
      $("jobsBody").innerHTML = rows.map((job) => {
        const rate = fmtPercent(job.success_rate);
        const episodes = fmtEpisodes(job.success, job.episodes);
        const duration = job.duration || (job.status === "running" ? "running" : "-");
        const title = [job.latest, job.source ? `source=${job.source}` : ""].filter(Boolean).join("\n");
        return `<tr title="${esc(title)}">
          <td>${statusPill(job.status)}</td>
          <td class="mono">${esc(job.task)}</td>
          <td>${esc(job.mode)}</td>
          <td>${esc(job.node || "-")}</td>
          <td>${esc(job.worker || "-")}</td>
          <td>${esc(rate)}</td>
          <td>${esc(episodes)}</td>
          <td>${esc(duration)}</td>
          <td>${logButton(job.log)}</td>
        </tr>`;
      }).join("") || `<tr><td colspan="9"><div class="empty">No work items found yet.</div></td></tr>`;
      bindLogButtons($("jobsBody"));
    }

    function renderNodes(data) {
      const nodes = data.nodes || [];
      if (!nodes.length) {
        $("nodes").innerHTML = `<div class="empty">No node*/ directories detected for this adapter.</div>`;
        return;
      }
      $("nodes").innerHTML = nodes.map((node) => {
        const servers = (node.servers || []).map((log) => `<div><span class="mono">${esc(log.name)}</span> ${logButton(log.rel)}<br><span>${esc(log.latest || "")}</span></div>`).join("");
        const workers = (node.workers || []).map((worker) => `<div><b>worker${esc(worker.worker)}</b> ok=${esc(worker.finished)} failed=${esc(worker.failed)} running=${esc(worker.running || 0)} ${logButton(worker.log)}<br><span>${esc(worker.latest || "")}</span></div>`).join("");
        return `<div class="node">
          <h3>node${esc(node.rank)} ${node.done ? statusPill("ok") : statusPill("running")}</h3>
          <div class="small-list">${servers || "<div>No server logs.</div>"}</div>
          <hr style="border:0;border-top:1px solid var(--line);margin:.55rem 0;">
          <div class="small-list">${workers || "<div>No workers.</div>"}</div>
        </div>`;
      }).join("");
      bindLogButtons($("nodes"));
    }

    function renderLogs(data) {
      const text = $("logFilter").value.trim().toLowerCase();
      const category = $("logCategoryFilter").value;
      const logs = (data.logs || []).filter((log) => {
        const hay = `${log.rel} ${log.latest} ${log.category}`.toLowerCase();
        return (!text || hay.includes(text)) && (!category || log.category === category);
      });
      $("logsBody").innerHTML = logs.map((log) => {
        const active = selectedFile === log.rel ? "active" : "";
        return `<tr class="log-row ${active}" data-log="${esc(log.rel)}">
          <td>${statusPill(log.category)}</td>
          <td class="mono">${esc(log.rel)}</td>
          <td>${esc(fmtBytes(log.size))}</td>
          <td>${esc(fmtTime(log.mtime))}</td>
          <td>${esc(log.success_rate == null ? (log.latest || "") : `[${fmtPercent(log.success_rate)}] ${log.latest || ""}`)}</td>
        </tr>`;
      }).join("") || `<tr><td colspan="5"><div class="empty">No log files found.</div></td></tr>`;
      [...$("logsBody").querySelectorAll("[data-log]")].forEach((row) => {
        row.addEventListener("click", () => selectLog(row.getAttribute("data-log")));
      });
    }

    function layoutMasonry() {
      const container = $("masonry");
      if (!container) return;
      const panels = [...container.querySelectorAll(":scope > .panel")]
        .sort((a, b) => Number(a.dataset.masonryPriority || 0) - Number(b.dataset.masonryPriority || 0));
      if (!panels.length) return;
      const gap = 16;
      if (window.matchMedia("(max-width: 1220px)").matches) {
        container.style.height = "";
        panels.forEach((panel) => {
          panel.style.transform = "";
          panel.style.width = "";
        });
        return;
      }

      const columnWidth = (container.clientWidth - gap) / 2;
      const heights = [0, 0];
      panels.forEach((panel) => {
        if (panel.classList.contains("wide")) {
          panel.style.width = `${container.clientWidth}px`;
          panel.style.transform = `translate3d(0px, ${Math.max(...heights)}px, 0)`;
          const nextY = Math.max(...heights) + panel.offsetHeight + gap;
          heights[0] = nextY;
          heights[1] = nextY;
          return;
        }
        const column = heights[0] <= heights[1] ? 0 : 1;
        const x = column * (columnWidth + gap);
        panel.style.width = `${columnWidth}px`;
        panel.style.transform = `translate3d(${x}px, ${heights[column]}px, 0)`;
        heights[column] += panel.offsetHeight + gap;
      });
      container.style.height = `${Math.max(...heights)}px`;
    }

    function scheduleMasonry() {
      requestAnimationFrame(() => requestAnimationFrame(layoutMasonry));
    }

    function bindLogButtons(root) {
      [...root.querySelectorAll("button[data-log]")].forEach((button) => {
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          selectLog(button.getAttribute("data-log"));
        });
      });
    }

    async function selectLog(rel) {
      if (!rel) return;
      selectedFile = rel;
      tailOffset = -1;
      $("tailTitle").textContent = rel;
      $("rawLink").href = `/raw?file=${encodeURIComponent(rel)}`;
      $("tailOutput").textContent = "";
      renderLogs(state || {logs: []});
      scheduleMasonry();
      await fetchTail();
    }

    async function fetchTail() {
      if (!selectedFile) return;
      const maxBytes = $("tailBytes") ? $("tailBytes").value : "";
      const url = `/api/tail?file=${encodeURIComponent(selectedFile)}&offset=${tailOffset}&max_bytes=${encodeURIComponent(maxBytes)}`;
      const res = await fetch(url, {cache: "no-store"});
      if (!res.ok) {
        const message = `[tail error] ${res.status} ${res.statusText}`;
        $("tailOutput").textContent += `\n${message}\n`;
        showBanner(message);
        scheduleMasonry();
        return;
      }
      showBanner("");
      const data = await res.json();
      tailOffset = data.next_offset;
      const pre = $("tailOutput");
      if (data.rotated) {
        pre.textContent += "\n[log rotated or truncated; resync]\n";
      }
      if (data.truncated && !pre.textContent) {
        pre.textContent += "[showing tail; earlier bytes omitted]\n";
      }
      if (data.data) {
        pre.textContent += data.data;
        if ($("followTail").checked) {
          pre.scrollTop = pre.scrollHeight;
        }
      }
      scheduleMasonry();
    }

    async function loadState() {
      const res = await fetch("/api/state", {cache: "no-store"});
      if (!res.ok) throw new Error(`state ${res.status}`);
      state = await res.json();
      showBanner("");
      $("logDir").textContent = state.log_dir || "";
      $("heartbeat").textContent = state.now || "";
      renderHero(state);
      renderCards(state);
      renderMeta(state);
      renderHealth(state);
      renderFailures(state);
      renderJobs(state);
      renderNodes(state);
      renderLogs(state);
      scheduleMasonry();
      if (!selectedFile) {
        const interesting = (state.jobs || []).find((job) => ["failed", "running"].includes(job.status) && job.log);
        if (interesting) {
          await selectLog(interesting.log);
        } else if ((state.logs || []).length) {
          await selectLog(state.logs[0].rel);
        }
      }
    }

    function startTailTimer() {
      if (tailTimer) clearInterval(tailTimer);
      tailTimer = setInterval(() => {
        if ($("followTail").checked) fetchTail().catch((err) => showBanner(String(err)));
      }, Math.max(750, REFRESH_SEC * 1000));
    }

    $("refreshBtn").addEventListener("click", () => loadState().catch((err) => showBanner(String(err))));
    $("jobFilter").addEventListener("input", () => {
      if (state) renderJobs(state);
      scheduleMasonry();
    });
    $("jobStatusFilter").addEventListener("change", () => state && renderJobs(state));
    $("jobStatusFilter").addEventListener("change", scheduleMasonry);
    $("logFilter").addEventListener("input", () => state && renderLogs(state));
    $("logFilter").addEventListener("input", scheduleMasonry);
    $("logCategoryFilter").addEventListener("change", () => state && renderLogs(state));
    $("logCategoryFilter").addEventListener("change", scheduleMasonry);
    $("tailBytes").addEventListener("change", () => {
      tailOffset = -1;
      $("tailOutput").textContent = "";
      fetchTail().catch((err) => showBanner(String(err)));
    });
    window.addEventListener("resize", scheduleMasonry);

    setInterval(() => {
      if ($("autoRefresh").checked) loadState().catch((err) => showBanner(String(err)));
    }, Math.max(1000, REFRESH_SEC * 1000));
    startTailTimer();
    loadState().catch((err) => {
      $("logDir").textContent = `Failed to load: ${err}`;
      showBanner(String(err));
      console.error(err);
    });
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        print(f"[ERROR] log_dir not found: {log_dir}", file=sys.stderr)
        return 1
    if args.port <= 0 or args.port > 65535:
        print(f"[ERROR] invalid port: {args.port}", file=sys.stderr)
        return 1

    handler = build_handler(
        log_dir,
        benchmark=args.benchmark,
        tail_bytes=args.tail_bytes,
        state_tail_bytes=args.state_tail_bytes,
        max_logs=args.max_logs,
        max_task_log_bytes=args.max_task_log_bytes,
        max_error_snippets=args.max_error_snippets,
        refresh_sec=args.refresh_sec,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{url_host}:{args.port}/"
    selected_benchmark = detect_benchmark(log_dir) if args.benchmark == "auto" else args.benchmark

    print(f"[INFO] benchmark web control serving: {url}")
    print(f"[INFO] benchmark adapter: {selected_benchmark}")
    print(f"[INFO] log_dir: {log_dir}")
    print("[INFO] Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n[INFO] stopping web console...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
