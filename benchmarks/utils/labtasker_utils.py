"""Small, benchmark-agnostic helpers for Labtasker entry points."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from labtasker import Client, Task


def new_submission_id() -> str:
    """Return a readable UTC timestamp with a short collision suffix."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"sub-{timestamp}-{secrets.token_hex(2)}"


def print_submission_id(submission_id: str) -> None:
    """Highlight the copyable ID in terminals, without escape codes in logs."""
    label = f"SUBMISSION_ID={submission_id}"
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
        label = f"\033[1;36m{label}\033[0m"
    print(f"\n{label}\n", flush=True)


def create_client(*, queue: str | None, auto_start_local_server: bool = False):
    """Create a current-Labtasker Client with explicit local-start authority."""

    import labtasker

    return labtasker.Client(queue=queue, auto_start_local_server=auto_start_local_server)


def print_summary_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Print a compact aligned table without adding a formatting dependency."""

    rendered = [[str(value) for value in row] for row in rows]
    if any(len(row) != len(headers) for row in rendered):
        raise ValueError("summary table rows must match the headers")
    widths = [max(len(header), *(len(row[index]) for row in rendered)) for index, header in enumerate(headers)]

    def line(values: Sequence[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True)).rstrip()

    print(title)
    print(line(headers))
    print(line(["-" * width for width in widths]))
    for row in rendered:
        print(line(row))


def derive_task_id(submission_id: str, task_index: int) -> str:
    """Derive a Labtasker ID so resubmitting one submission is idempotent."""

    digest = hashlib.sha256(f"{submission_id}\0{task_index}".encode()).digest()[:9]
    return "t_" + base64.urlsafe_b64encode(digest).decode()


def build_submission_filter(benchmark: str, submission_id: str, operation: str | None = None) -> str:
    """Build an exact filter without duplicating benchmark args in metadata."""

    clauses = [
        f"metadata.benchmark == {json.dumps(benchmark)}",
        f"metadata.submission_id == {json.dumps(submission_id)}",
    ]
    if operation is not None:
        clauses.append(f"args.operation == {json.dumps(operation)}")
    return " and ".join(clauses)


def list_submission_tasks(
    client: Client, benchmark: str, submission_id: str, operation: str | None = None
) -> list[Task]:
    """Read all pages so summaries and resubmission checks include every batch.

    Query the Server rather than local attempt files, which may contain retries
    or results whose completion was never accepted.
    """

    expression = build_submission_filter(benchmark, submission_id, operation)
    tasks: list[Task] = []
    cursor: str | None = None
    while True:
        page = client.list_tasks(filter=expression, limit=100, cursor=cursor)
        tasks.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            return tasks


def submit_tasks(
    client: Client,
    inputs: Iterable[dict[str, Any]],
    *,
    benchmark: str,
    submission_id: str,
    route: str,
    names: Sequence[str],
    max_attempts: int,
    priority: int,
) -> list[str]:
    """Resume partial submission using Labtasker's native idempotent Task creation.

    Resubmit the same definitions and IDs; existing Tasks keep their state and
    only missing Tasks are created. Reusing a submission ID with another
    definition is unsupported.
    """

    inputs = list(inputs)
    task_names = list(names)
    if len(task_names) != len(inputs):
        raise ValueError("Task names and inputs must have the same length")
    submission_unit_count = sum(item.get("num_episodes", item.get("num_trials", 0)) for item in inputs)

    def task_metadata(task_index: int, args: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            "benchmark": benchmark,
            "submission_id": submission_id,
            "submission_task_index": task_index,
            "submission_task_count": len(inputs),
            "submission_unit_count": submission_unit_count,
        }
        for kind, start_key, count_key, total_key in (
            ("episode", "episode_start", "num_episodes", "total_episodes"),
            ("trial", "trial_start", "num_trials", "total_trials"),
        ):
            if start_key in args:
                metadata.update(
                    range_kind=kind,
                    range_start=args[start_key],
                    range_count=args[count_key],
                    range_total=args[total_key],
                )
                break
        return metadata

    return [
        client.submit_task(
            args,
            task_id=derive_task_id(submission_id, task_index),
            name=task_name,
            routes=[route],
            metadata=task_metadata(task_index, args),
            max_attempts=max_attempts,
            priority=priority,
        ).id
        for task_index, (args, task_name) in enumerate(zip(inputs, task_names, strict=True))
    ]


def validate_submission_coverage(tasks: Sequence[Task], *, skip_coverage_check: bool = False) -> None:
    """Reject incomplete submissions unless explicitly allowed for summary."""

    if skip_coverage_check:
        print(
            "WARNING: task completeness is not checked (--skip-coverage-check); "
            "missing or overlapping ranges may go undetected. Summary may be partial "
            "or double-counted; result validation remains enabled.",
            file=sys.stderr,
        )
        return
    if not tasks:
        return
    if any("submission_task_count" not in (task.metadata or {}) for task in tasks):
        raise ValueError(
            "submission completeness is unverifiable because one or more Tasks "
            "predate persisted coverage metadata; backfill metadata or summarize with "
            "--skip-coverage-check if completeness cannot be verified"
        )
    expected_counts = {(task.metadata or {}).get("submission_task_count") for task in tasks}
    expected_units = {(task.metadata or {}).get("submission_unit_count") for task in tasks}
    if len(expected_counts) != 1 or len(expected_units) != 1:
        raise ValueError("submission Tasks disagree about expected coverage")
    expected = expected_counts.pop()
    unit_count = expected_units.pop()
    indices = {(task.metadata or {}).get("submission_task_index") for task in tasks}
    if type(expected) is not int or expected <= 0 or len(tasks) != expected or indices != set(range(expected)):
        raise ValueError(
            f"incomplete submission: found {len(tasks)} Tasks with {len(indices)} unique indices, "
            f"expected {expected}; "
            "rerun the original submit command with the same submission ID"
        )
    ranges: dict[tuple[object, ...], list[tuple[int, int, int]]] = {}
    observed_units = 0
    ranged_tasks = 0
    for task in tasks:
        metadata = task.metadata or {}
        range_specs = {
            "episode": ("episode_start", "num_episodes", "total_episodes"),
            "trial": ("trial_start", "num_trials", "total_trials"),
        }
        arg_kinds = [kind for kind, keys in range_specs.items() if any(key in task.args for key in keys)]
        metadata_has_range = any(
            key in metadata for key in ("range_kind", "range_start", "range_count", "range_total")
        )
        if not arg_kinds and not metadata_has_range:
            continue
        kind = metadata.get("range_kind")
        if len(arg_kinds) != 1 or kind != arg_kinds[0]:
            raise ValueError(f"submission Task {task.id} has incomplete persisted range coverage")
        start_key, count_key, total_key = range_specs[kind]
        if not all(key in task.args for key in (start_key, count_key, total_key)) or (
            metadata.get("range_start"),
            metadata.get("range_count"),
            metadata.get("range_total"),
        ) != (task.args.get(start_key), task.args.get(count_key), task.args.get(total_key)):
            raise ValueError(f"submission Task {task.id} has invalid persisted range coverage")
        start, count, total = (metadata.get("range_start"), metadata.get("range_count"), metadata.get("range_total"))
        if (
            type(start) is not int
            or type(count) is not int
            or type(total) is not int
            or start < 0
            or count <= 0
            or total <= 0
            or start + count > total
        ):
            raise ValueError(f"submission Task {task.id} has invalid persisted range bounds")
        if kind == "episode":
            group = (kind, task.args.get("mode"), task.args.get("task"), task.args.get("instruction_type"))
        else:
            group = (kind, task.args.get("suite"), task.args.get("task_id"))
        ranges.setdefault(group, []).append((start, count, total))
        ranged_tasks += 1
        observed_units += count
    if type(unit_count) is not int or unit_count < 0 or observed_units != unit_count:
        raise ValueError(f"incomplete submission ranges: found {observed_units}/{unit_count} expected units")
    if (unit_count > 0 and ranged_tasks != len(tasks)) or (unit_count == 0 and ranged_tasks != 0):
        raise ValueError("submission Tasks disagree about whether range coverage is required")
    for group, group_ranges in ranges.items():
        totals = {total for _, _, total in group_ranges}
        cursor = 0
        if len(totals) != 1:
            raise ValueError(f"submission range group {group!r} disagrees about its total")
        for start, count, _ in sorted(group_ranges):
            if start != cursor:
                raise ValueError(f"submission range group {group!r} is not contiguous at unit {cursor}")
            cursor += count
        total = totals.pop()
        if cursor != total:
            raise ValueError(f"submission range group {group!r} covers {cursor}/{total} expected units")


def count_task_statuses(tasks: Sequence[Task]) -> dict[str, int]:
    """Collapse cancelled into failed and keep the requested compact view."""

    counts = {"succeeded": 0, "pending": 0, "running": 0, "failed": 0}
    for task in tasks:
        status = "failed" if task.status == "cancelled" else task.status
        if status not in counts:
            raise ValueError(f"unsupported Labtasker status: {task.status}")
        counts[status] += 1
    return counts | {"expected": len(tasks)}


@contextmanager
def project_context() -> Iterator[None]:
    """Make local submissions and Workers select the same repository project.

    Entry scripts may be launched from different directories. Temporarily using
    the repository root avoids accidentally selecting separate local Servers.
    """
    previous = Path.cwd()
    os.chdir(Path(__file__).resolve().parents[2])
    try:
        yield
    finally:
        os.chdir(previous)
