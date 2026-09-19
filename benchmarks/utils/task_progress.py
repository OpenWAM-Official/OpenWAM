"""Attempt-local progress bridge for evaluators running in child environments."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def _validated(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    completed, total = value.get("completed"), value.get("total")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, (int, float))
        or isinstance(total, bool)
        or not isinstance(total, (int, float))
        or not math.isfinite(completed)
        or not math.isfinite(total)
        or total <= 0
        or completed < 0
        or completed > total
    ):
        raise ValueError("progress requires finite numbers with 0 <= completed <= total and total > 0")
    # Round-trip here also rejects Path, NumPy, tensor, and other non-JSON data.
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one child-to-Worker progress snapshot."""

    value = _validated(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ProgressFileReporter:
    """Forward changed child snapshots through Labtasker's progress API."""

    def __init__(
        self,
        path: Path,
        *,
        base: Mapping[str, Any],
    ) -> None:
        self.path = Path(path)
        self.base = dict(base)
        self._last_bytes: bytes | None = None

    def poll(self) -> bool:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return False
        if data == self._last_bytes:
            return False
        try:
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("progress snapshot must be an object")
            value = _validated(self.base | payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid evaluator progress snapshot {self.path}: {exc}") from exc
        try:
            # Evaluator environments only need write_progress, not Labtasker.
            import labtasker

            labtasker.report_progress(value)
        except ImportError:
            pass
        self._last_bytes = data
        return True
