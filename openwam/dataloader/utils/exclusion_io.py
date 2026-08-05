"""Shared transactional I/O for exclusion artifacts."""

from __future__ import annotations

import fcntl
import os
import re
import socket
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator


@contextmanager
def locked_exclusion_files(paths: Iterable[str | Path]) -> Iterator[None]:
    """Lock exclusion sidecars in a stable order for a read/merge/publish transaction."""
    targets = sorted({Path(path).resolve() for path in paths}, key=lambda path: str(path))
    handles = []
    try:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            lock_path = target.with_name(f".{target.name}.lock")
            handle = lock_path.open("a+")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def atomic_publish_text(path: str | Path, text: str) -> None:
    """Publish text with a host/PID/UUID-unique same-directory temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    hostname = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname()) or "unknown-host"
    tmp = target.with_name(f".{target.name}.{hostname}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
