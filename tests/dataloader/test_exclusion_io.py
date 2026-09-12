from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from openwam.dataloader.utils import exclusion_io


def test_lock_sidecar_blocks_another_process_and_remains_persistent(tmp_path):
    target = tmp_path / "excluded_episodes.json"
    ready = tmp_path / "ready"
    acquired = tmp_path / "acquired"
    script = """
import runpy
import sys
from pathlib import Path

# Exercise this stdlib-only utility without importing every dataset and torch.
locked_exclusion_files = runpy.run_path(sys.argv[1])["locked_exclusion_files"]
target, ready, acquired = map(Path, sys.argv[2:])
ready.write_text("ready")
with locked_exclusion_files([target]):
    acquired.write_text("acquired")
"""

    process = None
    try:
        with exclusion_io.locked_exclusion_files([target]):
            process = subprocess.Popen(
                [sys.executable, "-c", script, exclusion_io.__file__, str(target), str(ready), str(acquired)],
                cwd=Path(__file__).resolve().parents[2],
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 30
            while not ready.exists():
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    pytest.fail(f"Lock contender exited before becoming ready: {stderr}")
                assert time.monotonic() < deadline, "Lock contender did not become ready within 30 seconds"
                time.sleep(0.01)
            time.sleep(0.05)
            assert process.poll() is None
            assert not acquired.exists()

        _, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        assert acquired.read_text() == "acquired"
        assert target.with_name(f".{target.name}.lock").exists()
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.communicate()


def test_atomic_publish_uses_host_pid_uuid_unique_temp_names(tmp_path, monkeypatch):
    target = tmp_path / "excluded_episodes.json"
    uuids = iter([SimpleNamespace(hex="a" * 32), SimpleNamespace(hex="b" * 32)])
    real_replace = os.replace
    temp_names: list[str] = []

    def _capture_replace(source, destination):
        temp_names.append(Path(source).name)
        real_replace(source, destination)

    monkeypatch.setattr(exclusion_io.socket, "gethostname", lambda: "host/name")
    monkeypatch.setattr(exclusion_io.os, "getpid", lambda: 42)
    monkeypatch.setattr(exclusion_io.uuid, "uuid4", lambda: next(uuids))
    monkeypatch.setattr(exclusion_io.os, "replace", _capture_replace)

    exclusion_io.atomic_publish_text(target, "first")
    exclusion_io.atomic_publish_text(target, "second")

    assert temp_names == [
        f".{target.name}.host_name.42.{'a' * 32}.tmp",
        f".{target.name}.host_name.42.{'b' * 32}.tmp",
    ]
    assert target.read_text() == "second"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_atomic_publish_cleans_its_temp_after_replace_failure(tmp_path, monkeypatch):
    target = tmp_path / "excluded_episodes.json"
    monkeypatch.setattr(
        exclusion_io.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        exclusion_io.atomic_publish_text(target, "payload")

    assert not target.exists()
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
