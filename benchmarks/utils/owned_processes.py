"""Scoped child-process ownership and cleanup, independent of any benchmark."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


class ProcessRegistry:
    """Track only child processes owned by this caller for scoped cleanup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: list[subprocess.Popen] = []
        self._logs = []

    def add(self, process: subprocess.Popen, log_handle=None) -> None:
        with self._lock:
            self._processes.append(process)
            if log_handle is not None:
                self._logs.append(log_handle)

    def terminate_all(self, timeout=15) -> None:
        with self._lock:
            processes, self._processes = self._processes, []
            logs, self._logs = self._logs, []
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and any(p.poll() is None for p in processes):
            time.sleep(0.2)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in processes:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass

        for handle in logs:
            handle.close()


def spawn_resource(command, *, cwd=None, env=None, stdout=None):
    """Launch through the sibling guardian; unspecified options inherit normally."""
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("resource_process.py")), *command],
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
