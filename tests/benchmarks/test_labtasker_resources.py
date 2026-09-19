"""Exercise the owner lifeline and process-group cleanup without a simulator."""

import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "utils" / "resource_process.py"


def gone(pid):
    stat = Path(f"/proc/{pid}/stat")
    try:
        # A killed grandchild may remain as a zombie when container PID 1 does
        # not reap orphans. It is no longer running and therefore counts as gone.
        if stat.read_text().split(") ", 1)[1].split()[0] == "Z":
            return True
    except (FileNotFoundError, IndexError):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("resource did not reach expected state")


def test_owner_eof_kills_uncooperative_command(tmp_path):
    marker = tmp_path / "pid"
    command = f"import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(120)"
    with subprocess.Popen([sys.executable, SCRIPT, sys.executable, "-c", command], stdin=subprocess.PIPE) as owner:
        wait_for(marker.exists)
        pid = int(marker.read_text())
        owner.stdin.close()
        assert owner.wait(timeout=10) == 130
        wait_for(lambda: gone(pid))


def test_normal_exit_cleans_grandchild(tmp_path):
    marker = tmp_path / "grandchild"
    command = f"import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); Path({str(marker)!r}).write_text(str(p.pid))"
    with subprocess.Popen([sys.executable, SCRIPT, sys.executable, "-c", command], stdin=subprocess.PIPE) as owner:
        assert owner.wait(timeout=10) == 0
        wait_for(lambda: gone(int(marker.read_text())))
