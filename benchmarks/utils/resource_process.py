#!/usr/bin/env python3
"""Run one resource command; EOF on stdin means its owning Worker is gone."""

import os
import select
import signal
import subprocess
import sys
import time


def signal_group(pid, signum):
    # macOS can transiently return EPERM during process-group teardown.
    for attempt in range(3):
        try:
            os.killpg(pid, signum)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.1)


def main(command):
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # The command cannot inherit the owner's lifeline or consume terminal input.
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, close_fds=True, start_new_session=True)
    try:
        while child.poll() is None and not stopping:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable and not os.read(sys.stdin.fileno(), 1):
                stopping = True
        code = child.poll()
        return 130 if code is None else code if code >= 0 else 128 - code
    finally:
        # Shorter than ProcessRegistry's 15-second escalation budget.
        signal_group(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        signal_group(child.pid, signal.SIGKILL)
        child.wait()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
