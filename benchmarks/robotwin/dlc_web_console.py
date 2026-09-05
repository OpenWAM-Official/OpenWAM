#!/usr/bin/env python3
"""Compatibility wrapper for the unified benchmark web control console.

Prefer the generic entrypoint for new usage:

    python benchmarks/web_control.py <log_dir> --benchmark robotwin --host 0.0.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.web_control import (  # noqa: E402,F401
    BenchmarkConsoleAdapter,
    GenericLogAdapter,
    SnapshotBuilder,
    build_handler,
    create_adapter,
    detect_benchmark,
)
from benchmarks.web_control import main as _web_control_main  # noqa: E402


def main() -> int:
    if not any(arg == "--benchmark" or arg.startswith("--benchmark=") for arg in sys.argv[1:]):
        sys.argv[1:1] = ["--benchmark", "robotwin"]
    return _web_control_main()


if __name__ == "__main__":
    raise SystemExit(main())
