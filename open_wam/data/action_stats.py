"""Action statistics computation, re-exported from legacy module."""

import sys
from pathlib import Path

_WAM_DIR = str(Path(__file__).resolve().parent.parent.parent / "examples" / "wanvideo" / "wam")
if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)

from compute_action_stats import (  # noqa: E402
    compute_action_stats,
    compute_multitask_robotwin_stats,
    parse_tasks_file,
)

__all__ = [
    "compute_action_stats",
    "compute_multitask_robotwin_stats",
    "parse_tasks_file",
]
