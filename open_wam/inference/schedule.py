"""Denoising schedule generators, re-exported from legacy joint_inference."""

import sys
from pathlib import Path
from typing import List, Tuple

_WAM_DIR = str(Path(__file__).resolve().parent.parent.parent / "examples" / "wanvideo" / "wam")
if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)

from joint_inference import (  # noqa: E402
    Schedule,
    schedule_sync,
    schedule_video_leading,
    schedule_cascade,
    schedule_action_only,
    make_schedule,
)

__all__ = [
    "Schedule",
    "schedule_sync",
    "schedule_video_leading",
    "schedule_cascade",
    "schedule_action_only",
    "make_schedule",
]
