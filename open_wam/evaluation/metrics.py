"""Video and action quality metrics, re-exported from legacy eval_robotwin."""

import sys
from pathlib import Path

_WAM_DIR = str(Path(__file__).resolve().parent.parent.parent / "examples" / "wanvideo" / "wam")
if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)

from eval_robotwin import compute_video_metrics  # noqa: E402

__all__ = ["compute_video_metrics"]
