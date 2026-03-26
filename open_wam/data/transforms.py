"""Image transforms re-exported from the legacy dataset module."""

import sys
from pathlib import Path

_WAM_DIR = str(Path(__file__).resolve().parent.parent.parent / "examples" / "wanvideo" / "wam")
if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)

from video_action_dataset import (  # noqa: E402
    _crop_and_resize as crop_and_resize,
    _pad_and_resize as pad_and_resize,
    _resize_frame as resize_frame,
)

__all__ = ["crop_and_resize", "pad_and_resize", "resize_frame"]
