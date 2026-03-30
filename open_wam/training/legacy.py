"""Legacy training boundary helpers for OpenWAM's package-native entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WAM_DIR = str(_PROJECT_ROOT / "examples" / "wanvideo" / "wam")

if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train_video_action import FlowMatchVideoActionSFTLoss, VideoActionTrainingModule  # noqa: E402
from diffsynth.diffusion import ModelLogger  # noqa: E402
from diffsynth.diffusion.runner import launch_training_task  # noqa: E402

__all__ = [
    "FlowMatchVideoActionSFTLoss",
    "ModelLogger",
    "VideoActionTrainingModule",
    "launch_training_task",
]
