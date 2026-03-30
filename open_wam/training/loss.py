"""Loss functions for joint video-action training.

Provides both the standalone FlowMatchVideoActionLoss class and the legacy
FlowMatchVideoActionSFTLoss function for backward compatibility.
"""

from open_wam.training.flow_match_loss import FlowMatchVideoActionLoss

# Legacy re-export for backward compatibility
import sys
from pathlib import Path

_WAM_DIR = str(Path(__file__).resolve().parent.parent.parent / "examples" / "wanvideo" / "wam")
if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)

from train_video_action import FlowMatchVideoActionSFTLoss  # noqa: E402

__all__ = ["FlowMatchVideoActionLoss", "FlowMatchVideoActionSFTLoss"]
