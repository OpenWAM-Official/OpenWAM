"""Loss functions for joint video-action training.

Provides both the standalone FlowMatchVideoActionLoss class and the legacy
FlowMatchVideoActionSFTLoss function for backward compatibility.
"""

from open_wam.training.flow_match_loss import FlowMatchVideoActionLoss
from open_wam.training.legacy import FlowMatchVideoActionSFTLoss

__all__ = ["FlowMatchVideoActionLoss", "FlowMatchVideoActionSFTLoss"]
