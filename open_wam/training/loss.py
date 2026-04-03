"""Loss functions for joint video-action training."""

from open_wam.training.flow_match_loss import FlowMatchVideoActionLoss

# Backward-compat alias
FlowMatchVideoActionSFTLoss = FlowMatchVideoActionLoss

__all__ = ["FlowMatchVideoActionLoss", "FlowMatchVideoActionSFTLoss"]
