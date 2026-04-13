"""Loss functions for joint video-action training."""

from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss

# Backward-compat alias
FlowMatchVideoActionSFTLoss = FlowMatchVideoActionLoss

__all__ = ["FlowMatchVideoActionLoss", "FlowMatchVideoActionSFTLoss"]
