"""Training boundary helpers for OpenWAM's package-native entrypoints.

.. deprecated::
    This module wraps ``third_party/diffsynth`` training infrastructure.
    Use :class:`~open_wam.training.native_trainer.NativeTrainer` instead,
    which has no legacy dependencies.
"""

from __future__ import annotations

from open_wam.training.video_action_module import VideoActionTrainingModule
from open_wam.training.flow_match_loss import FlowMatchVideoActionLoss
from third_party.diffsynth.diffusion import ModelLogger
from third_party.diffsynth.diffusion.runner import launch_training_task

# Backward-compat alias for the old function-style loss
FlowMatchVideoActionSFTLoss = FlowMatchVideoActionLoss

__all__ = [
    "FlowMatchVideoActionLoss",
    "FlowMatchVideoActionSFTLoss",
    "ModelLogger",
    "VideoActionTrainingModule",
    "launch_training_task",
]
