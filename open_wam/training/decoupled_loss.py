"""Decoupled noise training loss for DreamZero-Flash style training.

During standard joint training, video and action timesteps are sampled
independently but from the same uniform distribution. Decoupled training
biases video timesteps toward high noise (via Beta distribution) while
keeping action timesteps uniform. This teaches the model to predict
clean actions from extremely noisy video context, enabling 1-4 step
action inference at deployment.

Usage:
    from open_wam.training.decoupled_loss import DecoupledFlowMatchLoss

    loss_fn = DecoupledFlowMatchLoss(
        video_beta_a=0.5, video_beta_b=1.0,
    )
    # Use in place of standard FlowMatchVideoActionSFTLoss
"""

import torch
import numpy as np
from typing import Optional

from open_wam.inference.optimizations.decoupled_schedule import sample_decoupled_timesteps


class DecoupledFlowMatchLoss:
    """Decoupled noise sampling strategy for flow matching training.

    Wraps the standard loss computation with Beta-distributed video
    timestep sampling. This is not a standalone loss function — it
    provides the timestep sampling logic that should be used with
    the existing FlowMatchVideoActionSFTLoss.

    Args:
        video_beta_a: Beta distribution alpha for video timesteps.
            Lower = more bias toward high noise. Default 0.5.
        video_beta_b: Beta distribution beta for video timesteps.
            Default 1.0.
        num_train_timesteps: Total timestep range. Default 1000.
        warmup_steps: Number of training steps to use standard uniform
            sampling before switching to decoupled. Default 0 (immediate).
    """

    def __init__(
        self,
        video_beta_a: float = 0.5,
        video_beta_b: float = 1.0,
        num_train_timesteps: int = 1000,
        warmup_steps: int = 0,
    ):
        self.video_beta_a = video_beta_a
        self.video_beta_b = video_beta_b
        self.num_train_timesteps = num_train_timesteps
        self.warmup_steps = warmup_steps

    def sample_timesteps(
        self,
        batch_size: int,
        current_step: int = 0,
        device: str = "cpu",
    ) -> tuple:
        """Sample decoupled video and action timesteps.

        During warmup, uses standard uniform sampling for both modalities.
        After warmup, switches to Beta-distributed video timesteps.

        Args:
            batch_size: Number of samples in the batch.
            current_step: Current training step (for warmup logic).
            device: Target device.

        Returns:
            (video_timesteps, action_timesteps) each (batch_size,).
        """
        if current_step < self.warmup_steps:
            # Standard uniform sampling during warmup
            video_t = torch.randint(0, self.num_train_timesteps, (batch_size,)).float()
            action_t = torch.randint(0, self.num_train_timesteps, (batch_size,)).float()
            return video_t.to(device), action_t.to(device)

        return sample_decoupled_timesteps(
            batch_size=batch_size,
            num_train_timesteps=self.num_train_timesteps,
            video_beta_a=self.video_beta_a,
            video_beta_b=self.video_beta_b,
            action_uniform=True,
            device=device,
        )
