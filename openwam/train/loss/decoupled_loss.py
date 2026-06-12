"""Decoupled timestep sampling for DreamZero-Flash style training.

During standard joint training, video and action timesteps are sampled
independently but from the same uniform distribution. Decoupled training
biases video timesteps toward high noise (via Beta distribution) while
keeping action timesteps uniform. This teaches the model to predict
clean actions from extremely noisy video context, enabling 1-4 step
action inference at deployment.

Usage:
    from openwam.train.loss.decoupled_loss import DecoupledFlowMatchLoss

    sampler = DecoupledFlowMatchLoss(
        video_beta_a=0.5, video_beta_b=1.0,
    )
    video_t, action_t = sampler.sample_timesteps(batch_size=B)

This module owns the entire training-side decoupled sampling logic.
The matching deployment-side schedules live in
the removed ``decoupled_schedule`` deploy module (see git history).
"""

import torch


class DecoupledFlowMatchLoss:
    """Decoupled noise timestep sampler for flow matching training.

    Despite the historical name, this is not a standalone loss — it
    provides Beta-distributed video timestep sampling (with uniform
    action timesteps) used by ``BaseWAMArchitecture.compute_loss``
    when ``training.decoupled.enabled=true``.

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
        After warmup, video timesteps are drawn from a Beta distribution
        biased toward high noise (samples cluster near ``num_train_timesteps``)
        while action timesteps stay uniform.

        Args:
            batch_size: Number of samples in the batch.
            current_step: Current training step (for warmup logic).
            device: Target device.

        Returns:
            (video_timesteps, action_timesteps) each (batch_size,),
            float tensors with values in [0, num_train_timesteps).
        """
        if current_step < self.warmup_steps:
            video_t = torch.randint(0, self.num_train_timesteps, (batch_size,)).float()
            action_t = torch.randint(0, self.num_train_timesteps, (batch_size,)).float()
            return video_t.to(device), action_t.to(device)

        # Beta(a, b) with a < b peaks near 0; flipping via (1 - x) makes
        # samples cluster near num_train_timesteps (i.e. high noise).
        beta_dist = torch.distributions.Beta(self.video_beta_a, self.video_beta_b)
        beta_samples = beta_dist.sample((batch_size,))
        video_t = ((1.0 - beta_samples) * self.num_train_timesteps).clamp(0, self.num_train_timesteps - 1)
        action_t = torch.randint(0, self.num_train_timesteps, (batch_size,)).float()
        return video_t.to(device), action_t.to(device)
