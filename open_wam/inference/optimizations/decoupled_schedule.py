"""Decoupled noise schedules for accelerated action inference.

DreamZero-Flash insight: by training with decoupled video/action noise
levels (video biased toward high noise via Beta distribution, action
sampled independently), the model learns to predict clean actions even
from extremely noisy video context. At inference time, this allows
action denoising in as few as 1-4 steps.

This module provides:
- Inference schedules: ``schedule_decoupled_flash`` for minimal-step action
- Training utilities: ``sample_decoupled_timesteps`` for Beta-distributed
  video noise during training
"""

from typing import List, Optional, Tuple

import torch
import numpy as np

# Type alias (same as joint_inference.py)
Schedule = List[Tuple[float, float]]


def _base_timesteps(num_steps: int, shift: float) -> List[float]:
    """Generate Wan-style descending timesteps."""
    from third_party.diffsynth.diffusion import FlowMatchScheduler
    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(num_steps, shift=shift)
    return scheduler.timesteps.tolist()


def schedule_decoupled_flash(
    action_steps: int = 1,
    shift: float = 5.0,
) -> Schedule:
    """DreamZero-Flash schedule: video is clean, action denoises in minimal steps.

    The video stream starts and stays at sigma=0 (clean). Only the action
    stream denoises, using ``action_steps`` steps. This is the fastest
    deployment schedule, enabled by decoupled noise training.

    Args:
        action_steps: Number of denoising steps for actions (1-4 typical).
        shift: Timestep shift parameter for Wan scheduler.

    Returns:
        Schedule of (t_video=0, t_action) pairs.
    """
    action_ts = _base_timesteps(action_steps, shift)
    schedule = [(0.0, t_a) for t_a in action_ts] + [(0.0, 0.0)]
    return schedule


def schedule_decoupled_asymmetric(
    video_steps: int = 10,
    action_steps: int = 2,
    shift: float = 5.0,
) -> Schedule:
    """Asymmetric schedule: video uses many steps, action uses few.

    A middle ground between full sync and flash mode. Video denoises
    with ``video_steps`` while action only needs ``action_steps`` thanks
    to decoupled training.

    The action denoising happens during the last ``action_steps`` of the
    video schedule, so total steps = video_steps.

    Args:
        video_steps: Number of video denoising steps.
        action_steps: Number of action denoising steps (must be <= video_steps).
        shift: Timestep shift parameter.

    Returns:
        Schedule with asymmetric video/action step counts.
    """
    assert action_steps <= video_steps, (
        f"action_steps ({action_steps}) must be <= video_steps ({video_steps})"
    )

    video_ts = _base_timesteps(video_steps, shift)
    action_ts = _base_timesteps(action_steps, shift)

    # Video denoises for all steps; action only activates for the last action_steps
    idle_steps = video_steps - action_steps
    t_action_max = action_ts[0] if action_ts else 0.0

    schedule = []
    for i, t_v in enumerate(video_ts):
        if i < idle_steps:
            schedule.append((t_v, t_action_max))  # action stays at max noise
        else:
            a_idx = i - idle_steps
            t_a = action_ts[a_idx] if a_idx < len(action_ts) else 0.0
            schedule.append((t_v, t_a))

    schedule.append((0.0, 0.0))
    return schedule


# ---------------------------------------------------------------------------
# Training-side utilities
# ---------------------------------------------------------------------------

def sample_decoupled_timesteps(
    batch_size: int,
    num_train_timesteps: int = 1000,
    video_beta_a: float = 0.5,
    video_beta_b: float = 1.0,
    action_uniform: bool = True,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample decoupled timesteps for video and action during training.

    DreamZero-Flash training: video timesteps are drawn from a Beta
    distribution biased toward high noise (near t=1000), while action
    timesteps are sampled uniformly. This trains the model to predict
    clean actions from very noisy video.

    Args:
        batch_size: Number of timesteps to sample.
        num_train_timesteps: Total training timesteps (typically 1000).
        video_beta_a: Beta distribution alpha parameter for video.
            Lower values bias toward higher noise. Default 0.5.
        video_beta_b: Beta distribution beta parameter for video.
            Default 1.0 gives strong high-noise bias.
        action_uniform: If True, sample action timesteps uniformly.
        device: Target device for output tensors.

    Returns:
        (video_timesteps, action_timesteps) each of shape (batch_size,),
        values in [0, num_train_timesteps).
    """
    # Video: Beta distribution biased toward high noise
    # Beta(0.5, 1.0) has PDF peaked at x=0, meaning samples cluster near 0
    # We flip: t_video = (1 - beta_sample) * num_train_timesteps
    # so samples cluster near high timesteps (high noise)
    # Use torch.distributions for reproducibility with torch.manual_seed
    beta_dist = torch.distributions.Beta(video_beta_a, video_beta_b)
    beta_samples = beta_dist.sample((batch_size,))
    video_timesteps = ((1.0 - beta_samples) * num_train_timesteps).clamp(0, num_train_timesteps - 1)

    # Action: uniform sampling (standard)
    if action_uniform:
        action_timesteps = torch.randint(0, num_train_timesteps, (batch_size,)).float()
    else:
        action_timesteps = torch.rand(batch_size) * num_train_timesteps
        action_timesteps = action_timesteps.clamp(0, num_train_timesteps - 1)

    return video_timesteps.to(device), action_timesteps.to(device)
