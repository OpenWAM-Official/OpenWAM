"""Decoupled noise schedules for accelerated action inference.

DreamZero-Flash insight: by training with decoupled video/action noise
levels (video biased toward high noise via Beta distribution, action
sampled independently), the model learns to predict clean actions even
from extremely noisy video context. At inference time, this allows
action denoising in as few as 1-4 steps.

This module provides the deployment-side schedules:
- ``schedule_decoupled_flash`` — minimal-step action denoising with
  clean video context
- ``schedule_decoupled_asymmetric`` — many video steps, few action steps

Both functions take two backbone-owned scheduler references (video and
action) and ask each to produce its own timestep series via the same
duck-typed interface as ``openwam.deploy.schedule``.

The matching training-side timestep sampler lives in
``openwam.train.loss.decoupled_loss.DecoupledFlowMatchLoss``.
"""

from typing import List, Tuple

Schedule = List[Tuple[float, float]]


def schedule_decoupled_flash(
    video_scheduler,  # noqa: ARG001 — kept for dispatcher signature symmetry
    action_scheduler,
    action_steps: int = 1,
    shift: float = 5.0,
    *,
    shift_video: float = None,  # noqa: ARG001 — video stays clean (σ=0); shift irrelevant
) -> Schedule:
    """DreamZero-Flash: video stays clean (σ=0); action denoises in 1-4 steps.

    Args:
        video_scheduler: Unused; signature symmetry with other schedules.
        action_scheduler: Action stream's scheduler.
        action_steps: Number of denoising steps for actions (1-4 typical).
        shift: Shifted-sigmoid shape parameter.
        shift_video: Ignored — video stays clean by construction, so its
            schedule discretization is irrelevant. Accepted for dispatcher
            signature symmetry with the other ``schedule_*`` functions.
    """
    action_scheduler.set_timesteps(action_steps, shift=shift)
    a_ts = action_scheduler.timesteps.tolist()
    return [(0.0, t_a) for t_a in a_ts] + [(0.0, 0.0)]


def schedule_decoupled_asymmetric(
    video_scheduler,
    action_scheduler,
    video_steps: int = 10,
    action_steps: int = 2,
    shift: float = 5.0,
    *,
    shift_video: float = None,
) -> Schedule:
    """Asymmetric: video uses many steps, action uses few (action joins late).

    Action denoising happens during the last ``action_steps`` of the video
    schedule, so total iteration count is ``video_steps``.

    ``shift_video`` (when set) overrides the video α-shift independently
    of the action α-shift, matching the
    :mod:`openwam.deploy.schedule` ``schedule_sync`` contract — the model
    was trained on independent ``(sigma_v, sigma_a)`` pairs so per-stream
    shift values are in-distribution.
    """
    assert action_steps <= video_steps, f"action_steps ({action_steps}) must be <= video_steps ({video_steps})"

    sv = shift if shift_video is None else shift_video
    video_scheduler.set_timesteps(video_steps, shift=sv)
    action_scheduler.set_timesteps(action_steps, shift=shift)
    v_ts = video_scheduler.timesteps.tolist()
    a_ts = action_scheduler.timesteps.tolist()

    idle_steps = video_steps - action_steps
    t_action_max = a_ts[0] if a_ts else 0.0

    schedule = []
    for i, t_v in enumerate(v_ts):
        if i < idle_steps:
            schedule.append((t_v, t_action_max))
        else:
            a_idx = i - idle_steps
            schedule.append((t_v, a_ts[a_idx] if a_idx < len(a_ts) else 0.0))

    schedule.append((0.0, 0.0))
    return schedule
