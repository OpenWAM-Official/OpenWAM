"""Schedule generator composing two backbone-owned schedulers.

This module knows nothing about flow-matching math or specific backbone
formulas. It receives two scheduler references (video and action,
typically pulled from the architecture) and asks each to produce its own
timestep series via the duck-typed minimum interface:

    scheduler.set_timesteps(num_inference_steps, shift=...)
    scheduler.timesteps    # 1-D tensor / array

``schedule_sync`` returns a list of ``(t_video, t_action)`` pairs
describing the per-iteration noise levels for the joint denoising loop,
terminated with a ``(0.0, 0.0)`` sentinel.

Only the ``sync`` strategy is supported. The removed strategies
(video_leading / cascade / action_only) live in git history; none of
them ever shipped in a config, and ``make_schedule`` raises
``NotImplementedError`` for them.
"""

from __future__ import annotations

from typing import List, Tuple

Schedule = List[Tuple[float, float]]


def schedule_sync(
    video_scheduler,
    action_scheduler,
    num_steps: int = 50,
    shift: float = 5.0,
    *,
    shift_video: float = None,
) -> Schedule:
    """Both streams advance in lockstep on their own timestep series.

    ``shift_video`` (when set, typically from ``arch.video_backbone.shift_video``)
    overrides the video scheduler's α-shift independently of the action
    scheduler. Action always uses ``shift`` — by design, since the
    Reconstruction-or-Semantics recipe (arXiv:2605.06388) applies
    dim-dependent shift to non-VAE video encoders only. The model was
    trained on independent ``(sigma_v, sigma_a)`` samples (independent
    randint per stream in ``compute_loss``), so any per-stream shift
    combination is in-distribution.
    """
    sv = shift if shift_video is None else shift_video
    video_scheduler.set_timesteps(num_steps, shift=sv)
    action_scheduler.set_timesteps(num_steps, shift=shift)
    v_ts = video_scheduler.timesteps.tolist()
    a_ts = action_scheduler.timesteps.tolist()
    return [(v, a) for v, a in zip(v_ts, a_ts)] + [(0.0, 0.0)]


def make_schedule(
    strategy: str,
    video_scheduler,
    action_scheduler,
    num_steps: int = 50,
    shift: float = 5.0,
    *,
    shift_video: float = None,
) -> Schedule:
    """Dispatcher kept as the single entry point; only ``sync`` is implemented.

    Args:
        strategy: Must be ``"sync"``. Any other value raises
            ``NotImplementedError`` (removed strategies live in git history).
        video_scheduler: Video stream's scheduler (e.g.
            ``architecture.video_scheduler``).
        action_scheduler: Action stream's scheduler (e.g.
            ``architecture.action_scheduler``).
        num_steps: Denoising step count for both streams.
        shift: Global α-shift; used by the action scheduler always, and by
            the video scheduler when ``shift_video`` is ``None``.
        shift_video: Optional override of the video α-shift only. Typically
            sourced from ``arch.video_backbone.shift_video`` so train and
            inference sigma grids match.
    """
    if strategy != "sync":
        raise NotImplementedError(
            f"schedule_type={strategy!r} has been removed; only 'sync' is supported. "
            "video_leading/cascade/action_only live in git history."
        )
    return schedule_sync(video_scheduler, action_scheduler, num_steps=num_steps, shift=shift, shift_video=shift_video)


__all__ = [
    "Schedule",
    "schedule_sync",
    "make_schedule",
]
