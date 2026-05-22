"""Schedule generators that compose two backbone-owned schedulers.

This module knows nothing about flow-matching math or specific backbone
formulas. It receives two scheduler references (video and action,
typically pulled from the architecture) and asks each to produce its own
timestep series via the duck-typed minimum interface:

    scheduler.set_timesteps(num_inference_steps, shift=...)
    scheduler.timesteps    # 1-D tensor / array

Each ``schedule_*`` function returns a list of ``(t_video, t_action)``
pairs describing the per-iteration noise levels for the joint denoising
loop, terminated with a ``(0.0, 0.0)`` sentinel.
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
    trained on independent ``(sigma_v, sigma_a)`` samples
    (openwam/train/loss/decoupled_loss.py), so any per-stream shift
    combination is in-distribution.
    """
    sv = shift if shift_video is None else shift_video
    video_scheduler.set_timesteps(num_steps, shift=sv)
    action_scheduler.set_timesteps(num_steps, shift=shift)
    v_ts = video_scheduler.timesteps.tolist()
    a_ts = action_scheduler.timesteps.tolist()
    return [(v, a) for v, a in zip(v_ts, a_ts)] + [(0.0, 0.0)]


def schedule_video_leading(
    video_scheduler,
    action_scheduler,
    num_steps: int = 50,
    lead_steps: int = 10,
    shift: float = 5.0,
    *,
    shift_video: float = None,
) -> Schedule:
    """Video denoises ``lead_steps`` steps before action joins."""
    sv = shift if shift_video is None else shift_video
    video_scheduler.set_timesteps(num_steps + lead_steps, shift=sv)
    action_scheduler.set_timesteps(num_steps, shift=shift)
    v_ts = video_scheduler.timesteps.tolist()
    a_ts = action_scheduler.timesteps.tolist()
    t_a_max = a_ts[0] if a_ts else 0.0

    schedule = []
    for i, v_t in enumerate(v_ts):
        if i < lead_steps:
            schedule.append((v_t, t_a_max))
        else:
            schedule.append((v_t, a_ts[i - lead_steps]))
    schedule.append((0.0, 0.0))
    return schedule


def schedule_cascade(
    video_scheduler,
    action_scheduler,
    video_steps: int = 50,
    action_steps: int = 50,
    shift: float = 5.0,
    *,
    shift_video: float = None,
) -> Schedule:
    """Video fully denoises, then action denoises (fully serial)."""
    sv = shift if shift_video is None else shift_video
    video_scheduler.set_timesteps(video_steps, shift=sv)
    action_scheduler.set_timesteps(action_steps, shift=shift)
    v_ts = video_scheduler.timesteps.tolist()
    a_ts = action_scheduler.timesteps.tolist()
    t_a_max = a_ts[0] if a_ts else 0.0

    schedule = [(v_t, t_a_max) for v_t in v_ts]
    schedule.extend((0.0, a_t) for a_t in a_ts)
    schedule.append((0.0, 0.0))
    return schedule


def schedule_action_only(
    video_scheduler,  # noqa: ARG001 — kept for dispatcher signature symmetry
    action_scheduler,
    num_steps: int = 50,
    shift: float = 5.0,
    *,
    shift_video: float = None,  # noqa: ARG001 — accepted for dispatcher symmetry
) -> Schedule:
    """Video stays clean (sigma=0); only action denoises.

    ``shift_video`` is ignored — video stays clean by construction, so its
    schedule discretization is irrelevant. Accepted for dispatcher
    signature symmetry with the other ``schedule_*`` functions.
    """
    action_scheduler.set_timesteps(num_steps, shift=shift)
    a_ts = action_scheduler.timesteps.tolist()
    return [(0.0, a_t) for a_t in a_ts] + [(0.0, 0.0)]


_SCHEDULE_REGISTRY = {
    "sync": schedule_sync,
    "video_leading": schedule_video_leading,
    "cascade": schedule_cascade,
    "action_only": schedule_action_only,
}


def _get_schedule_fn(strategy: str):
    """Resolve schedule function, lazily importing decoupled schedules to avoid circular imports."""
    if strategy in _SCHEDULE_REGISTRY:
        return _SCHEDULE_REGISTRY[strategy]
    from openwam.deploy.optimizations.decoupled_schedule import (
        schedule_decoupled_asymmetric,
        schedule_decoupled_flash,
    )

    _SCHEDULE_REGISTRY["decoupled_flash"] = schedule_decoupled_flash
    _SCHEDULE_REGISTRY["decoupled_asymmetric"] = schedule_decoupled_asymmetric
    return _SCHEDULE_REGISTRY.get(strategy)


def make_schedule(
    strategy: str,
    video_scheduler,
    action_scheduler,
    num_steps: int = 50,
    shift: float = 5.0,
    *,
    shift_video: float = None,
    **kwargs,
) -> Schedule:
    """Dispatcher: pick a schedule strategy and forward the two schedulers.

    Args:
        strategy: ``"sync" | "video_leading" | "cascade" | "action_only" |
            "decoupled_flash" | "decoupled_asymmetric"``.
        video_scheduler: Video stream's scheduler (e.g.
            ``architecture.video_scheduler``).
        action_scheduler: Action stream's scheduler (e.g.
            ``architecture.action_scheduler``).
        num_steps: Default step count; per-strategy overrides via kwargs.
        shift: Global α-shift; used by the action scheduler always, and by
            the video scheduler when ``shift_video`` is ``None``.
        shift_video: Optional override of the video α-shift only (Esser SD3
            dim-dependent shift). When set, video scheduler is discretized
            with this value while action stays at ``shift``. Typically
            sourced from ``arch.video_backbone.shift_video`` so train and
            inference sigma grids match. ``None`` (default) falls back to
            the global ``shift`` for both streams — bit-identical to
            pre-PR behavior.
        **kwargs: Strategy-specific knobs:
            - ``video_leading``: ``lead_steps`` (default 10)
            - ``cascade`` / ``decoupled_asymmetric``: ``video_steps`` and
              ``action_steps`` (each defaults to ``num_steps``)
            - ``decoupled_flash``: ``action_steps`` (default ``num_steps``)
    """
    fn = _get_schedule_fn(strategy)
    if fn is None:
        raise ValueError(
            f"Unknown schedule strategy '{strategy}'. Choose from: "
            f"{list(_SCHEDULE_REGISTRY.keys()) + ['decoupled_flash', 'decoupled_asymmetric']}"
        )

    call_kwargs = {"shift": shift, "shift_video": shift_video}
    if strategy == "cascade":
        call_kwargs["video_steps"] = kwargs.get("video_steps", num_steps)
        call_kwargs["action_steps"] = kwargs.get("action_steps", num_steps)
    elif strategy == "decoupled_flash":
        call_kwargs["action_steps"] = kwargs.get("action_steps", num_steps)
    elif strategy == "decoupled_asymmetric":
        call_kwargs["video_steps"] = kwargs.get("video_steps", num_steps)
        call_kwargs["action_steps"] = kwargs.get("action_steps", max(1, num_steps // 5))
    else:
        call_kwargs["num_steps"] = num_steps
    if strategy == "video_leading":
        call_kwargs["lead_steps"] = kwargs.get("lead_steps", 10)

    return fn(video_scheduler, action_scheduler, **call_kwargs)


__all__ = [
    "Schedule",
    "schedule_sync",
    "schedule_video_leading",
    "schedule_cascade",
    "schedule_action_only",
    "make_schedule",
]
