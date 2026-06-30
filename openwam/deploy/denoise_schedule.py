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

Two strategies are supported:

- ``sync``           — both streams advance in lockstep on their own
  deterministic timestep series (default; unchanged behavior).
- ``variance_shift`` — Latent-Forcing-style ordered trajectory: one
  stream denoises earlier than the other along an alpha-shift curve
  (``alpha``) and/or a linear ``offset`` (arXiv:2602.11401), with
  ``lead`` choosing which stream leads. ``alpha=1, offset=0`` degenerates
  to the ``sync`` diagonal.

The removed strategies (video_leading / cascade / action_only) live in
git history; ``make_schedule`` raises ``NotImplementedError`` for them.
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


def _alpha_shift(u: float, shift: float) -> float:
    """alpha-shift a uniform sample ``u`` in [0, 1] into a shifted sigma.

    ``f_alpha(u) = shift*u / (1 + (shift - 1)*u)`` -- the time shift that
    is informationally equivalent to scaling the latent variance by
    ``shift`` (Esser et al. 2024, SD3; Latent Forcing arXiv:2602.11401
    Eq. 4). This is the same closed form the backbone schedulers apply
    inside ``set_timesteps``; it is inlined here because random continuous
    sampling cannot reuse their fixed-grid ``linspace`` path. Keeping the
    formula identical preserves the train/inference alpha-shift contract.
    """
    return shift * u / (1.0 + (shift - 1.0) * u)


def schedule_variance_shift(
    video_scheduler,
    action_scheduler,
    num_steps: int = 50,
    *,
    lead: str = "action",
    alpha: float = 9.0,
    offset: float = 0.0,
) -> Schedule:
    """Latent-Forcing-style ordered schedule: one stream denoises earlier.

    Both streams share a global cleanness progress ``g`` that advances
    linearly over ``num_steps``. The **lead** stream follows the alpha-shift
    curve ``f_alpha(g)`` (Latent Forcing arXiv:2602.11401 Eq. 4) so it reaches
    "clean" earlier; the **lag** stream advances linearly, optionally delayed
    to start only after ``offset`` of the global progress (the piecewise /
    linear-offset variant). Sigma is ``1 - cleanness`` (noise -> clean), so
    both streams are monotonically decreasing and the schedule ends with the
    ``(0.0, 0.0)`` sentinel -- consumed by ``BaseWAMArchitecture.generate``
    exactly like ``sync`` (every supported architecture, unchanged).

    ``alpha=1, offset=0`` reduces to the ``sync`` diagonal. Unlike ``sync`` /
    ``independent`` this schedule defines its trajectory entirely from
    ``alpha`` / ``offset`` and does not consume the backbone alpha-shift.

    Args:
        video_scheduler: Video stream's scheduler (only its
            ``num_train_timesteps`` attribute is read).
        action_scheduler: Action stream's scheduler (only its
            ``num_train_timesteps`` attribute is read).
        num_steps: Number of denoising steps per stream.
        lead: Which stream denoises earlier -- ``"action"`` or ``"video"``.
        alpha: Lead-curve strength (``>1`` leads; ``1`` = linear/diagonal).
        offset: Fraction of global progress to delay the lagging stream's
            start (``0`` = pure curve; ``>0`` adds the piecewise offset).
    """
    if lead not in ("action", "video"):
        raise ValueError(f"variance_shift lead must be 'action' or 'video', got {lead!r}.")
    num_train_v = float(getattr(video_scheduler, "num_train_timesteps", 1000))
    num_train_a = float(getattr(action_scheduler, "num_train_timesteps", 1000))
    off = min(max(float(offset), 0.0), 0.999)

    def _lag_progress(g: float) -> float:
        if off <= 0.0:
            return g
        return min(max((g - off) / (1.0 - off), 0.0), 1.0)

    lead_sigmas: List[float] = []
    lag_sigmas: List[float] = []
    for k in range(num_steps):
        g = k / num_steps  # global cleanness progress in [0, 1)
        lead_sigmas.append(1.0 - _alpha_shift(g, alpha))  # lead reaches clean earlier
        lag_sigmas.append(1.0 - _lag_progress(g))

    if lead == "action":
        v_sigmas, a_sigmas = lag_sigmas, lead_sigmas
    else:
        v_sigmas, a_sigmas = lead_sigmas, lag_sigmas
    v_ts = [s * num_train_v for s in v_sigmas]
    a_ts = [s * num_train_a for s in a_sigmas]
    return [(v, a) for v, a in zip(v_ts, a_ts)] + [(0.0, 0.0)]


def make_schedule(
    strategy: str,
    video_scheduler,
    action_scheduler,
    num_steps: int = 50,
    shift: float = 5.0,
    *,
    shift_video: float = None,
    lead: str = "action",
    alpha: float = 9.0,
    offset: float = 0.0,
) -> Schedule:
    """Dispatcher kept as the single entry point for building a schedule.

    Args:
        strategy: ``"sync"`` (deterministic lockstep, default) or
            ``"variance_shift"`` (Latent-Forcing ordered curve/offset). Any
            other value raises ``NotImplementedError`` (the removed
            video_leading/cascade/action_only strategies live in git
            history).
        video_scheduler: Video stream's scheduler (e.g.
            ``architecture.video_scheduler``).
        action_scheduler: Action stream's scheduler (e.g.
            ``architecture.action_scheduler``).
        num_steps: Denoising step count for both streams.
        shift: Global α-shift; used by the action scheduler always, and by
            the video scheduler when ``shift_video`` is ``None`` (``sync``
            only).
        shift_video: Optional override of the video α-shift only. Typically
            sourced from ``arch.video_backbone.shift_video`` so train and
            inference sigma grids match (``sync`` only).
        lead: ``variance_shift`` only -- which stream denoises earlier
            (``"action"`` or ``"video"``).
        alpha: ``variance_shift`` only -- lead-curve strength (``>1`` leads;
            ``1`` = diagonal).
        offset: ``variance_shift`` only -- delay the lagging stream's start
            (``0`` = pure curve; ``>0`` = piecewise offset).
    """
    if strategy == "sync":
        return schedule_sync(
            video_scheduler, action_scheduler, num_steps=num_steps, shift=shift, shift_video=shift_video
        )
    if strategy == "variance_shift":
        return schedule_variance_shift(
            video_scheduler,
            action_scheduler,
            num_steps=num_steps,
            lead=lead,
            alpha=alpha,
            offset=offset,
        )
    raise NotImplementedError(
        f"schedule_type={strategy!r} is not supported; choose 'sync' or 'variance_shift'. "
        "video_leading/cascade/action_only live in git history."
    )


__all__ = [
    "Schedule",
    "schedule_sync",
    "schedule_variance_shift",
    "make_schedule",
]
