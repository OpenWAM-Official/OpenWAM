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

Two denoising modes are supported:

- ``sync``           — both streams advance in lockstep on their own
  deterministic timestep series (default; unchanged behavior).
- ``async``          — Latent-Forcing-style trajectory: one
  stream denoises earlier than the other along an alpha-shift curve
  (``alpha``, arXiv:2602.11401) and/or a linear ``offset`` delaying the
  lag stream, with ``lead`` choosing which stream leads. Each stream
  rides its own ``alpha_shift`` grid (matching training), and
  ``alpha=1, offset=0`` reproduces ``sync`` bit-for-bit.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

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


def _alpha_shift(u, shift: float):
    """alpha-shift ``u`` (float or tensor) in [0, 1] into a shifted sigma.

    ``f_alpha(u) = shift*u / (1 + (shift - 1)*u)`` -- the time shift that
    is informationally equivalent to scaling the latent variance by
    ``shift`` (Esser et al. 2024, SD3; Latent Forcing arXiv:2602.11401
    Eq. 4). Same closed form, same operation order as the backbone
    schedulers' ``set_timesteps``, so float32 tensor input reproduces
    their grids bit-for-bit.
    """
    return shift * u / (1.0 + (shift - 1.0) * u)


def schedule_variance_shift(
    video_scheduler,
    action_scheduler,
    num_steps: int = 50,
    *,
    lead: str = "video",
    alpha: float = 1.0,
    offset: float = 0.0,
    shift_video: float = 5.0,
    shift_action: float = 5.0,
) -> Schedule:
    """Latent-Forcing-style ordered schedule: one stream denoises earlier.

    Both streams share a global progress ``u = k / num_steps``. The **lead**
    stream takes cleanness ``f_alpha(u) >= u`` (Latent Forcing arXiv:2602.11401
    Eq. 4) so it reaches "clean" earlier; the **lag** stream takes ``u``,
    optionally delayed by ``offset``: cleanness stays 0 (sigma 1) until global
    progress passes ``offset``, then advances linearly (the piecewise variant).
    Each stream's sigma is ``alpha_shift(1 - cleanness, shift_stream)`` -- the
    SAME grid the backbone applies at training time (``set_timesteps_wan`` /
    ``ActionScheduler.set_timesteps``). A variance_shift-trained checkpoint and
    this schedule therefore stay point-wise in-distribution (the delayed head
    rides the grid's sigma=1 endpoint).

    Computed in float32 on the schedulers' own base grid
    (``linspace(1, 0, n+1)[:-1]``), with the lead curve applied as the
    algebraically identical ``1 - f_alpha(1 - s) == f_{1/alpha}(s)`` -- exact
    at ``alpha=1`` in floating point -- and the ``offset == 0`` path leaving
    the lag grid untouched, so ``alpha=1, offset=0`` reproduces
    ``schedule_sync`` bit-for-bit.

    Sigma is monotonically decreasing and the schedule ends with the
    ``(0.0, 0.0)`` sentinel -- consumed by ``BaseWAMArchitecture.generate``
    exactly like ``sync`` (every supported architecture, unchanged).

    Args:
        video_scheduler: Video stream's scheduler (only its
            ``num_train_timesteps`` attribute is read).
        action_scheduler: Action stream's scheduler (only its
            ``num_train_timesteps`` attribute is read).
        num_steps: Number of denoising steps per stream.
        lead: Which stream denoises earlier -- ``"action"`` or ``"video"``.
        alpha: Lead-curve strength, must be ``>= 1`` (``>1`` leads; ``1`` = sync diagonal; ``<1`` inverts lead/lag).
        offset: Fraction of pre-shift progress to delay the lag stream's start (clamped to [0, 0.999]; ``0`` = pure curve).
        shift_video: alpha-shift for the video stream's sigma grid.
        shift_action: alpha-shift for the action stream's sigma grid.
    """
    if lead not in ("action", "video"):
        raise ValueError(f"variance_shift lead must be 'action' or 'video', got {lead!r}.")
    num_train_v = float(getattr(video_scheduler, "num_train_timesteps", 1000))
    num_train_a = float(getattr(action_scheduler, "num_train_timesteps", 1000))

    # s[k] = 1 - k/num_steps: the schedulers' float32 base sigma grid.
    s = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
    # Lead pre-shift sigma 1 - f_alpha(1-s) rewritten as f_{1/alpha}(s), which
    # leaves s bitwise untouched at alpha=1; the lag stream stays on s unless
    # delayed by offset below.
    lead_sigma = _alpha_shift(s, 1.0 / alpha)
    off = min(max(float(offset), 0.0), 0.999)
    if off > 0.0:
        # lag cleanness = clamp((u - off)/(1 - off), 0, 1) in sigma form;
        # the off == 0 passthrough keeps alpha=1 bitwise == sync.
        lag_sigma = torch.clamp(s / (1.0 - off), max=1.0)
    else:
        lag_sigma = s
    if lead == "video":
        v_sigma, a_sigma = lead_sigma, lag_sigma
    else:
        v_sigma, a_sigma = lag_sigma, lead_sigma
    # Each stream's sigma rides its own alpha-shift grid (matches training).
    v_ts = (_alpha_shift(v_sigma, shift_video) * num_train_v).tolist()
    a_ts = (_alpha_shift(a_sigma, shift_action) * num_train_a).tolist()

    return [(v, a) for v, a in zip(v_ts, a_ts)] + [(0.0, 0.0)]


def make_schedule(
    mode: str,
    video_scheduler,
    action_scheduler,
    num_steps: int = 50,
    shift: float = 5.0,
    *,
    shift_video: float = None,
    lead: str = "video",
    alpha: float = 1.0,
    offset: float = 0.0,
) -> Schedule:
    """Build the schedule for a synchronous or asynchronous denoising trajectory.

    Args:
        mode: ``"sync"`` (lockstep) or ``"async"`` (shifted trajectory).
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
        lead: ``async`` only -- which stream denoises earlier
            (``"action"`` or ``"video"``).
        alpha: ``async`` only -- lead-curve strength (``>1`` leads;
            ``1`` = diagonal = sync).
        offset: ``async`` only -- delay the lag stream's start
            (``0`` = pure curve; ``>0`` = piecewise offset).
    """
    if mode == "sync":
        return schedule_sync(
            video_scheduler, action_scheduler, num_steps=num_steps, shift=shift, shift_video=shift_video
        )
    if mode == "async":
        return schedule_variance_shift(
            video_scheduler,
            action_scheduler,
            num_steps=num_steps,
            lead=lead,
            alpha=alpha,
            offset=offset,
            shift_video=shift if shift_video is None else shift_video,
            shift_action=shift,
        )
    raise NotImplementedError(f"denoise_mode={mode!r} is not supported; choose 'sync' or 'async'.")


__all__ = [
    "Schedule",
    "schedule_sync",
    "schedule_variance_shift",
    "make_schedule",
]
