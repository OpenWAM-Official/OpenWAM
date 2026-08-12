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

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

Schedule = List[Tuple[float, float]]
VALID_DENOISE_MODES = ("sync", "async")


@dataclass(frozen=True)
class DenoiseConfig:
    denoise_mode: str = "sync"
    lead_modality: str = "video"
    variance_shift_alpha: float = 1.0
    linear_offset: float = 0.0


def _config_value(cfg, name: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _finite_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return value


def normalize_denoise_config(cfg=None, *, reset_inactive: bool = False) -> DenoiseConfig:
    """Normalize and validate deploy denoising settings.

    ``reset_inactive`` drops the async-only controls back to their defaults
    under ``denoise_mode="sync"`` instead of rejecting them. Callers handed an
    already-resolved config — per-request ``conditions``, :func:`make_schedule`
    — cannot tell a user-written value from one inherited from the yaml, so
    they build what they are given; the config-shape check stays strict where
    "explicitly supplied" is knowable (startup validation and the CLI layer).
    """
    mode = str(_config_value(cfg, "denoise_mode", "sync")).strip().lower()
    if mode not in VALID_DENOISE_MODES:
        raise ValueError(f"Unsupported denoise mode {mode!r}; expected one of {VALID_DENOISE_MODES}")

    if mode == "sync" and reset_inactive:
        return DenoiseConfig()

    lead = str(_config_value(cfg, "lead_modality", "video")).strip().lower()
    if lead not in ("action", "video"):
        raise ValueError(f"lead_modality must be 'action' or 'video', got {lead!r}")

    alpha = _finite_float(_config_value(cfg, "variance_shift_alpha", 1.0), "variance_shift_alpha")
    if alpha < 1.0:
        raise ValueError(f"variance_shift_alpha must be >= 1, got {alpha!r}")

    offset = _finite_float(_config_value(cfg, "linear_offset", 0.0), "linear_offset")
    if not 0.0 <= offset < 1.0:
        raise ValueError(f"linear_offset must satisfy 0 <= value < 1, got {offset!r}")

    if mode == "sync":
        inactive = []
        if lead != "video":
            inactive.append("lead_modality")
        if alpha != 1.0:
            inactive.append("variance_shift_alpha")
        if offset != 0.0:
            inactive.append("linear_offset")
        if inactive:
            verb = "requires" if len(inactive) == 1 else "require"
            raise ValueError(f"{', '.join(inactive)} {verb} denoise_mode='async'")

    return DenoiseConfig(
        denoise_mode=mode,
        lead_modality=lead,
        variance_shift_alpha=alpha,
        linear_offset=offset,
    )


def denoise_async_is_noop(config: DenoiseConfig) -> bool:
    """True when ``async`` reproduces the ``sync`` trajectory bit-for-bit.

    ``alpha=1, offset=0`` is the diagonal, so a run labelled ``async`` at the
    shipped defaults produces exactly the ``sync`` schedule — worth a warning
    rather than silently attributing the numbers to a shifted trajectory.
    """
    defaults = DenoiseConfig()
    return (
        config.denoise_mode == "async"
        and config.variance_shift_alpha == defaults.variance_shift_alpha
        and config.linear_offset == defaults.linear_offset
    )


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
    Each stream's sigma stays on the backbone's training grid via
    ``alpha_shift(1 - cleanness, shift_stream)``.

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
        alpha: Lead-curve strength, must be ``>= 1`` (``>1`` leads;
            ``1`` = sync diagonal).
        offset: Fraction of pre-shift progress to delay the lag stream's start;
            must satisfy ``0 <= offset < 1``.
        shift_video: alpha-shift for the video stream's sigma grid.
        shift_action: alpha-shift for the action stream's sigma grid.
    """
    options = normalize_denoise_config(
        {
            "denoise_mode": "async",
            "lead_modality": lead,
            "variance_shift_alpha": alpha,
            "linear_offset": offset,
        }
    )
    lead = options.lead_modality
    alpha = options.variance_shift_alpha
    offset = options.linear_offset
    num_train_v = float(getattr(video_scheduler, "num_train_timesteps", 1000))
    num_train_a = float(getattr(action_scheduler, "num_train_timesteps", 1000))

    # s[k] = 1 - k/num_steps: the schedulers' float32 base sigma grid.
    s = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
    # Lead pre-shift sigma 1 - f_alpha(1-s) rewritten as f_{1/alpha}(s), which
    # leaves s bitwise untouched at alpha=1; the lag stream stays on s unless
    # delayed by offset below.
    lead_sigma = _alpha_shift(s, 1.0 / alpha)
    if offset > 0.0:
        # lag cleanness = clamp((u - off)/(1 - off), 0, 1) in sigma form;
        # the off == 0 passthrough keeps alpha=1 bitwise == sync.
        lag_sigma = torch.clamp(s / (1.0 - offset), max=1.0)
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

    ``mode="sync"`` ignores the three ``async`` arguments rather than
    rejecting them: this is the one call on the per-request path, so a
    config-shape rule here would surface as a per-request ``ValueError``.
    The shape check lives at startup and in the CLI layer instead.
    """
    options = normalize_denoise_config(
        {
            "denoise_mode": mode,
            "lead_modality": lead,
            "variance_shift_alpha": alpha,
            "linear_offset": offset,
        },
        reset_inactive=True,
    )
    if options.denoise_mode == "sync":
        return schedule_sync(
            video_scheduler, action_scheduler, num_steps=num_steps, shift=shift, shift_video=shift_video
        )
    return schedule_variance_shift(
        video_scheduler,
        action_scheduler,
        num_steps=num_steps,
        lead=options.lead_modality,
        alpha=options.variance_shift_alpha,
        offset=options.linear_offset,
        shift_video=shift if shift_video is None else shift_video,
        shift_action=shift,
    )


__all__ = [
    "Schedule",
    "DenoiseConfig",
    "VALID_DENOISE_MODES",
    "normalize_denoise_config",
    "denoise_async_is_noop",
    "schedule_sync",
    "schedule_variance_shift",
    "make_schedule",
]
