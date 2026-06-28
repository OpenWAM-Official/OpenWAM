"""Schedule generator composing two backbone-owned schedulers.

This module knows nothing about flow-matching math or specific backbone
formulas. It receives two scheduler references (video and action,
typically pulled from the architecture) and asks each to produce its own
timestep series via the duck-typed minimum interface:

    scheduler.set_timesteps(num_inference_steps, shift=...)
    scheduler.timesteps    # 1-D tensor / array

``schedule_sync`` returns a list of ``(t_video, t_action)`` pairs
describing the per-iteration noise levels for the joint denoising loop,
terminated with a ``(0.0, 0.0)`` sentinel. ``schedule_independent``
returns the same structure but draws each stream's trajectory from an
independent random sampler.

Two strategies are supported:

- ``sync``        — both streams advance in lockstep on their own
  deterministic timestep series (default; unchanged behavior).
- ``independent`` — video and action timesteps are sampled
  independently per stream (uniform -> alpha-shift -> sorted
  descending), the inference analogue of the independent per-modality
  timestep sampling used by Unified World Models (arXiv:2504.02792) and
  the multi-time / alpha-shift scheduling of Latent Forcing
  (arXiv:2602.11401). The model is trained on independent
  ``(sigma_v, sigma_a)`` samples, so a decoupled inference trajectory
  stays in-distribution.

The removed strategies (video_leading / cascade / action_only) live in
git history; ``make_schedule`` raises ``NotImplementedError`` for them.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

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


def schedule_independent(
    video_scheduler,
    action_scheduler,
    num_steps: int = 50,
    shift: float = 5.0,
    *,
    shift_video: float = None,
    seed: Optional[int] = None,
) -> Schedule:
    """Each stream follows its own independently sampled timestep trajectory.

    For each stream we draw ``num_steps`` uniform samples in ``[0, 1]``,
    alpha-shift them with that stream's shift (``shift_video`` for video
    when set, otherwise ``shift``; ``shift`` for action), sort the
    resulting sigmas descending, and scale by the scheduler's
    ``num_train_timesteps`` to obtain the timestep series. Video and action
    are drawn independently, so the two trajectories are decoupled -- the
    inference analogue of the independent per-modality timestep sampling
    used at training time (UWM arXiv:2504.02792; Latent Forcing
    arXiv:2602.11401).

    Each stream is monotonically decreasing and the schedule is terminated
    with a ``(0.0, 0.0)`` sentinel, so the joint denoising loop in
    ``BaseWAMArchitecture.generate`` consumes it exactly like ``sync``
    (every supported architecture, unchanged).

    Args:
        video_scheduler: Video stream's scheduler (only its
            ``num_train_timesteps`` attribute is read).
        action_scheduler: Action stream's scheduler (only its
            ``num_train_timesteps`` attribute is read).
        num_steps: Number of denoising steps per stream.
        shift: alpha-shift for the action stream (and the video stream
            when ``shift_video`` is ``None``).
        shift_video: Optional alpha-shift override for the video stream.
        seed: Optional RNG seed for a reproducible schedule. ``None`` draws
            a fresh nondeterministic trajectory each call.
    """
    sv = shift if shift_video is None else shift_video
    rng = random.Random(seed)

    def _stream(scheduler, stream_shift: float) -> List[float]:
        num_train = float(getattr(scheduler, "num_train_timesteps", 1000))
        # Independent uniform draws, alpha-shifted, sorted high->low so the
        # stream denoises monotonically from noise toward clean.
        us = sorted((rng.random() for _ in range(num_steps)), reverse=True)
        return [_alpha_shift(u, stream_shift) * num_train for u in us]

    v_ts = _stream(video_scheduler, sv)
    a_ts = _stream(action_scheduler, shift)
    return [(v, a) for v, a in zip(v_ts, a_ts)] + [(0.0, 0.0)]


def make_schedule(
    strategy: str,
    video_scheduler,
    action_scheduler,
    num_steps: int = 50,
    shift: float = 5.0,
    *,
    shift_video: float = None,
    seed: Optional[int] = None,
) -> Schedule:
    """Dispatcher kept as the single entry point for building a schedule.

    Args:
        strategy: ``"sync"`` (deterministic lockstep, default) or
            ``"independent"`` (per-stream randomly sampled timesteps). Any
            other value raises ``NotImplementedError`` (the removed
            video_leading/cascade/action_only strategies live in git
            history).
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
        seed: Reproducibility seed for ``strategy="independent"``; ignored
            by ``"sync"`` (deterministic). ``None`` samples a fresh
            trajectory each call.
    """
    if strategy == "sync":
        return schedule_sync(
            video_scheduler, action_scheduler, num_steps=num_steps, shift=shift, shift_video=shift_video
        )
    if strategy == "independent":
        return schedule_independent(
            video_scheduler,
            action_scheduler,
            num_steps=num_steps,
            shift=shift,
            shift_video=shift_video,
            seed=seed,
        )
    raise NotImplementedError(
        f"schedule_type={strategy!r} is not supported; choose 'sync' or 'independent'. "
        "video_leading/cascade/action_only live in git history."
    )


__all__ = [
    "Schedule",
    "schedule_sync",
    "schedule_independent",
    "make_schedule",
]
