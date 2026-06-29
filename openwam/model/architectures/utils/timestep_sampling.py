"""Pluggable timestep samplers for joint video-action flow-matching training.

``BaseWAMArchitecture.compute_loss`` (and ``DualSystemIDMArchitecture.compute_loss``)
expose a ``decoupled_sampler`` hook: when supplied, the per-sample video and
action diffusion timesteps come from ``sampler.sample_timesteps(...)`` instead
of the default ``torch.randint`` draw. This module implements that hook.

``IndependentTimestepSampler`` mirrors the independent per-modality timestep
sampling of Unified World Models (arXiv:2504.02792): the video and action
timesteps are drawn from independent uniform samples. The alpha-shift that maps
a uniform draw onto the shifted sigma curve is applied by the backbone
scheduler grid that ``compute_loss`` indexes -- the single source of truth for
the shift (mirroring ``openwam.deploy.denoise_schedule``) -- so this sampler
stays purely uniform and never double-shifts. A ``seed`` makes the draw
reproducible across runs.

This is the training-time counterpart to the deploy-side
``schedule_type="independent"`` / ``"variance_shift"`` schedules. Default
training behavior is unchanged: the trainer only builds a sampler when
``training.timestep_sampling`` selects one; otherwise ``compute_loss`` keeps its
legacy ``torch.randint`` path bit-for-bit.

Convention note (easy to trip on): the timesteps returned here are consumed by
``compute_loss`` as GRID-INDEX positions in ``[0, num_train_timesteps]`` -- a
*larger* returned value maps to a *higher* scheduler-grid index and thus a
*lower* sigma (cleaner). This is the OPPOSITE direction to the deploy-side
``openwam.deploy.denoise_schedule``, where a schedule entry IS the sigma value
(smaller == cleaner). Both sides are internally consistent; mind the direction
when comparing train vs inference code. ``VarianceShiftTimestepSampler`` only
aligns the lead/lag *ordering* with the deploy ``variance_shift`` schedule --
the exact distribution is the composition with the backbone grid's own
alpha-shift and carries no ``offset``, so it is order-aligned, not strictly
point-wise in-distribution.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

DEFAULT_NUM_TRAIN_TIMESTEPS = 1000


class IndependentTimestepSampler:
    """Independent per-stream uniform timestep sampler (UWM-style).

    Implements the ``decoupled_sampler`` contract consumed by
    ``BaseWAMArchitecture.compute_loss`` /
    ``DualSystemIDMArchitecture.compute_loss``:

    - ``num_train_timesteps``: int attribute used by ``compute_loss`` to
      normalize the returned timesteps into scheduler grid indices.
    - ``sample_timesteps(batch_size, *, current_step, device)``: returns a
      ``(video_t, action_t)`` pair of ``(batch_size,)`` float tensors in
      ``[0, num_train_timesteps]``, drawn independently per stream.

    Args:
        num_train_timesteps: Training timestep resolution (matches the
            schedulers' ``num_train_timesteps``; default 1000).
        seed: Optional RNG seed. ``None`` (default) uses the ambient global
            RNG, which the trainer already seeds from ``project.seed`` -- so
            the default is reproducible and varied per step. A non-``None``
            seed builds a private generator that advances across steps
            (reproducible run, distinct per-step draws).
    """

    def __init__(self, num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS, seed: Optional[int] = None):
        self.num_train_timesteps = int(num_train_timesteps)
        self._seed = None if seed is None else int(seed)
        # Per-device persistent generators. Seeded ONCE on first use and reused
        # (advancing) across calls, so a fixed seed yields a reproducible run
        # whose per-step draws still differ. Re-seeding every call would make
        # every training step sample identical timesteps (degenerate).
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, device) -> Optional[torch.Generator]:
        if self._seed is None:
            return None  # ambient global RNG (already seeded by the trainer run seed)
        key = str(device)
        gen = self._generators.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self._seed)
            self._generators[key] = gen
        return gen

    def sample_timesteps(
        self,
        batch_size: int,
        *,
        current_step: int = 0,
        device="cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw independent video and action timesteps for a batch.

        Each stream is sampled from an independent ``Uniform(0, 1)`` draw
        scaled to ``[0, num_train_timesteps]``. The backbone scheduler grid
        that ``compute_loss`` indexes then applies the alpha-shift, so the
        effective per-stream sigma is independent-uniform-then-alpha-shifted
        (UWM style). ``current_step`` is accepted for interface symmetry with
        schedule-annealing samplers and is unused here.
        """
        gen = self._generator(device)
        u_video = torch.rand(batch_size, generator=gen, device=device)
        u_action = torch.rand(batch_size, generator=gen, device=device)
        video_t = u_video * self.num_train_timesteps
        action_t = u_action * self.num_train_timesteps
        return video_t, action_t


class VarianceShiftTimestepSampler:
    """Latent-Forcing variance-shift correlated timestep sampler.

    Training counterpart to ``schedule_type="variance_shift"``: instead of two
    independent draws, draw one global ``u`` per sample and place the two
    streams on the alpha-shift curve -- the lead stream's cleanness is
    ``f_alpha(u) >= u`` so its sampled timestep is, on average, further denoised
    than the lag stream's. Training thus sees curve-correlated ``(t_v, t_a)``
    pairs matching the variance-shift inference path (Latent Forcing
    arXiv:2602.11401).

    Implements the same ``decoupled_sampler`` contract; ``compute_loss`` maps
    the returned timesteps onto the backbone sigma grid (which composes its own
    alpha-shift -- the lead/lag *ordering* is preserved, the exact curve is the
    composition).

    Args:
        num_train_timesteps: Training timestep resolution (default 1000).
        lead: Which stream denoises earlier -- ``"action"`` or ``"video"``.
        alpha: Lead-curve strength (``>1`` leads; ``1`` = uniform/diagonal).
        seed: Optional RNG seed (persistent generator; reproducible run).
    """

    def __init__(
        self,
        num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
        *,
        lead: str = "action",
        alpha: float = 9.0,
        seed: Optional[int] = None,
    ):
        if lead not in ("action", "video"):
            raise ValueError(f"variance_shift lead must be 'action' or 'video', got {lead!r}.")
        self.num_train_timesteps = int(num_train_timesteps)
        self._lead = lead
        self._alpha = float(alpha)
        self._seed = None if seed is None else int(seed)
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, device) -> Optional[torch.Generator]:
        if self._seed is None:
            return None
        key = str(device)
        gen = self._generators.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self._seed)
            self._generators[key] = gen
        return gen

    def sample_timesteps(
        self,
        batch_size: int,
        *,
        current_step: int = 0,
        device="cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw curve-correlated video and action timesteps for a batch.

        One global ``u ~ Uniform(0,1)`` per sample; the lead stream takes
        cleanness ``f_alpha(u) >= u`` (further denoised), the lag stream takes
        ``u``. Returned as ``(video_t, action_t)`` per ``lead``.
        """
        gen = self._generator(device)
        u = torch.rand(batch_size, generator=gen, device=device)
        a = self._alpha
        g_lead = (a * u) / (1.0 + (a - 1.0) * u)  # cleanness on the alpha-shift curve, >= u
        g_lag = u
        lead_t = g_lead * self.num_train_timesteps  # higher cleanness -> further denoised on the grid
        lag_t = g_lag * self.num_train_timesteps
        if self._lead == "action":
            return lag_t, lead_t  # (video_t, action_t): action leads
        return lead_t, lag_t  # video leads


def build_timestep_sampler(
    mode: Optional[str],
    *,
    num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
    seed: Optional[int] = None,
    lead: str = "action",
    alpha: float = 9.0,
):
    """Construct a training timestep sampler from a config mode string.

    Args:
        mode: ``None`` / ``"default"`` / ``"randint"`` / ``"independent_randint"``
            -> returns ``None`` (keep the legacy ``torch.randint`` path in
            ``compute_loss``, bit-identical to upstream).
            ``"independent_uniform_shift"`` -> :class:`IndependentTimestepSampler`.
            ``"variance_shift"`` -> :class:`VarianceShiftTimestepSampler`.
        num_train_timesteps: Forwarded to the sampler.
        seed: Forwarded to the sampler (reproducible draws).
        lead: ``variance_shift`` only -- which stream denoises earlier.
        alpha: ``variance_shift`` only -- lead-curve strength.

    Returns:
        A sampler instance, or ``None`` for the default/legacy path.
    """
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if normalized in ("", "default", "randint", "independent_randint", "none", "null"):
        return None
    if normalized == "independent_uniform_shift":
        return IndependentTimestepSampler(num_train_timesteps=num_train_timesteps, seed=seed)
    if normalized == "variance_shift":
        return VarianceShiftTimestepSampler(
            num_train_timesteps=num_train_timesteps, lead=lead, alpha=alpha, seed=seed
        )
    raise ValueError(
        f"Unknown training.timestep_sampling={mode!r}; expected 'default' (legacy randint), "
        "'independent_uniform_shift', or 'variance_shift'."
    )


__all__ = ["IndependentTimestepSampler", "VarianceShiftTimestepSampler", "build_timestep_sampler"]
