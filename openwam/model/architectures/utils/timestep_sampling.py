"""Pluggable timestep samplers for joint video-action flow-matching training.

``BaseWAMArchitecture.compute_loss`` (and ``DualSystemIDMArchitecture.compute_loss``)
expose a ``decoupled_sampler`` hook: when supplied, the per-sample video and
action diffusion timesteps come from ``sampler.sample_timesteps(...)`` instead
of the default ``torch.randint`` draw. This module implements that hook.

``VarianceShiftTimestepSampler`` is the training-time counterpart to the
deploy-side ``schedule_type="variance_shift"`` schedule. Default
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
        mode: ``None`` / ``"default"`` / ``"randint"`` -> returns ``None``
            (keep the legacy ``torch.randint`` path in ``compute_loss``,
            bit-identical to upstream).
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
    if normalized in ("", "default", "randint", "none", "null"):
        return None
    if normalized == "variance_shift":
        return VarianceShiftTimestepSampler(num_train_timesteps=num_train_timesteps, lead=lead, alpha=alpha, seed=seed)
    raise ValueError(
        f"Unknown training.timestep_sampling={mode!r}; expected 'default' (legacy randint) or 'variance_shift'."
    )


__all__ = ["VarianceShiftTimestepSampler", "build_timestep_sampler"]
