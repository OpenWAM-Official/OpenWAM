"""Flow-matching scheduler for the action stream.

Owned by ``action_backbone``. The sigma curve is hardcoded as a
shifted-sigmoid schedule (conceptually equivalent to the Wan video
scheduler), but this class deliberately does not expose any "template"
abstraction — action_backbone owns its own scheduler.

Self-contained: users may bypass ``openwam.deploy.denoise_schedule`` and use
this class directly (e.g. in receding-horizon control loops):

    from openwam.model.action_backbone.scheduler import ActionScheduler

    s = ActionScheduler()
    s.set_timesteps(num_inference_steps=4, shift=5.0)
    timesteps = s.timesteps  # tensor shape (4,)
"""

from __future__ import annotations

import torch


class ActionScheduler:
    """Flow-matching noise scheduler for the action stream.

    Provides the full math needed by both training and inference:
    timestep/sigma series generation, forward noise injection,
    training target/weight, and a flow-matching denoising step.

    The shifted-sigmoid sigma curve is intentionally hardcoded — there
    is no template knob. The ``num_train_timesteps = 1000`` constant
    matches the Wan video scheduler convention.
    """

    num_train_timesteps: int = 1000

    def __init__(self):
        self.timesteps: torch.Tensor | None = None
        self.sigmas: torch.Tensor | None = None
        self.linear_timesteps_weights: torch.Tensor | None = None
        self.training: bool = False

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        shift: float = 5.0,
        training: bool = False,
    ) -> None:
        """Generate the sigma/timestep series for ``num_inference_steps`` steps.

        Args:
            num_inference_steps: Number of discrete steps in the series
                (training typically passes 1000; inference passes 4-50).
            denoising_strength: Scales the sigma start point (1.0 = full
                noise; <1 when continuing from a partially denoised state).
            shift: Shifted-sigmoid shape parameter.
            training: If True, also computes ``linear_timesteps_weights``
                used for per-timestep loss weighting.
        """
        sigma_start = 1.0 * denoising_strength
        sigmas = torch.linspace(sigma_start, 0.0, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps
        if training:
            self._set_training_weight()
            self.training = True
        else:
            self.training = False

    def _set_training_weight(self) -> None:
        """Compute BSMNTW loss weights over ``self.timesteps``.

        Mirrors the legacy ``FlowMatchScheduler.set_training_weight``
        used by Wan-family video schedulers, so action and video losses
        share consistent timestep weighting.
        """
        steps = self.num_train_timesteps
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != steps:
            bsmntw = bsmntw * (len(self.timesteps) / steps)
            bsmntw = bsmntw + bsmntw[1]
        self.linear_timesteps_weights = bsmntw

    # ------------------------------------------------------------------
    # Training-side math
    # ------------------------------------------------------------------

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-matching forward noise injection: ``(1-σ)·orig + σ·noise``.

        Args:
            original_samples: Clean target tensor.
            noise: Gaussian noise tensor (same shape as ``original_samples``).
            sigma: Pre-broadcast sigma tensor.

        Returns:
            Noisy samples on the same device/dtype as inputs.
        """
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-matching training target: ``noise - original``."""
        return noise - original_samples

    def training_weight(self, timestep_ids: torch.Tensor) -> torch.Tensor:
        """Look up per-timestep loss weights by integer ID.

        Requires ``set_timesteps(..., training=True)`` to have been called.
        """
        if self.linear_timesteps_weights is None:
            raise RuntimeError("ActionScheduler.training_weight called before set_timesteps(..., training=True)")
        return self.linear_timesteps_weights[timestep_ids]

    # ------------------------------------------------------------------
    # Inference-side math
    # ------------------------------------------------------------------

    def flow_step(
        self,
        pred: torch.Tensor,
        sigma: float | torch.Tensor,
        sigma_next: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """One Euler denoising step in flow-matching form.

        ``sample(σ_next) = sample(σ) + (σ_next - σ) · pred``

        ``sigma`` / ``sigma_next`` are taken directly (no internal lookup
        in ``self.timesteps``), because the joint inference loop in
        ``BaseWAMArchitecture.generate`` drives its own (t_v, t_a)
        schedule whose timesteps are not necessarily indices into this
        scheduler's series.
        """
        return sample + pred * (sigma_next - sigma)
