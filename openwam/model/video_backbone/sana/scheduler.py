"""Flow-matching scheduler adapter for SANA-Video.

OpenWAM's training loop pulls four attributes off ``video_backbone.scheduler``:
``timesteps``, ``sigmas``, ``linear_timesteps_weights``, ``num_train_timesteps``
(see ``openwam/model/base.py``). SANA-Video is rectified flow with an optional
shift parameter; this adapter mirrors the Wan ``FlowMatchScheduler`` public
surface so SANA looks identical to the trainer.

Defaults are calibrated against upstream SANA-Video:

* ``flow_shift=3.0`` — matches
  ``third_party/Sana/diffusion/longsana/utils/model_wrapper.py:17``
  (``flow_shift: float = 3.0``) and the published SANA-Video 480p config
  ``third_party/Sana/configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml``.
* Training target ``noise - sample`` (``v_t = x_0 - x_1`` in flow matching)
  and noising ``x_t = (1-σ)·sample + σ·noise`` are identical to Wan's default
  fallback in ``BaseWAMArchitecture.compute_loss``, so ``SanaVideoBackbone``
  does **not** override ``add_training_noise`` / ``training_target``.
* Loss weighting is uniform — SANA upstream's training loop applies no
  per-step weighting at the trainer level (any sigma-dependent shaping is
  baked into ``flow_shift``).
"""

from __future__ import annotations

from typing import Optional

import torch


class SanaFlowSchedulerAdapter:
    """Wan-compatible flow-matching scheduler tuned for SANA-Video.

    Args:
        flow_shift: Rectified-flow shift parameter. Default ``3.0`` matches
            SANA-Video upstream (see module docstring).
        num_train_timesteps: Total training timestep budget (default 1000,
            matches Wan / FLUX / Cosmos conventions).
    """

    def __init__(self, *, flow_shift: float = 3.0, num_train_timesteps: int = 1000):
        self.flow_shift = float(flow_shift)
        self.num_train_timesteps = int(num_train_timesteps)
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.linear_timesteps_weights: Optional[torch.Tensor] = None
        self.training: bool = False

    @staticmethod
    def _flow_match_sigmas(num_inference_steps: int, shift: float, denoising_strength: float) -> torch.Tensor:
        sigma_min, sigma_max = 0.0, 1.0
        start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(start, sigma_min, num_inference_steps + 1)[:-1]
        return shift * sigmas / (1 + (shift - 1) * sigmas)

    def set_timesteps(
        self,
        num_inference_steps: int = 1000,
        denoising_strength: float = 1.0,
        shift: Optional[float] = None,
        training: bool = False,
        **_: object,
    ) -> None:
        shift_val = self.flow_shift if shift is None else float(shift)
        self.sigmas = self._flow_match_sigmas(num_inference_steps, shift_val, denoising_strength)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            self.linear_timesteps_weights = torch.ones_like(self.timesteps, dtype=torch.float32)
            self.training = True
        else:
            self.linear_timesteps_weights = None
            self.training = False

    # --- Inference / loss helpers (mirror Wan FlowMatchScheduler API) ---

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor, to_final: bool = False, **_):
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("SanaFlowSchedulerAdapter.step called before set_timesteps()")
        ts = timestep.cpu() if isinstance(timestep, torch.Tensor) else timestep
        idx = int(torch.argmin((self.timesteps - ts).abs()))
        sigma = self.sigmas[idx]
        sigma_next = 0 if to_final or idx + 1 >= len(self.timesteps) else self.sigmas[idx + 1]
        return sample + model_output * (sigma_next - sigma)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep) -> torch.Tensor:
        if self.sigmas is None:
            raise RuntimeError("SanaFlowSchedulerAdapter.add_noise called before set_timesteps()")
        ts = timestep.cpu() if isinstance(timestep, torch.Tensor) else timestep
        idx = int(torch.argmin((self.timesteps - ts).abs()))
        sigma = self.sigmas[idx]
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep) -> torch.Tensor:
        return noise - sample

    def training_weight(self, timestep) -> torch.Tensor:
        if self.linear_timesteps_weights is None:
            raise RuntimeError("training_weight requires set_timesteps(training=True) first")
        ts = timestep.to(self.timesteps.device) if isinstance(timestep, torch.Tensor) else timestep
        idx = torch.argmin((self.timesteps - ts).abs())
        return self.linear_timesteps_weights[idx]


__all__ = ["SanaFlowSchedulerAdapter"]
