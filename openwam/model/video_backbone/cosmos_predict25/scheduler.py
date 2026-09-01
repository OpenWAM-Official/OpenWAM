"""Flow-matching scheduler adapter for Cosmos-Predict2.5.

OpenWAM's training loop pulls four attributes off ``video_backbone.scheduler``:
``timesteps``, ``sigmas``, ``linear_timesteps_weights``, ``num_train_timesteps``
(see ``openwam/model/architectures/base.py``). Cosmos-Predict2.5 is rectified flow with an
optional shift parameter; this adapter mirrors the Wan ``FlowMatchScheduler``
public surface so both backbones look identical to the trainer.

Defaults are calibrated against upstream Cosmos-Predict2.5:

* ``shift_video=5.0`` — matches
  ``cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:99``
  (``shift: int = 5``) and the released checkpoint family
  ``rectified_flow_shift5_high_sigma``.
* Training target ``noise - sample`` (``v_t = x_0 - x_1`` in upstream) and
  noising ``x_t = (1-σ)·sample + σ·noise`` are identical to Wan's default
  fallback in ``BaseWAMArchitecture.compute_loss``, so
  ``CosmosPredict25VideoBackbone`` does **not** override ``add_training_noise`` /
  ``training_target``.
* Loss weighting is uniform — see ``_compute_training_weights`` below.
"""

from __future__ import annotations

from typing import Optional

import torch


class CosmosFlowSchedulerAdapter:
    """Wan-compatible flow-matching scheduler tuned for Cosmos-Predict2.5.

    Args:
        shift_video: Rectified-flow shift parameter. Default ``5.0`` matches
            Cosmos-Predict2.5 upstream (see module docstring).
        num_train_timesteps: Total training timestep budget (default 1000,
            matches Wan / FLUX / Cosmos conventions).
    """

    def __init__(self, *, shift_video: float = 5.0, num_train_timesteps: int = 1000):
        self.shift_video = float(shift_video)
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
        shift_val = self.shift_video if shift is None else float(shift)
        self.sigmas = self._flow_match_sigmas(num_inference_steps, shift_val, denoising_strength)
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            self.linear_timesteps_weights = self._compute_training_weights(self.timesteps)
            self.training = True
        else:
            self.linear_timesteps_weights = None
            self.training = False

    @staticmethod
    def _compute_training_weights(timesteps: torch.Tensor) -> torch.Tensor:
        """Uniform per-step weight, matching Cosmos-Predict2.5 upstream.

        Upstream uses ``TrainTimeWeight("uniform")`` which returns all-ones
        (``cosmos_predict2/_src/predict2/schedulers/rectified_flow.py:21-42``);
        the loss is then ``mean(weights * per_instance_loss)`` with
        ``weights = ones`` at
        ``cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py:793``.

        Wan's bell-shaped weighting deliberately diverges from this — keeping
        the Cosmos adapter on upstream-matching uniform weights avoids
        silent hyperparameter drift when porting LR / loss-scale settings
        from official Cosmos training runs.
        """
        return torch.ones_like(timesteps, dtype=torch.float32)

    # --- Inference / loss helpers (mirror Wan FlowMatchScheduler API) ---

    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor, to_final: bool = False, **_):
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("CosmosFlowSchedulerAdapter.step called before set_timesteps()")
        ts = timestep.cpu() if isinstance(timestep, torch.Tensor) else timestep
        idx = int(torch.argmin((self.timesteps - ts).abs()))
        sigma = self.sigmas[idx]
        sigma_next = 0 if to_final or idx + 1 >= len(self.timesteps) else self.sigmas[idx + 1]
        return sample + model_output * (sigma_next - sigma)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep) -> torch.Tensor:
        if self.sigmas is None:
            raise RuntimeError("CosmosFlowSchedulerAdapter.add_noise called before set_timesteps()")
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


__all__ = ["CosmosFlowSchedulerAdapter"]
