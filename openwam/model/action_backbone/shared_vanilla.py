"""Shared-backbone vanilla action backbone.

Used by ``SharedBackboneVanillaArchitecture``. Owns the action-specific
parameters (input projection, output head, normalization stats, action
scheduler) but does **not**
implement any control flow — the architecture's ``forward`` runs the
video DiT block loop with action tokens injected and calls these
helpers at the right moments.

API surface:
    encode(noisy_actions, timestep) -> tokens
    decode(final_hidden) -> action_prediction
"""

from __future__ import annotations

from typing import Optional

import torch

from openwam.model.action_backbone.base import SharedActionBackbone


class SharedVanillaActionBackbone(SharedActionBackbone):
    """Action-side I/O for SharedBackbone vanilla.

    Holds (via :class:`SharedActionBackbone`):
      - ``input_proj``: action_dim -> video_dim (fuses timestep)
      - ``action_output_head``: video_dim -> decoder_hidden_dim -> action_dim
      - ``action_mean`` / ``action_std`` persistent buffers
      - ``scheduler``: ActionScheduler

    No expert FFN — the raw shared DiT learns the modality boundary itself.
    """

    def __init__(
        self,
        action_dim: int,
        video_dim: int,
        max_action_len: int = 512,
        action_decoder_hidden_dim: Optional[int] = None,
        use_proprioception: bool = False,
        state_dim: int = 0,
    ):
        super().__init__(
            action_dim,
            video_dim,
            max_action_len=max_action_len,
            action_decoder_hidden_dim=action_decoder_hidden_dim,
            use_proprioception=use_proprioception,
            state_dim=state_dim,
        )
        self._init_action_input()
        self._init_action_output()

    @property
    def expert_layers(self) -> tuple[int, ...]:
        return ()

    def encode(self, noisy_actions: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Project noisy actions into video_dim space.

        Args:
            noisy_actions: (B, T, action_dim).
            timestep: action diffusion timestep, accepted shapes match
                ``ActionEncoder``: (1,), (B,), or (B, T).

        Returns:
            (B, T, video_dim) action tokens ready to be appended to the
            video sequence.
        """
        T = noisy_actions.shape[1]
        if T > self._max_action_len:
            raise ValueError(f"Action sequence length {T} exceeds max_action_len {self._max_action_len}.")
        return self.input_proj(noisy_actions, timestep)


__all__ = ["SharedVanillaActionBackbone"]
