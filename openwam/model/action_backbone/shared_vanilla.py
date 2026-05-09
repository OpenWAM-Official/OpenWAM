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

from openwam.model.action_backbone.backbone import ActionBackbone
from openwam.model.action_backbone.components import (
    DEFAULT_ACTION_DECODER_HIDDEN_DIM,
    ActionEncoder,
    ActionOutputMLP,
    StateEncoder,
)


class SharedVanillaActionBackbone(ActionBackbone):
    """Action-side I/O for SharedBackbone vanilla.

    Holds:
      - ``input_proj``: action_dim -> video_dim (fuses timestep)
      - ``action_output_head``: video_dim -> decoder_hidden_dim -> action_dim
      - ``action_mean`` / ``action_std`` persistent buffers
      - ``scheduler``: ActionScheduler (from ActionBackbone.__init__)
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
        super().__init__()
        self._action_dim = int(action_dim)
        self._video_dim = int(video_dim)
        self._max_action_len = int(max_action_len)
        self._action_decoder_hidden_dim = int(action_decoder_hidden_dim or DEFAULT_ACTION_DECODER_HIDDEN_DIM)
        self._use_proprioception = bool(use_proprioception)
        self.state_dim = int(state_dim or 0)
        if self._use_proprioception and self.state_dim <= 0:
            raise ValueError("use_proprioception=True requires state_dim > 0 for SharedBackbone state tokens.")

        self.input_proj = ActionEncoder(self._action_dim, self._video_dim)
        self.state_encoder = StateEncoder(self.state_dim, self._video_dim) if self._use_proprioception else None
        self.action_output_head = ActionOutputMLP(self._video_dim, self._action_decoder_hidden_dim, self._action_dim)
        self.register_buffer("action_mean", torch.zeros(self._action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(self._action_dim), persistent=True)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def expert_layers(self) -> tuple[int, ...]:
        return ()

    @property
    def uses_proprioception(self) -> bool:
        return self._use_proprioception

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

    def encode_state(self, proprio_state: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._use_proprioception:
            return None
        if proprio_state is None:
            raise ValueError("SharedBackbone use_proprioception=True requires `proprio_state`.")
        assert self.state_encoder is not None
        return self.state_encoder(proprio_state)

    def decode(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, video_dim) action tail -> (B, T, action_dim) prediction."""
        return self.action_output_head(action_tokens)


__all__ = ["SharedVanillaActionBackbone"]
