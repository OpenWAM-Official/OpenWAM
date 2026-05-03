"""Shared-backbone vanilla action backbone.

Used by ``SharedBackboneVanillaArchitecture``. Owns the action-specific
parameters (input projection, positional encoding, output head, modality
AdaLN bias, normalization stats, action scheduler) but does **not**
implement any control flow — the architecture's ``forward`` runs the
video DiT block loop with action tokens injected and calls these
helpers at the right moments.

API surface:
    encode(noisy_actions, timestep) -> tokens
    decode(final_hidden) -> action_prediction
    modality_tmod_bias                          (parameter)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from openwam.model.action_backbone.backbone import ActionBackbone
from openwam.model.action_backbone.components import ActionEncoder, ActionOutputMLP, LearnedPositionalEncoding


class SharedVanillaActionBackbone(ActionBackbone):
    """Action-side I/O for SharedBackbone vanilla.

    Holds:
      - ``input_proj``: action_dim -> video_dim (fuses timestep)
      - ``pos_encoding``: learned positional encoding
      - ``action_output_head``: video_dim -> 64 -> action_dim
      - ``modality_tmod_bias``: per-modality bias added to the video DiT's
        AdaLN modulation signal for action tokens. Shape ``(1, 1, 6, video_dim)``
        — the leading ``6`` mirrors the video DiT block's 6-chunk AdaLN layout
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp); if
        the video DiT changes that count this bias must be resized in lockstep.
      - ``action_mean`` / ``action_std`` persistent buffers
      - ``scheduler``: ActionScheduler (from ActionBackbone.__init__)
    """

    def __init__(self, action_dim: int, video_dim: int, max_action_len: int = 512):
        super().__init__()
        self._action_dim = int(action_dim)
        self._video_dim = int(video_dim)
        self._max_action_len = int(max_action_len)

        self.input_proj = ActionEncoder(self._action_dim, self._video_dim)
        self.pos_encoding = LearnedPositionalEncoding(max_action_len, self._video_dim)
        self.action_output_head = ActionOutputMLP(self._video_dim, 64, self._action_dim)
        self.modality_tmod_bias = nn.Parameter(torch.zeros(1, 1, 6, self._video_dim))
        self.register_buffer("action_mean", torch.zeros(self._action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(self._action_dim), persistent=True)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def bridge_layers(self) -> Tuple[int, ...]:
        return ()

    def encode(self, noisy_actions: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Project noisy actions into video_dim space and add positional encoding.

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
        x = self.input_proj(noisy_actions, timestep)
        x = self.pos_encoding(x)
        return x

    def decode(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, video_dim) action tail -> (B, T, action_dim) prediction."""
        return self.action_output_head(action_tokens)


__all__ = ["SharedVanillaActionBackbone"]
