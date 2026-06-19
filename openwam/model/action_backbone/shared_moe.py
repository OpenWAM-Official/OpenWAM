"""MoE Action Backbone: action-side helpers for SharedBackbone MoE.

Inspired by BAGEL's Mixture-of-Transformer-Experts (MoT) pattern:
- Action tokens are concatenated to the video token sequence (handled by
  the architecture, not this module).
- Shared self-attention: action and video tokens attend to each other
  using the VIDEO DiT's Q/K/V projections (shared representational
  space).
- Expert FFN: at designated ``expert_layers``, action tokens receive an
  additional FFN correction for modality-specific capacity.

This module owns the action-side parameters but **does not** drive the
video DiT block loop — the architecture's ``forward`` runs the loop and
calls ``apply_expert(layer_id, ...)`` exactly when ``layer_id in
expert_layers_set``.

API surface:
    encode(noisy_actions, timestep) -> (tokens, t_mod, t_embed)
    apply_expert(layer_id, x_action, t_mod) -> x_action
    decode(action_tokens) -> action_prediction
    expert_layers_set                        (attribute)

References:
- BAGEL (ByteDance Seed): Shared attention + expert FFN for multimodal
  understanding and generation (arXiv:2505.14683).
- DreamZero: Shared backbone WAM with action+video in same DiT.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from openwam.model.action_backbone.base import ActionBackbone
from openwam.model.action_backbone.components import (
    DEFAULT_ACTION_DECODER_HIDDEN_DIM,
    ActionEncoder,
    ActionOutputMLP,
    StateEncoder,
    TimestepEmbedding,
    TimestepModulation,
)


class ExpertFFNBlock(nn.Module):
    """Single expert FFN with AdaLN modulation, applied as a residual correction.

    Computation::

        h = LayerNorm(x) * (1 + scale) + shift
        x = x + gate * FFN(h)

    AdaLN params come from the action timestep; ``modulation`` is a
    learnable base offset added before chunking into (shift, scale, gate).
    Output linear is zero-initialized so the expert correction starts at
    zero, preserving the pretrained video DiT behavior at init.
    """

    def __init__(self, dim: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        # AdaLN base modulation (3 params: shift, scale, gate)
        self.modulation = nn.Parameter(torch.randn(1, 3, dim) / dim**0.5)

        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        """Apply AdaLN-modulated expert FFN as a residual.

        Args:
            x: (B, T_action, dim).
            t_mod: (B, 3, dim) per-sample, or (B, T_action, 3, dim) per-token.

        Returns:
            (B, T_action, dim).
        """
        base = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        if t_mod.dim() == 4:
            shift, scale, gate = (base.unsqueeze(1) + t_mod).chunk(3, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
            gate = gate.squeeze(2)
        else:
            shift, scale, gate = (base + t_mod).chunk(3, dim=1)

        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.ffn(h)


class SharedMoEActionBackbone(ActionBackbone):
    """Action-side helpers for SharedBackbone MoE.

    Owns:
      - ``input_proj`` / ``action_output_head``:
        encode/decode for action tokens (project into video_dim, decode back).
      - ``time_embedding`` / ``time_projection``: produce the AdaLN t_mod
        consumed by the expert FFN blocks. **Independent from the video DiT's
        own ``time_embedding`` / ``time_projection``**: the video DiT's
        modules feed the per-block self-attn AdaLN (via
        ``_build_action_t_mod`` in the adapter), while these ones drive only
        the expert-FFN AdaLN. Two separate routes is intentional —
        modality-specific modulation for the modality-specific FFN.
      - ``expert_blocks``: one ``ExpertFFNBlock`` per entry in ``expert_layers``.
      - ``action_mean`` / ``action_std``: normalization stats.

    Unlike ActionDiT, this module has no self-attention or cross-attention
    of its own — action tokens participate in the video DiT's shared
    self-attention by being concatenated to the video sequence.
    """

    def __init__(
        self,
        action_dim: int,
        video_dim: int,
        expert_ffn_dim: int,
        expert_layers: Tuple[int, ...],
        freq_dim: int = 256,
        max_action_len: int = 512,
        action_decoder_hidden_dim: Optional[int] = None,
        eps: float = 1e-6,
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
        self.expert_layers = tuple(int(i) for i in expert_layers)
        self.expert_layers_set = set(self.expert_layers)
        self.expert_layer_to_index = {layer_id: idx for idx, layer_id in enumerate(self.expert_layers)}
        self.num_experts = len(self.expert_layers)

        self.input_proj = ActionEncoder(self._action_dim, self._video_dim)
        self.state_encoder = StateEncoder(self.state_dim, self._video_dim) if self._use_proprioception else None
        self.time_embedding = TimestepEmbedding(freq_dim, self._video_dim)
        self.time_projection = TimestepModulation(self._video_dim, 3)
        self.expert_blocks = nn.ModuleList(
            [ExpertFFNBlock(self._video_dim, expert_ffn_dim, eps) for _ in range(self.num_experts)]
        )
        self.action_output_head = ActionOutputMLP(self._video_dim, self._action_decoder_hidden_dim, self._action_dim)
        self.register_buffer("action_mean", torch.zeros(self._action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(self._action_dim), persistent=True)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def uses_proprioception(self) -> bool:
        return self._use_proprioception

    def encode(self, noisy_actions: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project actions and build expert-FFN AdaLN modulation.

        Args:
            noisy_actions: (B, T, action_dim).
            timestep: action diffusion timestep with one of these shapes:
                (1,) scalar broadcast, (B,) per-sample, or (B, T) per-token.

        Returns:
            Tuple of:
                tokens: (B, T, video_dim) — projected action tokens.
                t_mod: (B, 3, dim) per-sample, or (B, T, 3, dim) per-token.
        """
        B, T, _ = noisy_actions.shape
        if T > self._max_action_len:
            raise ValueError(f"Action sequence length {T} exceeds max_action_len {self._max_action_len}.")

        x = self.input_proj(noisy_actions, timestep)

        per_token = timestep.dim() == 2 and timestep.shape == (B, T)
        if per_token:
            flat = timestep.reshape(B * T)
            t_flat = self.time_embedding(flat)
            t_mod_flat = self.time_projection(t_flat)
            t_mod = t_mod_flat.view(B, T, self.time_projection.n_params, -1)
        else:
            timestep_flat = timestep.flatten()
            if timestep_flat.numel() == 1:
                timestep_flat = timestep_flat.expand(B)
            elif timestep_flat.shape[0] != B:
                raise ValueError(
                    f"timestep has shape {tuple(timestep.shape)}; expected (1,), (B={B},), or (B={B}, T={T})."
                )
            t_embed = self.time_embedding(timestep_flat)
            t_mod = self.time_projection(t_embed)

        return x, t_mod

    def encode_state(self, proprio_state: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._use_proprioception:
            return None
        if proprio_state is None:
            raise ValueError("SharedBackbone use_proprioception=True requires `proprio_state`.")
        assert self.state_encoder is not None
        return self.state_encoder(proprio_state)

    def apply_expert(self, layer_id: int, x_action: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        """Apply the expert FFN at the given video DiT layer to action tokens.

        ``layer_id`` must be in ``expert_layers_set`` — the architecture's
        forward is responsible for the membership check before calling.

        Args:
            layer_id: Video DiT layer index where the expert is anchored.
            x_action: (B, T_action, video_dim) action slice after the video block.
            t_mod: AdaLN modulation produced by ``encode`` (per-sample or per-token).

        Returns:
            (B, T_action, video_dim) corrected action tokens.
        """
        return self.expert_blocks[self.expert_layer_to_index[layer_id]](x_action, t_mod)

    def decode(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """Project action tokens back from video_dim to action_dim."""
        return self.action_output_head(action_tokens)


__all__ = ["ExpertFFNBlock", "SharedMoEActionBackbone"]
