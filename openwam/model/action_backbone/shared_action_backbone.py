"""SingleSystem action backbones (vanilla + MoE).

Both share :class:`SharedActionBackbone` (action I/O: input projection, output
head, normalization buffers, scheduler) and differ only in how action tokens
couple to the video DiT:

- :class:`SharedVanillaActionBackbone` — no expert FFN; the raw shared DiT learns
  the modality boundary itself. ``encode`` returns tokens.
- :class:`SharedMoEActionBackbone` — adds a per-layer expert FFN (BAGEL-style
  Mixture-of-Transformer-Experts) at the video DiT layers named by
  ``bridge_layers``. ``encode`` also builds the expert-FFN AdaLN modulation.

Neither module drives the video DiT block loop — the architecture's ``forward``
runs the loop with action tokens injected and calls ``encode`` / ``decode`` (and
for MoE ``apply_expert(layer_id, ...)`` when ``layer_id in bridge_layers``).

References (MoE):
- BAGEL (ByteDance Seed): Shared attention + expert FFN for multimodal
  understanding and generation (arXiv:2505.14683).
- DreamZero: Single system WAM with action+video in same DiT.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from openwam.model.action_backbone.base import SharedActionBackbone
from openwam.model.action_backbone.components import (
    TimestepEmbedding,
    TimestepModulation,
)


class SharedVanillaActionBackbone(SharedActionBackbone):
    """Action-side I/O for SingleSystem vanilla.

    Holds (via :class:`SharedActionBackbone`):
      - ``input_proj``: action_dim -> video_dim (fuses timestep)
      - ``action_output_head``: video_dim -> decoder_hidden_dim -> action_dim
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


class SharedMoEActionBackbone(SharedActionBackbone):
    """Action-side helpers for SingleSystem MoE.

    Owns (beyond :class:`SharedActionBackbone`'s action I/O):
      - ``time_embedding`` / ``time_projection``: produce the AdaLN t_mod
        consumed by the expert FFN blocks. **Independent from the video DiT's
        own ``time_embedding`` / ``time_projection``**: the video DiT's
        modules feed the per-block self-attn AdaLN (via
        ``_build_action_t_mod`` in the adapter), while these ones drive only
        the expert-FFN AdaLN. Two separate routes is intentional —
        modality-specific modulation for the modality-specific FFN.
      - ``expert_blocks``: one ``ExpertFFNBlock`` per entry in ``bridge_layers``.

    Unlike ActionDiT, this module has no self-attention or cross-attention
    of its own — action tokens participate in the video DiT's shared
    self-attention by being concatenated to the video sequence.
    """

    def __init__(
        self,
        action_dim: int,
        video_dim: int,
        expert_ffn_dim: int,
        bridge_layers: Tuple[int, ...],
        freq_dim: int = 256,
        max_action_len: int = 512,
        action_decoder_hidden_dim: Optional[int] = None,
        eps: float = 1e-6,
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
        self._bridge_layers = tuple(int(i) for i in bridge_layers)
        self.expert_layer_to_index = {layer_id: idx for idx, layer_id in enumerate(self._bridge_layers)}
        self.num_experts = len(self._bridge_layers)

        # Module-creation order is load-bearing for parameter-init RNG: input
        # projection first, MoE-specific time/expert modules next, output head
        # last — matching the pre-refactor sequence so weights stay identical.
        self._init_action_input()
        self.time_embedding = TimestepEmbedding(freq_dim, self._video_dim)
        self.time_projection = TimestepModulation(self._video_dim, 3)
        self.expert_blocks = nn.ModuleList(
            [ExpertFFNBlock(self._video_dim, expert_ffn_dim, eps) for _ in range(self.num_experts)]
        )
        self._init_action_output()

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

    def apply_expert(self, layer_id: int, x_action: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        """Apply the expert FFN at the given video DiT layer to action tokens.

        ``layer_id`` must be in ``bridge_layers`` — the architecture's
        forward is responsible for the membership check before calling.

        Args:
            layer_id: Video DiT layer index where the expert is anchored.
            x_action: (B, T_action, video_dim) action slice after the video block.
            t_mod: AdaLN modulation produced by ``encode`` (per-sample or per-token).

        Returns:
            (B, T_action, video_dim) corrected action tokens.
        """
        return self.expert_blocks[self.expert_layer_to_index[layer_id]](x_action, t_mod)


__all__ = ["ExpertFFNBlock", "SharedMoEActionBackbone", "SharedVanillaActionBackbone"]
