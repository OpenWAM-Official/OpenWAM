"""Learnable-query cross-attention decoder: latent action -> real action.

Used only in dual-system latent mode (model.action_backbone.type=latent) to
decode ActionDiT's predicted latent action tokens into executable robot actions.
``num_query`` learnable queries cross-attend the latent sequence (KV) and
self-attend each other, then project to ``real_action_dim``. Deterministic
(no diffusion / no timestep).

When ``use_proprioception`` is on, the current normalized proprio state is
encoded into one extra KV token appended to the latent sequence — latent action
is proprio-agnostic, so proprio anchors the decode to the robot's current pose.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from openwam.model.action_backbone.action_dit import ActionSelfAttention, BridgeCrossAttention


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class _DecoderBlock(nn.Module):
    """DETR-style block: self-attn(query) -> cross-attn(query->latent) -> FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.self_attn = ActionSelfAttention(hidden_dim, num_heads, attn_head_dim, eps)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.cross_attn = BridgeCrossAttention(hidden_dim, num_heads, attn_head_dim, eps, kv_hidden_dim=hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Learnable queries carry their own position, so self-attn uses no RoPE.
        # ``kv`` is the projected latent sequence, optionally with a proprio token.
        query = query + self.self_attn(self.self_attn_norm(query))
        query = query + self.cross_attn(self.cross_attn_norm(query), kv)
        query = query + self.ffn(self.ffn_norm(query))
        return query


class LatentQueryDecoder(nn.Module):
    """Decode latent action tokens into real actions via learnable queries."""

    def __init__(
        self,
        latent_dim: int,
        real_action_dim: int,
        num_query: int,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        attn_head_dim: int = 128,
        ffn_dim: int = 4096,
        eps: float = 1e-6,
        use_proprioception: bool = False,
        proprio_dim: int = 0,
    ):
        super().__init__()
        self.num_query = int(num_query)
        self.real_action_dim = int(real_action_dim)
        self.use_proprioception = bool(use_proprioception)
        self.query = nn.Parameter(torch.randn(self.num_query, hidden_dim) * hidden_dim**-0.5)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.proprio_proj = None
        if self.use_proprioception:
            if int(proprio_dim) <= 0:
                raise ValueError("LatentQueryDecoder use_proprioception=True requires proprio_dim > 0.")
            self.proprio_proj = nn.Linear(int(proprio_dim), hidden_dim)
        self.blocks = nn.ModuleList(
            [_DecoderBlock(hidden_dim, num_heads, attn_head_dim, ffn_dim, eps) for _ in range(int(num_layers))]
        )
        self.norm_out = nn.LayerNorm(hidden_dim, eps=eps)
        self.head = nn.Linear(hidden_dim, self.real_action_dim)

    def forward(self, latent: torch.Tensor, proprio: torch.Tensor | None = None) -> torch.Tensor:
        """latent (B, T_latent, latent_dim) [+ proprio (B, 1, proprio_dim)]
        -> action (B, num_query, real_action_dim). proprio must be normalized."""
        kv = self.latent_proj(latent)
        if self.use_proprioception:
            if proprio is None:
                raise ValueError("LatentQueryDecoder use_proprioception=True requires a proprio input.")
            proprio = proprio.to(dtype=kv.dtype)
            if proprio.dim() == 1:  # (D,) -> (B, 1, D)
                proprio = proprio.view(1, 1, -1).expand(kv.shape[0], -1, -1)
            elif proprio.dim() == 2:  # (B, D) -> (B, 1, D)
                proprio = proprio.unsqueeze(1)
            proprio_h = self.proprio_proj(proprio)
            kv = torch.cat([kv, proprio_h], dim=1)
        query = self.query.unsqueeze(0).expand(latent.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, kv)
        return self.head(self.norm_out(query))


def build_latent_action_decoder(cfg: Any, *, latent_dim: int) -> LatentQueryDecoder:
    name = str(_cfg_get(cfg, "name", "cross_attn_query"))
    if name != "cross_attn_query":
        raise ValueError(f"Unsupported latent action decoder: {name!r}")
    real_action_dim = _cfg_get(cfg, "real_action_dim")
    num_query = _cfg_get(cfg, "num_query")
    if real_action_dim is None or num_query is None:
        raise ValueError("model.action_backbone.latent_decoder requires real_action_dim and num_query.")
    use_proprioception = bool(_cfg_get(cfg, "use_proprioception", False))
    return LatentQueryDecoder(
        latent_dim=int(latent_dim),
        real_action_dim=int(real_action_dim),
        num_query=int(num_query),
        hidden_dim=int(_cfg_get(cfg, "hidden_dim", 1024)),
        num_layers=int(_cfg_get(cfg, "num_layers", 4)),
        num_heads=int(_cfg_get(cfg, "num_heads", 8)),
        attn_head_dim=int(_cfg_get(cfg, "attn_head_dim", 128)),
        ffn_dim=int(_cfg_get(cfg, "ffn_dim", 4096)),
        use_proprioception=use_proprioception,
        proprio_dim=int(_cfg_get(cfg, "proprio_dim", 0)),
    )
