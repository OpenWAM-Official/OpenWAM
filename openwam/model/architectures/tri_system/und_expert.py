"""Understanding Expert for tri_system: 30-layer transformer participating
in trimodal joint attention via separate QKV in WAN head space.

No AdaLN, no decoder, no registers. Aligned with Motus/models/und_expert.py.
"""

# Source: https://github.com/thu-ml/Motus (models/und_expert.py).
# Upstream revision: UNKNOWN (the original internal import did not record a commit SHA).
# License: Apache-2.0; see the repository-level LICENSE (Apache-2.0).
# Modified by OpenWAM contributors: rewritten around OpenWAM tensor, mask,
# configuration, and backbone contracts.

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from openwam.model.action_backbone.components import RMSNorm


class _WanCompatLayerNorm(nn.LayerNorm):
    """LayerNorm without elementwise affine, matching Motus WanLayerNorm semantics.

    PyTorch's CUDA LayerNorm kernel already uses fp32 accumulation internally
    for bf16/fp16 inputs (verified numerically identical to the explicit
    ``.float()`` cast on CUDA, PyTorch >= 2.1). The previous implementation
    did ``super().forward(x.float()).type_as(x)`` which added 60 dtype
    conversions per forward (30 layers × 2 norms) with no benefit on GPU.

    Note: on CPU the native kernel does NOT do fp32 accumulation, so CPU-only
    unit tests may see small numerical differences vs the old path. All
    training and inference runs on CUDA where the two are bitwise identical.
    CPU-only tests (the default CI path) verify structural correctness
    (shapes, grad flow, mask layout) but not numerical equivalence with
    the CUDA fp32 accumulation path.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__(dim, elementwise_affine=False, eps=eps)


@dataclass
class UnderstandingExpertConfig:
    dim: int = 512
    ffn_dim: int = 2048
    num_layers: int = 30
    vlm_input_dim: int = 2048
    vlm_projector_type: str = "mlp3x_silu"
    eps: float = 1e-5


@dataclass
class UnderstandingState:
    und_tokens: torch.Tensor
    # [B, Su] bool, True = valid token; None means all positions valid (zero-cost path).
    # Sourced from the VLM's attention_mask so padded und positions do not leak into
    # video/action queries through the trimodal joint attention.
    und_mask: Optional[torch.Tensor] = None


class UnderstandingExpertBlock(nn.Module):
    def __init__(self, cfg: UnderstandingExpertConfig, wan_dim: int, wan_num_heads: int):
        super().__init__()
        wan_head_dim = wan_dim // wan_num_heads
        self.norm1 = _WanCompatLayerNorm(cfg.dim, eps=cfg.eps)
        self.norm2 = _WanCompatLayerNorm(cfg.dim, eps=cfg.eps)
        self.wan_und_qkv = nn.Parameter(
            torch.randn(3, wan_num_heads, cfg.dim, wan_head_dim) / (cfg.dim * wan_head_dim) ** 0.5
        )
        self.wan_und_o = nn.Linear(wan_dim, cfg.dim, bias=False)
        self.wan_und_norm_q = RMSNorm(wan_dim, eps=cfg.eps)
        self.wan_und_norm_k = RMSNorm(wan_dim, eps=cfg.eps)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.dim, cfg.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.ffn_dim, cfg.dim),
        )


class UnderstandingExpert(nn.Module):
    def __init__(self, cfg: UnderstandingExpertConfig, wan_dim: int, wan_num_heads: int):
        super().__init__()
        self.cfg = cfg
        self._num_heads = wan_num_heads
        self._head_dim = wan_dim // wan_num_heads
        self.vlm_projector = self._build_mlp(cfg.vlm_projector_type, cfg.vlm_input_dim, cfg.dim)
        # Init variance 1/(cfg.dim * wan_head_dim) for wan_und_qkv (see UnderstandingExpertBlock)
        # follows the Motus understanding expert reference (Motus/models/und_expert.py:
        # WanUndQKV.__init__). Smaller than standard Xavier (1/cfg.dim) by sqrt(head_dim);
        # intentional to keep early und→{v,a} contribution small while the projector warms up.
        # Do not change without re-tuning warmup.
        self.blocks = nn.ModuleList(
            [UnderstandingExpertBlock(cfg, wan_dim, wan_num_heads) for _ in range(cfg.num_layers)]
        )

    @property
    def num_layers(self) -> int:
        return self.cfg.num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @staticmethod
    def _build_mlp(projector_type: str, in_features: int, out_features: int) -> nn.Module:
        if projector_type == "linear":
            return nn.Linear(in_features, out_features)
        m = re.match(r"^mlp(\d+)x_silu$", projector_type)
        if m:
            depth = int(m.group(1))
            modules = [nn.Linear(in_features, out_features)]
            for _ in range(1, depth):
                modules.append(nn.SiLU())
                modules.append(nn.Linear(out_features, out_features))
            return nn.Sequential(*modules)
        raise ValueError(f"Unknown projector type: {projector_type}")

    def prepare_state(
        self,
        vlm_hidden: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        vlm_attention_mask: Optional[torch.Tensor] = None,
    ) -> UnderstandingState:
        target_dtype = dtype if dtype is not None else next(self.vlm_projector.parameters()).dtype
        und_tokens = self.vlm_projector(vlm_hidden.to(dtype=target_dtype))
        und_mask: Optional[torch.Tensor] = None
        if vlm_attention_mask is not None:
            if vlm_attention_mask.ndim != 2:
                raise ValueError(
                    "vlm_attention_mask must be 2D [B, L] aligned with vlm_hidden's sequence dim, "
                    f"got shape {tuple(vlm_attention_mask.shape)}"
                )
            if vlm_attention_mask.shape != und_tokens.shape[:2]:
                raise ValueError(
                    "vlm_attention_mask shape must match vlm_hidden [B, L], got "
                    f"{tuple(vlm_attention_mask.shape)} vs {tuple(und_tokens.shape[:2])}"
                )
            und_mask = vlm_attention_mask.to(device=und_tokens.device, dtype=torch.bool)
        return UnderstandingState(und_tokens=und_tokens, und_mask=und_mask)

    def pre_attn_at_layer(
        self, layer_id: int, state: UnderstandingState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        q, k, v, post_tuple = self.pre_attn_at_layer_for_compile(layer_id, state)
        (residual_x,) = post_tuple
        return q, k, v, {"residual_x": residual_x}

    def pre_attn_at_layer_for_compile(
        self, layer_id: int, state: UnderstandingState
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor]]:
        block = self.blocks[layer_id]
        residual_x = state.und_tokens
        norm_und = block.norm1(residual_x)
        qkv = torch.einsum("BTD,KNDE->KBTNE", norm_und, block.wan_und_qkv)
        q_h, k_h, v_h = qkv[0], qkv[1], qkv[2]
        q = block.wan_und_norm_q(q_h.flatten(-2))
        k = block.wan_und_norm_k(k_h.flatten(-2))
        v = v_h.flatten(-2)
        return q, k, v, (residual_x,)

    def post_attn_at_layer(
        self,
        layer_id: int,
        state: UnderstandingState,
        attn_out: torch.Tensor,
        post_state: dict,
    ) -> UnderstandingState:
        if isinstance(post_state, dict):
            post_state = (post_state["residual_x"],)
        return self.post_attn_at_layer_for_compile(layer_id, state, attn_out, post_state)

    def post_attn_at_layer_for_compile(
        self,
        layer_id: int,
        state: UnderstandingState,
        attn_out: torch.Tensor,
        post_state: tuple[torch.Tensor],
    ) -> UnderstandingState:
        block = self.blocks[layer_id]
        (residual_x,) = post_state
        x = residual_x + block.wan_und_o(attn_out)
        x = x + block.ffn(block.norm2(x))
        state.und_tokens = x
        return state
