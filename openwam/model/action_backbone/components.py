"""Shared components for action model architectures.

These building blocks are reused across DualSystem (``ActionDiT``) and
SingleSystem (``SharedVanillaActionBackbone`` / ``SharedMoEActionBackbone``).
"""

import logging
import math
import os
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional embedding for timestep conditioning."""
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_1d(head_dim: int, max_len: int = 1024, theta: float = 10000.0) -> torch.Tensor:  # noqa: B008
    """Precompute complex rotary frequencies for 1D RoPE.

    Returns a complex64 tensor of shape (max_len, head_dim // 2).

    Note:
        This mirrors FastWAM's `precompute_freqs_cis` in `wan_video_dit.py`.
        The default `max_len=1024` matches FastWAM to support action sequences
        up to 1024 steps (e.g., ~3.5s at 300Hz or ~7s at 150Hz).
    """
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).double() / head_dim))
    freqs = torch.outer(torch.arange(max_len).double(), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_apply_1d(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply 1D rotary position embedding to Q or K.

    Mirrors FastWAM/Wan RoPE numerics by doing the complex multiply through a
    float64 complex view, then restoring the caller's dtype.

    Args:
        x:     (B, H, S, D) head-split tensor.
        freqs: (S, D // 2) complex frequencies from ``precompute_freqs_cis_1d``.

    Returns:
        Rotated tensor with the same dtype and shape as ``x``.
    """
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.to(x_c.device).view(1, 1, x_c.shape[-2], x_c.shape[-1])
    return torch.view_as_real(x_c * freqs).flatten(-2).to(x.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by MLP projection.

    Input: (B,) timestep scalar
    Output: (B, dim) timestep embedding
    """

    def __init__(self, freq_dim: int, dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        t = sinusoidal_embedding_1d(self.freq_dim, timestep)
        return self.mlp(t)


class TimestepModulation(nn.Module):
    """Projects timestep embedding to per-block modulation parameters.

    Input: (B, dim) from TimestepEmbedding
    Output: (B, n_params, dim)
    """

    def __init__(self, dim: int, n_params: int):
        super().__init__()
        self.n_params = n_params
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * n_params),
        )

    def forward(self, t_embed: torch.Tensor) -> torch.Tensor:
        return self.proj(t_embed).unflatten(-1, (self.n_params, -1))


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding for timestep conditioning inside ActionEncoder.

    Generates a (B, T, embedding_dim) encoding from (B, T) timesteps —
    distinct from `sinusoidal_embedding_1d` above which operates on (B,)
    timesteps and returns (B, dim).
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        assert embedding_dim % 2 == 0, f"SinusoidalPositionalEncoding requires even embedding_dim got {embedding_dim}"
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B, T) float tensor.
        Returns:
            (B, T, embedding_dim) tensor.
        """
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (math.log(10000.0) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class ActionEncoder(nn.Module):
    """3-layer MLP action projector with timestep sinusoid fusion.

    Fuses the diffusion timestep into the action embedding inside the
    encoder itself, in addition to any AdaLN modulation that may happen
    downstream.

    Accepts timesteps of shape (1,), (B,), or (B, T) — broadcasts (1,) or
    (B,) to (B, T).
    """

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(action_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions:   (B, T, action_dim).
            timesteps: (1,) scalar broadcast, (B,) per-sample, or (B, T) per-token diffusion timestep.
        Returns:
            (B, T, hidden_dim).
        """
        B, T, _ = actions.shape
        if timesteps.dim() == 1:
            if timesteps.numel() == 1:
                # (1,) -> broadcast to (B, T)
                timesteps = timesteps.expand(B, T)
            elif timesteps.shape[0] != B:
                raise ValueError(f"timesteps {tuple(timesteps.shape)} does not match actions batch {B}")
            else:
                # (B,) -> expand to (B, T)
                timesteps = timesteps.unsqueeze(1).expand(B, T)
        elif timesteps.shape != (B, T):
            raise ValueError(
                f"timesteps {tuple(timesteps.shape)} must be (1,), (B,) or (B, T) with actions (B={B}, T={T})"
            )

        a_emb = self.W1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = F.silu(self.W2(x))
        x = self.W3(x)
        return x


class StateEncoder(nn.Module):
    """Project proprioceptive state into one single-system state token.

    The first single-system proprio path uses a single current-state token:
    input ``(D,)``, ``(B, D)`` or ``(B, 1, D)`` and output
    ``(B, 1, hidden_dim)``.
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        elif state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0, :]
        if state.ndim != 2:
            raise ValueError(f"proprio must have shape [D], [B, D] or [B, 1, D], got {tuple(state.shape)}")
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"proprio last dim must be {self.state_dim}, got {state.shape[-1]}")
        return self.proj(state).unsqueeze(1)


DEFAULT_ACTION_DECODER_HIDDEN_DIM = 1024


class ActionOutputMLP(nn.Module):
    """2-layer MLP output projection for action prediction.

    A plain Linear -> ReLU -> Linear stack with small-random weight
    initialization (N(0, 0.02), zero bias) on both layers. Used by
    SingleSystem / MoE architectures; DualSystem's ActionDiT decodes with a
    single ``Linear(dim, action_dim)``.

    Args:
        input_dim:  Hidden size of incoming action tokens (= video_dim).
        hidden_dim: Hidden width.
        action_dim: Raw action vector dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, action_dim)
        nn.init.normal_(self.layer1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.layer1.bias)
        nn.init.normal_(self.layer2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) hidden states.
        Returns:
            (B, T, action_dim) predicted action noise.
        """
        return self.layer2(F.relu(self.layer1(x)))


# ================================================================
# Attention backend auto-selection
# ================================================================
# Automatically selects the most efficient available attention backend,
# following the priority: Flash Attention 3 > Flash Attention 2 >
# Sage Attention > xFormers > PyTorch native SDPA.
#
# Override via the ``WAM_ATTENTION_IMPL`` environment variable:
#     flash3, flash2, sage, xformers, sdpa

_ATTENTION_FN: Callable | None = None


def _try_flash_attn_3() -> Callable | None:
    try:
        from flash_attn_interface import flash_attn_func as flash3_fn  # type: ignore

        def _flash3(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            if q.device.type != "cuda":
                return _sdpa(q, k, v)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash3_fn(q, k, v)
            return out.transpose(1, 2)

        return _flash3
    except ImportError:
        return None


def _try_flash_attn_2() -> Callable | None:
    try:
        from flash_attn import flash_attn_func  # type: ignore

        def _flash2(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            if q.device.type != "cuda":
                return _sdpa(q, k, v)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = flash_attn_func(q, k, v)
            return out.transpose(1, 2)

        return _flash2
    except ImportError:
        return None


def _try_sage_attention() -> Callable | None:
    try:
        from sageattention import sageattn  # type: ignore

        def _sage(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            if q.device.type != "cuda":
                return _sdpa(q, k, v)
            return sageattn(q, k, v)

        return _sage
    except ImportError:
        return None


def _try_xformers() -> Callable | None:
    try:
        from xformers.ops import memory_efficient_attention  # type: ignore

        def _xformers(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            if q.device.type != "cuda":
                return _sdpa(q, k, v)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = memory_efficient_attention(q, k, v)
            return out.transpose(1, 2)

        return _xformers
    except ImportError:
        return None


def _sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return F.scaled_dot_product_attention(q, k, v)


_BACKEND_MAP = {
    "flash3": _try_flash_attn_3,
    "flash2": _try_flash_attn_2,
    "sage": _try_sage_attention,
    "xformers": _try_xformers,
    "sdpa": lambda: _sdpa,
}

_AUTO_PRIORITY = ["flash3", "flash2", "sage", "xformers", "sdpa"]


def get_attention_fn() -> Callable:
    """Return the best available attention function.

    Signature: ``(q, k, v) -> out`` where tensors are ``(B, H, S, D)``.
    """
    global _ATTENTION_FN
    if _ATTENTION_FN is not None:
        return _ATTENTION_FN

    override = os.environ.get("WAM_ATTENTION_IMPL", "").strip().lower()

    if override:
        if override not in _BACKEND_MAP:
            raise ValueError(f"Unknown WAM_ATTENTION_IMPL='{override}'. Choose from: {list(_BACKEND_MAP.keys())}")
        fn = _BACKEND_MAP[override]()
        if fn is None:
            logger.warning(
                "WAM_ATTENTION_IMPL='%s' requested but not available, falling back to auto-detect",
                override,
            )
        else:
            logger.info("Using attention backend: %s (explicit)", override)
            _ATTENTION_FN = fn
            return _ATTENTION_FN

    for name in _AUTO_PRIORITY:
        fn = _BACKEND_MAP[name]()
        if fn is not None:
            logger.info("Using attention backend: %s (auto-detected)", name)
            _ATTENTION_FN = fn
            return _ATTENTION_FN

    _ATTENTION_FN = _sdpa
    return _ATTENTION_FN
