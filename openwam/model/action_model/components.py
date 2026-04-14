"""Shared components for action model architectures.

These building blocks are used identically across DualSystem (ActionDiT),
MoE Expert (MoEExpertDiT), and SharedBackbone architectures.
"""

import torch
import torch.nn as nn


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional embedding for timestep conditioning."""
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


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


class ActionEmbedding(nn.Module):
    """Projects raw action vectors to hidden dimension.

    Architecture: Linear → GELU → Linear
    """

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for action token sequences."""

    def __init__(self, max_len: int, dim: int, scale: float = 0.02):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(1, max_len, dim) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor (B, T, dim)."""
        return x + self.embedding[:, : x.shape[1], :]


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


class ActionOutputHead(nn.Module):
    """Output head for action prediction with AdaLN modulation.

    Applies LayerNorm → AdaLN (shift + scale from timestep) → Linear.
    Output weights are zero-initialized for stable training start.
    """

    def __init__(self, dim: int, action_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, action_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """Apply modulated output head.

        Args:
            x: (B, T, dim) hidden states
            t_embed: (B, dim) timestep embedding for modulation
        """
        shift, scale = (self.modulation + t_embed.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm(x) * (1 + scale) + shift
        return self.head(x)
