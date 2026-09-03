"""Action / state token t_mod + 1D RoPE helpers for single-system
(DreamZero-style) action injection.

Free functions consumed by ``WanBase.inject_shared_tokens`` — the DiT
is passed explicitly so this module stays Wan-internal (no backbone/ABC import;
``is_per_token_t_mod_active`` takes the ``time_mod`` tensor, not BlockLoopState).
"""

from __future__ import annotations

import torch
from torch import Tensor

from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d


def is_per_token_t_mod_active(time_mod: Tensor) -> bool:
    return time_mod.dim() == 4


def build_action_t_mod(
    action_timestep: Tensor,
    n_action_tokens: int,
    *,
    dit,
    batch_size: int,
) -> Tensor:
    if action_timestep.dim() == 2:
        if action_timestep.shape != (batch_size, n_action_tokens):
            raise ValueError(
                f"action_timestep has shape {tuple(action_timestep.shape)}; expected "
                f"(B={batch_size}, n_action_tokens={n_action_tokens})."
            )
        B_t = action_timestep.shape[0]
        flat = action_timestep.reshape(B_t * n_action_tokens)
        t_emb = sinusoidal_embedding_1d(dit.freq_dim, flat)
        dtype = next(dit.time_embedding.parameters()).dtype
        t = dit.time_embedding(t_emb.to(dtype=dtype, device=t_emb.device))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        t_mod = t_mod.view(B_t, n_action_tokens, 6, dit.dim)
    else:
        timestep_flat = action_timestep.flatten()
        if timestep_flat.numel() == 1:
            timestep_flat = timestep_flat.expand(batch_size)
        elif timestep_flat.numel() != batch_size:
            raise ValueError(
                f"action_timestep has shape {tuple(action_timestep.shape)}; expected scalar, "
                f"(B={batch_size},), or (B={batch_size}, n_action_tokens={n_action_tokens})."
            )
        t_emb = sinusoidal_embedding_1d(dit.freq_dim, timestep_flat)
        dtype = next(dit.time_embedding.parameters()).dtype
        t = dit.time_embedding(t_emb.to(dtype=dtype, device=t_emb.device))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        t_mod = t_mod.unsqueeze(1).expand(-1, n_action_tokens, -1, -1)
    return t_mod


def build_sample_t_mod(
    timestep: Tensor,
    n_tokens: int,
    *,
    dit,
    batch_size: int,
) -> Tensor:
    timestep_flat = timestep.flatten()
    if timestep_flat.numel() == 1:
        timestep_flat = timestep_flat.expand(batch_size)
    elif timestep_flat.numel() == batch_size:
        pass
    elif timestep.dim() == 2 and timestep.shape[0] == batch_size:
        timestep_flat = timestep[:, 0]
    else:
        raise ValueError(
            f"timestep has shape {tuple(timestep.shape)}; expected scalar, (B={batch_size},), "
            f"or (B={batch_size}, T) for sample-level state t_mod."
        )
    t_emb = sinusoidal_embedding_1d(dit.freq_dim, timestep_flat)
    dtype = next(dit.time_embedding.parameters()).dtype
    t = dit.time_embedding(t_emb.to(dtype=dtype, device=t_emb.device))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    t_mod = t_mod.unsqueeze(1).expand(-1, n_tokens, -1, -1)
    return t_mod


def extend_freqs_with_action_tokens(freqs: Tensor, n_action_tokens: int) -> Tensor:
    """Append 1D action RoPE frequencies to ``freqs`` (DreamZero's separate
    action RoPE — 1D positions in action-horizon space). Tied to Wan's
    complex-form RoPE; a different RoPE representation needs this changed too.
    """
    if n_action_tokens <= 0:
        return freqs
    return torch.cat([freqs, build_1d_action_freqs(freqs, n_action_tokens)], dim=0)


def extend_freqs_with_shared_tokens(freqs: Tensor, n_action_tokens: int, n_state_tokens: int = 0) -> Tensor:
    pieces = [freqs]
    if n_action_tokens > 0:
        pieces.append(build_1d_action_freqs(freqs, n_action_tokens))
    if n_state_tokens > 0:
        pieces.append(build_1d_state_freqs(freqs, n_state_tokens))
    return torch.cat(pieces, dim=0)


def build_1d_action_freqs(freqs: Tensor, n_action_tokens: int, theta: float = 10000.0) -> Tensor:
    head_dim = int(freqs.shape[-1]) * 2
    positions = torch.arange(n_action_tokens, dtype=torch.float64, device=freqs.device)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64, device=freqs.device) / head_dim))
    angles = torch.outer(positions, inv_freq)
    action_freqs = torch.polar(torch.ones_like(angles), angles).view(n_action_tokens, 1, -1)
    return action_freqs.to(dtype=freqs.dtype)


def build_1d_state_freqs(freqs: Tensor, n_state_tokens: int, theta: float = 10000.0) -> Tensor:
    head_dim = int(freqs.shape[-1]) * 2
    positions = torch.arange(n_state_tokens, dtype=torch.float64, device=freqs.device)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64, device=freqs.device) / head_dim))
    angles = torch.outer(positions, inv_freq)
    state_freqs = torch.polar(torch.ones_like(angles), angles).view(n_state_tokens, 1, -1)
    return state_freqs.to(dtype=freqs.dtype)
