"""Single-system token injection helpers for the Cosmos3-Edge backbone.

The single-system architectures ride action (and an optional state) token on
the video DiT's own sequence. On Cosmos3 this is unusually direct: the gen
stream is already a flat ``(B, S, D)`` sequence and the model has **no AdaLN**
(timestep conditioning is an additive embedding on the token itself), so
injection is a plain concat plus:

- rotary extension — the injected tokens get the identity rotation
  (``cos = 1``, ``sin = 0``), the Cosmos3 analogue of predict2.5 appending
  zero *angles*: both mean "no positional rotation", but Cosmos3's rotary
  module hands back cos/sin already evaluated, so the identity is 1/0 rather
  than 0/0.
- additive timestep embedding for the injected tokens, matching what
  ``prepare_block_loop`` does for noisy video tokens (same ``timestep_scale``,
  same ``time_proj``/``time_embedder`` path, so they land in one time domain).

There is deliberately no ``run_block_3d`` analogue here: the injected tokens are
ordinary gen tokens, so ``dit_forward.run_block`` handles them unchanged once the
cross-modal mask is stashed on ``extras['shared_attention_mask']``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "extend_rotary_with_shared_tokens",
    "shared_token_timestep_embedding",
    "validate_shared_tokens",
]


def extend_rotary_with_shared_tokens(cos: Tensor, sin: Tensor, n_shared: int) -> Tuple[Tensor, Tensor]:
    """Append ``n_shared`` identity-rotation rows to the gen rotary tensors.

    ``cos``/``sin`` are ``(B, S, head_dim)``; the appended rows are ones/zeros so
    the injected tokens are position-agnostic (they carry no place in the video
    space-time grid).
    """
    if n_shared <= 0:
        return cos, sin
    b, _, d = cos.shape
    ones = torch.ones((b, n_shared, d), dtype=cos.dtype, device=cos.device)
    zeros = torch.zeros((b, n_shared, d), dtype=sin.dtype, device=sin.device)
    return torch.cat([cos, ones], dim=1), torch.cat([sin, zeros], dim=1)


def shared_token_timestep_embedding(
    net, timestep: Tensor, n_tokens: int, batch_size: int, dtype: torch.dtype
) -> Tensor:
    """Additive timestep embedding for ``n_tokens`` injected tokens.

    Accepts a scalar, a per-sample ``(B,)`` timestep, or a per-token
    ``(B, n_tokens)`` tensor. Returns ``(B, n_tokens, D)`` in ``dtype``.
    """
    if timestep.dim() == 2:
        if tuple(timestep.shape) != (batch_size, n_tokens):
            raise ValueError(
                f"shared-token timestep has shape {tuple(timestep.shape)}; expected "
                f"(B={batch_size}, n_tokens={n_tokens}) for the per-token form."
            )
        ts_tok = timestep
    else:
        ts = timestep.flatten()
        if ts.numel() == 1:
            ts = ts.expand(batch_size)
        elif ts.numel() != batch_size:
            raise ValueError(
                f"shared-token timestep has {ts.numel()} elements; expected 1, "
                f"batch_size={batch_size}, or a (B={batch_size}, n_tokens={n_tokens}) per-token tensor."
            )
        ts_tok = ts.view(batch_size, 1).expand(batch_size, n_tokens)

    # Same scaling + embedder the video tokens go through in prepare_block_loop.
    ts_eff = ts_tok.reshape(-1).to(torch.float32) * float(net.config.timestep_scale)
    te_dtype = next(net.time_embedder.parameters()).dtype
    emb = net.time_embedder(net.time_proj(ts_eff).to(te_dtype)).to(dtype)
    return emb.view(batch_size, n_tokens, -1)


def validate_shared_tokens(tokens: Optional[Tensor], n_tok: int, name: str, batch: int, dim: int) -> Tensor:
    """Shape-check an injected token tensor and return it."""
    if tokens is None:
        raise ValueError(f"n_{name} > 0 requires {name}_tokens.")
    if tokens.shape[0] != batch or tokens.shape[1] != n_tok or tokens.shape[2] != dim:
        raise ValueError(
            f"{name}_tokens shape {tuple(tokens.shape)} does not match (B={batch}, n_{name}={n_tok}, dim={dim})."
        )
    return tokens
