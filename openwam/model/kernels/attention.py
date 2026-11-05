"""Masked scaled-dot-product attention with the cuDNN kernel preferred (``OPENWAM_SDPA_CUDNN=1``).

PyTorch's default SDPA selection sends every call with an ``attn_mask`` to the
memory-efficient (CUTLASS) kernel. On recent GPUs cuDNN's fused attention is
much faster at this model's shapes (about 2.5x, forward + backward). ``sdpa``
asks for cuDNN first, then the default order. Same inputs and bf16 precision
(fp32 softmax and accumulation in both kernels); a different kernel, so the
rounding can differ.

cuDNN attention has no deterministic implementation (PyTorch finds no kernel
under ``torch.use_deterministic_algorithms(True)``), so in deterministic mode
``sdpa`` is the default call. Off by default: PyTorch's backend selection.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_PREFER_CUDNN = os.environ.get("OPENWAM_SDPA_CUDNN", "0") == "1"
_ORDER = None


def _order():
    global _ORDER
    if _ORDER is None:
        from torch.nn.attention import SDPBackend

        _ORDER = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    return _ORDER


def _safe_cudnn_mask(mask: torch.Tensor):
    """Keep empty attention rows away from cuDNN, without a host synchronization.

    Native SDPA defines fully masked rows as zero. cuDNN can instead return
    nonzero values and NaN gradients. Give those rows an attendable mask for
    the kernel, then zero their outputs (and thus their backward contribution).
    Do not mask NaNs after running the original, unsupported attention.
    """
    # Context masks are often expanded across queries/heads. Keep broadcast
    # dimensions compact rather than materializing the same row many times.
    mask = mask[tuple(slice(0, 1) if stride == 0 else slice(None) for stride in mask.stride())]
    if mask.dtype == torch.bool:
        nonempty = mask.any(dim=-1, keepdim=True)
        safe_mask = mask | ~nonempty
    else:
        # Only -inf masks a key completely; finite additive biases, however
        # negative, must retain their ordinary softmax semantics.
        nonempty = ~torch.isneginf(mask).all(dim=-1, keepdim=True)
        safe_mask = torch.where(nonempty, mask, 0.0)
    return safe_mask, nonempty


def _zero_masked_rows(out: torch.Tensor, nonempty: torch.Tensor) -> torch.Tensor:
    # cuDNN returns a dense output that can have BSHD physical layout. A plain
    # masked_fill would make its backward gradient BHSD-contiguous instead.
    # PyTorch 2.7's cuDNN backward plan cache does not distinguish dO strides;
    # keep the original physical layout in both directions. Permutations are
    # views, and masking in physical axis order supplies that gradient layout.
    order = sorted(range(out.ndim), key=out.stride().__getitem__, reverse=True)
    inverse = [order.index(axis) for axis in range(out.ndim)]
    empty = (~nonempty).expand_as(out).permute(order)
    return out.permute(order).masked_fill(empty, 0.0).permute(inverse)


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None) -> torch.Tensor:
    """``F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)``, cuDNN first when enabled."""
    if not _PREFER_CUDNN or not q.is_cuda or torch.are_deterministic_algorithms_enabled():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    from torch.nn.attention import sdpa_kernel

    nonempty = None
    if attn_mask is not None:
        attn_mask, nonempty = _safe_cudnn_mask(attn_mask)
    with sdpa_kernel(_order(), set_priority=True):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out if nonempty is None else _zero_masked_rows(out, nonempty)
