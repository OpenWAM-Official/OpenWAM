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


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None) -> torch.Tensor:
    """``F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)``, cuDNN first when enabled."""
    if not _PREFER_CUDNN or not q.is_cuda or torch.are_deterministic_algorithms_enabled():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    from torch.nn.attention import sdpa_kernel

    with sdpa_kernel(_order(), set_priority=True):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
