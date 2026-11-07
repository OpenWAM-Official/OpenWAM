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

When cuDNN is preferred, calls that autograd will differentiate pad K/V to a
multiple of the cuDNN key tile (see ``_pad_kv_for_cudnn``): the backward kernels
shipped with PyTorch 2.7 can otherwise return NaN query gradients when the key
length is not a multiple of 128.
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


# The cuDNN SDPA backward bundled with PyTorch 2.7 (cudnn-frontend 1.10, cuDNN 9.7) computes the last key tile
# without a padding mask when s_kv is not a multiple of 128, so the out-of-range key columns score 0 there. The
# forward pass is unaffected, but for a query row whose visible logits are all very negative (logsumexp below about
# -88.7, e.g. keys sharing a large component) the backward evaluates exp(0 - lse) = inf and inf * 0 = NaN in that
# row of dQ, which then poisons every parameter upstream of Q (NVIDIA/cudnn-frontend#160, pytorch/pytorch#166211;
# cudnn-frontend >= 1.15 applies a padding mask itself). Padding K/V with zero keys that the mask hides removes the
# out-of-range columns; the hidden keys get exactly zero attention, so outputs and K/V/Q gradients are unchanged.
# Only calls that will be differentiated are padded: the forward kernel is not affected.
_CUDNN_KV_TILE = 128


def _pad_seq(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Append ``pad`` zero rows to the sequence axis (-2) of ``x``, keeping its physical axis order."""
    # q/k/v are often BSHD buffers viewed as BHSD; keep that layout (cf. ``_zero_masked_rows``). Broadcast
    # (stride-0) axes go outermost so they do not become the innermost axis of the new buffer.
    order = sorted(range(x.ndim), key=lambda axis: (x.stride(axis) == 0, x.stride(axis)), reverse=True)
    inverse = [order.index(axis) for axis in range(x.ndim)]
    physical = x.permute(order)
    axis = order.index(x.ndim - 2)
    shape = list(physical.shape)
    shape[axis] += pad
    # Zero-filled buffer + copy_ (vectorized for bf16, unlike torch.cat in PyTorch 2.7); autograd routes the
    # gradient of the copied slice back to ``x``.
    padded = physical.new_zeros(shape)
    padded.narrow(axis, 0, physical.shape[axis]).copy_(physical)
    return padded.permute(inverse)


def _pad_kv_for_cudnn(k: torch.Tensor, v: torch.Tensor, attn_mask):
    """Pad K/V to a multiple of the cuDNN key tile and hide the padded keys in the (bool or additive) mask."""
    s_kv = k.shape[-2]
    pad = -s_kv % _CUDNN_KV_TILE
    if pad == 0:
        return k, v, attn_mask
    k, v = _pad_seq(k, pad), _pad_seq(v, pad)
    if attn_mask is None:  # key-padding mask that broadcasts against any query shape of the same rank
        attn_mask = torch.ones(*([1] * (k.ndim - 1)), s_kv + pad, dtype=torch.bool, device=k.device)
        attn_mask[..., s_kv:] = False
        return k, v, attn_mask
    if attn_mask.shape[-1] != s_kv:  # broadcast over keys: materialize the key axis before extending it
        attn_mask = attn_mask.expand(*attn_mask.shape[:-1], s_kv)
    hidden = attn_mask.new_full((*attn_mask.shape[:-1], pad), False if attn_mask.dtype == torch.bool else -torch.inf)
    return k, v, torch.cat([attn_mask, hidden], dim=-1)


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None) -> torch.Tensor:
    """``F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)``, cuDNN first when enabled."""
    if not _PREFER_CUDNN or not q.is_cuda or torch.are_deterministic_algorithms_enabled():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    from torch.nn.attention import sdpa_kernel

    nonempty = None
    if attn_mask is not None:
        attn_mask, nonempty = _safe_cudnn_mask(attn_mask)
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        k, v, attn_mask = _pad_kv_for_cudnn(k, v, attn_mask)  # backward-only issue: inference stays unpadded
    with sdpa_kernel(_order(), set_priority=True):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out if nonempty is None else _zero_masked_rows(out, nonempty)
