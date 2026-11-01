"""Concatenate / split the joint [video, action] token sequence along dim 1 with one Triton copy pass each
(``OPENWAM_FAST_CAT=1``).

The MoT joint attention concatenates the two branches' q/k/v ``(B, S_v, C)`` + ``(B, S_a, C)`` along the sequence
and splits the attention output back (``split`` + ``.contiguous()``); in the backward these become the mirror
split / cat of the gradients. PyTorch's ``cat`` along dim 1 of bf16 tensors copies element by element (about 4x the
time of a plain copy of the same bytes at this model's shapes), and the strided slice copies are similarly slow.
These kernels copy whole contiguous rows with vectorized loads and stores, at the speed of a clone. Pure data
movement: bitwise identical results and gradients. Guards: CUDA, same dtype, 3-D, equal batch and last dims;
otherwise ``torch.cat`` / ``split`` + ``contiguous``.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False

_ENABLED = os.environ.get("OPENWAM_FAST_CAT", "0") == "1"
_BLOCK = 1024

if _HAVE_TRITON:

    @triton.jit
    def _cat2_kernel(A, Bt, O, S1, S2, C, BLOCK: tl.constexpr):
        row = tl.program_id(0)  # over B * (S1 + S2) output rows
        blk = tl.program_id(1)
        S = S1 + S2
        b = row // S
        s = row % S
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        m = cols < C
        if s < S1:
            v = tl.load(A + (b.to(tl.int64) * S1 + s) * C + cols, mask=m)
        else:
            v = tl.load(Bt + (b.to(tl.int64) * S2 + (s - S1)) * C + cols, mask=m)
        tl.store(O + row.to(tl.int64) * C + cols, v, mask=m)

    @triton.jit
    def _split2_kernel(X, A, Bt, S1, S2, C, BLOCK: tl.constexpr):
        row = tl.program_id(0)  # over B * (S1 + S2) input rows
        blk = tl.program_id(1)
        S = S1 + S2
        b = row // S
        s = row % S
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        m = cols < C
        v = tl.load(X + row.to(tl.int64) * C + cols, mask=m)
        if s < S1:
            tl.store(A + (b.to(tl.int64) * S1 + s) * C + cols, v, mask=m)
        else:
            tl.store(Bt + (b.to(tl.int64) * S2 + (s - S1)) * C + cols, v, mask=m)


def _cat2(a, b):
    a = a.contiguous()
    b = b.contiguous()
    n, s1, c = a.shape
    s2 = b.shape[1]
    out = torch.empty((n, s1 + s2, c), device=a.device, dtype=a.dtype)
    _cat2_kernel[(n * (s1 + s2), triton.cdiv(c, _BLOCK))](a, b, out, s1, s2, c, BLOCK=_BLOCK)
    return out


def _split2(x, s1):
    x = x.contiguous()
    n, s, c = x.shape
    s2 = s - s1
    a = torch.empty((n, s1, c), device=x.device, dtype=x.dtype)
    b = torch.empty((n, s2, c), device=x.device, dtype=x.dtype)
    _split2_kernel[(n * s, triton.cdiv(c, _BLOCK))](x, a, b, s1, s2, c, BLOCK=_BLOCK)
    return a, b


class _Cat2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.s1 = a.shape[1]
        return _cat2(a, b)

    @staticmethod
    def backward(ctx, g):
        return _split2(g, ctx.s1)


class _Split2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s1):
        return _split2(x, s1)

    @staticmethod
    def backward(ctx, ga, gb):
        return _cat2(ga, gb), None


def _ok(*ts) -> bool:
    t0 = ts[0]
    return (
        _ENABLED
        and _HAVE_TRITON
        and all(t.is_cuda and t.dim() == 3 and t.dtype == t0.dtype for t in ts)
        and all(t.shape[0] == t0.shape[0] and t.shape[2] == t0.shape[2] for t in ts)
    )


def cat_seq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch.cat([a, b], dim=1)`` for ``(B, S, C)`` tensors."""
    if _ok(a, b):
        return _Cat2.apply(a, b)
    return torch.cat([a, b], dim=1)


def split_seq(x: torch.Tensor, s1: int):
    """``[t.contiguous() for t in x.split([s1, S - s1], dim=1)]`` for a ``(B, S, C)`` tensor."""
    if _ok(x) and 0 < s1 < x.shape[1]:
        return _Split2.apply(x, s1)
    a, b = x.split([s1, x.shape[1] - s1], dim=1)
    return a.contiguous(), b.contiguous()
