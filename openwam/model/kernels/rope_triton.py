"""Fused Wan 3-D RoPE (``wan.models.dit.rope_apply``) forward/backward in one Triton pass each
(``OPENWAM_FUSED_ROPE=1``).

``rope_apply`` computes, per call, ``x.to(float64)`` -> complex view -> ``* freqs`` (complex128)
-> real view -> ``.to(bf16)``: four full-size passes with float64 intermediates, and the
mirror chain in the backward (``ToCopyBackward`` / complex ``MulBackward``: ``grad * conj(freqs)``).

These kernels compute the same float64 products with the same rounding steps:
  real = a*c - b*d, imag = a*d + b*c   (fp64, the FMA pattern of PyTorch's complex multiply,
  selected by ``OPENWAM_ROPE_FMA_MODE``), then float64 -> float32 -> bfloat16 (PyTorch's
  double -> BFloat16 conversion goes through float), so the outputs and input gradients are
  bitwise those of ``rope_apply`` (``tests/test_fast_kernels.py``). The first call of a process
  also runs ``rope_apply`` and compares; on a mismatch the fused path stays off.

Guards: CUDA, bf16, contiguous ``(B, S, H*D)`` input, ``freqs`` complex128 ``(S, 1, D/2)``;
otherwise the caller falls back to ``rope_apply``.
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

# 0: no FMA (a*c and b*d rounded, then subtract); 1: fma(a, c, -(b*d)) / fma(a, d, b*c);
# 2: fma(-b, d, a*c) / fma(b, c, a*d)
_FMA_MODE = int(os.environ.get("OPENWAM_ROPE_FMA_MODE", "1"))

if _HAVE_TRITON:

    @triton.jit
    def _rope_kernel(
        X,  # (rows, H*D) bf16, rows = B*S
        F,  # (S, HALF, 2) float64 (cos, sin) for each pair
        O,  # (rows, H*D) bf16
        S,
        NPAIRS: tl.constexpr,  # H * HALF pairs per row
        HALF: tl.constexpr,  # D / 2
        BLOCK: tl.constexpr,
        CONJ: tl.constexpr,
        MODE: tl.constexpr,
    ):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        s = row % S
        p = blk * BLOCK + tl.arange(0, BLOCK)
        m = p < NPAIRS
        j = p % HALF
        xbase = X + row.to(tl.int64) * (2 * NPAIRS)
        a = tl.load(xbase + 2 * p, mask=m, other=0.0).to(tl.float64)
        b = tl.load(xbase + 2 * p + 1, mask=m, other=0.0).to(tl.float64)
        fbase = F + s.to(tl.int64) * (2 * HALF)
        c = tl.load(fbase + 2 * j, mask=m, other=0.0)
        d = tl.load(fbase + 2 * j + 1, mask=m, other=0.0)
        if CONJ:
            d = -d
        if MODE == 0:
            ac = a * c
            bd = b * d
            ad = a * d
            bc = b * c
            re = ac - bd
            im = ad + bc
        elif MODE == 1:
            re = tl.fma(a, c, -(b * d))
            im = tl.fma(a, d, b * c)
        else:
            re = tl.fma(-b, d, a * c)
            im = tl.fma(b, c, a * d)
        obase = O + row.to(tl.int64) * (2 * NPAIRS)
        tl.store(obase + 2 * p, re.to(tl.float32).to(tl.bfloat16), mask=m)
        tl.store(obase + 2 * p + 1, im.to(tl.float32).to(tl.bfloat16), mask=m)


def _launch(x: torch.Tensor, f_real: torch.Tensor, conj: bool, mode: int) -> torch.Tensor:
    B, S, C = x.shape
    half = f_real.shape[1]
    npairs = C // 2
    out = torch.empty_like(x)
    BLOCK = 512
    grid = (B * S, triton.cdiv(npairs, BLOCK))
    _rope_kernel[grid](
        x, f_real, out, S, NPAIRS=npairs, HALF=half, BLOCK=BLOCK, CONJ=conj, MODE=mode, enable_fp_fusion=False
    )
    return out


class _RopeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f_real, mode):
        ctx.save_for_backward(f_real)
        ctx.mode = mode
        return _launch(x, f_real, False, mode)

    @staticmethod
    def backward(ctx, g):
        (f_real,) = ctx.saved_tensors
        return _launch(g.contiguous(), f_real, True, ctx.mode), None, None


def _freqs_real(freqs: torch.Tensor):
    # (S, 1, HALF) complex128 -> (S, HALF, 2) float64 contiguous (cached per tensor)
    cache = _freqs_real.__dict__.setdefault("cache", {})
    key = (freqs.data_ptr(), tuple(freqs.shape), freqs.device)
    hit = cache.get(key)
    if hit is not None and hit[0] is freqs:
        return hit[1]
    fr = torch.view_as_real(freqs.reshape(freqs.shape[0], freqs.shape[-1])).contiguous()
    if len(cache) > 16:
        cache.clear()
    cache[key] = (freqs, fr)
    return fr


def fused_rope_ok(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> bool:
    return (
        _HAVE_TRITON
        and os.environ.get("OPENWAM_FUSED_ROPE", "0") == "1"
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 3
        and x.is_contiguous()
        and freqs.dtype == torch.complex128
        and freqs.dim() == 3
        and freqs.shape[0] == x.shape[1]
        and freqs.shape[1] == 1
        and x.shape[2] == num_heads * 2 * freqs.shape[2]
        and freqs.device == x.device
    )


_STATE = {"checked": False, "enabled": True}


def fused_rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    return _RopeFn.apply(x, _freqs_real(freqs), _FMA_MODE)


def rope_apply_auto(x: torch.Tensor, freqs: torch.Tensor, num_heads: int, reference_fn) -> torch.Tensor:
    """``reference_fn(x, freqs, num_heads)`` (``rope_apply``) through the fused kernel
    when its guards hold. The first call also runs the reference and compares bitwise; on a
    mismatch the fused path is disabled for the process (logged)."""
    if not (_STATE["enabled"] and fused_rope_ok(x, freqs, num_heads)):
        return reference_fn(x, freqs, num_heads)
    if not _STATE["checked"]:
        _STATE["checked"] = True
        with torch.no_grad():
            ref = reference_fn(x.detach(), freqs, num_heads)
            out = fused_rope_apply(x.detach(), freqs, num_heads)
            same = torch.equal(ref.view(torch.int16), out.view(torch.int16))
        if not same:
            _STATE["enabled"] = False
            import logging

            logging.getLogger(__name__).warning("fused RoPE differs from rope_apply on first call; disabled")
            return reference_fn(x, freqs, num_heads)
    return fused_rope_apply(x, freqs, num_heads)
