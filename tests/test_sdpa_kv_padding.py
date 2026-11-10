"""K/V padding that keeps the cuDNN SDPA backward finite when s_kv is not a multiple of the key tile."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from openwam.model.kernels import attention


def _mask(kind, sq, skv, *, device="cpu", dtype=torch.float64):
    if kind is None:
        return None
    keep = torch.ones(2, 1, sq, skv, dtype=torch.bool, device=device)
    keep[:, :, : sq // 3, skv // 2 :] = False
    if kind == "bool":
        return keep
    bias = torch.zeros(keep.shape, dtype=dtype, device=device).masked_fill(~keep, -torch.inf)
    bias[0, :, :3, :5] = -0.25  # finite biases must survive the padding unchanged
    return bias


def _grads(fn, qkv, mask, upstream):
    q, k, v = [x.detach().clone().requires_grad_() for x in qkv]
    out = fn(q, k, v, mask)
    return out, torch.autograd.grad(out, (q, k, v), upstream)


def _math(q, k, v, mask):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@pytest.mark.parametrize("transposed", [False, True])
def test_pad_seq_appends_zero_keys_and_keeps_the_physical_layout(transposed):
    x = torch.randn(2, 3, 5, 4)
    if transposed:  # BSHD buffer viewed as BHSD
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    padded = attention._pad_seq(x, 3)
    assert padded.shape == (2, 3, 8, 4)
    torch.testing.assert_close(padded[:, :, :5], x, rtol=0, atol=0)
    assert torch.count_nonzero(padded[:, :, 5:]) == 0
    order = sorted(range(4), key=x.stride().__getitem__, reverse=True)
    assert sorted(range(4), key=padded.stride().__getitem__, reverse=True) == order


@pytest.mark.parametrize("skv", [128, 256])
def test_tile_multiples_are_left_untouched(skv):
    k = torch.randn(1, 2, skv, 8)
    v = torch.randn(1, 2, skv, 8)
    mask = torch.ones(1, 1, 3, skv, dtype=torch.bool)
    pk, pv, pmask = attention._pad_kv_for_cudnn(k, v, mask)
    assert pk is k and pv is v and pmask is mask


@pytest.mark.parametrize("kind", ["bool", "additive", None])
@pytest.mark.parametrize("skv", [77, 392, 64])
def test_padding_preserves_outputs_and_gradients_cpu(kind, skv):
    torch.manual_seed(3)
    sq = 50
    qkv = [torch.randn(2, 4, s, 16, dtype=torch.float64) for s in (sq, skv, skv)]
    mask = _mask(kind, sq, skv)
    upstream = torch.randn(2, 4, sq, 16, dtype=torch.float64)

    def padded(q, k, v, m):
        k, v, m = attention._pad_kv_for_cudnn(k, v, m)
        assert k.shape[-2] % attention._CUDNN_KV_TILE == 0
        return _math(q, k, v, m)

    ref, ref_grads = _grads(_math, qkv, mask, upstream)
    out, grads = _grads(padded, qkv, mask, upstream)
    torch.testing.assert_close(out, ref, rtol=1e-12, atol=1e-12)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_three_dimensional_inputs_without_mask_keep_their_rank():
    torch.manual_seed(5)
    q, k, v = (torch.randn(3, s, 8, dtype=torch.float64) for s in (6, 77, 77))
    pk, pv, pmask = attention._pad_kv_for_cudnn(k, v, None)
    assert pmask.shape == (1, 1, 128)  # same rank as the (B, L, S) logits of a 3-D call
    assert pmask[..., :77].all() and not pmask[..., 77:].any()
    out = _math(q, pk, pv, pmask)
    assert out.shape == (3, 6, 8)
    torch.testing.assert_close(out, _math(q, k, v, None), rtol=1e-12, atol=1e-12)


def test_key_broadcast_mask_is_materialized_before_padding():
    k, v = torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4)
    mask = torch.tensor([True, False])[:, None, None, None].expand(2, 1, 3, 1)
    _, _, pmask = attention._pad_kv_for_cudnn(k, v, mask)
    assert pmask.shape == (2, 1, 3, 128)
    assert pmask[0, ..., :5].all() and not pmask[0, ..., 5:].any()
    assert not pmask[1].any()


def _low_lse_inputs(sq, skv, *, device="cuda", dtype=torch.bfloat16, bshd=False, head_dim=128, row=5):
    """Keys sharing a large component and one query row anti-aligned with it: that row's logits are all about
    -136, so its logsumexp is far below -88.7 - the condition that turns the out-of-range key columns of the last
    cuDNN tile into exp(0 - lse) = inf in the backward pass."""
    torch.manual_seed(0)
    common = torch.ones(head_dim, device=device, dtype=dtype)
    q = torch.randn(2, 4, sq, head_dim, device=device, dtype=dtype)
    q[:, :, row] = -(136.0 / head_dim**0.5) * common
    k = torch.randn(2, 4, skv, head_dim, device=device, dtype=dtype) * 0.5 + common
    v = torch.randn(2, 4, skv, head_dim, device=device, dtype=dtype)
    if bshd:  # the callers pass BSHD buffers viewed as BHSD
        q, k, v = (x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q, k, v))
    return q, k, v


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/cuDNN")
@pytest.mark.parametrize("bshd", [False, True])
@pytest.mark.parametrize("kind", ["bool", "additive", None])
@pytest.mark.parametrize("sq,skv", [(392, 392), (360, 64), (100, 77), (256, 392), (128, 64)])
def test_cudnn_low_logsumexp_rows_have_finite_gradients(monkeypatch, kind, sq, skv, bshd):
    monkeypatch.setattr(attention, "_PREFER_CUDNN", True)
    # Force the affected backend; a silent fallback to another kernel cannot pass this test.
    monkeypatch.setattr(attention, "_order", lambda: [SDPBackend.CUDNN_ATTENTION])
    qkv = _low_lse_inputs(sq, skv, bshd=bshd)
    mask = _mask(kind, sq, skv, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn(2, 4, sq, 128, device="cuda", dtype=torch.bfloat16)

    def wrapper(q, k, v, m):
        return attention.sdpa(q, k, v, attn_mask=m)

    def reference(q, k, v, m):
        m32 = m if m is None or m.dtype == torch.bool else m.float()
        return _math(q.float(), k.float(), v.float(), m32).to(q.dtype)

    out, grads = _grads(wrapper, qkv, mask, upstream)
    ref, ref_grads = _grads(reference, qkv, mask, upstream)
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(g).all() for g in grads)
    torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got.float(), want.float(), rtol=0.05, atol=0.05 * want.float().abs().max().item())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/cuDNN")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("sq,skv", [(392, 392), (100, 77)])
def test_cudnn_low_logsumexp_row_in_last_query_tile(monkeypatch, head_dim, sq, skv):
    """Same regression with the low row in the last, partial query tile and 64-wide heads (Cosmos bridges)."""
    monkeypatch.setattr(attention, "_PREFER_CUDNN", True)
    monkeypatch.setattr(attention, "_order", lambda: [SDPBackend.CUDNN_ATTENTION])
    qkv = _low_lse_inputs(sq, skv, head_dim=head_dim, row=sq - 1)
    mask = _mask("bool", sq, skv, device="cuda")
    upstream = torch.randn(2, 4, sq, head_dim, device="cuda", dtype=torch.bfloat16)

    def reference(q, k, v, m):
        return _math(q.float(), k.float(), v.float(), m).to(q.dtype)

    out, grads = _grads(lambda q, k, v, m: attention.sdpa(q, k, v, attn_mask=m), qkv, mask, upstream)
    ref, ref_grads = _grads(reference, qkv, mask, upstream)
    assert all(torch.isfinite(g).all() for g in grads)
    torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got.float(), want.float(), rtol=0.05, atol=0.05 * want.float().abs().max().item())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/cuDNN")
@pytest.mark.parametrize(
    "grad_enabled,requires_grad,padded", [(False, True, False), (True, False, False), (True, True, True)]
)
def test_only_differentiated_calls_are_padded(monkeypatch, grad_enabled, requires_grad, padded):
    """The issue is backward-only: calls autograd will not differentiate keep the original cuDNN forward bit for bit."""
    monkeypatch.setattr(attention, "_PREFER_CUDNN", True)
    monkeypatch.setattr(attention, "_order", lambda: [SDPBackend.CUDNN_ATTENTION])
    calls = []
    real = attention._pad_kv_for_cudnn

    def spy(*args):
        calls.append(1)
        return real(*args)

    monkeypatch.setattr(attention, "_pad_kv_for_cudnn", spy)
    q, k, v = (x.requires_grad_(requires_grad) for x in _low_lse_inputs(392, 392))
    mask = _mask("bool", 392, 392, device="cuda")
    with torch.set_grad_enabled(grad_enabled):
        out = attention.sdpa(q, k, v, attn_mask=mask)
    assert bool(calls) == padded
    with torch.no_grad(), sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if padded:
        torch.testing.assert_close(out, reference, rtol=0.02, atol=0.02)
    else:
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
