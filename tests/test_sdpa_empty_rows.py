"""Empty attention rows retain native SDPA's finite zero-output/gradient contract."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from openwam.model.kernels import attention


def _mask(kind, *, device="cpu", dtype=torch.float64):
    mask = torch.ones(2, 1, 64, 96, dtype=torch.bool, device=device)
    mask[:, :, :, -9:] = False
    mask[0, :, 3, :] = False
    mask[1] = False  # an entire sample can have no context
    if kind == "bool":
        return mask
    bias = torch.zeros(mask.shape, dtype=dtype, device=device).masked_fill(~mask, -torch.inf)
    bias[0, :, :3, :7] = -0.25  # preserve finite attention biases too
    return bias


def _forward_backward(qkv, mask, *, safe=False, wrapper=False, upstream=None):
    q, k, v = [x.detach().clone().requires_grad_() for x in qkv]
    if wrapper:
        out = attention.sdpa(q, k, v, attn_mask=mask)
    elif safe:
        safe_mask, nonempty = attention._safe_cudnn_mask(mask)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_mask)
        out = attention._zero_masked_rows(out, nonempty)
    else:
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    # Nonzero upstream gradients even on zero outputs: the wrapper must block
    # the empty rows itself, rather than relying on the chosen loss to do so.
    gradients = torch.autograd.grad(out, (q, k, v), torch.ones_like(out) if upstream is None else upstream)
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(g).all() for g in gradients)
    return out, gradients


@pytest.mark.parametrize("kind", ["bool", "additive"])
def test_safe_mask_preserves_empty_row_outputs_and_gradients_cpu(kind):
    torch.manual_seed(7)
    qkv = [torch.randn(2, 4, s, 8, dtype=torch.float64) for s in (64, 96, 96)]
    mask = _mask(kind)
    with sdpa_kernel(SDPBackend.MATH):
        actual, grads = _forward_backward(qkv, mask, safe=True)
    reference, ref_grads = _forward_backward(qkv, mask)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(actual[0, :, 3]) == 0
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.count_nonzero(grads[0][0, :, 3]) == 0
    assert all(torch.count_nonzero(g[1]) == 0 for g in grads)


def test_expanded_context_mask_stays_compact():
    mask = torch.tensor([[True, False, True], [False, False, False]])[:, None, None, :].expand(2, 4, 64, 3)
    safe_mask, nonempty = attention._safe_cudnn_mask(mask)
    assert safe_mask.shape == (2, 1, 1, 3)
    assert nonempty.shape == (2, 1, 1, 1)
    assert nonempty.flatten().tolist() == [True, False]
    assert safe_mask[1].all()


def test_finite_additive_bias_is_not_an_empty_row():
    mask = torch.full((2, 3), torch.finfo(torch.float32).min)
    safe_mask, nonempty = attention._safe_cudnn_mask(mask)
    assert nonempty.all()
    assert torch.equal(mask, safe_mask)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("shape", [(2, 3, 7, 8), (1, 3, 7, 8), (2, 1, 7, 8), (2, 3, 1, 8)])
def test_zero_rows_preserves_output_and_backward_physical_layout(transposed, shape):
    out = torch.randn(shape)
    if transposed:
        out = out.transpose(1, 2).contiguous().transpose(1, 2)
    out.requires_grad_()
    nonempty = torch.ones(shape[0], 1, shape[2], 1, dtype=torch.bool)
    nonempty[0, :, min(2, shape[2] - 1)] = False
    upstream = torch.randn_like(out).contiguous()
    result = attention._zero_masked_rows(out, nonempty)
    gradient = torch.autograd.grad(result, out, upstream)[0]
    # Singleton-axis strides may differ without changing the physical layout.
    for axis, size in enumerate(shape):
        if size > 1:
            assert result.stride(axis) == out.stride(axis)
            assert gradient.stride(axis) == out.stride(axis)
    torch.testing.assert_close(result, out.masked_fill(~nonempty, 0.0), rtol=0, atol=0)
    torch.testing.assert_close(gradient, upstream.masked_fill(~nonempty, 0.0), rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/cuDNN")
@pytest.mark.parametrize("kind", ["bool", "additive"])
def test_cudnn_empty_rows_have_finite_zero_gradients(monkeypatch, kind):
    torch.manual_seed(7)
    monkeypatch.setattr(attention, "_PREFER_CUDNN", True)
    # Force the actual affected backend; a silent fallback cannot pass this test.
    monkeypatch.setattr(attention, "_order", lambda: [SDPBackend.CUDNN_ATTENTION])
    qkv = [torch.randn(2, 4, s, 64, device="cuda", dtype=torch.bfloat16) for s in (64, 96, 96)]
    mask = _mask(kind, device="cuda", dtype=torch.bfloat16)
    actual, grads = _forward_backward(qkv, mask, wrapper=True)
    reference, ref_grads = _forward_backward(qkv, mask)
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got, want, rtol=0.05, atol=0.02)
    assert torch.count_nonzero(actual[0, :, 3]) == 0
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.count_nonzero(grads[0][0, :, 3]) == 0
    assert all(torch.count_nonzero(g[1]) == 0 for g in grads)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/cuDNN")
def test_nonempty_cudnn_path_is_unchanged(monkeypatch):
    monkeypatch.setattr(attention, "_PREFER_CUDNN", True)
    monkeypatch.setattr(attention, "_order", lambda: [SDPBackend.CUDNN_ATTENTION])
    q, k, v = [torch.randn(2, 4, s, 64, device="cuda", dtype=torch.bfloat16) for s in (64, 96, 96)]
    mask = torch.ones(2, 1, 64, 96, device="cuda", dtype=torch.bool)
    mask[..., -7:] = False
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    actual = attention.sdpa(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/cuDNN")
@pytest.mark.parametrize("kind", ["bool", "additive"])
def test_cudnn_expanded_context_masks_preserve_broadcast_gradients(monkeypatch, kind):
    monkeypatch.setattr(attention, "_PREFER_CUDNN", True)
    monkeypatch.setattr(attention, "_order", lambda: [SDPBackend.CUDNN_ATTENTION])
    qkv = [torch.randn(2, 4, s, 64, device="cuda", dtype=torch.bfloat16) for s in (64, 96, 96)]
    base = torch.ones(2, 96, device="cuda", dtype=torch.bool)
    base[0, -7:] = False
    base[1] = False
    if kind == "additive":
        base = torch.zeros_like(base, dtype=torch.bfloat16).masked_fill(~base, -torch.inf)
    mask = base[:, None, None, :].expand(2, 4, 64, 96)
    actual, grads = _forward_backward(qkv, mask, wrapper=True)
    reference, ref_grads = _forward_backward(qkv, mask)
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)
    for got, want in zip(grads, ref_grads):
        torch.testing.assert_close(got, want, rtol=0.05, atol=0.02)
    assert torch.count_nonzero(actual[1]) == 0
    assert all(torch.count_nonzero(g[1]) == 0 for g in grads)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/cuDNN")
@pytest.mark.parametrize("kind", ["bool", "additive"])
@pytest.mark.parametrize("empty_rows", [False, True])
def test_cudnn_transposed_qkv_random_gradients_and_reused_plans(monkeypatch, kind, empty_rows):
    torch.manual_seed(1759)
    monkeypatch.setattr(attention, "_PREFER_CUDNN", True)
    monkeypatch.setattr(attention, "_order", lambda: [SDPBackend.CUDNN_ATTENTION])
    qkv = [torch.randn(2, s, 4, 64, device="cuda", dtype=torch.bfloat16).transpose(1, 2) for s in (64, 96, 96)]
    mask = _mask(kind, device="cuda", dtype=torch.bfloat16)
    if not empty_rows:
        mask = torch.ones_like(mask) if kind == "bool" else torch.zeros_like(mask)
    upstream = torch.randn_like(qkv[0])
    reference, ref_grads = _forward_backward(qkv, mask, upstream=upstream)
    # Reuse the same cuDNN plan with both common incoming gradient layouts.
    # Uniform upstream gradients would hide an incorrect stride descriptor.
    for grad in (upstream, upstream.contiguous(), upstream):
        actual, grads = _forward_backward(qkv, mask, wrapper=True, upstream=grad)
        torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)
        for got, want in zip(grads, ref_grads):
            torch.testing.assert_close(got, want, rtol=0.05, atol=0.02)
        assert actual.stride() == qkv[0].stride()
        if empty_rows:
            assert torch.count_nonzero(actual[0, :, 3]) == 0
            assert torch.count_nonzero(actual[1]) == 0
            assert torch.count_nonzero(grads[0][0, :, 3]) == 0
            assert all(torch.count_nonzero(g[1]) == 0 for g in grads)
