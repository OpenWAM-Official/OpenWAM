"""Regression tests for fused-attention backend dispatch.

FA2/FA3/sage are CUDA half-precision kernels: they raise rather than degrade.
Every dispatch site must therefore check device *and* dtype before selecting
one, otherwise merely having ``flash-attn`` importable breaks fp32 and CPU
inputs -- including the CPU tensors the rest of this suite runs on.

The backends are faked here, so these tests exercise the "flash-attn is
installed" configuration without it being installed. That matters: CI runs on
CPU-only torch and would otherwise never cover this dispatch at all.
"""

from __future__ import annotations

import logging
import types

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange


class _FakeQuery:
    """Stand-in exposing only the two attributes the dispatch predicates read.

    Lets the CUDA-side branches be covered on a CPU-only box.
    """

    def __init__(self, is_cuda: bool, dtype: torch.dtype):
        self.is_cuda = is_cuda
        self.dtype = dtype


def _sdpa_reference(q, k, v, num_heads):
    qq, kk, vv = (rearrange(t, "b s (n d) -> b n s d", n=num_heads) for t in (q, k, v))
    return rearrange(F.scaled_dot_product_attention(qq, kk, vv), "b n s d -> b s (n d)", n=num_heads)


def _install_fake_flash_attn_2(monkeypatch, vdit, kernel):
    """Make ``dit.flash_attention`` believe FA2 -- and only FA2 -- is installed."""
    monkeypatch.setattr(vdit, "flash_attn", types.SimpleNamespace(flash_attn_func=kernel), raising=False)
    monkeypatch.setattr(vdit, "FLASH_ATTN_2_AVAILABLE", True)
    monkeypatch.setattr(vdit, "FLASH_ATTN_3_AVAILABLE", False)
    monkeypatch.setattr(vdit, "SAGE_ATTN_AVAILABLE", False)


# --------------------------------------------------------------- Wan video DiT


def test_wan_dit_falls_back_to_sdpa_on_cpu(monkeypatch):
    """CPU tensors must never reach the fused kernel."""
    import openwam.model.video_backbone.wan.models.dit as vdit

    def _reject(*args, **kwargs):
        raise AssertionError("fused kernel must not receive CPU tensors")

    _install_fake_flash_attn_2(monkeypatch, vdit, _reject)

    num_heads = 4
    q, k, v = (torch.randn(2, 8, num_heads * 16) for _ in range(3))

    out = vdit.flash_attention(q, k, v, num_heads=num_heads)

    assert torch.allclose(out, _sdpa_reference(q, k, v, num_heads), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_wan_dit_falls_back_to_sdpa_for_cuda_fp32(monkeypatch):
    """fp32 is rejected by the fused kernels even on CUDA."""
    import openwam.model.video_backbone.wan.models.dit as vdit

    def _reject(*args, **kwargs):
        raise AssertionError("fused kernel must not receive fp32 tensors")

    _install_fake_flash_attn_2(monkeypatch, vdit, _reject)

    num_heads = 4
    q, k, v = (torch.randn(2, 8, num_heads * 16, device="cuda", dtype=torch.float32) for _ in range(3))

    out = vdit.flash_attention(q, k, v, num_heads=num_heads)

    assert torch.allclose(out, _sdpa_reference(q, k, v, num_heads), atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_wan_dit_keeps_fused_path_for_cuda_half(monkeypatch):
    """The guard must not cost us the fast path it is protecting."""
    import openwam.model.video_backbone.wan.models.dit as vdit

    calls = []

    def _record(q, k, v, *args, **kwargs):
        calls.append(1)
        return q

    _install_fake_flash_attn_2(monkeypatch, vdit, _record)

    num_heads = 4
    q, k, v = (torch.randn(2, 8, num_heads * 16, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    out = vdit.flash_attention(q, k, v, num_heads=num_heads)

    assert calls, "cuda/bf16 input should still reach the fused kernel"
    assert torch.equal(out, q)


# ------------------------------------------------------------------ ActionDiT


def test_action_fused_kernel_usable_requires_cuda_and_half():
    from openwam.model.action_backbone.components import _fused_kernel_usable

    assert _fused_kernel_usable(_FakeQuery(True, torch.float16))
    assert _fused_kernel_usable(_FakeQuery(True, torch.bfloat16))
    # the case the device-only guard used to let through
    assert not _fused_kernel_usable(_FakeQuery(True, torch.float32))
    assert not _fused_kernel_usable(_FakeQuery(False, torch.float32))
    assert not _fused_kernel_usable(_FakeQuery(False, torch.bfloat16))


def test_action_sage_backend_falls_back_on_cpu(monkeypatch):
    """Same contract as the flash-attn backends, for the sage path."""
    import sys

    from openwam.model.action_backbone import components

    def _cuda_only_sageattn(*args, **kwargs):
        raise AssertionError("sageattn should not receive CPU tensors")

    monkeypatch.setitem(sys.modules, "sageattention", types.SimpleNamespace(sageattn=_cuda_only_sageattn))

    fn = components._try_sage_attention()
    assert fn is not None

    q, k, v = (torch.randn(1, 2, 3, 4) for _ in range(3))
    assert fn(q, k, v).shape == q.shape


# ----------------------------------------------------------- Wan shared core


@pytest.mark.parametrize("impl", ["flash_attention_3", "flash_attention_2", "sage_attention"])
def test_shared_core_downgrades_fused_backends_it_cannot_run(monkeypatch, impl):
    from openwam.model.video_backbone.wan.shared.core.attention import attention as shared

    monkeypatch.setattr(shared, "ATTENTION_IMPLEMENTATION", impl)

    assert shared.resolve_implementation(_FakeQuery(True, torch.bfloat16)) == impl
    assert shared.resolve_implementation(_FakeQuery(True, torch.float32)) == "torch"
    assert shared.resolve_implementation(_FakeQuery(False, torch.float32)) == "torch"


def test_shared_core_keeps_xformers_for_cuda_fp32(monkeypatch):
    """xformers is CUDA-only but does support fp32, so it must not be downgraded for it."""
    from openwam.model.video_backbone.wan.shared.core.attention import attention as shared

    monkeypatch.setattr(shared, "ATTENTION_IMPLEMENTATION", "xformers")

    assert shared.resolve_implementation(_FakeQuery(True, torch.float32)) == "xformers"
    assert shared.resolve_implementation(_FakeQuery(False, torch.float32)) == "torch"


def test_shared_core_attention_forward_runs_on_cpu(monkeypatch):
    """End-to-end: a CPU call must produce the SDPA result, not raise."""
    from openwam.model.video_backbone.wan.shared.core.attention import attention as shared

    monkeypatch.setattr(shared, "ATTENTION_IMPLEMENTATION", "flash_attention_2")
    monkeypatch.setattr(shared, "flash_attn_interface", None, raising=False)

    q, k, v = (torch.randn(2, 4, 8, 16) for _ in range(3))

    out = shared.attention_forward(q, k, v)

    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=0, rtol=0)


# ------------------------------------------------------- deploy diagnostics


def test_fused_backend_name_follows_availability(monkeypatch):
    import openwam.model.video_backbone.wan.models.dit as vdit

    for flag in ("FLASH_ATTN_3_AVAILABLE", "FLASH_ATTN_2_AVAILABLE", "SAGE_ATTN_AVAILABLE"):
        monkeypatch.setattr(vdit, flag, False)
    assert vdit.fused_backend_name() == "torch_sdpa"

    monkeypatch.setattr(vdit, "SAGE_ATTN_AVAILABLE", True)
    assert vdit.fused_backend_name() == "sage_attention"

    monkeypatch.setattr(vdit, "FLASH_ATTN_2_AVAILABLE", True)
    assert vdit.fused_backend_name() == "flash_attention_2"

    monkeypatch.setattr(vdit, "FLASH_ATTN_3_AVAILABLE", True)
    assert vdit.fused_backend_name() == "flash_attention_3"


def test_diagnostics_do_not_claim_a_backend_unconditionally(monkeypatch, caplog):
    """The report must not read as 'flash_attention_2' on a run that uses SDPA."""
    import openwam.model.video_backbone.wan.models.dit as vdit
    from openwam.deploy.server import _log_attention_backends

    monkeypatch.setattr(vdit, "FLASH_ATTN_2_AVAILABLE", True)
    monkeypatch.setattr(vdit, "FLASH_ATTN_3_AVAILABLE", False)
    monkeypatch.setattr(vdit, "SAGE_ATTN_AVAILABLE", False)

    with caplog.at_level(logging.INFO):
        _log_attention_backends(logging.getLogger(__name__))

    video_line = next(line for line in caplog.text.splitlines() if "Video DiT" in line)
    assert "flash_attention_2" in video_line
    assert "CUDA fp16/bf16 only" in video_line
