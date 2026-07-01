"""CPU smoke for Phase 4 VAE freeze + dtype/device propagation in
``CosmosPredict25VideoBackbone``.

``Wan2pt1VAEInterface`` is not an ``nn.Module``, so the adapter has to
explicitly walk the inner nn.Module + mean/std tensor attributes on every
``set_dtype_device`` call. These tests pin both code paths against a fake
``Wan2pt1VAEInterface``-shaped object.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from openwam.model.video_backbone.cosmos_predict25 import CosmosFlowSchedulerAdapter
from openwam.model.video_backbone.cosmos_predict25._vae_utils import (
    _move_cosmos_vae,
    _vae_inner_module,
)
from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone


class _FakeWanVAE:
    """Mimics ``Wan2pt1VAEInterface.model`` (the inner ``WanVAE`` wrapper)."""

    def __init__(self) -> None:
        # `model.model` is the actual nn.Module we need to walk/move.
        self.model: nn.Module = nn.Linear(4, 4)
        # The six mean/std tensors that live on `WanVAE`.
        self.mean = torch.zeros(16)
        self.std = torch.ones(16)
        self.img_mean = torch.zeros(1, 16, 1, 1, 1)
        self.img_std = torch.ones(1, 16, 1, 1, 1)
        self.video_mean = torch.zeros(1, 16, 9, 1, 1)
        self.video_std = torch.ones(1, 16, 9, 1, 1)
        # `WanVAE.device` / `.dtype` attributes.
        self.device = torch.device("cpu")
        self.dtype = torch.float32


class _FakeWan2pt1Interface:
    """Mimics ``Wan2pt1VAEInterface``."""

    def __init__(self) -> None:
        self.model = _FakeWanVAE()


class _ParamNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))


def _build_backbone_with_fake_vae(*, freeze: bool) -> CosmosPredict25VideoBackbone:
    pipe = CosmosPredict25VideoBackbone(
        net=_ParamNet(),
        vae=_FakeWan2pt1Interface(),
        text_encoder=None,
        dim=2048,
        num_layers=28,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        shift_video=5.0,
    )
    return CosmosPredict25VideoBackbone(
        net=pipe.dit,
        vae=getattr(pipe, "_vae_iface", None),
        text_encoder=getattr(pipe, "text_encoder", None),
        shift_video=pipe._shift_video,
        dim=2048,
        num_layers=28,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        scheduler=CosmosFlowSchedulerAdapter(shift_video=5.0),
        freeze=freeze,
    )


def test_vae_inner_module_helper_returns_the_real_module():
    iface = _FakeWan2pt1Interface()
    inner = _vae_inner_module(iface)
    assert inner is iface.model.model
    assert _vae_inner_module(None) is None
    assert _vae_inner_module(object()) is None


def test_freeze_walks_fake_vae_inner_module_params():
    bb = _build_backbone_with_fake_vae(freeze=True)
    inner = bb.vae  # registered inner nn.Module (facade lives on bb._vae_iface)
    assert inner is not None
    for p in inner.parameters():
        assert p.requires_grad is False, f"freeze missed VAE param {p.shape}"
    # And the net params are frozen too (existing contract).
    for p in bb.dit.parameters():
        assert p.requires_grad is False


def test_unfrozen_backbone_leaves_fake_vae_params_trainable():
    bb = _build_backbone_with_fake_vae(freeze=False)
    inner = bb.vae  # registered inner nn.Module (facade lives on bb._vae_iface)
    assert inner is not None
    # default Linear params start with requires_grad=True; freeze=False MUST NOT touch them.
    for p in inner.parameters():
        assert p.requires_grad is True


def test_set_dtype_device_moves_inner_module_and_mean_std_tensors():
    bb = _build_backbone_with_fake_vae(freeze=True)
    bb.set_dtype_device(torch.bfloat16, torch.device("cpu"))

    iface = bb._vae_iface
    inner = bb.vae
    assert inner is not None
    for p in inner.parameters():
        assert p.dtype == torch.bfloat16
        assert p.device == torch.device("cpu")
    for attr in ("mean", "std", "img_mean", "img_std", "video_mean", "video_std"):
        t = getattr(iface.model, attr)
        assert t.dtype == torch.bfloat16, f"{attr} dtype not converted"
        assert t.device == torch.device("cpu"), f"{attr} device not moved"
    # Cached attributes also kept in sync.
    assert iface.model.device == torch.device("cpu")
    assert iface.model.dtype == torch.bfloat16


def test_move_cosmos_vae_is_noop_on_none():
    """Helper must tolerate ``vae=None`` without exception."""
    _move_cosmos_vae(None, dtype=torch.bfloat16, device=torch.device("cpu"))
