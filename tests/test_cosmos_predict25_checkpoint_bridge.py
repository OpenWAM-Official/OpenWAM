"""Flat checkpoint round-trip lock for ``CosmosPredict25VideoBackbone``.

The backbone holds flat named children (``dit.*`` / ``vae.*`` / ``reason1.*``):
the DiT directly, plus the inner ``nn.Module`` of the (plain-object) VAE and
Reason1 facades registered under the clean child names so their weights enter
the unified ``state_dict``. These tests pin that the flat layout saves/loads
strict, including the ``assign=True`` deploy path and the architecture's
``video_backbone.``-prefixed unified load.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone


class _ParamNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(3))


class _FakeWanVAE:
    """Mimics ``Wan2pt1VAEInterface.model`` — its ``.model`` is the inner nn.Module."""

    def __init__(self) -> None:
        self.model: nn.Module = nn.Linear(4, 4)


class _FakeVAEInterface:
    def __init__(self) -> None:
        self.model = _FakeWanVAE()


class _FakeReason1:
    def __init__(self) -> None:
        self.model = nn.Linear(3, 3)


def _build(seed: int) -> CosmosPredict25VideoBackbone:
    torch.manual_seed(seed)
    return CosmosPredict25VideoBackbone(
        net=_ParamNet(),
        vae=_FakeVAEInterface(),
        text_encoder=_FakeReason1(),
        dim=16,
        num_layers=1,
        num_heads=4,
        head_dim=4,
        context_dim=12,
    )


def test_fresh_save_has_flat_keys():
    bb = _build(0)
    keys = list(bb.state_dict().keys())
    assert keys, "state_dict unexpectedly empty"
    # The flat children are present under their clean child names...
    assert any(k.startswith("dit.") for k in keys)
    assert any(k.startswith("vae.") for k in keys)
    assert any(k.startswith("reason1.") for k in keys)
    # ...and no stale wrapper / inner-suffix prefixes leak through.
    assert not any(k.startswith("_pipe") for k in keys), f"stale _pipe keys: {keys}"
    assert not any("_vae_inner" in k or "_reason1_inner" in k for k in keys), keys


def test_flat_checkpoint_roundtrip():
    """The flat layout round-trips strict and the weights actually transfer."""
    src = _build(1)
    flat_sd = {k: v.clone() for k, v in src.state_dict().items()}

    dst = _build(2)  # different random init
    incompatible = dst.load_state_dict(flat_sd, strict=True)
    assert not incompatible.missing_keys, incompatible.missing_keys
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys

    # Weights transferred for every flat child.
    torch.testing.assert_close(dst.dit.w, src.dit.w)
    torch.testing.assert_close(dst.vae.weight, src.vae.weight)
    torch.testing.assert_close(dst.reason1.weight, src.reason1.weight)


def test_flat_load_with_assign_true_deploy_path():
    """Deploy materialises a meta-device shell and loads with ``assign=True``."""
    src = _build(3)
    flat_sd = {k: v.clone() for k, v in src.state_dict().items()}
    dst = _build(4)
    inc = dst.load_state_dict(flat_sd, strict=True, assign=True)
    assert not inc.missing_keys and not inc.unexpected_keys
    torch.testing.assert_close(dst.dit.w, src.dit.w)
    torch.testing.assert_close(dst.reason1.weight, src.reason1.weight)


class _Parent(nn.Module):
    """Mimics the architecture holding the backbone as ``video_backbone``, so the
    load runs under the production ``video_backbone.`` prefix (not empty)."""

    def __init__(self, bb: CosmosPredict25VideoBackbone) -> None:
        super().__init__()
        self.video_backbone = bb


def test_flat_load_under_video_backbone_prefix():
    """The flat layout strict-loads under the real ``video_backbone.`` load
    prefix (the architecture's unified load), not just the standalone case."""
    src = _build(5)
    prefixed = {"video_backbone." + k: v.clone() for k, v in src.state_dict().items()}

    parent = _Parent(_build(6))
    inc = parent.load_state_dict(prefixed, strict=True)
    assert not inc.missing_keys, inc.missing_keys
    assert not inc.unexpected_keys, inc.unexpected_keys
    torch.testing.assert_close(parent.video_backbone.dit.w, src.dit.w)
    torch.testing.assert_close(parent.video_backbone.reason1.weight, src.reason1.weight)
