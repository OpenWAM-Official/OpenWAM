"""Checkpoint compatibility lock for the `_pipe` → flat refactor.

An earlier layout nested everything under a `_pipe.` wrapper child
(`_pipe.net.* / _pipe._vae_inner.* / _pipe._reason1_inner.*`). The backbone now
holds flat children (`dit.* / _vae_inner.* / _reason1_inner.*`) and a
`_register_load_state_dict_pre_hook` remaps legacy keys so old checkpoints still
strict-load. These tests pin both halves of that contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from openwam.model.video_backbone.cosmos25_backbone import Cosmos25VideoBackbone


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


def _build(seed: int) -> Cosmos25VideoBackbone:
    torch.manual_seed(seed)
    return Cosmos25VideoBackbone(
        net=_ParamNet(),
        vae=_FakeVAEInterface(),
        text_encoder=_FakeReason1(),
        dim=16,
        num_layers=1,
        num_heads=4,
        head_dim=4,
        context_dim=12,
    )


def _to_legacy_pipe_layout(flat_sd: dict) -> dict:
    """flat (`dit.* / _vae_inner.* / _reason1_inner.*`) → legacy (`_pipe.*`)."""
    legacy = {}
    for k, v in flat_sd.items():
        if k.startswith("dit."):
            legacy["_pipe.net." + k[len("dit.") :]] = v
        elif k.startswith("_vae_inner.") or k.startswith("_reason1_inner."):
            legacy["_pipe." + k] = v
        else:
            legacy[k] = v
    return legacy


def test_fresh_save_has_no_pipe_keys():
    bb = _build(0)
    keys = list(bb.state_dict().keys())
    assert keys, "state_dict unexpectedly empty"
    assert not any(k.startswith("_pipe") for k in keys), f"stale _pipe keys: {keys}"
    # The flat children are present.
    assert any(k.startswith("dit.") for k in keys)
    assert any(k.startswith("_vae_inner.") for k in keys)
    assert any(k.startswith("_reason1_inner.") for k in keys)


def test_legacy_pipe_checkpoint_strict_loads_into_flat_backbone():
    """A checkpoint saved under the OLD `_pipe.*` prefixes loads strict via the
    pre-hook, and the weights actually transfer."""
    src = _build(1)
    legacy_sd = _to_legacy_pipe_layout({k: v.clone() for k, v in src.state_dict().items()})
    assert all(k.startswith("_pipe.") for k in legacy_sd if not k.endswith("_extra_state"))

    dst = _build(2)  # different random init
    incompatible = dst.load_state_dict(legacy_sd, strict=True)
    assert not incompatible.missing_keys, incompatible.missing_keys
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys

    # Weights transferred: the flat children now match the source.
    torch.testing.assert_close(dst.dit.w, src.dit.w)
    torch.testing.assert_close(dst._vae_inner.weight, src._vae_inner.weight)
    torch.testing.assert_close(dst._reason1_inner.weight, src._reason1_inner.weight)


def test_flat_checkpoint_still_loads():
    """The new flat layout round-trips normally (the pre-hook is a no-op on it)."""
    src = _build(3)
    flat_sd = {k: v.clone() for k, v in src.state_dict().items()}
    dst = _build(4)
    incompatible = dst.load_state_dict(flat_sd, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    torch.testing.assert_close(dst.dit.w, src.dit.w)


class _Parent(nn.Module):
    """Mimics the architecture holding the backbone as `video_backbone`, so the
    pre-hook fires with the production `video_backbone.` prefix (not empty)."""

    def __init__(self, bb: Cosmos25VideoBackbone) -> None:
        super().__init__()
        self.video_backbone = bb


def test_legacy_load_under_video_backbone_prefix():
    """The remap must work under the real `video_backbone.` load prefix (the
    architecture's unified load), not just the standalone empty-prefix case."""
    src = _build(5)
    legacy = _to_legacy_pipe_layout({k: v.clone() for k, v in src.state_dict().items()})
    prefixed = {"video_backbone." + k: v for k, v in legacy.items()}
    assert all(k.startswith("video_backbone._pipe.") for k in prefixed if not k.endswith("_extra_state"))

    parent = _Parent(_build(6))
    inc = parent.load_state_dict(prefixed, strict=True)
    assert not inc.missing_keys, inc.missing_keys
    assert not inc.unexpected_keys, inc.unexpected_keys
    torch.testing.assert_close(parent.video_backbone.dit.w, src.dit.w)
    torch.testing.assert_close(parent.video_backbone._reason1_inner.weight, src._reason1_inner.weight)


def test_legacy_load_with_assign_true_deploy_path():
    """Deploy materialises a meta-device shell and loads with assign=True; the
    legacy remap must survive that path too."""
    src = _build(7)
    legacy = _to_legacy_pipe_layout({k: v.clone() for k, v in src.state_dict().items()})
    dst = _build(8)
    inc = dst.load_state_dict(legacy, strict=True, assign=True)
    assert not inc.missing_keys and not inc.unexpected_keys
    torch.testing.assert_close(dst.dit.w, src.dit.w)
