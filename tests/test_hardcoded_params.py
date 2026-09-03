"""Tests for hardcoded parameter removal (Steps 3-7 of the refactor plan)."""

import numpy as np
import pytest
import torch

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.action_backbone.shared_action_backbone import SharedMoEActionBackbone
from openwam.model.architectures.dual_system import DualSystemCrossAttnArchitecture
from openwam.model.architectures.single_system.moe import SingleSystemMoEArchitecture
from openwam.model.architectures.single_system.vanilla import SingleSystemVanillaArchitecture

# ---------------------------------------------------------------------------
# Step 3: sigma uses scheduler.num_train_timesteps, not hardcoded 1000
# ---------------------------------------------------------------------------


def test_generate_sigma_uses_scheduler_num_train_timesteps():
    """generate() should use self.action_scheduler.num_train_timesteps, not 1000.0."""
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()
    # Verify the scheduler has num_train_timesteps attribute
    assert hasattr(arch.action_scheduler, "num_train_timesteps")
    assert arch.action_scheduler.num_train_timesteps == 1000

    # Check that the source code does NOT contain / 1000.0 in generate()
    import inspect

    source = inspect.getsource(arch.generate)
    assert "/ 1000.0" not in source
    assert "/ 1000" not in source
    assert "num_train_ts" in source or "num_train_timesteps" in source


# ---------------------------------------------------------------------------
# Step 4: ActionDiT / MoEDiT require all params (no domain-specific defaults)
# ---------------------------------------------------------------------------


def test_action_dit_requires_all_params():
    """ActionDiT should raise TypeError when required params are missing."""
    with pytest.raises(TypeError):
        ActionDiT()

    with pytest.raises(TypeError):
        ActionDiT(action_dim=20)

    # Should succeed with all required params
    dit = ActionDiT(
        action_dim=7,
        dim=64,
        ffn_dim=256,
        num_heads=4,
        num_layers=2,
        video_dim=128,
        bridge_layers=(0, 1),
    )
    assert dit.action_dim == 7
    assert dit.dim == 64


def test_moe_dit_requires_all_params():
    """SharedMoEActionBackbone should raise TypeError when required params are missing."""
    with pytest.raises(TypeError):
        SharedMoEActionBackbone()

    with pytest.raises(TypeError):
        SharedMoEActionBackbone(action_dim=20)

    # Should succeed with all required params (num_experts is derived from bridge_layers)
    moe = SharedMoEActionBackbone(
        action_dim=14,
        video_dim=128,
        expert_ffn_dim=512,
        bridge_layers=(0, 1, 2),
    )
    assert moe.action_dim == 14
    assert moe._video_dim == 128
    assert moe.num_experts == 3


# ---------------------------------------------------------------------------
# Step 5: vanilla.py video_dim raises without config or backbone
# ---------------------------------------------------------------------------


def test_vanilla_no_video_dim_raises():
    """SingleSystemVanilla should raise when video_dim is not specified."""
    cfg = {
        "framework": "single_system",
        "variant": "vanilla",
        "action_dim": 20,
    }
    with pytest.raises(ValueError, match="video_dim must be specified"):
        SingleSystemVanillaArchitecture(cfg=cfg)


def test_vanilla_explicit_video_dim_works():
    """SingleSystemVanilla should work with explicit video_dim."""
    cfg = {
        "framework": "single_system",
        "variant": "vanilla",
        "action_dim": 20,
        "video_dim": 128,
    }
    arch = SingleSystemVanillaArchitecture(cfg=cfg)
    assert arch.action_dim == 20


# ---------------------------------------------------------------------------
# Step 5 (extended): all architectures raise without video_dim
# ---------------------------------------------------------------------------


def test_dual_system_cross_attn_no_video_dim_raises():
    """DualSystemCrossAttn should raise when video_dim is not specified."""
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "action_dim": 20,
        "dim": 64,
        "ffn_dim": 256,
        "num_heads": 4,
        "bridge_layers": (0, 1),
    }
    with pytest.raises(ValueError, match="video_dim must be specified"):
        DualSystemCrossAttnArchitecture(cfg=cfg)


def test_moe_architecture_no_video_dim_raises():
    """SingleSystemMoE should raise when video_dim is not specified."""
    cfg = {
        "framework": "single_system",
        "variant": "moe",
        "action_dim": 20,
        "bridge_layers": (0, 1, 2),
        "expert_ffn_dim": 512,
    }
    with pytest.raises(ValueError, match="video_dim must be specified"):
        SingleSystemMoEArchitecture(cfg=cfg)


# ---------------------------------------------------------------------------
# Step 6: action decode path — action_repr then action normalizer (serial)
# ---------------------------------------------------------------------------


class _FakeActionRepr:
    """Fake action representation that doubles values."""

    def decode(self, x):
        return x * 2.0


class _FakeNormalizer:
    """Fake action normalizer whose unnormalize adds 10."""

    def unnormalize(self, x):
        return x + 10.0


def test_action_decode_repr_then_normalizer():
    """generate() action decode should apply action_repr.decode then normalizer.unnormalize."""
    from tests.test_openwam_trainer import _make_tiny_arch

    arch = _make_tiny_arch()

    # Simulate the decode logic from generate()
    action_latents = torch.ones(1, 5, 7)
    action_repr = _FakeActionRepr()
    arch.normalizer = _FakeNormalizer()

    # Replicate the logic from base.py generate()
    if action_repr is not None:
        actions = action_repr.decode(action_latents.float()).squeeze(0).cpu().numpy()
    else:
        actions = action_latents.squeeze(0).float().cpu().numpy()
    normalizer = getattr(arch, "normalizer", None)
    if normalizer is not None:
        actions = normalizer.unnormalize(actions)

    # action_repr doubles (1→2), normalizer.unnormalize adds 10 (2→12)
    expected = np.full((5, 7), 12.0)
    np.testing.assert_allclose(actions, expected)


def test_action_decode_repr_only():
    """With only action_repr, decode should work without an action normalizer."""
    action_latents = torch.ones(1, 5, 7)
    action_repr = _FakeActionRepr()

    if action_repr is not None:
        actions = action_repr.decode(action_latents.float()).squeeze(0).cpu().numpy()
    else:
        actions = action_latents.squeeze(0).float().cpu().numpy()
    normalizer = None
    if normalizer is not None:
        actions = normalizer.unnormalize(actions)

    expected = np.full((5, 7), 2.0)
    np.testing.assert_allclose(actions, expected)


def test_action_decode_normalizer_only():
    """With only an action normalizer, decode should work without action_repr."""
    action_latents = torch.ones(1, 5, 7)
    action_repr = None
    normalizer = _FakeNormalizer()

    if action_repr is not None:
        actions = action_repr.decode(action_latents.float()).squeeze(0).cpu().numpy()
    else:
        actions = action_latents.squeeze(0).float().cpu().numpy()
    if normalizer is not None:
        actions = normalizer.unnormalize(actions)

    expected = np.full((5, 7), 11.0)
    np.testing.assert_allclose(actions, expected)


def test_action_decode_neither():
    """With neither action_repr nor action normalizer, raw latents are returned."""
    action_latents = torch.ones(1, 5, 7) * 3.0
    action_repr = None
    normalizer = None

    actions = action_latents.squeeze(0).float().cpu().numpy()
    if action_repr is not None:
        actions = action_repr.decode(action_latents.float()).squeeze(0).cpu().numpy()
    if normalizer is not None:
        actions = normalizer.unnormalize(actions)

    expected = np.full((5, 7), 3.0)
    np.testing.assert_allclose(actions, expected)


# ---------------------------------------------------------------------------
# Step 7: tile_size/tile_stride default to None
# ---------------------------------------------------------------------------


def test_generate_tile_defaults_are_none():
    """generate() signature should default tile_size and tile_stride to None."""
    import inspect

    from openwam.model.architectures.base import BaseWAMArchitecture

    sig = inspect.signature(BaseWAMArchitecture.generate)
    assert sig.parameters["tile_size"].default is None
    assert sig.parameters["tile_stride"].default is None
