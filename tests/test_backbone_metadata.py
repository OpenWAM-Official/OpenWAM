"""Tests that architecture __init__ derives video_dim and num_dit_layers from config/backbone."""

from openwam.model.architectures.dual_system import DualSystemCrossAttnArchitecture


def test_architecture_uses_explicit_video_dim_without_backbone():
    """Without video_backbone, architecture constructs with explicit video_dim."""
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "action_dim": 14,
        "dim": 64,
        "ffn_dim": 256,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": (0, 1),
    }
    arch = DualSystemCrossAttnArchitecture(cfg=cfg)
    assert arch.action_backbone is not None
    assert arch.video_backbone is None


def test_architecture_raises_without_video_dim():
    """Without video_backbone and without video_dim, architecture raises."""
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "action_dim": 14,
        "dim": 64,
        "ffn_dim": 256,
        "num_heads": 4,
        "bridge_layers": (0, 1),
    }
    import pytest

    with pytest.raises(ValueError, match="video_dim must be specified"):
        DualSystemCrossAttnArchitecture(cfg=cfg)


def test_architecture_derives_bridge_layers_from_backbone_num_layers():
    """bridge_interval should use video_backbone.num_layers when num_dit_layers not in cfg."""
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "action_dim": 14,
        "dim": 64,
        "ffn_dim": 256,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_interval": 5,
        "num_dit_layers": 20,
    }
    arch = DualSystemCrossAttnArchitecture(cfg=cfg)
    assert arch.bridge_layers == (0, 5, 10, 15)
