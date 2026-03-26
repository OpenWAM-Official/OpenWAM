"""Tests for WAM Architecture registry and implementations."""

import torch


def test_architecture_registry_populated():
    from open_wam.models.architectures import ARCHITECTURE_REGISTRY
    assert "dual_system" in ARCHITECTURE_REGISTRY
    assert "moe_expert" in ARCHITECTURE_REGISTRY
    assert "shared_backbone" in ARCHITECTURE_REGISTRY
    assert len(ARCHITECTURE_REGISTRY) == 3


def test_build_architecture_dual_system():
    from open_wam.models.architectures import build_architecture
    cfg = {
        "action_dim": 7,
        "dim": 128,
        "ffn_dim": 256,
        "num_heads": 4,
        "num_layers": 2,
        "video_dim": 256,
        "bridge_layers": (0, 1),
        "bridge_type": "cross_attn_detach",
    }
    arch = build_architecture("dual_system", cfg)
    assert arch.action_dim == 7
    assert arch.bridge_layers == (0, 1)
    assert arch.action_dit is not None


def test_build_architecture_moe():
    from open_wam.models.architectures import build_architecture
    cfg = {"action_dim": 7, "num_action_tokens": 10, "expert_layers": (1, 3)}
    arch = build_architecture("moe_expert", cfg)
    assert arch.action_dim == 7
    assert arch.bridge_layers == (1, 3)


def test_build_architecture_shared():
    from open_wam.models.architectures import build_architecture
    cfg = {"action_dim": 7, "video_dim": 256, "num_action_tokens": 10}
    arch = build_architecture("shared_backbone", cfg)
    assert arch.action_dim == 7
    assert arch.bridge_layers == ()


def test_build_architecture_unknown():
    from open_wam.models.architectures import build_architecture
    import pytest
    with pytest.raises(KeyError, match="Unknown architecture"):
        build_architecture("nonexistent", {})


def test_dual_system_prepare_and_extract():
    """Smoke test: prepare action tokens and extract prediction (cross_attn)."""
    from open_wam.models.architectures import build_architecture
    cfg = {
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "video_dim": 128,
        "bridge_layers": (0, 1),
        "bridge_type": "cross_attn",
    }
    arch = build_architecture("dual_system", cfg)
    arch.eval()

    B, T_action = 1, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([500.0])

    state = arch.prepare_action_tokens(noisy_actions, timestep)
    assert state.action_latents is not None
    assert "bridge_features" in state.extra

    # Simulate two DiT blocks producing video hidden states
    video_hidden = torch.randn(B, 20, 128)
    for block_id in range(2):
        video_hidden, state = arch.on_dit_block(block_id, video_hidden, state)

    # Extract action prediction
    with torch.no_grad():
        action_pred = arch.extract_action_prediction(state)
    assert action_pred.shape == (B, T_action, 7)


def test_register_custom_architecture():
    """Verify that custom architectures can be registered."""
    from open_wam.models.architectures.base import BaseWAMArchitecture, ActionState
    from open_wam.models.architectures.registry import register_architecture, ARCHITECTURE_REGISTRY

    @register_architecture("test_custom")
    class TestArch(BaseWAMArchitecture):
        def prepare_action_tokens(self, noisy_actions, timestep, **kw):
            return ActionState(action_latents=noisy_actions, timestep=timestep)
        def on_dit_block(self, block_id, video_hidden, action_state):
            return video_hidden, action_state
        def extract_action_prediction(self, action_state):
            return action_state.action_latents
        @property
        def action_dim(self): return 7
        @property
        def bridge_layers(self): return ()

    assert "test_custom" in ARCHITECTURE_REGISTRY

    # Cleanup
    del ARCHITECTURE_REGISTRY["test_custom"]
