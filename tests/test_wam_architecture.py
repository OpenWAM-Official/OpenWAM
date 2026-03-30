"""Tests for WAM Architecture registry and implementations."""

import torch


def test_architecture_registry_populated():
    from open_wam.models.architectures import ARCHITECTURE_REGISTRY, ARCHITECTURE_SUPPORT
    assert "dual_system" in ARCHITECTURE_REGISTRY
    assert "moe_expert" in ARCHITECTURE_REGISTRY
    assert "shared_backbone" in ARCHITECTURE_REGISTRY
    assert len(ARCHITECTURE_REGISTRY) == 3
    assert ARCHITECTURE_SUPPORT["dual_system"].supported is True
    assert ARCHITECTURE_SUPPORT["moe_expert"].supported is True
    assert ARCHITECTURE_SUPPORT["shared_backbone"].supported is False


def test_architecture_support_lists():
    from open_wam.models.architectures import (
        get_architecture_support,
        list_experimental_architectures,
        list_supported_architectures,
    )

    assert list_supported_architectures() == ("dual_system", "moe_expert")
    assert list_experimental_architectures() == ("shared_backbone",)
    assert get_architecture_support("shared_backbone").status == "experimental"


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
    cfg = {
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "num_experts": 2,
        "expert_layers": (1, 3),
    }
    arch = build_architecture("moe_expert", cfg)
    assert arch.action_dim == 7
    assert arch.bridge_layers == (1, 3)
    assert arch.moe_dit is not None
    assert len(arch.moe_dit.expert_blocks) == 2


def test_build_architecture_shared():
    """Experimental architectures should fail fast on the default path."""
    from open_wam.models.architectures import build_architecture
    import pytest
    cfg = {"action_dim": 7, "video_dim": 256, "num_action_tokens": 10}
    with pytest.raises(NotImplementedError, match="experimental"):
        build_architecture("shared_backbone", cfg)


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


def test_moe_expert_prepare_and_extract():
    """Smoke test: MoE prepare action tokens, expert FFN, and extract prediction."""
    from open_wam.models.architectures import build_architecture
    cfg = {
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "num_experts": 2,
        "expert_layers": (0, 1),
    }
    arch = build_architecture("moe_expert", cfg)
    arch.eval()

    B, T_action, T_video = 1, 10, 20

    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([500.0])

    state = arch.prepare_action_tokens(noisy_actions, timestep)
    assert "moe_state" in state.extra

    moe_state = state.extra["moe_state"]
    assert moe_state.action_tokens.shape == (B, T_action, 128)
    assert moe_state.n_action_tokens == T_action

    # Simulate video DiT blocks: action tokens are part of the combined sequence
    # In real pipeline, concatenation happens in model_fn_wan_video.
    # Here we simulate by creating a combined hidden state.
    video_hidden = torch.randn(B, T_video + T_action, 128)  # combined sequence

    for block_id in range(2):
        video_hidden, state = arch.on_dit_block(block_id, video_hidden, state)

    assert moe_state.expert_block_counter == 2

    # Extract action prediction
    with torch.no_grad():
        action_pred = arch.extract_action_prediction(state)
    assert action_pred.shape == (B, T_action, 7)


def test_moe_expert_ffn_zero_init():
    """Verify expert FFN and output head are zero-initialized."""
    from diffsynth.models.moe_action_expert import MoEExpertDiT
    import sys
    from pathlib import Path
    _tp = str(Path(__file__).resolve().parent.parent / "third_party")
    if _tp not in sys.path:
        sys.path.insert(0, _tp)

    dit = MoEExpertDiT(
        action_dim=7, video_dim=64, expert_ffn_dim=128,
        num_experts=2, expert_layers=(0, 1),
    )
    # Expert FFN output layer should be zero
    for block in dit.expert_blocks:
        assert torch.all(block.ffn[2].weight == 0)
        assert torch.all(block.ffn[2].bias == 0)
    # Output head should be zero
    assert torch.all(dit.output_head.weight == 0)
    assert torch.all(dit.output_head.bias == 0)


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
