"""Smoke tests for WAM architecture composition.

Tests prepare_action_tokens / on_dit_block / extract_action_prediction
flow for each architecture with fake tensors. No GPU required.
"""

import torch


def _make_dual_system(bridge_type="cross_attn_detach"):
    from openwam.model import build_architecture

    cfg = {
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": (0, 2),
        "bridge_type": bridge_type,
    }
    return build_architecture("dual_system", cfg)


def test_dual_system_cross_attn_flow():
    """DualSystem cross_attn: collect features then run ActionDiT."""
    arch = _make_dual_system("cross_attn_detach")
    B, T_action, T_video = 2, 5, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    state = arch.prepare_action_tokens(noisy_actions, timestep)
    assert state.action_latents is not None

    # Simulate 3 video DiT blocks (only block 0 and 2 are bridge layers)
    video_hidden = torch.randn(B, T_video, 128)
    for block_id in range(3):
        video_hidden, state = arch.on_dit_block(block_id, video_hidden, state)

    pred = arch.extract_action_prediction(state)
    assert pred.shape == (B, T_action, 7)


def test_dual_system_joint_self_attn_flow():
    """DualSystem joint_self_attn: interleaved execution."""
    arch = _make_dual_system("joint_self_attn")
    B, T_action, T_video = 2, 5, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    state = arch.prepare_action_tokens(noisy_actions, timestep)

    video_hidden = torch.randn(B, T_video, 128)
    for block_id in range(3):
        video_hidden, state = arch.on_dit_block(block_id, video_hidden, state)

    pred = arch.extract_action_prediction(state)
    assert pred.shape == (B, T_action, 7)


def test_moe_expert_flow():
    """MoE Expert: action tokens in video sequence with expert FFN."""
    from openwam.model import build_architecture

    cfg = {
        "action_dim": 7,
        "video_dim": 64,
        "expert_ffn_dim": 128,
        "bridge_layers": (0, 2),
    }
    arch = build_architecture("moe_expert", cfg)

    B, T_action, T_video = 2, 5, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    state = arch.prepare_action_tokens(noisy_actions, timestep)

    # Simulate combined video+action sequence
    video_hidden = torch.randn(B, T_video + T_action, 64)
    for block_id in range(3):
        video_hidden, state = arch.on_dit_block(block_id, video_hidden, state)

    pred = arch.extract_action_prediction(state)
    assert pred.shape == (B, T_action, 7)


def test_shared_backbone_flow():
    """SharedBackbone: action tokens processed by shared DiT."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5}
    arch = build_architecture("shared_backbone", cfg)

    B, T_action = 2, 5
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    state = arch.prepare_action_tokens(noisy_actions, timestep)
    assert state.action_latents.shape == (B, T_action, 64)

    # SharedBackbone: on_dit_block is no-op, need final_hidden for extract
    state.extra["final_hidden"] = torch.randn(B, 20 + T_action, 64)
    pred = arch.extract_action_prediction(state)
    assert pred.shape == (B, T_action, 7)


def test_all_architectures_registered():
    """All three architectures should be in the registry."""
    from openwam.model.registry import list_supported_architectures

    supported = list_supported_architectures()
    assert "dual_system" in supported
    assert "moe_expert" in supported
    assert "shared_backbone" in supported
