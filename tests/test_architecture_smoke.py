"""Smoke tests for WAM architecture composition.

Drives the new ActionBackbone block-loop adapter (prepare_state /
before_loop / run_block / after_loop / extract_prediction) for each
architecture with fake tensors. No GPU required.
"""

import torch


def _make_dual_system(variant="joint_cross_attn", detach_bridge=True):
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": variant,
        "detach_bridge": detach_bridge,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": (0, 2),
    }
    return build_architecture(
        "dual_system_cross_attn" if variant == "joint_cross_attn" else "dual_system_self_attn", cfg
    )


def test_dual_system_joint_cross_attn_flow():
    """DualSystem joint_cross_attn: collect features then run ActionDiT."""
    arch = _make_dual_system("joint_cross_attn", detach_bridge=True)
    B, T_action, T_video = 2, 5, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    ab = arch.action_backbone
    state = ab.prepare_state(noisy_actions, timestep)
    assert state.action_latents is not None

    # Simulate bridge feature collection at every bridge_layer (sorted order).
    for _ in ab.bridge_layers:
        state.bridge_features.append(torch.randn(B, T_video, 128))

    pred = ab.extract_prediction(state)
    assert pred.shape == (B, T_action, 7)


def test_dual_system_joint_self_attn_flow():
    """DualSystem joint_self_attn: interleaved execution exercised via run_block."""

    class _MockVState:
        def __init__(self, x, t_mod, f, h, w):
            self.x = x
            self.reference_prefix_len = 0
            self.t_mod = t_mod
            self.f = f
            self.h = h
            self.w = w

    arch = _make_dual_system("joint_self_attn", detach_bridge=False)
    B, T_action, T_video = 2, 5, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    ab = arch.action_backbone
    state = ab.prepare_state(noisy_actions, timestep)
    # Treat T_video=10 as a 1D temporal axis (f=10, h=1, w=1).
    vstate = _MockVState(torch.randn(B, T_video, 128), t_mod=torch.zeros(B, 6, 128), f=T_video, h=1, w=1)

    class _MockVB:
        def run_block(self, _bid, vs):
            return vs

    vb = _MockVB()
    for block_id in range(3):
        vstate, state = ab.run_block(block_id, vb, vstate, state)

    pred = ab.extract_prediction(state)
    assert pred.shape == (B, T_action, 7)


def test_shared_backbone_moe_flow():
    """SharedBackbone MoE variant: action tokens in video sequence with expert FFN."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 64,
        "expert_ffn_dim": 128,
        "bridge_layers": (0, 2),
    }
    arch = build_architecture("shared_backbone_moe", cfg)

    B, T_action, T_video = 2, 5, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    ab = arch.action_backbone
    state = ab.prepare_state(noisy_actions, timestep)

    # Drive the run_block loop with a mock video state where action tokens
    # sit at the tail of state.x (matching what before_loop would produce).
    class _MockVState:
        def __init__(self, x):
            self.x = x

    payload = state.runtime_state.payload
    vstate = _MockVState(torch.cat([torch.randn(B, T_video, 64), payload.action_tokens], dim=1))

    class _MockVB:
        def run_block(self, _bid, vs):
            return vs

    vb = _MockVB()
    for block_id in range(3):
        vstate, state = ab.run_block(block_id, vb, vstate, state)

    pred = ab.extract_prediction(state)
    assert pred.shape == (B, T_action, 7)


def test_shared_backbone_flow():
    """SharedBackbone vanilla: action tokens processed by shared DiT."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5}
    arch = build_architecture("shared_backbone_vanilla", cfg)

    B, T_action = 2, 5
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    ab = arch.action_backbone
    state = ab.prepare_state(noisy_actions, timestep)
    assert state.action_latents.shape == (B, T_action, 64)

    # SharedBackbone: per-block is no-op; supply a final_hidden as if extracted
    # from the video stream tail.
    state.final_hidden = torch.randn(B, 20 + T_action, 64)
    pred = ab.extract_prediction(state)
    assert pred.shape == (B, T_action, 7)


def test_all_architectures_registered():
    """Supported top-level architecture families should be in the registry."""
    from openwam.model import list_supported_architectures

    supported = list_supported_architectures()
    assert "dual_system_cross_attn" in supported
    assert "dual_system_self_attn" in supported
    assert "shared_backbone_vanilla" in supported
    assert "shared_backbone_moe" in supported
