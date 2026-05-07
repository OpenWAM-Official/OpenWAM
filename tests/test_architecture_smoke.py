"""Smoke tests for WAM architecture composition.

Each architecture exposes a different action-backbone contract:
  - joint_cross_attn → ActionBackbone.forward(action_tokens, bridges, timestep)
  - joint_self_attn  → MoTJointDriver coordinates pre/post_attn_at_layer
  - shared_backbone vanilla → encode / decode helpers; architecture forward drives vb loop
  - shared_backbone moe → encode / apply_expert(layer_id, ...) / decode helpers

These tests exercise each path with fake tensors; no GPU required.
"""

import torch


def _make_dual_system(variant="joint_cross_attn", detach_bridge=True, dim=64, video_dim=64):
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": variant,
        "detach_bridge": detach_bridge,
        "action_dim": 7,
        "dim": dim,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": video_dim,
        "bridge_layers": (0, 2),
    }
    return build_architecture(
        "dual_system_cross_attn" if variant == "joint_cross_attn" else "dual_system_self_attn", cfg
    )


def test_dual_system_joint_cross_attn_flow():
    """DualSystem joint_cross_attn: ActionDiT.forward(action_tokens, bridges, timestep)."""
    arch = _make_dual_system("joint_cross_attn", detach_bridge=True, dim=64, video_dim=128)
    B, T_action, T_video = 2, 5, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    ab = arch.action_backbone
    bridges = {bid: torch.randn(B, T_video, 128) for bid in ab.bridge_layers}
    pred = ab(noisy_actions, bridges, timestep)
    assert pred.shape == (B, T_action, 7)


def test_dual_system_joint_self_attn_flow():
    """DualSystem joint_self_attn: MoT pre/post_attn_at_layer round-trip."""
    # joint_self_attn requires action dim == video_dim (the MoT driver shares
    # per-head space across modalities).
    arch = _make_dual_system("joint_self_attn", detach_bridge=False, dim=64, video_dim=64)
    B, T_action = 2, 5
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    ab = arch.action_backbone
    astate = ab.prepare_state(noisy_actions, timestep)

    # Drive the action half-step at each layer with a fake mixed-attention
    # output. The driver's job (plumbing video Q/K/V, mixed attention, splitting)
    # is covered separately in test_mot_driver.
    for layer_id in range(ab.num_layers):
        q, k, v, post = ab.pre_attn_at_layer(layer_id, astate)
        assert q.shape == k.shape == v.shape == (B, T_action, ab.num_heads * ab.head_dim)
        attn_out = torch.randn_like(q)
        astate = ab.post_attn_at_layer(layer_id, astate, attn_out, post)

    pred = ab.extract_prediction(astate)
    assert pred.shape == (B, T_action, 7)


def test_dual_system_self_attn_has_mot_driver():
    """DualSystemSelfAttn must construct a MoTJointDriver when wired with a video backbone."""
    # The __init__ short-circuits when video_backbone is None, so this just checks
    # the attribute is present on the architecture (MoT driver itself is exercised
    # in tests/test_mot_driver.py).
    arch = _make_dual_system("joint_self_attn", detach_bridge=False, dim=64, video_dim=64)
    # No video backbone in the test factory → driver is None, but the slot exists.
    assert hasattr(arch, "_mot_driver")


def test_shared_backbone_moe_flow():
    """SharedBackbone MoE: encode → apply_expert at each expert layer → decode."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 64,
        "expert_ffn_dim": 128,
        "expert_layers": (0, 2),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    ab = arch.action_backbone

    B, T_action = 2, 5
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    tokens, t_mod = ab.encode(noisy_actions, timestep)
    assert tokens.shape == (B, T_action, 64)

    x_action = tokens
    for block_id in ab.expert_layers:
        x_action = ab.apply_expert(block_id, x_action, t_mod)
    pred = ab.decode(x_action)
    assert pred.shape == (B, T_action, 7)


def test_shared_backbone_flow():
    """SharedBackbone vanilla: encode → decode roundtrip."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5}
    arch = build_architecture("shared_backbone_vanilla", cfg)
    ab = arch.action_backbone

    B, T_action = 2, 5
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    tokens = ab.encode(noisy_actions, timestep)
    assert tokens.shape == (B, T_action, 64)
    pred = ab.decode(tokens)
    assert pred.shape == (B, T_action, 7)


def test_all_architectures_registered():
    """Supported top-level architecture families should be in the registry."""
    from openwam.model import list_supported_architectures

    supported = list_supported_architectures()
    assert "dual_system_cross_attn" in supported
    assert "dual_system_self_attn" in supported
    assert "shared_backbone_vanilla" in supported
    assert "shared_backbone_moe" in supported
