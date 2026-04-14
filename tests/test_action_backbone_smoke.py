"""Smoke tests for action backbone components.

Verifies that all action model classes can be imported, instantiated with
small parameters, and produce correct output shapes. No GPU required.
"""

import torch


def test_components_import():
    """Shared components should be importable."""
    from openwam.model.action_model.components import (
        ActionEmbedding,
        ActionOutputHead,
        LearnedPositionalEncoding,
        RMSNorm,
        TimestepEmbedding,
        TimestepModulation,
        sinusoidal_embedding_1d,
    )

    assert all(
        c is not None
        for c in [
            ActionEmbedding,
            ActionOutputHead,
            LearnedPositionalEncoding,
            RMSNorm,
            TimestepEmbedding,
            TimestepModulation,
            sinusoidal_embedding_1d,
        ]
    )


def test_action_embedding_shape():
    """ActionEmbedding should project action_dim -> hidden_dim."""
    from openwam.model.action_model.components import ActionEmbedding

    embed = ActionEmbedding(action_dim=7, hidden_dim=64)
    x = torch.randn(2, 10, 7)
    out = embed(x)
    assert out.shape == (2, 10, 64)


def test_timestep_embedding_shape():
    """TimestepEmbedding should produce (B, dim) from (B,) timestep."""
    from openwam.model.action_model.components import TimestepEmbedding

    te = TimestepEmbedding(freq_dim=32, dim=64)
    t = torch.tensor([0.5, 0.8])
    out = te(t)
    assert out.shape == (2, 64)


def test_timestep_modulation_shape():
    """TimestepModulation should produce (B, n_params, dim)."""
    from openwam.model.action_model.components import TimestepModulation

    mod = TimestepModulation(dim=64, n_params=9)
    t_embed = torch.randn(2, 64)
    out = mod(t_embed)
    assert out.shape == (2, 9, 64)


def test_action_output_head_shape():
    """ActionOutputHead should produce (B, T, action_dim)."""
    from openwam.model.action_model.components import ActionOutputHead

    head = ActionOutputHead(dim=64, action_dim=7)
    x = torch.randn(2, 10, 64)
    t_embed = torch.randn(2, 64)
    out = head(x, t_embed)
    assert out.shape == (2, 10, 7)


def test_action_output_head_zero_init():
    """ActionOutputHead weights should be zero-initialized."""
    from openwam.model.action_model.components import ActionOutputHead

    head = ActionOutputHead(dim=64, action_dim=7)
    assert torch.all(head.head.weight == 0)
    assert torch.all(head.head.bias == 0)


def test_action_dit_small_instantiate():
    """ActionDiT should instantiate with small parameters."""
    from openwam.model.action_model.action_dit import ActionDiT

    dit = ActionDiT(
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=128,
        bridge_layers=(0, 1),
        bridge_type="cross_attn_detach",
    )
    assert dit.action_dim == 7
    assert dit.num_layers == 2


def test_action_dit_forward_shape():
    """ActionDiT forward should produce (B, T, action_dim)."""
    from openwam.model.action_model.action_dit import ActionDiT

    dit = ActionDiT(
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=128,
        bridge_layers=(0, 1),
        bridge_type="cross_attn_detach",
    )
    actions = torch.randn(2, 5, 7)
    video_features = [torch.randn(2, 10, 128) for _ in range(2)]
    timestep = torch.tensor([0.5, 0.8])

    out = dit(actions, video_features, timestep)
    assert out.shape == (2, 5, 7)


def test_moe_expert_dit_instantiate():
    """MoEExpertDiT should instantiate with small parameters."""
    from openwam.model.action_model.moe_expert_dit import MoEExpertDiT

    dit = MoEExpertDiT(
        action_dim=7,
        video_dim=64,
        expert_ffn_dim=128,
        num_experts=2,
        expert_layers=(0, 1),
    )
    assert dit.action_dim == 7
    assert dit.num_experts == 2


def test_shared_backbone_instantiate():
    """SharedBackboneArchitecture should instantiate."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5}
    arch = build_architecture("shared_backbone", cfg)
    assert arch.action_dim == 7
