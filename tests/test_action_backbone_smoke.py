"""Smoke tests for action backbone components.

Verifies that all action model classes can be imported, instantiated with
small parameters, and produce correct output shapes. No GPU required.
"""

import pytest
import torch


def test_components_import():
    """Shared components should be importable."""
    from openwam.model.action_backbone.components import (
        ActionEncoder,
        ActionOutputMLP,
        RMSNorm,
        SinusoidalPositionalEncoding,
        TimestepEmbedding,
        TimestepModulation,
        sinusoidal_embedding_1d,
    )

    assert all(
        c is not None
        for c in [
            ActionEncoder,
            ActionOutputMLP,
            RMSNorm,
            SinusoidalPositionalEncoding,
            TimestepEmbedding,
            TimestepModulation,
            sinusoidal_embedding_1d,
        ]
    )


def test_timestep_embedding_shape():
    """TimestepEmbedding should produce (B, dim) from (B,) timestep."""
    from openwam.model.action_backbone.components import TimestepEmbedding

    te = TimestepEmbedding(freq_dim=32, dim=64)
    t = torch.tensor([0.5, 0.8])
    out = te(t)
    assert out.shape == (2, 64)


def test_timestep_modulation_shape():
    """TimestepModulation should produce (B, n_params, dim)."""
    from openwam.model.action_backbone.components import TimestepModulation

    mod = TimestepModulation(dim=64, n_params=9)
    t_embed = torch.randn(2, 64)
    out = mod(t_embed)
    assert out.shape == (2, 9, 64)


def test_action_dit_small_instantiate():
    """ActionDiT should instantiate with small parameters."""
    import torch.nn as nn

    from openwam.model.action_backbone.separate_action_dit import ActionDiT

    dit = ActionDiT(
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=128,
        bridge_layers=(0, 1),
        variant="joint_cross_attn",
    )
    assert dit.action_dim == 7
    assert dit.num_layers == 2
    assert isinstance(dit.action_encoder, nn.Linear)
    assert dit.action_encoder.in_features == 7
    assert dit.action_encoder.out_features == 64
    assert isinstance(dit.text_embedding, nn.Sequential)
    assert dit.text_embedding[0].in_features == 4096
    assert dit.text_embedding[-1].out_features == 64
    assert isinstance(dit.action_decoder, nn.Linear)
    assert dit.action_decoder.in_features == 64
    assert dit.action_decoder.out_features == 7
    assert not torch.all(dit.action_decoder.weight == 0)
    assert not hasattr(dit, "action_embedding")
    assert not hasattr(dit, "action_output_head")


def test_action_dit_forward_shape():
    """ActionDiT forward should produce (B, T, action_dim)."""
    from openwam.model.action_backbone.separate_action_dit import ActionDiT

    dit = ActionDiT(
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=128,
        bridge_layers=(0, 1),
        variant="joint_cross_attn",
    )
    actions = torch.randn(2, 5, 7)
    bridges = {bid: torch.randn(2, 10, 128) for bid in (0, 1)}
    timestep = torch.tensor([0.5, 0.8])

    out = dit(actions, bridges, timestep)
    assert out.shape == (2, 5, 7)


def test_action_dit_joint_cross_attn_context_shape_and_effect():
    """joint_cross_attn should accept action-owned raw text/proprio context."""
    from openwam.model.action_backbone.separate_action_dit import ActionDiT

    torch.manual_seed(0)
    dit = ActionDiT(
        action_dim=7,
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=2,
        video_dim=48,
        bridge_layers=(0, 1),
        variant="joint_cross_attn",
        text_dim=16,
    ).eval()
    actions = torch.randn(2, 5, 7)
    bridges = {bid: torch.randn(2, 10, 48) for bid in (0, 1)}
    timestep = torch.tensor([0.5, 0.8])
    context_a = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(11))
    context_b = torch.randn(2, 4, 16, generator=torch.Generator().manual_seed(22))
    context_mask = torch.ones(2, 4, dtype=torch.bool)

    with torch.no_grad():
        out_a = dit(actions, bridges, timestep, context=context_a, context_mask=context_mask)
        out_b = dit(actions, bridges, timestep, context=context_b, context_mask=context_mask)

    assert out_a.shape == (2, 5, 7)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_moe_dit_instantiate():
    """SharedMoEActionBackbone should instantiate with small parameters."""
    from openwam.model.action_backbone.shared_action_backbone import SharedMoEActionBackbone

    dit = SharedMoEActionBackbone(
        action_dim=7,
        video_dim=64,
        expert_ffn_dim=128,
        bridge_layers=(0, 1),
    )
    assert dit.action_dim == 7
    assert dit.num_experts == 2


def test_single_system_instantiate():
    """SingleSystemArchitecture should instantiate."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5}
    arch = build_architecture("single_system_vanilla", cfg)
    assert arch.action_dim == 7


def test_sinusoidal_positional_encoding_shape():
    """SinusoidalPositionalEncoding maps (B, T) timesteps to (B, T, dim)."""
    from openwam.model.action_backbone.components import SinusoidalPositionalEncoding

    pe = SinusoidalPositionalEncoding(embedding_dim=64)
    t = torch.rand(2, 10)
    out = pe(t)
    assert out.shape == (2, 10, 64)


def test_action_encoder_shape_per_sample_timestep():
    """ActionEncoder should accept (B,) timestep and broadcast to (B, T)."""
    from openwam.model.action_backbone.components import ActionEncoder

    enc = ActionEncoder(action_dim=14, hidden_dim=64)
    actions = torch.randn(2, 10, 14)
    t = torch.rand(2)
    out = enc(actions, t)
    assert out.shape == (2, 10, 64)


def test_action_encoder_shape_per_token_timestep():
    """ActionEncoder should accept (B, T) timestep directly."""
    from openwam.model.action_backbone.components import ActionEncoder

    enc = ActionEncoder(action_dim=14, hidden_dim=64)
    actions = torch.randn(2, 10, 14)
    t = torch.rand(2, 10)
    out = enc(actions, t)
    assert out.shape == (2, 10, 64)


def test_action_encoder_timestep_mismatch_raises():
    """ActionEncoder should reject mismatched timestep shapes."""
    from openwam.model.action_backbone.components import ActionEncoder

    enc = ActionEncoder(action_dim=14, hidden_dim=64)
    actions = torch.randn(2, 10, 14)
    with pytest.raises(ValueError):
        enc(actions, torch.rand(3))
    with pytest.raises(ValueError):
        enc(actions, torch.rand(2, 7))


def test_single_system_encode_uses_action_encoder():
    """SharedVanillaActionBackbone.encode should run ActionEncoder without learned PE."""
    from openwam.model.action_backbone.components import ActionEncoder
    from openwam.model.architectures.single_system.vanilla import SingleSystemVanillaArchitecture

    arch = SingleSystemVanillaArchitecture(cfg={"action_dim": 14, "video_dim": 128, "max_action_len": 64})
    assert isinstance(arch.action_backbone.input_proj, ActionEncoder)
    assert not hasattr(arch.action_backbone, "pos_encoding")

    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2)
    tokens = arch.action_backbone.encode(actions, timestep)
    assert tokens.shape == (2, 16, 128)


def test_single_system_encode_per_token_timestep():
    """SharedVanillaActionBackbone.encode should accept (B, T) per-token timestep."""
    from openwam.model.architectures.single_system.vanilla import SingleSystemVanillaArchitecture

    arch = SingleSystemVanillaArchitecture(cfg={"action_dim": 14, "video_dim": 128, "max_action_len": 64})
    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2, 16)  # per-token
    tokens = arch.action_backbone.encode(actions, timestep)
    assert tokens.shape == (2, 16, 128)


def test_moe_encode_uses_action_encoder():
    """SharedMoEActionBackbone.encode wires ActionEncoder."""
    from openwam.model.action_backbone.components import ActionEncoder
    from openwam.model.architectures.single_system.moe import SingleSystemMoEArchitecture

    arch = SingleSystemMoEArchitecture(
        cfg={
            "action_dim": 14,
            "video_dim": 128,
            "expert_ffn_dim": 256,
            "bridge_layers": [0, 1, 2],
        }
    )
    assert isinstance(arch.action_backbone.input_proj, ActionEncoder)
    assert not hasattr(arch.action_backbone, "pos_encoding")

    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2)
    tokens, t_mod = arch.action_backbone.encode(actions, timestep)
    assert tokens.shape == (2, 16, 128)
    assert t_mod.shape == (2, 3, 128)


def test_moe_encode_per_token_timestep():
    """MoE encode should build per-token ExpertFFN AdaLN under (B, T) timestep."""
    from openwam.model.architectures.single_system.moe import SingleSystemMoEArchitecture

    arch = SingleSystemMoEArchitecture(
        cfg={
            "action_dim": 14,
            "video_dim": 128,
            "expert_ffn_dim": 256,
            "bridge_layers": [0, 1, 2],
        }
    )
    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2, 16)
    tokens, t_mod = arch.action_backbone.encode(actions, timestep)
    assert tokens.shape == (2, 16, 128)
    # ExpertFFN AdaLN t_mod must be per-token so each action token's
    # shift/scale/gate tracks its own noise level.
    assert t_mod.shape == (2, 16, 3, 128)


def test_expert_ffn_block_per_token_tmod_shape():
    """ExpertFFNBlock should accept per-token t_mod (B, T, 3, dim)."""
    from openwam.model.action_backbone.shared_action_backbone import ExpertFFNBlock

    block = ExpertFFNBlock(dim=64, ffn_dim=128)
    x = torch.randn(2, 8, 64)
    t_mod = torch.randn(2, 8, 3, 64)
    out = block(x, t_mod)
    assert out.shape == (2, 8, 64)


def test_expert_ffn_block_per_token_matches_broadcast():
    """Broadcasting a per-sample t_mod to per-token must give identical output.

    Sanity-checks that the new per-token AdaLN branch introduces no semantic
    drift vs. the per-sample branch when all T tokens share the same t_mod.
    """
    from openwam.model.action_backbone.shared_action_backbone import ExpertFFNBlock

    torch.manual_seed(0)
    block = ExpertFFNBlock(dim=64, ffn_dim=128)
    # Force non-zero FFN output so the gate path is exercised.
    for p in block.ffn[2].parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)

    x = torch.randn(3, 5, 64)
    t_mod_per_sample = torch.randn(3, 3, 64)
    t_mod_per_token = t_mod_per_sample.unsqueeze(1).expand(3, 5, 3, 64).contiguous()

    out_per_sample = block(x, t_mod_per_sample)
    out_per_token = block(x, t_mod_per_token)
    assert torch.allclose(out_per_sample, out_per_token, atol=1e-6)


def test_moe_encode_per_sample_keeps_tmod_rank3():
    """Per-sample timestep must still produce (B, 3, dim) t_mod (no regression)."""
    from openwam.model.architectures.single_system.moe import SingleSystemMoEArchitecture

    arch = SingleSystemMoEArchitecture(
        cfg={
            "action_dim": 14,
            "video_dim": 128,
            "expert_ffn_dim": 256,
            "bridge_layers": [0, 1, 2],
        }
    )
    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2)  # per-sample
    _, t_mod = arch.action_backbone.encode(actions, timestep)
    assert t_mod.shape == (2, 3, 128)


def test_action_output_mlp_shape():
    """ActionOutputMLP should produce (B, T, action_dim)."""
    from openwam.model.action_backbone.components import ActionOutputMLP

    head = ActionOutputMLP(input_dim=128, hidden_dim=64, action_dim=14)
    x = torch.randn(2, 10, 128)
    out = head(x)
    assert out.shape == (2, 10, 14)


def test_state_encoder_accepts_single_deploy_state():
    """SingleSystem state encoder should match dual-system's [D] deploy proprio input."""
    from openwam.model.action_backbone.components import StateEncoder

    enc = StateEncoder(state_dim=14, hidden_dim=64)
    out = enc(torch.randn(14))
    assert out.shape == (1, 1, 64)


def test_action_output_mlp_small_random_init():
    """Weights should be small-random (std=0.02), biases zero on both layers."""
    from openwam.model.action_backbone.components import ActionOutputMLP

    head = ActionOutputMLP(input_dim=128, hidden_dim=64, action_dim=14)
    assert torch.all(head.layer1.bias == 0)
    assert torch.all(head.layer2.bias == 0)
    for w in (head.layer1.weight, head.layer2.weight):
        assert not torch.all(w == 0), "weights should be small-random, not zero"
        assert w.abs().max() < 0.2, "weights should be small (std ~ 0.02)"


def test_single_system_uses_action_output_mlp():
    """SingleSystem wires ActionOutputMLP as its output head."""
    from openwam.model.action_backbone.components import DEFAULT_ACTION_DECODER_HIDDEN_DIM, ActionOutputMLP
    from openwam.model.architectures.single_system.vanilla import SingleSystemVanillaArchitecture

    arch = SingleSystemVanillaArchitecture(cfg={"action_dim": 14, "video_dim": 128, "max_action_len": 64})
    assert isinstance(arch.action_backbone.action_output_head, ActionOutputMLP)
    assert arch.action_backbone.action_output_head.layer1.out_features == DEFAULT_ACTION_DECODER_HIDDEN_DIM

    # Simulate the (B, T_action, video_dim) action tail extracted by
    # vb.extract_action_tokens after the DiT loop.
    final_hidden = torch.randn(2, 16, 128)
    pred = arch.action_backbone.decode(final_hidden)
    assert pred.shape == (2, 16, 14)


def test_single_system_action_decoder_hidden_dim_can_be_overridden():
    """SingleSystem decoder default aligns with dual-system dim but still supports explicit config."""
    from openwam.model.architectures.single_system.vanilla import SingleSystemVanillaArchitecture

    arch = SingleSystemVanillaArchitecture(
        cfg={
            "action_dim": 14,
            "video_dim": 128,
            "max_action_len": 64,
            "action_decoder_hidden_dim": 256,
        }
    )

    assert arch.action_backbone.action_output_head.layer1.out_features == 256


def test_moe_expert_uses_action_output_mlp():
    """MoE architecture wires ActionOutputMLP as its output head."""
    from openwam.model.action_backbone.components import DEFAULT_ACTION_DECODER_HIDDEN_DIM, ActionOutputMLP
    from openwam.model.architectures.single_system.moe import SingleSystemMoEArchitecture

    arch = SingleSystemMoEArchitecture(
        cfg={
            "action_dim": 14,
            "video_dim": 128,
            "expert_ffn_dim": 256,
            "bridge_layers": [0, 1, 2],
        }
    )
    assert isinstance(arch.action_backbone.action_output_head, ActionOutputMLP)
    assert arch.action_backbone.action_output_head.layer1.out_features == DEFAULT_ACTION_DECODER_HIDDEN_DIM

    # encode → simulate per-block updates → decode roundtrip.
    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2)
    tokens, _ = arch.action_backbone.encode(actions, timestep)
    pred = arch.action_backbone.decode(tokens)
    assert pred.shape == (2, 16, 14)
