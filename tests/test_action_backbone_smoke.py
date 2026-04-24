"""Smoke tests for action backbone components.

Verifies that all action model classes can be imported, instantiated with
small parameters, and produce correct output shapes. No GPU required.
"""

import pytest
import torch


def test_components_import():
    """Shared components should be importable."""
    from openwam.model.action_model.components import (
        ActionEmbedding,
        ActionEncoder,
        ActionOutputHead,
        ActionOutputMLP,
        LearnedPositionalEncoding,
        RMSNorm,
        SinusoidalPositionalEncoding,
        TimestepEmbedding,
        TimestepModulation,
        sinusoidal_embedding_1d,
    )

    assert all(
        c is not None
        for c in [
            ActionEmbedding,
            ActionEncoder,
            ActionOutputHead,
            ActionOutputMLP,
            LearnedPositionalEncoding,
            RMSNorm,
            SinusoidalPositionalEncoding,
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


def test_sinusoidal_positional_encoding_shape():
    """SinusoidalPositionalEncoding maps (B, T) timesteps to (B, T, dim)."""
    from openwam.model.action_model.components import SinusoidalPositionalEncoding

    pe = SinusoidalPositionalEncoding(embedding_dim=64)
    t = torch.rand(2, 10)
    out = pe(t)
    assert out.shape == (2, 10, 64)


def test_action_encoder_shape_per_sample_timestep():
    """ActionEncoder should accept (B,) timestep and broadcast to (B, T)."""
    from openwam.model.action_model.components import ActionEncoder

    enc = ActionEncoder(action_dim=14, hidden_dim=64)
    actions = torch.randn(2, 10, 14)
    t = torch.rand(2)
    out = enc(actions, t)
    assert out.shape == (2, 10, 64)


def test_action_encoder_shape_per_token_timestep():
    """ActionEncoder should accept (B, T) timestep directly."""
    from openwam.model.action_model.components import ActionEncoder

    enc = ActionEncoder(action_dim=14, hidden_dim=64)
    actions = torch.randn(2, 10, 14)
    t = torch.rand(2, 10)
    out = enc(actions, t)
    assert out.shape == (2, 10, 64)


def test_action_encoder_timestep_mismatch_raises():
    """ActionEncoder should reject mismatched timestep shapes."""
    from openwam.model.action_model.components import ActionEncoder

    enc = ActionEncoder(action_dim=14, hidden_dim=64)
    actions = torch.randn(2, 10, 14)
    with pytest.raises(ValueError):
        enc(actions, torch.rand(3))
    with pytest.raises(ValueError):
        enc(actions, torch.rand(2, 7))


def test_shared_backbone_prepare_action_tokens_with_action_encoder():
    """SharedBackbone.prepare_action_tokens should use ActionEncoder internally."""
    from openwam.model.action_model.components import ActionEncoder
    from openwam.model.shared_backbone import SharedBackboneArchitecture

    arch = SharedBackboneArchitecture(cfg={"action_dim": 14, "video_dim": 128, "max_action_len": 64})
    assert isinstance(arch.input_proj, ActionEncoder)

    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2)
    state = arch.prepare_action_tokens(actions, timestep)
    assert state.action_latents.shape == (2, 16, 128)
    assert state.extra["num_action_tokens"] == 16


def test_shared_backbone_prepare_action_tokens_per_token_timestep():
    """SharedBackbone should accept (B, T) per-token timestep via ActionEncoder."""
    from openwam.model.shared_backbone import SharedBackboneArchitecture

    arch = SharedBackboneArchitecture(cfg={"action_dim": 14, "video_dim": 128, "max_action_len": 64})
    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2, 16)  # per-token
    state = arch.prepare_action_tokens(actions, timestep)
    assert state.action_latents.shape == (2, 16, 128)
    # ActionState.timestep is still collapsed to per-sample (used by the
    # dead t_embed / t_mod path and DualSystem's ActionDiT).
    assert state.timestep.shape == (2,)
    # SharedBackboneState.timestep preserves the raw (B, T) granularity so
    # _build_action_t_mod can build per-token AdaLN modulation in
    # model_fn_wan_video (see wan_video.py _build_action_t_mod).
    assert state.extra["shared_backbone_state"].timestep.shape == (2, 16)


def test_moe_expert_prepare_state_with_action_encoder():
    """MoEActionExpertArchitecture.prepare_action_tokens wires ActionEncoder."""
    from openwam.model.action_model.components import ActionEncoder
    from openwam.model.moe_expert import MoEActionExpertArchitecture

    arch = MoEActionExpertArchitecture(cfg={
        "action_dim": 14,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "bridge_layers": [0, 1, 2],
    })
    assert isinstance(arch.moe_dit.action_input_proj, ActionEncoder)

    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2)
    state = arch.prepare_action_tokens(actions, timestep)
    assert "moe_state" in state.extra
    assert state.extra["moe_state"].action_tokens.shape == (2, 16, 128)


def test_moe_expert_prepare_state_per_token_timestep():
    """MoE prepare_state should build per-token ExpertFFN AdaLN under (B, T)."""
    from openwam.model.moe_expert import MoEActionExpertArchitecture

    arch = MoEActionExpertArchitecture(cfg={
        "action_dim": 14,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "bridge_layers": [0, 1, 2],
    })
    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2, 16)
    state = arch.prepare_action_tokens(actions, timestep)
    moe_state = state.extra["moe_state"]
    assert moe_state.action_tokens.shape == (2, 16, 128)
    # ExpertFFN AdaLN t_mod must be per-token so each action token's
    # shift/scale/gate tracks its own noise level.
    assert moe_state.t_embed.shape == (2, 16, 128)
    assert moe_state.t_mod.shape == (2, 16, 3, 128)
    # State.timestep preserves the raw (B, T) shape for _build_action_t_mod.
    assert moe_state.timestep.shape == (2, 16)


def test_expert_ffn_block_per_token_tmod_shape():
    """ExpertFFNBlock should accept per-token t_mod (B, T, 3, dim)."""
    from openwam.model.action_model.moe_expert_dit import ExpertFFNBlock

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
    from openwam.model.action_model.moe_expert_dit import ExpertFFNBlock

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


def test_moe_expert_prepare_state_per_sample_keeps_tmod_rank3():
    """Per-sample timestep must still produce (B, 3, dim) t_mod (no regression)."""
    from openwam.model.moe_expert import MoEActionExpertArchitecture

    arch = MoEActionExpertArchitecture(cfg={
        "action_dim": 14,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "bridge_layers": [0, 1, 2],
    })
    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2)  # per-sample
    state = arch.prepare_action_tokens(actions, timestep)
    moe_state = state.extra["moe_state"]
    assert moe_state.t_mod.shape == (2, 3, 128)
    assert moe_state.t_embed.shape == (2, 128)
    assert moe_state.timestep.shape == (2,)


def test_action_output_mlp_shape():
    """ActionOutputMLP should produce (B, T, action_dim)."""
    from openwam.model.action_model.components import ActionOutputMLP

    head = ActionOutputMLP(input_dim=128, hidden_dim=64, action_dim=14)
    x = torch.randn(2, 10, 128)
    out = head(x)
    assert out.shape == (2, 10, 14)


def test_action_output_mlp_small_random_init():
    """Weights should be small-random (std=0.02), biases zero on both layers."""
    from openwam.model.action_model.components import ActionOutputMLP

    head = ActionOutputMLP(input_dim=128, hidden_dim=64, action_dim=14)
    assert torch.all(head.layer1.bias == 0)
    assert torch.all(head.layer2.bias == 0)
    for w in (head.layer1.weight, head.layer2.weight):
        assert not torch.all(w == 0), "weights should be small-random, not zero"
        assert w.abs().max() < 0.2, "weights should be small (std ~ 0.02)"


def test_shared_backbone_uses_action_output_mlp():
    """SharedBackbone wires ActionOutputMLP as its output head."""
    from openwam.model.action_model.components import ActionOutputMLP
    from openwam.model.shared_backbone import SharedBackboneArchitecture

    arch = SharedBackboneArchitecture(cfg={"action_dim": 14, "video_dim": 128, "max_action_len": 64})
    assert isinstance(arch.action_output_head, ActionOutputMLP)

    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2)
    state = arch.prepare_action_tokens(actions, timestep)
    # Simulate the combined video+action hidden output after the DiT loop.
    state.extra["final_hidden"] = torch.randn(2, 32, 128)
    pred = arch.extract_action_prediction(state)
    assert pred.shape == (2, 16, 14)


def test_moe_expert_uses_action_output_mlp():
    """MoE architecture wires ActionOutputMLP as its output head."""
    from openwam.model.action_model.components import ActionOutputMLP
    from openwam.model.moe_expert import MoEActionExpertArchitecture

    arch = MoEActionExpertArchitecture(cfg={
        "action_dim": 14,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "bridge_layers": [0, 1, 2],
    })
    assert isinstance(arch.moe_dit.action_output_head, ActionOutputMLP)

    actions = torch.randn(2, 16, 14)
    timestep = torch.rand(2)
    state = arch.prepare_action_tokens(actions, timestep)
    pred = arch.extract_action_prediction(state)
    assert pred.shape == (2, 16, 14)
