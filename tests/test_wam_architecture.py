"""Tests for WAM Architecture registry and implementations."""

import torch


def test_architecture_registry_populated():
    from openwam.model import ARCHITECTURE_REGISTRY, ARCHITECTURE_SUPPORT

    assert "dual_system" in ARCHITECTURE_REGISTRY
    assert "moe_expert" in ARCHITECTURE_REGISTRY
    assert "shared_backbone" in ARCHITECTURE_REGISTRY
    assert len(ARCHITECTURE_REGISTRY) == 3
    assert ARCHITECTURE_SUPPORT["dual_system"].supported is True
    assert ARCHITECTURE_SUPPORT["moe_expert"].supported is True
    assert ARCHITECTURE_SUPPORT["shared_backbone"].supported is True


def test_architecture_support_lists():
    from openwam.model import (
        get_architecture_support,
        list_supported_architectures,
    )

    supported = list_supported_architectures()
    assert "dual_system" in supported
    assert "moe_expert" in supported
    assert "shared_backbone" in supported
    assert get_architecture_support("shared_backbone").status == "supported"


def test_build_architecture_dual_system():
    from openwam.model import build_architecture

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
    from openwam.model import build_architecture

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
    """Shared backbone should build successfully."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 10}
    arch = build_architecture("shared_backbone", cfg)
    assert arch.action_dim == 7
    assert arch.bridge_layers == ()
    assert arch.is_interleaved is True


def test_build_architecture_unknown():
    import pytest

    from openwam.model import build_architecture

    with pytest.raises(KeyError, match="Unknown architecture"):
        build_architecture("nonexistent", {})


def test_dual_system_prepare_and_extract():
    """Smoke test: prepare action tokens and extract prediction (cross_attn)."""
    from openwam.model import build_architecture

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
    from openwam.model import build_architecture

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


def test_shared_backbone_prepare_and_extract():
    """Smoke test: shared backbone prepare, on_dit_block (no-op), and extract."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 10}
    arch = build_architecture("shared_backbone", cfg)
    arch.eval()

    B, T_action, T_video = 1, 10, 20

    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([500.0])

    state = arch.prepare_action_tokens(noisy_actions, timestep)
    assert state.action_latents.shape == (B, T_action, 128)
    assert state.extra["num_action_tokens"] == T_action

    # Simulate video DiT blocks: action tokens are part of the combined sequence
    video_hidden = torch.randn(B, T_video + T_action, 128)
    for block_id in range(2):
        video_hidden, state = arch.on_dit_block(block_id, video_hidden, state)

    # Store final hidden state for extraction
    state.extra["final_hidden"] = video_hidden

    with torch.no_grad():
        action_pred = arch.extract_action_prediction(state)
    assert action_pred.shape == (B, T_action, 7)


def test_shared_backbone_output_head_init():
    """SharedBackbone output head (ActionOutputMLP) uses small-random init."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5}
    arch = build_architecture("shared_backbone", cfg)
    head = arch.action_output_head
    assert torch.all(head.layer1.bias == 0)
    assert torch.all(head.layer2.bias == 0)
    for w in (head.layer1.weight, head.layer2.weight):
        assert not torch.all(w == 0), "weights should be small-random, not zero"
        assert w.abs().max() < 0.2, "weights should be small (std ~ 0.02)"


def test_moe_expert_ffn_and_output_head_init():
    """MoE: expert FFN output stays zero-init; action output head uses small-random init."""
    from openwam.model.action_model.moe_expert_dit import MoEExpertDiT

    dit = MoEExpertDiT(
        action_dim=7,
        video_dim=64,
        expert_ffn_dim=128,
        num_experts=2,
        expert_layers=(0, 1),
    )
    # Expert FFN output layer: zero-init preserved (pretrained video DiT
    # behavior at init for action tokens).
    for block in dit.expert_blocks:
        assert torch.all(block.ffn[2].weight == 0)
        assert torch.all(block.ffn[2].bias == 0)
    # Action output head (ActionOutputMLP): small-random, not zero.
    head = dit.action_output_head
    assert torch.all(head.layer1.bias == 0)
    assert torch.all(head.layer2.bias == 0)
    for w in (head.layer1.weight, head.layer2.weight):
        assert not torch.all(w == 0)
        assert w.abs().max() < 0.2


def test_dual_system_bridge_interval_resolves():
    """bridge_layers: null + bridge_interval resolves to range(0, N, step)."""
    from openwam.model import build_architecture

    cfg = {
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": None,
        "bridge_interval": 2,
        "num_dit_layers": 30,
        "bridge_type": "cross_attn_detach",
    }
    arch = build_architecture("dual_system", cfg)
    assert arch.bridge_layers == tuple(range(0, 30, 2))
    assert len(arch.bridge_layers) == 15

    cfg["bridge_interval"] = 1
    arch_full = build_architecture("dual_system", cfg)
    assert arch_full.bridge_layers == tuple(range(30))


def test_dual_system_bridge_interval_backward_compat():
    """bridge_layers: null without bridge_interval keeps legacy 8-layer default."""
    from openwam.model import build_architecture

    cfg = {
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": None,
        "bridge_type": "cross_attn_detach",
    }
    arch = build_architecture("dual_system", cfg)
    assert arch.bridge_layers == (3, 7, 11, 15, 19, 23, 26, 29)


def test_action_self_attention_rope_breaks_permutation_equivariance():
    """Without positional info, self-attention is permutation-equivariant.
    RoPE injects absolute position into Q/K, so permuting the input tokens
    must NOT merely permute the output (the model distinguishes positions).
    Also: supplying RoPE freqs must change the output relative to freqs=None.
    """
    from openwam.model.action_model.action_dit import ActionSelfAttention
    from openwam.model.action_model.components import precompute_freqs_cis_1d

    dim, num_heads, seq = 32, 4, 4
    head_dim = dim // num_heads
    attn = ActionSelfAttention(dim=dim, num_heads=num_heads).eval()

    x = torch.randn(1, seq, dim)
    freqs = precompute_freqs_cis_1d(head_dim, max_len=seq)
    perm = torch.tensor([3, 1, 2, 0])  # non-identity permutation
    x_perm = x[:, perm, :]

    with torch.no_grad():
        out_plain = attn(x, freqs=None)
        out_plain_perm = attn(x_perm, freqs=None)
        out_rope = attn(x, freqs=freqs)
        out_rope_perm = attn(x_perm, freqs=freqs)

    # Sanity: freqs=None is permutation-equivariant.
    assert torch.allclose(out_plain_perm, out_plain[:, perm, :], atol=1e-5)
    # RoPE must break that symmetry (content same, positions shuffled → non-permute-equivalent output).
    assert not torch.allclose(out_rope_perm, out_rope[:, perm, :], atol=1e-5), (
        "RoPE had no effect: permuted output equals permuted-input output"
    )
    # And RoPE output must differ from no-freqs output on the same input.
    assert not torch.allclose(out_rope, out_plain, atol=1e-5)


def test_action_dit_rope_cross_attn_path():
    """ActionDiT cross_attn_detach path runs end-to-end with RoPE instead of
    LearnedPositionalEncoding.
    """
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
    assert dit.pos_encoding is None  # RoPE path
    assert hasattr(dit, "freqs") and dit.freqs.shape == (1024, 64 // 4 // 2)

    actions = torch.randn(2, 5, 7)
    video_features = [torch.randn(2, 10, 128) for _ in range(2)]
    timestep = torch.tensor([0.5, 0.8])
    out = dit(actions, video_features, timestep)
    assert out.shape == (2, 5, 7)


def test_action_dit_joint_self_attn_keeps_learned_pe():
    """joint_self_attn keeps learned action PE while also preparing joint RoPE."""
    from openwam.model.action_model.action_dit import ActionDiT

    dit = ActionDiT(
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=64,  # match dim so no video_projs Linear mismatch
        bridge_layers=(0, 1),
        bridge_type="joint_self_attn",
    )
    assert dit.pos_encoding is not None

    state = dit.prepare_action_state(torch.randn(2, 5, 7), torch.tensor([0.5, 0.8]))
    assert state.use_joint_rope is True


def test_dual_system_joint_self_attn_production_path_applies_rope():
    """Interleaved joint_self_attn path should wire RoPE into production blocks."""
    from openwam.model import build_architecture

    cfg = {
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "bridge_layers": (0,),
        "bridge_type": "joint_self_attn",
    }
    arch = build_architecture("dual_system", cfg)
    arch.eval()

    captured = {}
    original_forward = arch.action_dit.blocks[0].joint_attn.forward

    def wrapped_forward(x_action, x_video, freqs_action=None, freqs_video=None):
        captured["freqs_action"] = freqs_action
        captured["freqs_video"] = freqs_video
        return original_forward(x_action, x_video, freqs_action=freqs_action, freqs_video=freqs_video)

    arch.action_dit.blocks[0].joint_attn.forward = wrapped_forward
    try:
        noisy_actions = torch.randn(2, 5, 7)
        timestep = torch.tensor([0.5, 0.8])
        state = arch.prepare_action_tokens(noisy_actions, timestep)
        video_hidden = torch.randn(2, 9, 32)
        _, state = arch.on_dit_block(0, video_hidden, state)
        with torch.no_grad():
            action_pred = arch.extract_action_prediction(state)
    finally:
        arch.action_dit.blocks[0].joint_attn.forward = original_forward

    assert action_pred.shape == (2, 5, 7)
    assert captured["freqs_action"] is not None
    assert captured["freqs_video"] is not None
    assert captured["freqs_action"].shape[0] == state.extra["dit_state"].x_action.shape[1]
    assert captured["freqs_video"].shape[0] == state.extra["dit_state"].x_video_proj.shape[1]


def test_register_custom_architecture():
    """Verify that custom architectures can be registered."""
    from openwam.model.base import ActionState, BaseWAMArchitecture
    from openwam.model.registry import ARCHITECTURE_REGISTRY, register_architecture

    @register_architecture("test_custom")
    class TestArch(BaseWAMArchitecture):
        def prepare_action_tokens(self, noisy_actions, timestep, **kw):
            return ActionState(action_latents=noisy_actions, timestep=timestep)

        def on_dit_block(self, block_id, video_hidden, action_state):
            return video_hidden, action_state

        def extract_action_prediction(self, action_state):
            return action_state.action_latents

        @property
        def action_dim(self):
            return 7

        @property
        def bridge_layers(self):
            return ()

    assert "test_custom" in ARCHITECTURE_REGISTRY

    # Cleanup
    del ARCHITECTURE_REGISTRY["test_custom"]
