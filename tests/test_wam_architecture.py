"""Tests for WAM Architecture registry and implementations."""

import pytest
import torch

from openwam.model.base import ExecutionPlan


def test_architecture_module_layout_imports():
    from openwam.model.architectures.dual_system import (
        DualSystemCrossAttnArchitecture,
        DualSystemSelfAttnArchitecture,
    )
    from openwam.model.architectures.shared_backbone.moe import SharedBackboneMoEArchitecture
    from openwam.model.architectures.shared_backbone.vanilla import SharedBackboneVanillaArchitecture

    assert DualSystemCrossAttnArchitecture is not None
    assert DualSystemSelfAttnArchitecture is not None
    assert SharedBackboneMoEArchitecture is not None
    assert SharedBackboneVanillaArchitecture is not None


def test_architecture_state_types_import():
    from openwam.model.action_backbone.action_dit import ActionDiTState
    from openwam.model.action_backbone.shared_vanilla import SharedVanillaState

    assert ActionDiTState is not None
    assert SharedVanillaState is not None


def test_architecture_support_lists():
    from openwam.model import (
        get_architecture_support,
        list_supported_architectures,
    )

    supported = list_supported_architectures()
    assert "dual_system_cross_attn" in supported
    assert "dual_system_self_attn" in supported
    assert "shared_backbone_vanilla" in supported
    assert "shared_backbone_moe" in supported
    assert get_architecture_support("shared_backbone_moe").status == "supported"
    assert get_architecture_support("shared_backbone_vanilla").status == "supported"


def test_build_architecture_dual_system():
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": 7,
        "dim": 128,
        "ffn_dim": 256,
        "num_heads": 4,
        "num_layers": 2,
        "video_dim": 256,
        "bridge_layers": (0, 1),
    }
    arch = build_architecture("dual_system_cross_attn", cfg)
    assert arch.action_dim == 7
    assert arch.bridge_layers == (0, 1)
    assert arch.action_backbone is not None


def test_build_architecture_shared_backbone_moe():
    from openwam.model import build_architecture

    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "num_experts": 2,
        "bridge_layers": (1, 3),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    assert arch.action_dim == 7
    assert arch.bridge_layers == (1, 3)
    assert arch.action_backbone is not None
    assert len(arch.action_backbone.expert_blocks) == 2


def test_build_architecture_shared():
    """Shared backbone vanilla should build successfully."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 10}
    arch = build_architecture("shared_backbone_vanilla", cfg)
    assert arch.action_dim == 7
    assert arch.bridge_layers == ()
    assert arch.execution_plan == ExecutionPlan.INTERLEAVED_WHOLE_BLOCK


def test_build_architecture_unknown():
    import pytest

    from openwam.model import build_architecture

    with pytest.raises(KeyError, match="Unknown architecture"):
        build_architecture("nonexistent", {})


def test_dual_system_prepare_and_extract():
    """Smoke test: prepare action tokens and extract prediction (cross_attn)."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "video_dim": 128,
        "bridge_layers": (0, 1),
    }
    arch = build_architecture("dual_system_cross_attn", cfg)
    arch.eval()

    B, T_action = 1, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([500.0])

    state = arch.action_backbone.prepare_state(noisy_actions, timestep)
    assert state.action_latents is not None
    assert state.bridge_features == []

    # Simulate two DiT blocks producing video hidden states (cross_attn collects
    # features at every bridge_layer; ordering matches sorted bridge_layers).
    for _ in arch.action_backbone.bridge_layers:
        state.bridge_features.append(torch.randn(B, 20, 128))

    # Extract action prediction
    with torch.no_grad():
        action_pred = arch.action_backbone.extract_prediction(state)
    assert action_pred.shape == (B, T_action, 7)


def test_shared_backbone_moe_prepare_and_extract():
    """Smoke test: shared_backbone moe variant prepare action tokens, expert FFN, and extract prediction."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "num_experts": 2,
        "bridge_layers": (0, 1),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    arch.eval()

    B, T_action, T_video = 1, 10, 20

    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([500.0])

    state = arch.action_backbone.prepare_state(noisy_actions, timestep)
    assert state.runtime_state is not None

    moe_state = state.runtime_state.payload
    assert moe_state.action_tokens.shape == (B, T_action, 128)
    assert moe_state.n_action_tokens == T_action

    # Drive run_block with a mock video state where action tokens sit at the
    # tail of vstate.x (matching what before_loop would inject).
    class _MockVState:
        def __init__(self, x):
            self.x = x

    vstate = _MockVState(torch.cat([torch.randn(B, T_video, 128), moe_state.action_tokens], dim=1))

    class _MockVB:
        def run_block(self, _bid, vs):
            return vs

    vb = _MockVB()
    ab = arch.action_backbone
    for block_id in range(2):
        vstate, state = ab.run_block(block_id, vb, vstate, state)

    assert moe_state.expert_block_counter == 2

    # Extract action prediction
    with torch.no_grad():
        action_pred = ab.extract_prediction(state)
    assert action_pred.shape == (B, T_action, 7)


def test_shared_backbone_prepare_and_extract():
    """Smoke test: shared backbone prepare, run_block (no-op), and extract."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 10}
    arch = build_architecture("shared_backbone_vanilla", cfg)
    arch.eval()

    B, T_action, T_video = 1, 10, 20

    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([500.0])

    state = arch.action_backbone.prepare_state(noisy_actions, timestep)
    assert state.action_latents.shape == (B, T_action, 128)
    assert state.num_action_tokens == T_action

    # SharedBackbone vanilla: per-block is no-op; extract uses final_hidden.
    state.final_hidden = torch.randn(B, T_video + T_action, 128)

    with torch.no_grad():
        action_pred = arch.action_backbone.extract_prediction(state)
    assert action_pred.shape == (B, T_action, 7)


def test_shared_backbone_output_head_init():
    """SharedBackbone output head (ActionOutputMLP) uses small-random init."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5}
    arch = build_architecture("shared_backbone_vanilla", cfg)
    head = arch.action_backbone.action_output_head
    assert torch.all(head.layer1.bias == 0)
    assert torch.all(head.layer2.bias == 0)
    for w in (head.layer1.weight, head.layer2.weight):
        assert not torch.all(w == 0), "weights should be small-random, not zero"
        assert w.abs().max() < 0.2, "weights should be small (std ~ 0.02)"


def test_moe_expert_ffn_and_output_head_init():
    """MoE: expert FFN output stays zero-init; action output head uses small-random init."""
    from openwam.model.action_backbone.moe_dit import MoEExpertDiT

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
    """bridge_layers: null + bridge_interval resolves from injected num_dit_layers."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": None,
        "bridge_interval": 2,
        "num_dit_layers": 30,
    }
    arch = build_architecture("dual_system_cross_attn", cfg)
    assert arch.bridge_layers == tuple(range(0, 30, 2))
    assert len(arch.bridge_layers) == 15

    cfg["bridge_interval"] = 1
    arch_full = build_architecture("dual_system_cross_attn", cfg)
    assert arch_full.bridge_layers == tuple(range(30))


def test_dual_system_bridge_interval_missing_raises():
    """bridge_layers: null without bridge_interval should fail explicitly."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": None,
    }
    with pytest.raises(ValueError, match="bridge_layers is null but bridge_interval is not set"):
        build_architecture("dual_system_cross_attn", cfg)


def test_action_self_attention_rope_breaks_permutation_equivariance():
    """Without positional info, self-attention is permutation-equivariant.
    RoPE injects absolute position into Q/K, so permuting the input tokens
    must NOT merely permute the output (the model distinguishes positions).
    Also: supplying RoPE freqs must change the output relative to freqs=None.
    """
    from openwam.model.action_backbone.action_dit import ActionSelfAttention
    from openwam.model.action_backbone.components import precompute_freqs_cis_1d

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


def test_action_dit_joint_cross_attn_detach_path_uses_rope():
    """ActionDiT joint_cross_attn with detach_bridge=True runs end-to-end with RoPE instead of
    LearnedPositionalEncoding.
    """
    from openwam.model.action_backbone.action_dit import ActionDiT

    dit = ActionDiT(
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=128,
        bridge_layers=(0, 1),
        variant="joint_cross_attn",
        detach_bridge=True,
    )
    assert dit.pos_encoding is None  # RoPE path
    assert hasattr(dit, "freqs") and dit.freqs.shape == (1024, 64 // 4 // 2)

    actions = torch.randn(2, 5, 7)
    video_features = [torch.randn(2, 10, 128) for _ in range(2)]
    timestep = torch.tensor([0.5, 0.8])
    out = dit(actions, video_features, timestep)
    assert out.shape == (2, 5, 7)


def test_action_dit_joint_self_attn_uses_only_rope():
    """joint_self_attn relies solely on RoPE (no learned absolute PE)."""
    from openwam.model.action_backbone.action_dit import ActionDiT

    dit = ActionDiT(
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=64,  # match dim so no video_projs Linear mismatch
        bridge_layers=(0, 1),
        variant="joint_self_attn",
    )
    assert dit.pos_encoding is None

    state = dit.prepare_action_state(torch.randn(2, 5, 7), torch.tensor([0.5, 0.8]))
    assert state.use_joint_rope is True


def test_dual_system_joint_self_attn_production_path_applies_rope():
    """Interleaved joint_self_attn path should wire RoPE into production blocks."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "bridge_layers": (0,),
    }
    arch = build_architecture("dual_system_self_attn", cfg)
    arch.eval()

    captured = {}
    original_forward = arch.action_backbone.blocks[0].joint_attn.forward

    def wrapped_forward(x_action, x_video, freqs_action=None, freqs_video=None):
        captured["freqs_action"] = freqs_action
        captured["freqs_video"] = freqs_video
        return original_forward(
            x_action,
            x_video,
            freqs_action=freqs_action,
            freqs_video=freqs_video,
        )

    arch.action_backbone.blocks[0].joint_attn.forward = wrapped_forward
    try:
        noisy_actions = torch.randn(2, 5, 7)
        timestep = torch.tensor([0.5, 0.8])
        state = arch.action_backbone.prepare_state(noisy_actions, timestep)

        class _MockVState:
            def __init__(self, x, t_mod, f, h, w):
                self.x = x
                self.reference_prefix_len = 0
                self.t_mod = t_mod
                self.f = f
                self.h = h
                self.w = w

        class _MockVB:
            def run_block(self, _bid, vs):
                return vs

        # 9 video tokens treated as 1D temporal axis.
        vstate = _MockVState(torch.randn(2, 9, 32), t_mod=torch.zeros(2, 6, 32), f=9, h=1, w=1)
        _, state = arch.action_backbone.run_block(0, _MockVB(), vstate, state)
        with torch.no_grad():
            action_pred = arch.action_backbone.extract_prediction(state)
    finally:
        arch.action_backbone.blocks[0].joint_attn.forward = original_forward

    assert action_pred.shape == (2, 5, 7)
    assert captured["freqs_action"] is not None
    assert captured["freqs_video"] is not None
    assert captured["freqs_action"].shape[0] == state.runtime_state.payload.x_action.shape[1]
    assert captured["freqs_video"].shape[0] == state.runtime_state.payload.x_video_proj.shape[1]


def test_dual_system_joint_cross_attn_not_interleaved_and_collects_bridge_features():
    """Joint cross-attn stays in bridge-collection mode and records features."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": (0, 2),
    }
    arch = build_architecture("dual_system_cross_attn", cfg)
    assert arch.execution_plan == ExecutionPlan.BRIDGE_COLLECTION

    state = arch.action_backbone.prepare_state(torch.randn(1, 5, 7), torch.tensor([0.5]))
    assert state.bridge_features == []
    assert state.runtime_state.payload is state

    class _MockVState:
        def __init__(self, x):
            self.x = x

    class _MockVB:
        def run_block(self, _bid, vs):
            return vs

    vstate = _MockVState(torch.randn(1, 9, 128))
    vb = _MockVB()
    for block_id in range(3):
        vstate, state = arch.action_backbone.run_block(block_id, vb, vstate, state)
    assert len(state.bridge_features) == 2


def test_dual_system_detached_joint_cross_attn_not_interleaved_and_collects_bridge_features():
    """Detached joint cross-attn shares the same structure and still uses bridge collection."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": (1,),
    }
    arch = build_architecture("dual_system_cross_attn", cfg)
    assert arch.execution_plan == ExecutionPlan.BRIDGE_COLLECTION

    state = arch.action_backbone.prepare_state(torch.randn(1, 5, 7), torch.tensor([0.5]))
    assert state.bridge_features == []
    assert state.runtime_state.payload is state


def test_dual_system_joint_self_attn_execution_plan_and_creates_dit_state():
    """joint_self_attn switches DualSystem into interleaved mode."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "bridge_layers": (0, 1),
    }
    arch = build_architecture("dual_system_self_attn", cfg)
    assert arch.execution_plan == ExecutionPlan.INTERLEAVED_SPLIT_SELF_ATTENTION

    state = arch.action_backbone.prepare_state(torch.randn(1, 5, 7), torch.tensor([0.5]))
    assert state.runtime_state is not None
    assert state.runtime_state.variant == "joint_self_attn"
    assert state.bridge_features == []


def test_moe_execution_plan_and_uses_expert_layers_as_bridge_layers():
    """SharedBackbone moe variant remains interleaved and exposes expert_layers via bridge_layers."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "num_experts": 2,
        "bridge_layers": (1, 3),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    assert arch.execution_plan == ExecutionPlan.INTERLEAVED_SPLIT_FFN
    assert arch.bridge_layers == (1, 3)

    state = arch.action_backbone.prepare_state(torch.randn(1, 5, 7), torch.tensor([0.5]))
    assert state.runtime_state is not None
    assert state.runtime_state.variant == "moe"


def test_shared_backbone_execution_plan_and_has_no_bridge_layers():
    """SharedBackbone is interleaved but has no explicit bridge layers."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 5}
    arch = build_architecture("shared_backbone_vanilla", cfg)
    assert arch.execution_plan == ExecutionPlan.INTERLEAVED_WHOLE_BLOCK
    assert arch.bridge_layers == ()


def test_normalize_architecture_spec_shared_backbone():
    from openwam.model.architectures.registry import normalize_architecture_spec

    spec = normalize_architecture_spec("shared_backbone_vanilla", {"action_dim": 7})
    assert spec.framework == "shared_backbone"
    assert spec.variant == "vanilla"
    assert spec.options == {}


def test_normalize_architecture_spec_shared_backbone_moe():
    from openwam.model.architectures.registry import normalize_architecture_spec

    spec = normalize_architecture_spec("shared_backbone_moe", {"action_dim": 7})
    assert spec.framework == "shared_backbone"
    assert spec.variant == "moe"
    assert spec.options == {}


def test_normalize_architecture_spec_dual_system_joint_cross_attn_detach_false():
    from openwam.model.architectures.registry import normalize_architecture_spec

    spec = normalize_architecture_spec("dual_system_cross_attn", {"detach_bridge": False})
    assert spec.framework == "dual_system"
    assert spec.variant == "joint_cross_attn"
    assert spec.options == {"detach_bridge": False}


def test_normalize_architecture_spec_dual_system_joint_cross_attn_detach_true():
    from openwam.model.architectures.registry import normalize_architecture_spec

    spec = normalize_architecture_spec("dual_system_cross_attn", {"detach_bridge": True})
    assert spec.framework == "dual_system"
    assert spec.variant == "joint_cross_attn"
    assert spec.options == {"detach_bridge": True}


def test_normalize_architecture_spec_dual_system_joint_self_attn():
    from openwam.model.architectures.registry import normalize_architecture_spec

    spec = normalize_architecture_spec("dual_system_self_attn")
    assert spec.framework == "dual_system"
    assert spec.variant == "joint_self_attn"
    assert spec.options == {}


def test_resolve_architecture_config_from_canonical_dual_system_fields():
    from types import SimpleNamespace

    from openwam.model.registry import resolve_architecture_config

    model_cfg = SimpleNamespace(
        architecture={
            "framework": "dual_system",
            "variant": "joint_cross_attn",
            "detach_bridge": True,
            "action_dim": 20,
        },
        action_backbone={"dim": 128, "num_heads": 4},
    )
    resolved = resolve_architecture_config(model_cfg, video_dim=256)

    assert resolved.registry_name == "dual_system_cross_attn"
    assert resolved.canonical.framework == "dual_system"
    assert resolved.canonical.variant == "joint_cross_attn"
    assert resolved.params["detach_bridge"] is True
    assert resolved.params["video_dim"] == 256
    assert resolved.params["dim"] == 128


def test_resolve_architecture_config_from_canonical_fields():
    from types import SimpleNamespace

    from openwam.model.registry import resolve_architecture_config

    model_cfg = SimpleNamespace(
        architecture={
            "framework": "shared_backbone",
            "variant": "moe",
            "action_dim": 20,
            "expert_ffn_dim": 512,
        },
        action_backbone={},
    )
    resolved = resolve_architecture_config(model_cfg, video_dim=192)

    assert resolved.registry_name == "shared_backbone_moe"
    assert resolved.canonical.framework == "shared_backbone"
    assert resolved.canonical.variant == "moe"
    assert resolved.params["framework"] == "shared_backbone"
    assert resolved.params["variant"] == "moe"
    assert resolved.params["video_dim"] == 192


def test_build_architecture_injects_framework_and_variant_for_shared_backbone():
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5}
    arch = build_architecture("shared_backbone_vanilla", cfg)
    assert arch.cfg["framework"] == "shared_backbone"
    assert arch.cfg["variant"] == "vanilla"


def test_build_architecture_injects_framework_variant_and_detach_option():
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": (0, 1),
    }
    arch = build_architecture("dual_system_cross_attn", cfg)
    assert arch.cfg["framework"] == "dual_system"
    assert arch.cfg["variant"] == "joint_cross_attn"
    assert arch.cfg["detach_bridge"] is True


def test_build_architecture_shared_backbone_moe_canonical_config():
    from openwam.model import build_architecture

    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "num_experts": 2,
        "bridge_layers": (1, 3),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    assert arch.cfg["framework"] == "shared_backbone"
    assert arch.cfg["variant"] == "moe"


def test_dual_system_runtime_state_cross_attn():
    from openwam.model import build_architecture
    from openwam.model.architectures.base import ExecutionPlan

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 7,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "video_dim": 128,
        "bridge_layers": (0, 1),
    }
    arch = build_architecture("dual_system_cross_attn", cfg)
    state = arch.action_backbone.prepare_state(torch.randn(1, 5, 7), torch.tensor([0.5]))
    assert state.runtime_state is not None
    assert state.runtime_state.framework == "dual_system"
    assert state.runtime_state.variant == "joint_cross_attn"
    assert state.runtime_state.execution_plan == ExecutionPlan.BRIDGE_COLLECTION
    assert state.runtime_state.options == {"detach_bridge": False}


def test_dual_system_runtime_state_joint_self_attn():
    from openwam.model import build_architecture
    from openwam.model.action_backbone.action_dit import ActionDiTState
    from openwam.model.architectures.base import ExecutionPlan

    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "bridge_layers": (0, 1),
    }
    arch = build_architecture("dual_system_self_attn", cfg)
    state = arch.action_backbone.prepare_state(torch.randn(1, 5, 7), torch.tensor([0.5]))
    assert state.runtime_state is not None
    assert state.runtime_state.framework == "dual_system"
    assert state.runtime_state.variant == "joint_self_attn"
    assert state.runtime_state.execution_plan == ExecutionPlan.INTERLEAVED_SPLIT_SELF_ATTENTION
    assert isinstance(state.runtime_state.payload, ActionDiTState)


def test_shared_backbone_runtime_state_typed():
    from openwam.model import build_architecture
    from openwam.model.action_backbone.shared_vanilla import SharedVanillaState
    from openwam.model.architectures.base import ExecutionPlan

    arch = build_architecture("shared_backbone_vanilla", {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5})
    state = arch.action_backbone.prepare_state(torch.randn(1, 5, 7), torch.tensor([0.5]))
    assert state.runtime_state is not None
    assert state.runtime_state.framework == "shared_backbone"
    assert state.runtime_state.variant == "vanilla"
    assert state.runtime_state.execution_plan == ExecutionPlan.INTERLEAVED_WHOLE_BLOCK
    assert isinstance(state.runtime_state.payload, SharedVanillaState)


def test_moe_runtime_state_typed():
    from openwam.model import build_architecture
    from openwam.model.action_backbone.moe_dit import MoEExpertState
    from openwam.model.architectures.base import ExecutionPlan

    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "num_experts": 2,
        "bridge_layers": (1, 3),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    state = arch.action_backbone.prepare_state(torch.randn(1, 5, 7), torch.tensor([0.5]))
    assert state.runtime_state is not None
    assert state.runtime_state.framework == "shared_backbone"
    assert state.runtime_state.variant == "moe"
    assert state.runtime_state.execution_plan == ExecutionPlan.INTERLEAVED_SPLIT_FFN
    assert isinstance(state.runtime_state.payload, MoEExpertState)


def test_register_custom_architecture():
    """Verify that custom architectures can be registered."""
    from openwam.model.architectures.base import ActionState, BaseWAMArchitecture
    from openwam.model.architectures.registry import ARCHITECTURE_METADATA, ARCHITECTURE_REGISTRY, register_architecture

    @register_architecture("test_custom", framework="test", variant="custom")
    class TestArch(BaseWAMArchitecture):
        def prepare_action_tokens(self, noisy_actions, timestep, **_kw):
            return ActionState(action_latents=noisy_actions, timestep=timestep)

        def on_dit_block(self, _block_id, video_hidden, action_state):
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
    assert "test_custom" in ARCHITECTURE_METADATA

    # Cleanup
    del ARCHITECTURE_REGISTRY["test_custom"]
    ARCHITECTURE_METADATA.pop("test_custom", None)
