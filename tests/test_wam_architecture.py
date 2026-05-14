"""Tests for WAM Architecture registry and implementations."""

import sys

import pytest
import torch


def _action_context(ab, batch_size: int, seq_len: int = 4):
    return torch.randn(batch_size, seq_len, ab.text_dim), torch.ones(batch_size, seq_len, dtype=torch.bool)


def _make_dual_system_self_attn_mot_fixture():
    from openwam.model import build_architecture
    from tests.test_openwam_trainer import _MockVideoBackbone

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
    arch.video_backbone = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    driver = arch.build_mot_driver()
    arch.eval()
    return arch, driver


def _make_dual_system_cross_attn_fixture():
    from openwam.model import build_architecture

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 48,
        "bridge_layers": (0, 1),
        "text_dim": 16,
    }
    arch = build_architecture("dual_system_cross_attn", cfg)
    arch.eval()
    return arch


def _dual_system_self_attn_mot_states(arch, seed: int):
    from openwam.model.video_backbone.adapter import BlockLoopState

    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(1, 3, 7, generator=g)
    context = torch.randn(1, 4, arch.action_backbone.text_dim, generator=g)
    context_mask = torch.ones(1, 4, dtype=torch.bool)
    astate = arch.action_backbone.prepare_state(
        actions,
        torch.tensor([0.5]),
        context=context,
        context_mask=context_mask,
    )
    vstate = BlockLoopState(
        x=torch.randn(1, 4, 32, generator=g),
        t_mod=torch.zeros(1, 6, 32),
        freqs=torch.zeros(4, 1, 1),
        context=torch.randn(1, 4, 32, generator=g),
        context_mask=torch.ones(1, 4, dtype=torch.bool),
        f=4,
        h=1,
        w=1,
        extras={},
    )
    return vstate, astate


def _dual_system_cross_attn_inputs(arch, seed: int):
    g = torch.Generator().manual_seed(seed)
    actions = torch.randn(1, 3, 7, generator=g)
    timestep = torch.tensor([0.5])
    bridges = {bid: torch.randn(1, 4, 48, generator=g) for bid in arch.action_backbone.bridge_layers}
    context = torch.randn(1, 4, arch.action_backbone.text_dim, generator=g)
    context_mask = torch.ones(1, 4, dtype=torch.bool)
    return actions, bridges, timestep, context, context_mask


def test_architecture_module_layout_imports():
    from openwam.model.architectures.dual_system import (
        DualSystemCrossAttnArchitecture,
        DualSystemIDMArchitecture,
        DualSystemSelfAttnArchitecture,
    )
    from openwam.model.architectures.shared_backbone.moe import SharedBackboneMoEArchitecture
    from openwam.model.architectures.shared_backbone.vanilla import SharedBackboneVanillaArchitecture

    assert DualSystemCrossAttnArchitecture is not None
    assert DualSystemIDMArchitecture is not None
    assert DualSystemSelfAttnArchitecture is not None
    assert SharedBackboneMoEArchitecture is not None
    assert SharedBackboneVanillaArchitecture is not None


def test_architecture_state_types_import():
    from openwam.model.action_backbone.dualsystem_dit import ActionDiTState

    assert ActionDiTState is not None


def test_architecture_support_lists():
    from openwam.model import (
        get_architecture_support,
        list_supported_architectures,
    )

    supported = list_supported_architectures()
    assert "dual_system_cross_attn" in supported
    assert "dual_system_self_attn" in supported
    assert "dual_system_idm" in supported
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
        "expert_layers": (1, 3),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    assert arch.action_dim == 7
    assert arch.expert_layers == (1, 3)
    assert arch.action_backbone is not None
    assert len(arch.action_backbone.expert_blocks) == 2


def test_build_architecture_shared():
    """Shared backbone vanilla should build successfully."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 10}
    arch = build_architecture("shared_backbone_vanilla", cfg)
    assert arch.action_dim == 7
    assert arch.expert_layers == ()


def test_build_architecture_unknown():
    import pytest

    from openwam.model import build_architecture

    with pytest.raises(KeyError, match="Unknown architecture"):
        build_architecture("nonexistent", {})


def test_dual_system_prepare_and_extract():
    """Smoke test: cross_attn ActionDiT.forward(action, bridges, timestep, context)."""
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

    bridges = {bid: torch.randn(B, 20, 128) for bid in arch.action_backbone.bridge_layers}
    context, context_mask = _action_context(arch.action_backbone, B)

    with torch.no_grad():
        action_pred = arch.action_backbone(
            noisy_actions,
            bridges,
            timestep,
            context=context,
            context_mask=context_mask,
        )
    assert action_pred.shape == (B, T_action, 7)


def test_shared_backbone_moe_encode_apply_decode():
    """Smoke test: MoE encode → apply_expert at expert layers → decode."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "expert_layers": (0, 1),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    arch.eval()
    ab = arch.action_backbone

    B, T_action = 1, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([500.0])

    tokens, t_mod = ab.encode(noisy_actions, timestep)
    assert tokens.shape == (B, T_action, 128)

    # Apply expert at each expert layer; output shape preserved.
    x_action = tokens
    for block_id in ab.expert_layers:
        x_action = ab.apply_expert(block_id, x_action, t_mod)
    assert x_action.shape == (B, T_action, 128)

    with torch.no_grad():
        action_pred = ab.decode(x_action)
    assert action_pred.shape == (B, T_action, 7)


def test_shared_backbone_encode_decode():
    """Smoke test: vanilla encode + decode shapes."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 10}
    arch = build_architecture("shared_backbone_vanilla", cfg)
    arch.eval()
    ab = arch.action_backbone

    B, T_action = 1, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([500.0])
    tokens = ab.encode(noisy_actions, timestep)
    assert tokens.shape == (B, T_action, 128)

    with torch.no_grad():
        action_pred = ab.decode(tokens)
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
    from openwam.model.action_backbone.shared_moe import SharedMoEActionBackbone

    dit = SharedMoEActionBackbone(
        action_dim=7,
        video_dim=64,
        expert_ffn_dim=128,
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
    from openwam.model.action_backbone.components import precompute_freqs_cis_1d
    from openwam.model.action_backbone.dualsystem_dit import ActionSelfAttention

    dim, num_heads, seq = 32, 4, 4
    head_dim = dim // num_heads
    attn = ActionSelfAttention(hidden_dim=dim, num_heads=num_heads, attn_head_dim=head_dim).eval()

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


def test_action_flash_attention_backend_falls_back_to_sdpa_on_cpu(monkeypatch):
    """Flash-style ActionDiT backends are CUDA-only; CPU unit tests need SDPA."""
    import types

    from openwam.model.action_backbone import components

    def _cuda_only_flash_attn(*args, **kwargs):
        raise AssertionError("flash_attn_func should not receive CPU tensors")

    monkeypatch.setitem(
        sys.modules,
        "flash_attn",
        types.SimpleNamespace(flash_attn_func=_cuda_only_flash_attn),
    )

    fn = components._try_flash_attn_2()
    assert fn is not None
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    out = fn(q, k, v)
    assert out.shape == q.shape


def test_action_dit_joint_cross_attn_uses_rope():
    """ActionDiT joint_cross_attn runs end-to-end with RoPE (no learned absolute PE)."""
    from openwam.model.action_backbone.dualsystem_dit import ActionDiT

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
    assert not hasattr(dit, "pos_encoding")
    assert hasattr(dit, "freqs") and dit.freqs.shape == (1024, 64 // 4 // 2)

    actions = torch.randn(2, 5, 7)
    bridges = {bid: torch.randn(2, 10, 128) for bid in (0, 1)}
    timestep = torch.tensor([0.5, 0.8])
    out = dit(actions, bridges, timestep)
    assert out.shape == (2, 5, 7)


def test_action_dit_joint_self_attn_uses_only_rope():
    """joint_self_attn relies solely on RoPE (no learned absolute PE)."""
    from openwam.model.action_backbone.dualsystem_dit import ActionDiT

    dit = ActionDiT(
        action_dim=7,
        dim=64,
        ffn_dim=128,
        num_heads=4,
        num_layers=2,
        video_dim=64,
        bridge_layers=(0, 1),
        variant="joint_self_attn",
    )
    assert not hasattr(dit, "pos_encoding")

    context, context_mask = _action_context(dit, 2)
    astate = dit.prepare_state(
        torch.randn(2, 5, 7), torch.tensor([0.5, 0.8]), context=context, context_mask=context_mask
    )
    payload = astate.payload
    # RoPE freqs on the action stream are pre-computed and stashed for
    # consumption by pre_attn_at_layer; their length equals the (proprio +
    # action) prefix that the attention sees.
    assert payload.action_freqs is not None
    assert payload.action_freqs.shape[0] == payload.x_action.shape[1]
    assert payload.action_freqs.device == payload.x_action.device


def test_dual_system_joint_self_attn_pre_post_attn_round_trip():
    """joint_self_attn pre/post_attn_at_layer round trip applies RoPE on Q/K."""
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

    noisy_actions = torch.randn(2, 5, 7)
    timestep = torch.tensor([0.5, 0.8])
    context, context_mask = _action_context(arch.action_backbone, 2)
    astate = arch.action_backbone.prepare_state(noisy_actions, timestep, context=context, context_mask=context_mask)

    q, k, v, post = arch.action_backbone.pre_attn_at_layer(0, astate)
    # Q/K go through RoPE (different from V even when the same input feeds q/k/v):
    assert not torch.allclose(q, v)
    assert not torch.allclose(k, v)
    # Q/K/V are in (B, S, H*D) layout, ready for the driver to concat with
    # the video stream.
    B, S = noisy_actions.shape[0], noisy_actions.shape[1]
    assert q.shape == (B, S, arch.action_backbone.num_heads * arch.action_backbone.head_dim)

    # Round-trip through post_attn so we know the slot is wired.
    astate2 = arch.action_backbone.post_attn_at_layer(0, astate, torch.randn_like(q), post)
    assert astate2 is astate
    pred = arch.action_backbone.extract_prediction(astate2)
    assert pred.shape == (2, 5, 7)


def test_dual_system_joint_cross_attn_no_per_layer_state():
    """joint_cross_attn skips prepare_state and runs ActionDiT.forward directly."""
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

    # cross_attn calls ab.forward(actions, bridges, timestep) directly —
    # no per-layer state machinery on the action backbone.
    bridges = {bid: torch.randn(1, 9, 128) for bid in arch.action_backbone.bridge_layers}
    context, context_mask = _action_context(arch.action_backbone, 1)
    with torch.no_grad():
        out = arch.action_backbone(
            torch.randn(1, 5, 7),
            bridges,
            torch.tensor([0.5]),
            context=context,
            context_mask=context_mask,
        )
    assert out.shape == (1, 5, 7)


def test_dual_system_joint_cross_attn_passes_appended_proprio_context_to_action():
    """cross_attn architecture should pass raw text+proprio context to ActionDiT."""
    from openwam.model import build_architecture
    from tests.test_openwam_trainer import _MockVideoBackbone

    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "bridge_layers": (0,),
        "use_proprioception": True,
        "state_dim": 7,
        "text_dim": 16,
    }
    arch = build_architecture("dual_system_cross_attn", cfg)
    arch.video_backbone = _MockVideoBackbone(dim=32, num_layers=1, num_heads=4)
    arch.eval()

    seen = {}
    orig_forward = arch.action_backbone.forward

    def _capture(*args, **kwargs):
        seen["context"] = kwargs.get("context")
        seen["context_mask"] = kwargs.get("context_mask")
        return orig_forward(*args, **kwargs)

    arch.action_backbone.forward = _capture

    B = 1
    with torch.no_grad():
        video_pred, action_pred = arch(
            torch.randn(B, 5, 7),
            torch.tensor([0.5]),
            proprio_state=torch.randn(B, 7),
            latents=torch.randn(B, 16, 1, 2, 2),
            timestep=torch.tensor([0.5]),
            context=torch.randn(B, 3, 16),
            seq_lens=torch.tensor([2]),
        )

    assert video_pred.shape[0] == B
    assert action_pred.shape == (B, 5, 7)
    assert seen["context"].shape == (B, 4, 16)
    assert seen["context_mask"].shape == (B, 4)
    assert seen["context_mask"].tolist() == [[True, True, False, True]]


def test_dual_system_detached_joint_cross_attn_blocks_grad_to_video():
    """detach_bridge=True: action loss must not produce grad on the bridge tensors."""
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
        "bridge_layers": (0,),
    }
    arch = build_architecture("dual_system_cross_attn", cfg)
    # detach is owned by the architecture; the action backbone has no such flag.
    assert arch.detach_bridge is True
    assert not hasattr(arch.action_backbone, "detach_bridge")


def test_action_dit_cross_attn_bridge_tuple_matches_dict():
    """Ordered bridge tuple path should preserve the existing dict-path result."""

    arch = _make_dual_system_cross_attn_fixture()
    ab = arch.action_backbone
    actions, bridges, timestep, context, context_mask = _dual_system_cross_attn_inputs(arch, 7)
    bridge_tuple = ab.bridge_tuple_from_dict(bridges)

    with torch.no_grad():
        dict_out = ab(actions, bridges, timestep, context=context, context_mask=context_mask)
        tuple_out = ab.forward_with_bridge_tuple(
            actions,
            bridge_tuple,
            timestep,
            context=context,
            context_mask=context_mask,
        )

    assert torch.allclose(tuple_out, dict_out, atol=1e-6)


def test_dual_system_cross_attn_compile_helper_matches_eager(monkeypatch):
    """auto compile mode should select the cross-attn action-side tuple path."""
    from omegaconf import OmegaConf

    arch = _make_dual_system_cross_attn_fixture()
    ab = arch.action_backbone
    actions, bridges, timestep, context, context_mask = _dual_system_cross_attn_inputs(arch, 13)

    arch.apply_compile_optimizations(
        OmegaConf.create(
            {
                "mode": "auto",
                "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )
    )
    compiled_action = arch._compiled_cross_attn_action
    assert compiled_action is not None

    compile_calls = []

    def _fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)

    with torch.no_grad():
        eager_out = ab(actions, bridges, timestep, context=context, context_mask=context_mask)
        compiled_out = compiled_action.run(actions, bridges, timestep, context=context, context_mask=context_mask)

    assert compile_calls == [{"dynamic": False, "mode": "reduce-overhead"}]
    assert torch.allclose(compiled_out, eager_out, atol=1e-6)


def test_dual_system_cross_attn_mode_none_stays_eager(monkeypatch):
    """mode=none must not activate the cross-attn action helper."""
    from omegaconf import OmegaConf

    arch = _make_dual_system_cross_attn_fixture()
    compile_calls = []

    def _fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)

    arch.apply_compile_optimizations(OmegaConf.create({"mode": "none"}))
    assert arch._compiled_cross_attn_action is None

    assert compile_calls == []


def test_dual_system_cross_attn_auto_mode_enables_helper():
    """mode=auto should infer cross_attn for dual_system_cross_attn checkpoints."""
    from omegaconf import OmegaConf

    arch = _make_dual_system_cross_attn_fixture()

    arch.apply_compile_optimizations(
        OmegaConf.create(
            {
                "mode": "auto",
                "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )
    )

    assert arch._compiled_cross_attn_action is not None


def test_dual_system_cross_attn_compile_failure_falls_back_to_eager(monkeypatch):
    """Runtime compile failures should disable the cross-attn fast path and continue eager."""
    from omegaconf import OmegaConf

    arch = _make_dual_system_cross_attn_fixture()
    ab = arch.action_backbone
    actions, bridges, timestep, context, context_mask = _dual_system_cross_attn_inputs(arch, 17)

    arch.apply_compile_optimizations(
        OmegaConf.create(
            {
                "mode": "auto",
                "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )
    )
    compiled_action = arch._compiled_cross_attn_action
    assert compiled_action is not None

    compile_calls = []

    def _fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)

        def _broken(*args, **kwargs):
            raise RuntimeError("inductor unavailable")

        return _broken

    monkeypatch.setattr(torch, "compile", _fake_compile)

    with torch.no_grad():
        eager_out = ab(actions, bridges, timestep, context=context, context_mask=context_mask)
        fallback_out = compiled_action.run(actions, bridges, timestep, context=context, context_mask=context_mask)

    assert compile_calls == [{"dynamic": False, "mode": "reduce-overhead"}]
    assert compiled_action._compile_disabled is True
    assert torch.allclose(fallback_out, eager_out, atol=1e-6)


def test_dual_system_cross_attn_bad_request_does_not_disable_compile(monkeypatch):
    """Input errors should preserve eager semantics without poisoning later cross-attn compile calls."""
    from omegaconf import OmegaConf

    arch = _make_dual_system_cross_attn_fixture()
    ab = arch.action_backbone

    arch.apply_compile_optimizations(
        OmegaConf.create(
            {
                "mode": "auto",
                "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )
    )
    compiled_action = arch._compiled_cross_attn_action
    assert compiled_action is not None

    compile_calls = []
    compiled_invocations = 0

    def _fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)

        def _wrapped(*args, **kwargs):
            nonlocal compiled_invocations
            compiled_invocations += 1
            return fn(*args, **kwargs)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    actions_ok, bridges_ok, timestep_ok, context_ok, context_mask_ok = _dual_system_cross_attn_inputs(arch, 19)
    with torch.no_grad():
        compiled_action.run(actions_ok, bridges_ok, timestep_ok, context=context_ok, context_mask=context_mask_ok)

    actions_bad, bridges_bad, timestep_bad, context_bad, context_mask_bad = _dual_system_cross_attn_inputs(arch, 23)
    first_bridge = ab.bridge_layers[0]
    bridges_bad[first_bridge] = bridges_bad[first_bridge].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="bridge dtype mismatch"):
        with torch.no_grad():
            compiled_action.run(
                actions_bad,
                bridges_bad,
                timestep_bad,
                context=context_bad,
                context_mask=context_mask_bad,
            )

    assert compiled_action._compile_disabled is False

    actions_next, bridges_next, timestep_next, context_next, context_mask_next = _dual_system_cross_attn_inputs(arch, 29)
    with torch.no_grad():
        eager_out = ab(actions_next, bridges_next, timestep_next, context=context_next, context_mask=context_mask_next)
        compiled_out = compiled_action.run(
            actions_next,
            bridges_next,
            timestep_next,
            context=context_next,
            context_mask=context_mask_next,
        )

    assert compile_calls == [
        {"dynamic": False, "mode": "reduce-overhead"},
        {"dynamic": False, "mode": "reduce-overhead"},
    ]
    assert compiled_invocations == 3
    assert compiled_action._compile_disabled is False
    assert torch.allclose(compiled_out, eager_out, atol=1e-6)


def test_dual_system_joint_self_attn_creates_dit_state():
    """joint_self_attn populates ActionDiTState payload via prepare_state."""
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

    context, context_mask = _action_context(arch.action_backbone, 1)
    state = arch.action_backbone.prepare_state(
        torch.randn(1, 5, 7), torch.tensor([0.5]), context=context, context_mask=context_mask
    )
    payload = state.payload
    assert payload is not None
    assert payload.x_action.shape == (1, 5, 32)
    assert payload.action_freqs is not None


def test_dual_system_mot_loop_compile_helper_matches_eager(monkeypatch):
    """auto compile mode should preserve the eager joint-loop result."""
    from omegaconf import OmegaConf

    arch, driver = _make_dual_system_self_attn_mot_fixture()

    compile_path_calls = []

    def _v_pre_compile(layer_id, state):
        q, k, v, post = arch.video_backbone.pre_attn_at_layer(layer_id, state)
        compile_path_calls.append("v_pre")
        return q, k, v, (post["residual"],)

    def _v_post_compile(layer_id, state, attn_out, post_state):
        compile_path_calls.append("v_post")
        state.x = post_state[0] + attn_out
        return state

    arch.video_backbone.pre_attn_at_layer_for_compile = _v_pre_compile
    arch.video_backbone.post_attn_at_layer_for_compile = _v_post_compile

    vstate_eager, astate_eager = _dual_system_self_attn_mot_states(arch, 11)
    with torch.no_grad():
        vstate_eager, astate_eager = driver.run_joint_loop(vstate_eager, astate_eager)

    arch.apply_compile_optimizations(
        OmegaConf.create(
            {
                "mode": "auto",
                "self_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )
    )
    assert arch._compiled_mot_loop is not None

    compile_calls = []

    def _fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)

    vstate_compiled, astate_compiled = _dual_system_self_attn_mot_states(arch, 11)
    with torch.no_grad():
        vstate_compiled, astate_compiled = arch._compiled_mot_loop.run(vstate_compiled, astate_compiled)

    assert compile_calls == [{"dynamic": False, "mode": "reduce-overhead"}]
    assert compile_path_calls == ["v_pre", "v_post", "v_pre", "v_post"]
    assert torch.allclose(vstate_compiled.x, vstate_eager.x, atol=1e-6)
    assert torch.allclose(astate_compiled.payload.x_action, astate_eager.payload.x_action, atol=1e-6)


def test_dual_system_self_attn_compile_mode_none_disables_mot_loop():
    """mode=none should keep the self-attn architecture on the eager MoT loop."""
    from omegaconf import OmegaConf

    arch, _driver = _make_dual_system_self_attn_mot_fixture()

    arch.apply_compile_optimizations(
        OmegaConf.create(
            {
                "mode": "none",
                "self_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )
    )

    assert arch._compiled_mot_loop is None


def test_dual_system_self_attn_auto_mode_enables_mot_loop():
    """mode=auto should infer self_attn for dual_system_self_attn checkpoints."""
    from omegaconf import OmegaConf

    arch, _driver = _make_dual_system_self_attn_mot_fixture()

    arch.apply_compile_optimizations(
        OmegaConf.create(
            {
                "mode": "auto",
                "self_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )
    )

    assert arch._compiled_mot_loop is not None


def test_dual_system_self_attn_legacy_cross_attn_mode_is_rejected(monkeypatch):
    """Architecture-specific mode names are no longer public compile modes."""
    from omegaconf import OmegaConf

    arch, _driver = _make_dual_system_self_attn_mot_fixture()

    compile_calls = []

    def _fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)
    with pytest.raises(ValueError, match="Unknown compile mode"):
        arch.apply_compile_optimizations(
            OmegaConf.create(
                {
                    "mode": "cross_attn",
                    "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
                }
            )
        )

    assert arch._compiled_mot_loop is None
    assert compile_calls == []


def test_dual_system_mot_loop_compile_failure_falls_back_to_eager(monkeypatch):
    """Runtime torch.compile failures should disable the MoT fast path and continue eager."""
    from omegaconf import OmegaConf

    arch, driver = _make_dual_system_self_attn_mot_fixture()

    arch.apply_compile_optimizations(
        OmegaConf.create(
            {
                "mode": "auto",
                "self_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )
    )
    compiled_loop = arch._compiled_mot_loop
    assert compiled_loop is not None

    compile_calls = []

    def _fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)

        def _broken(*args, **kwargs):
            raise RuntimeError("inductor unavailable")

        return _broken

    monkeypatch.setattr(torch, "compile", _fake_compile)

    vstate_eager, astate_eager = _dual_system_self_attn_mot_states(arch, 17)
    with torch.no_grad():
        vstate_eager, astate_eager = driver.run_joint_loop(vstate_eager, astate_eager)

    vstate_fallback, astate_fallback = _dual_system_self_attn_mot_states(arch, 17)
    with torch.no_grad():
        vstate_fallback, astate_fallback = compiled_loop.run(vstate_fallback, astate_fallback)

    assert compile_calls == [{"dynamic": False, "mode": "reduce-overhead"}]
    assert compiled_loop._compile_disabled is True
    assert torch.allclose(vstate_fallback.x, vstate_eager.x, atol=1e-6)
    assert torch.allclose(astate_fallback.payload.x_action, astate_eager.payload.x_action, atol=1e-6)

    vstate_eager2, astate_eager2 = _dual_system_self_attn_mot_states(arch, 23)
    with torch.no_grad():
        vstate_eager2, astate_eager2 = driver.run_joint_loop(vstate_eager2, astate_eager2)

    vstate_fallback2, astate_fallback2 = _dual_system_self_attn_mot_states(arch, 23)
    with torch.no_grad():
        vstate_fallback2, astate_fallback2 = compiled_loop.run(vstate_fallback2, astate_fallback2)

    assert compile_calls == [{"dynamic": False, "mode": "reduce-overhead"}]
    assert torch.allclose(vstate_fallback2.x, vstate_eager2.x, atol=1e-6)
    assert torch.allclose(astate_fallback2.payload.x_action, astate_eager2.payload.x_action, atol=1e-6)


def test_dual_system_mot_loop_bad_request_does_not_disable_compile(monkeypatch):
    """Input errors should preserve eager semantics without poisoning future compile calls."""
    from omegaconf import OmegaConf

    arch, driver = _make_dual_system_self_attn_mot_fixture()

    arch.apply_compile_optimizations(
        OmegaConf.create(
            {
                "mode": "auto",
                "self_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )
    )
    compiled_loop = arch._compiled_mot_loop
    assert compiled_loop is not None

    compile_calls = []
    compiled_invocations = 0

    def _fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)

        def _wrapped(*args, **kwargs):
            nonlocal compiled_invocations
            compiled_invocations += 1
            return fn(*args, **kwargs)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    vstate_ok, astate_ok = _dual_system_self_attn_mot_states(arch, 29)
    with torch.no_grad():
        compiled_loop.run(vstate_ok, astate_ok)

    vstate_bad, astate_bad = _dual_system_self_attn_mot_states(arch, 31)
    vstate_bad.x = vstate_bad.x.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        with torch.no_grad():
            compiled_loop.run(vstate_bad, astate_bad)

    assert compiled_loop._compile_disabled is False

    vstate_eager, astate_eager = _dual_system_self_attn_mot_states(arch, 37)
    with torch.no_grad():
        vstate_eager, astate_eager = driver.run_joint_loop(vstate_eager, astate_eager)

    vstate_compiled, astate_compiled = _dual_system_self_attn_mot_states(arch, 37)
    with torch.no_grad():
        vstate_compiled, astate_compiled = compiled_loop.run(vstate_compiled, astate_compiled)

    assert compile_calls == [
        {"dynamic": False, "mode": "reduce-overhead"},
        {"dynamic": False, "mode": "reduce-overhead"},
    ]
    assert compiled_invocations == 3
    assert compiled_loop._compile_disabled is False
    assert torch.allclose(vstate_compiled.x, vstate_eager.x, atol=1e-6)
    assert torch.allclose(astate_compiled.payload.x_action, astate_eager.payload.x_action, atol=1e-6)


def test_dual_system_mot_loop_compile_setup_bad_request_does_not_disable_compile(monkeypatch):
    """Compile setup errors on invalid inputs should not poison later valid requests."""
    from omegaconf import OmegaConf

    arch, driver = _make_dual_system_self_attn_mot_fixture()

    arch.apply_compile_optimizations(
        OmegaConf.create(
            {
                "mode": "auto",
                "self_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )
    )
    compiled_loop = arch._compiled_mot_loop
    assert compiled_loop is not None

    compile_calls = 0

    def _fake_compile(fn, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls == 1:
            raise RuntimeError("compile setup unavailable")
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)

    vstate_bad, astate_bad = _dual_system_self_attn_mot_states(arch, 41)
    vstate_bad.x = vstate_bad.x.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        with torch.no_grad():
            compiled_loop.run(vstate_bad, astate_bad)

    assert compiled_loop._compile_disabled is False

    vstate_eager, astate_eager = _dual_system_self_attn_mot_states(arch, 43)
    with torch.no_grad():
        vstate_eager, astate_eager = driver.run_joint_loop(vstate_eager, astate_eager)

    vstate_compiled, astate_compiled = _dual_system_self_attn_mot_states(arch, 43)
    with torch.no_grad():
        vstate_compiled, astate_compiled = compiled_loop.run(vstate_compiled, astate_compiled)

    assert compile_calls == 2
    assert compiled_loop._compile_disabled is False
    assert torch.allclose(vstate_compiled.x, vstate_eager.x, atol=1e-6)
    assert torch.allclose(astate_compiled.payload.x_action, astate_eager.payload.x_action, atol=1e-6)


def test_moe_uses_expert_layers():
    """SharedBackbone moe variant exposes expert layer ids."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "expert_layers": (1, 3),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    assert arch.expert_layers == (1, 3)
    assert arch.action_backbone.expert_layers_set == {1, 3}


def test_shared_backbone_has_no_expert_layers():
    """SharedBackbone vanilla has no expert layers."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 5}
    arch = build_architecture("shared_backbone_vanilla", cfg)
    assert arch.expert_layers == ()


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
        "expert_layers": (1, 3),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    assert arch.cfg["framework"] == "shared_backbone"
    assert arch.cfg["variant"] == "moe"


def test_dual_system_self_attn_payload_is_action_dit_state():
    """joint_self_attn populates a flat payload of type ActionDiTState."""
    from openwam.model import build_architecture
    from openwam.model.action_backbone.dualsystem_dit import ActionDiTState

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
    context, context_mask = _action_context(arch.action_backbone, 1)
    state = arch.action_backbone.prepare_state(
        torch.randn(1, 5, 7), torch.tensor([0.5]), context=context, context_mask=context_mask
    )
    assert isinstance(state.payload, ActionDiTState)


def test_register_custom_architecture():
    """Verify that custom architectures can be registered."""
    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.model.architectures.registry import ARCHITECTURE_METADATA, ARCHITECTURE_REGISTRY, register_architecture

    @register_architecture("test_custom", framework="test", variant="custom")
    class TestArch(BaseWAMArchitecture):
        def __init__(self, cfg=None):
            super().__init__(cfg)

        def forward(self, noisy_actions, action_timestep, **_kw):
            return None, noisy_actions

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
