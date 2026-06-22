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
    from openwam.model.video_backbone.base import BlockLoopState

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
        hidden_states=torch.randn(1, 4, 32, generator=g),
        time_mod=torch.zeros(1, 6, 32),
        rope_freqs=torch.zeros(4, 1, 1),
        context=torch.randn(1, 4, 32, generator=g),
        context_mask=torch.ones(1, 4, dtype=torch.bool),
        grid_frames=4,
        grid_height=1,
        grid_width=1,
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
    from openwam.model.action_backbone.separate_action_dit import ActionDiTState

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


def test_tri_system_rejects_vlm_freeze_in_model_config():
    """VLM freezing is trainer-owned via the model freeze list."""
    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture

    cfg = {
        "framework": "tri_system",
        "variant": "joint_self_attn",
        "video_dim": 32,
        "num_heads": 4,
        "attn_head_dim": 8,
        "vlm_backbone": {
            "checkpoint_path": "unused",
            "freeze": True,
            "load_pretrained": False,
        },
    }

    with pytest.raises(ValueError, match="moved to the model"):
        TriSystemJointSelfAttnArchitecture(cfg)


def test_freeze_modules_supports_vlm_dotted_path():
    from openwam.model.architectures.base import BaseWAMArchitecture

    class TestArch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self.vlm_backbone = torch.nn.Module()
            self.vlm_backbone.vlm_model = torch.nn.Linear(2, 2)

        def forward(self, noisy_actions, action_timestep, **_kw):
            return None, noisy_actions

    arch = TestArch()
    frozen = arch.freeze_modules(["vlm_backbone.vlm_model"])

    assert frozen == ["vlm_backbone.vlm_model"]
    assert not any(p.requires_grad for p in arch.vlm_backbone.vlm_model.parameters())


def test_openwam_trainer_uses_strategy_freeze_for_tri_system_vlm(monkeypatch):
    """Tri-system VLM freeze is owned by the model freeze list."""

    from omegaconf import OmegaConf

    from openwam.train.openwam_trainer import OpenWAMTrainer

    class _Arch(torch.nn.Module):
        def __init__(self, cfg=None):  # noqa: ARG002
            super().__init__()
            self.dtype = torch.float32
            self.device = torch.device("cpu")
            self.vlm_backbone = torch.nn.Module()
            self.vlm_backbone.vlm_model = torch.nn.Linear(2, 2)
            self._freeze_calls = []

        def set_dtype_device(self, dtype, device):  # noqa: ARG002
            return None

        @property
        def backbones(self):
            return {"vlm_backbone": self.vlm_backbone}

        def freeze_modules(self, names):
            self._freeze_calls.append(list(names))
            frozen = []
            for name in names:
                if name == "vlm_backbone.vlm_model":
                    self.vlm_backbone.vlm_model.requires_grad_(False)
                    frozen.append(name)
            return frozen

        def init_training_schedulers(self, num_timesteps=1000):  # noqa: ARG002
            return None

        def set_training_runtime(self, **kwargs):  # noqa: ARG002
            return None

    holder = {}

    def _fake_resolve(model_cfg):  # noqa: ARG001
        return type(
            "Resolved",
            (),
            {
                "registry_name": "tri_system_joint_self_attn",
                "params": {},
                "canonical": type("Canonical", (), {"framework": "tri_system", "variant": "joint_self_attn"})(),
            },
        )()

    def _fake_build(name, params):  # noqa: ARG001
        arch = _Arch()
        holder["arch"] = arch
        return arch

    monkeypatch.setattr("openwam.model.resolve_architecture_config", _fake_resolve)
    monkeypatch.setattr("openwam.model.build_architecture", _fake_build)

    cfg = OmegaConf.create(
        {
            "training": {
                "initialize_model_on_cpu": False,
                "use_gradient_checkpointing": False,
                "use_gradient_checkpointing_offload": False,
                "max_timestep_boundary": 1.0,
                "min_timestep_boundary": 0.0,
                "lambda_video": 1.0,
                "lambda_action": 1.0,
            },
            "model": {
                "architecture": {"framework": "tri_system", "variant": "joint_self_attn"},
                "freeze": ["vlm_backbone.vlm_model"],
            },
        }
    )

    trainer = OpenWAMTrainer(cfg, accelerator=None, dataset=None)
    arch = holder["arch"]

    assert trainer.architecture is arch
    assert arch._freeze_calls == [["vlm_backbone.vlm_model"]]
    assert not any(p.requires_grad for p in arch.vlm_backbone.vlm_model.parameters())


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
        "bridge_layers": (0, 1),
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
    for block_id in ab.bridge_layers:
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
    from openwam.model.action_backbone.shared_action_backbone import SharedMoEActionBackbone

    dit = SharedMoEActionBackbone(
        action_dim=7,
        video_dim=64,
        expert_ffn_dim=128,
        bridge_layers=(0, 1),
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
    from openwam.model.action_backbone.separate_action_dit import ActionSelfAttention

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
    assert not hasattr(dit, "pos_encoding")
    assert hasattr(dit, "freqs") and dit.freqs.shape == (1024, 64 // 4 // 2)

    actions = torch.randn(2, 5, 7)
    bridges = {bid: torch.randn(2, 10, 128) for bid in (0, 1)}
    timestep = torch.tensor([0.5, 0.8])
    out = dit(actions, bridges, timestep)
    assert out.shape == (2, 5, 7)


def test_action_dit_joint_self_attn_uses_only_rope():
    """joint_self_attn relies solely on RoPE (no learned absolute PE)."""
    from openwam.model.action_backbone.separate_action_dit import ActionDiT

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
            proprio=torch.randn(B, 7),
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


def test_moe_uses_expert_layers():
    """SharedBackbone moe variant exposes expert layer ids."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "bridge_layers": (1, 3),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    assert arch.bridge_layers == (1, 3)
    assert frozenset(arch.action_backbone.bridge_layers) == {1, 3}


def test_shared_backbone_has_no_expert_layers():
    """SharedBackbone vanilla has no expert layers."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 5}
    arch = build_architecture("shared_backbone_vanilla", cfg)
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

    from openwam.model.architectures.registry import resolve_architecture_config

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

    from openwam.model.architectures.registry import resolve_architecture_config

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
        "bridge_layers": (1, 3),
    }
    arch = build_architecture("shared_backbone_moe", cfg)
    assert arch.cfg["framework"] == "shared_backbone"
    assert arch.cfg["variant"] == "moe"


def test_dual_system_self_attn_payload_is_action_dit_state():
    """joint_self_attn populates a flat payload of type ActionDiTState."""
    from openwam.model import build_architecture
    from openwam.model.action_backbone.separate_action_dit import ActionDiTState

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


def test_freeze_modules_disables_grad_and_wraps_forward_in_no_grad():
    """``freeze_modules`` is the single API for freezing: it both sets
    ``requires_grad=False`` AND wraps the named module's forward in
    ``torch.no_grad`` so the subtree never builds a backward graph.

    This is the contract every backbone and trainer relies on — there is no
    backbone-side freeze detection in OpenWAM.
    """
    from torch import nn

    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self.frozen_mod = nn.Linear(3, 4)
            self.trainable_mod = nn.Linear(3, 4)

        def forward(self, *args, **kwargs):  # pragma: no cover - unused stub
            raise NotImplementedError

    arch = _Arch()
    frozen = arch.freeze_modules(["frozen_mod"])
    assert frozen == ["frozen_mod"]

    # 1) Params no longer require grad
    assert not any(p.requires_grad for p in arch.frozen_mod.parameters())
    # Trainable module untouched
    assert all(p.requires_grad for p in arch.trainable_mod.parameters())

    # 2) Forward wraps in no_grad — output does not require grad, even with grad-tracking inputs.
    x = torch.randn(2, 3, requires_grad=True)
    assert arch.frozen_mod(x).requires_grad is False
    # Trainable module still builds graph
    assert arch.trainable_mod(x).requires_grad is True

    # 3) Idempotent — second call doesn't double-wrap
    arch.freeze_modules(["frozen_mod"])
    out = arch.frozen_mod(x)
    assert out.requires_grad is False
    assert getattr(arch.frozen_mod, "_openwam_no_grad_wrapped", False) is True


def test_freeze_modules_skips_unknown_paths():
    """Freeze list may mention modules absent on the current architecture
    (e.g. ``vlm_backbone.vlm_model`` on dual_system) — they are silently
    skipped, so each model yaml's freeze list only needs its own components.
    """
    from torch import nn

    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self.real = nn.Linear(2, 2)

        def forward(self, *args, **kwargs):  # pragma: no cover - unused stub
            raise NotImplementedError

    arch = _Arch()
    frozen = arch.freeze_modules(["real", "vlm_backbone.vlm_model", "does.not.exist"])
    assert frozen == ["real"]
    assert not any(p.requires_grad for p in arch.real.parameters())


def test_freeze_modules_wraps_all_descendants_in_no_grad():
    """``freeze_modules`` must wrap forward on every submodule in the frozen subtree,
    so calls that bypass the root (e.g. ``self.vlm_model.model(...)`` in
    ``Qwen3VLBackbone.extract_features``, which skips ``vlm_model.forward`` to
    avoid the LM head) also see ``no_grad``. Without recursive wrap, the bypass
    silently leaves the frozen subtree grad-tracking and the activation memory
    isn't reclaimed.
    """
    from torch import nn

    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 4)

        def forward(self, x):
            return self.lin(x)

    class _Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _Inner()
            self.head = nn.Linear(4, 5)

        def forward(self, x):
            return self.head(self.inner(x))

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self.outer = _Outer()

        def forward(self, *args, **kwargs):  # pragma: no cover - unused stub
            raise NotImplementedError

    arch = _Arch()
    arch.freeze_modules(["outer"])
    x = torch.randn(2, 3, requires_grad=True)
    # Top-level call: wrapped
    assert arch.outer(x).requires_grad is False
    # Nested submodule called directly (bypasses outer.forward): also wrapped
    assert arch.outer.inner(x).requires_grad is False
    # Leaf submodule called directly: also wrapped
    assert arch.outer.inner.lin(x).requires_grad is False


def test_freeze_parent_blocks_trainable_child_grad():
    """Document limitation: freezing a parent freezes ALL descendants.

    _wrap_forward_in_no_grad is subtree-level: a trainable child registered
    under a frozen parent will NOT receive gradients. This is by design —
    partial-freeze (e.g. LoRA on a frozen base) requires freezing specific
    leaves, not the parent. See base.py:75-81.
    """
    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self.trunk = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
            # Register a trainable adapter UNDER the trunk
            self.trunk.adapter = torch.nn.Linear(8, 4)

        def forward(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    arch = _Arch()
    assert any(p.requires_grad for p in arch.trunk.adapter.parameters()), "adapter should start trainable"

    arch.freeze_modules(["trunk"])

    x = torch.randn(1, 8, requires_grad=True)
    out = arch.trunk.adapter(x)
    # Adapter under frozen parent does NOT track grad — documented limitation
    assert not out.requires_grad, (
        "Expected trainable child under frozen parent to NOT track grad. "
        "This is by design; use leaf-level freeze for partial-freeze setups."
    )


def test_base_generate_signature_takes_extra_pipeline_inputs():
    """``BaseWAMArchitecture.generate`` accepts arbitrary keyword args via
    ``**extra_pipeline_inputs`` and forwards non-None values to
    ``architecture.forward`` via ``inputs_shared``. End-to-end validation lives
    in ``test_tri_system_generate_reuses_cached_vlm_hidden`` (tri_system uses
    this mechanism to thread ``vlm_hidden`` / ``vlm_attention_mask`` through);
    this test just guards the signature itself so future refactors don't
    accidentally revert to explicit per-arch params.
    """
    import inspect

    from openwam.model.architectures.base import BaseWAMArchitecture

    sig = inspect.signature(BaseWAMArchitecture.generate)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    assert has_var_keyword, "BaseWAMArchitecture.generate must accept **extra_pipeline_inputs"
    # And the tri_system-specific kwargs are NOT in the explicit signature anymore.
    explicit_names = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    for arch_specific in ("vlm_inputs", "vlm_hidden", "vlm_attention_mask"):
        assert arch_specific not in explicit_names, (
            f"'{arch_specific}' must not be in base.generate's explicit signature — "
            "it's tri_system-specific and should flow through **extra_pipeline_inputs."
        )
