"""Tests for WAM Architecture registry and implementations."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

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
    from openwam.model.architectures.single_system.moe import SingleSystemMoEArchitecture
    from openwam.model.architectures.single_system.vanilla import SingleSystemVanillaArchitecture

    assert DualSystemCrossAttnArchitecture is not None
    assert DualSystemIDMArchitecture is not None
    assert DualSystemSelfAttnArchitecture is not None
    assert SingleSystemMoEArchitecture is not None
    assert SingleSystemVanillaArchitecture is not None


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
    assert "single_system_vanilla" in supported
    assert "single_system_moe" in supported
    assert get_architecture_support("single_system_moe").status == "supported"
    assert get_architecture_support("single_system_vanilla").status == "supported"


def _make_tiny_wan_backbone_for_compile_test():
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    class _Block(torch.nn.Module):
        def forward(self, x, context, t_mod, freqs, context_mask=None, self_attn_mask=None):  # noqa: ARG002
            return x + 1

    class _Dit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = 8
            self.freq_dim = 4
            self.fuse_vae_embedding_in_latents = False
            self.has_image_input = False
            self.blocks = torch.nn.ModuleList([_Block()])
            self.text_embedding = torch.nn.Linear(8, 8)

    holder = SimpleNamespace(
        dit=_Dit(),
        scheduler=None,
        tokenizer=None,
        height_division_factor=1,
        width_division_factor=1,
        time_division_factor=1,
        time_division_remainder=0,
        latent_spec=None,
    )
    return Wan22Ti2v(holder)


def _tiny_wan_block_state(vb, *, use_gradient_checkpointing=False):
    from openwam.model.video_backbone.base import BlockLoopState

    return BlockLoopState(
        hidden_states=torch.zeros(1, 2, 8),
        time_mod=torch.zeros(1, 6, 8),
        rope_freqs=torch.zeros(2, 1, 1),
        context=torch.zeros(1, 3, 8),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
        grid_frames=2,
        grid_height=1,
        grid_width=1,
        use_gradient_checkpointing=use_gradient_checkpointing,
        extras={"dit": vb.dit},
    )


def test_wan_backbone_compile_auto_lazily_compiles_run_block():
    from omegaconf import OmegaConf

    vb = _make_tiny_wan_backbone_for_compile_test()
    cfg = OmegaConf.create({"mode": "auto", "wan_blocks": {}})
    calls = {"compile": 0, "wrapped": 0}

    def _identity_compile(fn, **kwargs):
        calls["compile"] += 1
        assert kwargs == {"dynamic": False, "mode": "default"}

        def _wrapped(*args, **kw):
            calls["wrapped"] += 1
            return fn(*args, **kw)

        return _wrapped

    with patch("torch.compile", side_effect=_identity_compile):
        vb.apply_compile_optimizations(cfg)
        assert calls["compile"] == 0

        state = _tiny_wan_block_state(vb)
        state = vb.run_block(0, state)
        state = vb.run_block(0, state)

    assert calls == {"compile": 1, "wrapped": 2}
    assert torch.equal(state.hidden_states, torch.full((1, 2, 8), 2.0))


def test_wan_backbone_compile_skips_gradient_checkpointed_run_block():
    from omegaconf import OmegaConf

    vb = _make_tiny_wan_backbone_for_compile_test()
    vb.apply_compile_optimizations(OmegaConf.create({"mode": "auto", "wan_blocks": {}}))

    with patch("torch.compile") as mock_compile:
        state = vb.run_block(0, _tiny_wan_block_state(vb, use_gradient_checkpointing=True))

    mock_compile.assert_not_called()
    assert torch.equal(state.hidden_states, torch.ones(1, 2, 8))


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


def test_build_architecture_single_system_moe():
    from openwam.model import build_architecture

    cfg = {
        "framework": "single_system",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "bridge_layers": (1, 3),
    }
    arch = build_architecture("single_system_moe", cfg)
    assert arch.action_dim == 7
    assert arch.bridge_layers == (1, 3)
    assert arch.action_backbone is not None
    assert len(arch.action_backbone.expert_blocks) == 2


def test_build_architecture_shared():
    """Single system vanilla should build successfully."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 10}
    arch = build_architecture("single_system_vanilla", cfg)
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


def test_single_system_moe_encode_apply_decode():
    """Smoke test: MoE encode → apply_expert at expert layers → decode."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "single_system",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "bridge_layers": (0, 1),
    }
    arch = build_architecture("single_system_moe", cfg)
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


def test_single_system_encode_decode():
    """Smoke test: vanilla encode + decode shapes."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 10}
    arch = build_architecture("single_system_vanilla", cfg)
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


def test_single_system_output_head_init():
    """SingleSystem output head (ActionOutputMLP) uses small-random init."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5}
    arch = build_architecture("single_system_vanilla", cfg)
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


def test_action_dit_rope_freqs_preserve_complex_dtype_on_dtype_to():
    """ActionDiT RoPE cache must not be cast to real by dtype-only Module.to()."""
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
    original_dtype = dit.freqs.dtype

    dit.to(dtype=torch.bfloat16)

    assert dit.freqs.dtype == original_dtype
    assert dit.freqs.is_complex()
    assert "freqs" not in dit.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for ActionDiT RoPE device migration check")
def test_action_dit_rope_freqs_follow_cuda_to_without_dtype_cast():
    """ActionDiT.to(cuda, bf16) should move RoPE freqs to CUDA but keep them complex."""
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
        text_dim=16,
    ).to(device="cuda", dtype=torch.bfloat16)

    assert dit.freqs.device.type == "cuda"
    assert dit.freqs.is_complex()
    assert dit.freqs.dtype.is_complex
    assert "freqs" not in dit.state_dict()

    context = torch.randn(2, 4, 16, device="cuda", dtype=torch.bfloat16)
    context_mask = torch.ones(2, 4, dtype=torch.bool, device="cuda")
    astate = dit.prepare_state(
        torch.randn(2, 5, 7, device="cuda", dtype=torch.bfloat16),
        torch.tensor([0.5, 0.8], device="cuda", dtype=torch.bfloat16),
        context=context,
        context_mask=context_mask,
    )
    assert astate.payload.action_freqs.device.type == "cuda"
    assert astate.payload.action_freqs.is_complex()


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


def test_dual_system_joint_cross_attn_compile_none_stays_eager():
    from omegaconf import OmegaConf

    arch = _make_dual_system_cross_attn_fixture()
    with patch("torch.compile") as mock_compile:
        arch.apply_compile_optimizations(OmegaConf.create({"mode": "none"}))

    mock_compile.assert_not_called()
    assert arch._compiled_action_forward is None


def test_dual_system_joint_cross_attn_compile_auto_sets_action_forward():
    from omegaconf import OmegaConf

    arch = _make_dual_system_cross_attn_fixture()

    def _identity_compile(fn, **_kwargs):
        return fn

    cfg = OmegaConf.create(
        {
            "mode": "auto",
            "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
        }
    )
    with patch("torch.compile", side_effect=_identity_compile) as mock_compile:
        arch.apply_compile_optimizations(cfg)

    mock_compile.assert_called_once()
    assert mock_compile.call_args.kwargs == {"dynamic": False, "mode": "reduce-overhead"}
    assert arch._compiled_action_forward is not None


def test_dual_system_joint_cross_attn_compiled_action_path_matches_eager():
    from omegaconf import OmegaConf

    arch = _make_dual_system_cross_attn_fixture()
    actions, bridges, timestep, context, context_mask = _dual_system_cross_attn_inputs(arch, 11)

    with torch.no_grad():
        eager_out = arch._predict_actions_from_bridges(
            actions,
            bridges,
            timestep,
            context=context,
            context_mask=context_mask,
        )

    calls = {"n": 0}

    def _counting_compile(fn, **_kwargs):
        def _wrapped(*args, **kwargs):
            calls["n"] += 1
            return fn(*args, **kwargs)

        return _wrapped

    with patch("torch.compile", side_effect=_counting_compile):
        arch.apply_compile_optimizations(OmegaConf.create({"mode": "auto", "cross_attn": {}}))

    with torch.no_grad():
        compiled_out = arch._predict_actions_from_bridges(
            actions,
            bridges,
            timestep,
            context=context,
            context_mask=context_mask,
        )

    assert calls["n"] == 1
    assert torch.allclose(compiled_out, eager_out, atol=1e-6)


def test_dual_system_joint_cross_attn_compile_skips_checkpointed_path():
    from omegaconf import OmegaConf

    arch = _make_dual_system_cross_attn_fixture()
    arch.action_backbone.eval()
    actions, bridges, timestep, context, context_mask = _dual_system_cross_attn_inputs(arch, 11)
    calls = {"n": 0}

    def _counting_compile(fn, **_kwargs):
        def _wrapped(*args, **kwargs):
            calls["n"] += 1
            return fn(*args, **kwargs)

        return _wrapped

    with patch("torch.compile", side_effect=_counting_compile):
        arch.apply_compile_optimizations(OmegaConf.create({"mode": "auto", "cross_attn": {}}))

    with torch.no_grad():
        out = arch._predict_actions_from_bridges(
            actions,
            bridges,
            timestep,
            context=context,
            context_mask=context_mask,
            use_gradient_checkpointing=True,
        )

    assert calls["n"] == 0
    assert out.shape == actions.shape


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


def test_dual_system_joint_self_attn_compile_auto_sets_mot_loop():
    from omegaconf import OmegaConf

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
        "text_dim": 16,
    }
    arch = build_architecture("dual_system_self_attn", cfg)
    arch.video_backbone = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    arch.build_mot_driver()

    def _identity_compile(fn, **_kwargs):
        return fn

    with patch("torch.compile", side_effect=_identity_compile) as mock_compile:
        arch.apply_compile_optimizations(OmegaConf.create({"mode": "auto", "self_attn": {}}))

    mock_compile.assert_called_once()
    assert mock_compile.call_args.kwargs == {"dynamic": False, "mode": "reduce-overhead"}
    assert arch._compiled_mot_run_joint_loop is not None


def test_dual_system_idm_compile_auto_sets_action_cache_loop():
    from omegaconf import OmegaConf

    from openwam.model import build_architecture
    from tests.test_openwam_trainer import _MockVideoBackbone

    cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "bridge_layers": (0, 1),
        "text_dim": 16,
    }
    arch = build_architecture("dual_system_idm", cfg)
    arch.video_backbone = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    arch.build_mot_driver()

    def _identity_compile(fn, **_kwargs):
        return fn

    with patch("torch.compile", side_effect=_identity_compile) as mock_compile:
        arch.apply_compile_optimizations(OmegaConf.create({"mode": "auto", "idm": {"action_cache": {}}}))

    mock_compile.assert_called_once()
    assert mock_compile.call_args.kwargs == {"dynamic": False, "mode": "reduce-overhead"}
    assert arch._compiled_idm_action_cache_loop is not None


def test_dual_system_idm_action_cache_tensor_loop_matches_eager_full_mask_path():
    from openwam.model import build_architecture
    from openwam.model.video_backbone.base import BlockLoopState
    from tests.test_openwam_trainer import _MockVideoBackbone

    cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": 7,
        "dim": 32,
        "ffn_dim": 64,
        "num_heads": 4,
        "video_dim": 32,
        "bridge_layers": (0, 1),
        "text_dim": 16,
    }
    arch = build_architecture("dual_system_idm", cfg)
    arch.video_backbone = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    driver = arch.build_mot_driver()
    arch.eval()

    ab = arch.action_backbone
    g = torch.Generator().manual_seed(123)
    batch_size = 2
    video_seq_len = 5
    action_seq_len = 4
    action_latents = torch.randn(batch_size, action_seq_len, ab.action_dim, generator=g)
    action_timestep = torch.tensor([0.25, 0.75])
    context = torch.randn(batch_size, 3, ab.text_dim, generator=g)
    context_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    vstate = BlockLoopState(
        hidden_states=torch.randn(batch_size, video_seq_len, 32, generator=g),
        time_mod=torch.zeros(batch_size, video_seq_len, 6, 32),
        rope_freqs=torch.zeros(video_seq_len, 1, 1),
        context=torch.randn(batch_size, 4, 32, generator=g),
        context_mask=torch.ones(batch_size, 4, dtype=torch.bool),
        grid_frames=video_seq_len,
        grid_height=1,
        grid_width=1,
        extras={},
    )
    video_kv_cache, _, _ = driver.prefill_video_cache(vstate)

    with torch.no_grad():
        eager_state = ab.prepare_state(
            action_latents,
            action_timestep,
            context=context,
            context_mask=context_mask,
        )
        eager_state = driver.run_action_with_video_cache(
            eager_state,
            video_kv_cache=video_kv_cache,
            video_seq_len=video_seq_len,
        )
        eager_x = eager_state.payload.x_action
        eager_pred = ab.extract_prediction(eager_state)

        x_action = ab._embed_actions(action_latents)
        prepared_timestep = ab._prepare_timestep(action_timestep, batch_size)
        t_mod = ab.time_projection(ab.time_embedding(prepared_timestep))
        action_freqs = ab._get_rope_freqs(action_seq_len).to(device=x_action.device)
        context_emb, context_attn_mask = ab._prepare_context(
            context,
            context_mask,
            batch_size=batch_size,
            seq_len=action_seq_len,
            dtype=x_action.dtype,
            device=x_action.device,
        )
        video_k_tuple, video_v_tuple = driver.video_kv_cache_to_tuples(video_kv_cache)
        action_mask = torch.ones(
            (action_seq_len, video_seq_len + action_seq_len),
            dtype=torch.bool,
            device=x_action.device,
        )
        tensor_x = driver.run_action_with_video_cache_tensor_loop(
            x_action,
            t_mod,
            action_freqs,
            context_emb,
            context_attn_mask,
            action_mask,
            video_k_tuple,
            video_v_tuple,
        )
        tensor_pred = ab.action_decoder(tensor_x)

    assert torch.allclose(tensor_x, eager_x, atol=1e-6)
    assert torch.allclose(tensor_pred, eager_pred, atol=1e-6)


def test_tri_system_joint_self_attn_compile_auto_sets_mot_loop():
    from omegaconf import OmegaConf

    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture

    class _Driver:
        def run_joint_loop(self, vstate, astate, ustate, **_kwargs):
            raise AssertionError("eager tri-system loop should not run through the compiled wrapper")

        def run_joint_loop_for_compile(self, vstate, astate, ustate, *, attn_mask):
            assert attn_mask == "mask"
            return vstate, astate, ustate

    arch = TriSystemJointSelfAttnArchitecture(cfg=None)
    arch.video_backbone = torch.nn.Module()
    arch.action_backbone = torch.nn.Module()
    arch.understanding_expert = torch.nn.Module()
    arch._mot_driver = _Driver()

    def _identity_compile(fn, **_kwargs):
        return fn

    with patch("torch.compile", side_effect=_identity_compile) as mock_compile:
        arch.apply_compile_optimizations(OmegaConf.create({"mode": "auto", "tri_system": {}}))

    mock_compile.assert_called_once()
    assert mock_compile.call_args.kwargs == {"dynamic": False, "mode": "reduce-overhead"}
    assert arch._compiled_mot_run_joint_loop is not None
    assert arch._compiled_mot_run_joint_loop("v", "a", "u", "mask") == ("v", "a", "u")


def test_moe_uses_expert_layers():
    """SingleSystem moe variant exposes expert layer ids."""
    from openwam.model import build_architecture

    cfg = {
        "framework": "single_system",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "bridge_layers": (1, 3),
    }
    arch = build_architecture("single_system_moe", cfg)
    assert arch.bridge_layers == (1, 3)
    assert frozenset(arch.action_backbone.bridge_layers) == {1, 3}


def test_single_system_has_no_expert_layers():
    """SingleSystem vanilla has no expert layers."""
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 128, "num_action_tokens": 5}
    arch = build_architecture("single_system_vanilla", cfg)
    assert arch.bridge_layers == ()


def test_normalize_architecture_spec_single_system():
    from openwam.model.architectures.registry import normalize_architecture_spec

    spec = normalize_architecture_spec("single_system_vanilla", {"action_dim": 7})
    assert spec.framework == "single_system"
    assert spec.variant == "vanilla"
    assert spec.options == {}


def test_normalize_architecture_spec_single_system_moe():
    from openwam.model.architectures.registry import normalize_architecture_spec

    spec = normalize_architecture_spec("single_system_moe", {"action_dim": 7})
    assert spec.framework == "single_system"
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
            "framework": "single_system",
            "variant": "moe",
            "action_dim": 20,
            "expert_ffn_dim": 512,
        },
        action_backbone={},
    )
    resolved = resolve_architecture_config(model_cfg, video_dim=192)

    assert resolved.registry_name == "single_system_moe"
    assert resolved.canonical.framework == "single_system"
    assert resolved.canonical.variant == "moe"
    assert resolved.params["framework"] == "single_system"
    assert resolved.params["variant"] == "moe"
    assert resolved.params["video_dim"] == 192


def test_build_architecture_injects_framework_and_variant_for_single_system():
    from openwam.model import build_architecture

    cfg = {"action_dim": 7, "video_dim": 64, "num_action_tokens": 5}
    arch = build_architecture("single_system_vanilla", cfg)
    assert arch.cfg["framework"] == "single_system"
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


def test_build_architecture_single_system_moe_canonical_config():
    from openwam.model import build_architecture

    cfg = {
        "framework": "single_system",
        "variant": "moe",
        "action_dim": 7,
        "video_dim": 128,
        "expert_ffn_dim": 256,
        "bridge_layers": (1, 3),
    }
    arch = build_architecture("single_system_moe", cfg)
    assert arch.cfg["framework"] == "single_system"
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


def test_base_generate_dit_cache_reuses_joint_action_prediction(monkeypatch):
    """A joint cache hit must skip the whole video/action forward."""

    from openwam.deploy.optimizations.dit_cache import DiTVelocityCache
    from openwam.model.architectures.base import BaseWAMArchitecture

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)

    class _Scheduler:
        num_train_timesteps = 1000

        @staticmethod
        def flow_step(model_output, sigma, sigma_next, sample):
            return sample + model_output * (sigma_next - sigma)

    class _VideoBackbone(torch.nn.Module):
        scheduler = _Scheduler()
        external_encoder = None

        @staticmethod
        def preprocess_input_for_inference(**_kwargs):
            return {"latents": torch.zeros(1, 1, 1, 1, 1)}

    class _ActionBackbone(torch.nn.Module):
        scheduler = _Scheduler()
        action_dim = 3
        bridge_layers = ()
        uses_proprioception = False

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            self.video_backbone = _VideoBackbone()
            self.action_backbone = _ActionBackbone()
            self.forward_calls = 0

        def forward(self, noisy_actions, action_timestep, **kwargs):  # noqa: ARG002
            self.forward_calls += 1
            return torch.ones_like(kwargs["latents"]), torch.ones_like(noisy_actions) * 2

    arch = _Arch()
    cache = DiTVelocityCache(cosine_threshold=0.0)
    schedule = [(1000, 1000), (800, 800), (600, 600), (400, 400)]

    result = arch.generate(
        schedule=schedule,
        prompt="",
        num_frames=4,
        action_num_frames=4,
        seed=0,
        dit_cache=cache,
        decode_video=False,
    )

    assert arch.forward_calls == 2
    assert cache.stats["total_steps"] == 3
    assert cache.stats["total_skips"] == 1
    assert result["actions"].shape == (3, 3)


def test_base_generate_frozen_lag_stream_stays_in_forward_context(monkeypatch):
    """A sigma plateau (linear_offset delay) freezes a stream's *update* only: its
    tokens/timestep must stay in every forward, because v→a-visible attention
    masks (mutual / video_sees_action) read them."""

    from openwam.model.architectures.base import BaseWAMArchitecture

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)

    forward_inputs = []
    flow_steps = []

    class _Scheduler:
        num_train_timesteps = 1000

        @staticmethod
        def flow_step(model_output, sigma, sigma_next, sample):
            flow_steps.append((sigma, sigma_next))
            return sample + model_output * (sigma_next - sigma)

    class _VideoBackbone(torch.nn.Module):
        scheduler = _Scheduler()
        external_encoder = None

        @staticmethod
        def preprocess_input_for_inference(**_kwargs):
            return {"latents": torch.zeros(1, 1, 1, 1, 1)}

    class _ActionBackbone(torch.nn.Module):
        scheduler = _Scheduler()
        action_dim = 3
        bridge_layers = ()
        uses_proprioception = False

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            self.video_backbone = _VideoBackbone()
            self.action_backbone = _ActionBackbone()

        def forward(self, noisy_actions, action_timestep, **kwargs):
            forward_inputs.append((noisy_actions is not None, float(action_timestep.item())))
            return torch.zeros_like(kwargs["latents"]), torch.zeros_like(noisy_actions)

    class _RecordingCache:
        def __init__(self):
            self.action_updates = []

        def should_recompute(self, sigma, *, require_action=False):  # noqa: ARG002
            return True

        def update(self, velocity, sigma, action_velocity=None):  # noqa: ARG002
            self.action_updates.append(action_velocity is not None)

    # linear_offset=0.5 shape: sigma_a plateaus at 1.0 for the first
    # two transitions, then catches up.
    schedule = [(1000.0, 1000.0), (750.0, 1000.0), (500.0, 1000.0), (250.0, 500.0), (0.0, 0.0)]
    cache = _RecordingCache()

    result = _Arch().generate(
        schedule=schedule,
        prompt="",
        num_frames=4,
        action_num_frames=4,
        seed=0,
        dit_cache=cache,
        decode_video=False,
    )

    # Every step is a joint forward with action tokens present; the frozen
    # head rides t_a=1000 (the training grid's sigma=1 endpoint).
    assert forward_inputs == [(True, 1000.0), (True, 1000.0), (True, 1000.0), (True, 500.0)]
    # The plateau gates only the update: action flow_steps once it moves.
    assert flow_steps == [(1.0, 0.5), (0.5, 0.0)]
    # Plateau-step predictions are cached with the action branch present.
    assert cache.action_updates == [True, True, True, True]
    assert result["actions"].shape == (3, 3)


_PIN_ACTION_DIM = 8
_PIN_ACTIVE_IDX = [0, 2, 5]  # non-contiguous, mimics a unify_action scatter map
_PIN_INACTIVE_IDX = [1, 3, 4, 6, 7]


def _make_inactive_dim_probe_arch():
    """Fake arch whose forward emits huge garbage flow on the non-active action
    channels — an adversarial stand-in for the unsupervised padding-dim flow of
    a unify_action checkpoint. Records every action forward input."""
    from openwam.model.architectures.base import BaseWAMArchitecture

    class _Scheduler:
        num_train_timesteps = 1000

        @staticmethod
        def flow_step(model_output, sigma, sigma_next, sample):
            return sample + model_output * (sigma_next - sigma)

    class _VideoBackbone(torch.nn.Module):
        scheduler = _Scheduler()
        external_encoder = None

        @staticmethod
        def preprocess_input_for_inference(**_kwargs):
            return {"latents": torch.zeros(1, 1, 1, 1, 1)}

    class _ActionBackbone(torch.nn.Module):
        scheduler = _Scheduler()
        action_dim = _PIN_ACTION_DIM
        bridge_layers = ()
        uses_proprioception = False

    class _Arch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            self.video_backbone = _VideoBackbone()
            self.action_backbone = _ActionBackbone()
            self.seen_action_inputs = []

        def forward(self, noisy_actions, action_timestep, **kwargs):  # noqa: ARG002
            self.seen_action_inputs.append(noisy_actions.detach().clone())
            pred = torch.empty_like(noisy_actions)
            pred[..., _PIN_ACTIVE_IDX] = 1.0
            pred[..., _PIN_INACTIVE_IDX] = 1000.0
            return torch.ones_like(kwargs["latents"]), pred

    return _Arch()


def test_base_generate_pins_inactive_action_dims_on_noise_path(monkeypatch):
    """Unsupervised (non-scattered) unified-action dims must ride the analytic
    sigma * eps0 noise path through the denoising loop instead of integrating
    unconstrained flow back into the next forward."""
    import numpy as np

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)

    schedule = [(1000.0, 1000.0), (750.0, 750.0), (500.0, 500.0), (250.0, 250.0), (0.0, 0.0)]
    sigmas = [t_a / 1000.0 for _, t_a in schedule]
    gen_kwargs = dict(schedule=schedule, prompt="", num_frames=5, action_num_frames=5, seed=7, decode_video=False)
    eps0 = torch.randn(1, 4, _PIN_ACTION_DIM, generator=torch.Generator(device="cpu").manual_seed(7))

    baseline = _make_inactive_dim_probe_arch().generate(**gen_kwargs)["actions"]
    # Unpinned, the garbage flow drags inactive dims far off the noise path.
    assert np.abs(baseline[:, _PIN_INACTIVE_IDX]).max() > 100

    mask = torch.zeros(_PIN_ACTION_DIM, dtype=torch.bool)
    mask[_PIN_ACTIVE_IDX] = True
    pinned_arch = _make_inactive_dim_probe_arch()
    pinned = pinned_arch.generate(**gen_kwargs, active_action_mask=mask)["actions"]

    # Every forward saw inactive dims exactly on sigma_k * eps0 ...
    for k, seen in enumerate(pinned_arch.seen_action_inputs):
        torch.testing.assert_close(
            seen[..., _PIN_INACTIVE_IDX], sigmas[k] * eps0[..., _PIN_INACTIVE_IDX], rtol=0, atol=1e-7
        )
    # ... the final state lands exactly on 0 (sigma_end == 0) ...
    assert np.abs(pinned[:, _PIN_INACTIVE_IDX]).max() == 0.0
    # ... and active dims are untouched by the pin (input-independent fake pred).
    np.testing.assert_array_equal(pinned[:, _PIN_ACTIVE_IDX], baseline[:, _PIN_ACTIVE_IDX])


def test_base_generate_infers_inactive_dims_from_unify_normalizer(monkeypatch):
    """The pin auto-enables from the attached unify normalizer's scatter map and
    stays off for plain (non-unify) normalizers or width mismatches."""
    import numpy as np

    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)

    schedule = [(1000.0, 1000.0), (500.0, 500.0), (0.0, 0.0)]
    gen_kwargs = dict(schedule=schedule, prompt="", num_frames=5, action_num_frames=5, seed=7, decode_video=False)

    class _UnifyNormalizer:  # duck-types _UnifyAwareNormalizer's private surface
        _dst_index = np.asarray(_PIN_ACTIVE_IDX, dtype=np.int64)
        _unify_dim = _PIN_ACTION_DIM

        @staticmethod
        def unnormalize(x):
            return x  # keep the unified width so inactive dims stay observable

    baseline = _make_inactive_dim_probe_arch().generate(**gen_kwargs)["actions"]

    mask = torch.zeros(_PIN_ACTION_DIM, dtype=torch.bool)
    mask[_PIN_ACTIVE_IDX] = True
    explicit = _make_inactive_dim_probe_arch().generate(**gen_kwargs, active_action_mask=mask)["actions"]

    auto_arch = _make_inactive_dim_probe_arch()
    auto_arch.attach_normalizer(_UnifyNormalizer())
    np.testing.assert_array_equal(auto_arch.generate(**gen_kwargs)["actions"], explicit)

    class _PlainNormalizer:
        @staticmethod
        def unnormalize(x):
            return x

    plain_arch = _make_inactive_dim_probe_arch()
    plain_arch.attach_normalizer(_PlainNormalizer())
    np.testing.assert_array_equal(plain_arch.generate(**gen_kwargs)["actions"], baseline)

    class _MismatchNormalizer(_UnifyNormalizer):
        _unify_dim = _PIN_ACTION_DIM + 1

    mismatch_arch = _make_inactive_dim_probe_arch()
    mismatch_arch.attach_normalizer(_MismatchNormalizer())
    np.testing.assert_array_equal(mismatch_arch.generate(**gen_kwargs)["actions"], baseline)
