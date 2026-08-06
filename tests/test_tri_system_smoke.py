"""Smoke tests for the tri_system architecture.

CPU tests cover the trimodal MoT driver, action/understanding pre/post
interfaces, Qwen3-VL feature extraction with fake modules, and lightweight
checkpoint invariants.
"""

import copy
import os

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from openwam.model.action_backbone.separate_action_dit import ActionDiT, ActionDiTState
from openwam.model.architectures.tri_system.mot_driver import TriSystemMoTDriver
from openwam.model.architectures.tri_system.und_expert import (
    UnderstandingExpert,
    UnderstandingExpertConfig,
)
from openwam.model.architectures.utils.mask_modes import (
    ACTION_SEES_VIDEO,
    ISOLATED,
    MUTUAL,
    VIDEO_SEES_ACTION,
    build_cross_modal_attention_mask,
)
from openwam.model.video_backbone.base import BlockLoopState
from openwam.model.video_backbone.wan.models.dit import DiTBlock
from openwam.model.video_backbone.wan_backbone import Wan21
from openwam.model.vlm_backbone import Qwen3VLBackbone

WAN22_TI2V_5B = os.environ.get("OPENWAM_WAN22_TI2V_5B", "/path/to/Wan2.2-TI2V-5B")


# ---------------------------------------------------------------------------
# Helpers — tiny Wan-style stubs
# ---------------------------------------------------------------------------


class _FakePipe:
    """Minimal duck-typed pipeline for Wan21.

    Mirrors the shortcut in ``tests/test_video_backbone_consistency.py``:
    only ``dit`` is populated so ``WanVideoBackbone`` can run the block
    loop without a real VAE / text encoder / clip image encoder.
    """

    def __init__(self, dit):
        self.dit = dit
        self.use_unified_sequence_parallel = False
        self.in_iteration_models = ["dit"]


# ---------------------------------------------------------------------------
# CPU micro-tests: trimodal driver and interface invariants
# ---------------------------------------------------------------------------


def _make_tiny_wan_backbone(num_layers=2, dim=32, num_heads=4, ffn_dim=64):
    blocks = nn.ModuleList(
        [
            DiTBlock(has_image_input=False, dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, eps=1e-6)
            for _ in range(num_layers)
        ]
    )

    class _StubDit(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = dim
            self.freq_dim = 256
            self.blocks = blocks
            self.head = nn.Identity()

    return Wan21(_FakePipe(_StubDit()))


def _make_tiny_video_state(dit, *, batch=2, grid_frames=1, grid_height=2, grid_width=3, dtype=torch.float32):
    dim = dit.dim
    seq_len = grid_frames * grid_height * grid_width
    # RoPE freqs are multiplicative complex ones, so they preserve Q/K while
    # still satisfying Wan's expected shape.
    freqs = torch.ones(seq_len, 1, dim // dit.blocks[0].num_heads // 2, dtype=torch.complex64)
    return BlockLoopState(
        hidden_states=torch.randn(batch, seq_len, dim, dtype=dtype),
        time_mod=torch.randn(batch, 6, dim, dtype=dtype),
        rope_freqs=freqs,
        context=torch.randn(batch, 4, dim, dtype=dtype),
        grid_frames=grid_frames,
        grid_height=grid_height,
        grid_width=grid_width,
        vace_hints=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extras={"dit": dit, "time_embed": torch.randn(batch, dim, dtype=dtype)},
    )


def _make_tiny_trimodal_components(num_layers=2, dim=32, num_heads=4, action_dim=7, und_dim=16):
    vb = _make_tiny_wan_backbone(num_layers=num_layers, dim=dim, num_heads=num_heads, ffn_dim=64)
    ab = ActionDiT(
        action_dim=action_dim,
        dim=24,
        ffn_dim=48,
        num_heads=num_heads,
        num_layers=num_layers,
        video_dim=dim,
        bridge_layers=tuple(range(num_layers)),
        variant="joint_self_attn",
        attn_head_dim=dim // num_heads,
        text_dim=16,
        eps=1.0e-6,
    )
    ub = UnderstandingExpert(
        UnderstandingExpertConfig(
            dim=und_dim,
            ffn_dim=und_dim * 2,
            num_layers=num_layers,
            vlm_input_dim=20,
            vlm_projector_type="linear",
        ),
        wan_dim=dim,
        wan_num_heads=num_heads,
    )
    return vb, ab, ub


def _make_forward_tri_arch(vb, ab, ub):
    """Minimal tri arch wired from pre-built components for full-``forward()``
    tests: no VLM backbone, MoT driver without mixed-attn checkpointing."""
    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture

    class _Arch(TriSystemJointSelfAttnArchitecture):
        device = torch.device("cpu")

        def __init__(self):
            nn.Module.__init__(self)
            self.video_backbone = vb
            self.action_backbone = ab
            self.understanding_expert = ub
            self.vlm_backbone = None
            self._proprio_context = None
            self._mot_driver = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False)

    return _Arch()


def test_tri_system_action_pre_post_round_trip():
    torch.manual_seed(0)
    vb, ab, _ = _make_tiny_trimodal_components()
    device = torch.device("cpu")
    noisy_actions = torch.randn(2, 4, ab.action_dim, device=device)
    timestep = torch.tensor([10.0, 20.0], device=device)
    context = torch.randn(2, 3, ab.text_dim, device=device)
    context_mask = torch.ones(2, 3, dtype=torch.bool, device=device)
    astate = ab.prepare_state(noisy_actions, timestep, context=context, context_mask=context_mask)
    assert isinstance(astate.payload, ActionDiTState)

    q, k, v, post = ab.pre_attn_at_layer(0, astate)
    assert q.shape == k.shape == v.shape
    assert q.shape[:2] == astate.payload.x_action.shape[:2]
    assert q.shape[-1] == vb.dim
    out = torch.randn_like(q)
    astate = ab.post_attn_at_layer(0, astate, out, post)
    assert astate.payload.x_action.shape[0] == 2
    assert torch.isfinite(astate.payload.x_action).all()


def test_understanding_expert_pre_post_round_trip_and_grad():
    torch.manual_seed(0)
    _, _, ub = _make_tiny_trimodal_components()
    vlm_hidden = torch.randn(2, 5, ub.cfg.vlm_input_dim)
    ustate = ub.prepare_state(vlm_hidden)
    q, k, v, post = ub.pre_attn_at_layer(0, ustate)
    assert q.shape == k.shape == v.shape == (2, 5, ub.num_heads * ub.head_dim)

    ustate = ub.post_attn_at_layer(0, ustate, torch.randn_like(q), post)
    loss = ustate.und_tokens.float().pow(2).mean()
    loss.backward()
    grad_norm = sum(float(p.grad.float().norm()) for p in ub.parameters() if p.grad is not None)
    assert grad_norm > 0


def test_understanding_expert_config_accepts_explicit_ffn_dim(monkeypatch):
    from openwam.model.architectures.tri_system import joint_self_attn as tri_mod
    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture

    class _StubVLM(nn.Module):
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            super().__init__()

        @property
        def hidden_size(self):
            return 20

    class _Arch(TriSystemJointSelfAttnArchitecture):
        def _init_video_backbone(self, cfg):  # noqa: ARG002
            self.video_backbone, _, _ = _make_tiny_trimodal_components(num_layers=1)

    monkeypatch.setattr(tri_mod, "build_vlm_backbone", lambda *a, **k: _StubVLM())

    cfg = {
        "vlm_backbone": {"checkpoint_path": "", "load_pretrained": False},
        "understanding_expert": {
            "dim": 16,
            "ffn_dim": 96,
            "vlm_projector_type": "linear",
        },
        "action_dim": 7,
        "dim": 24,
        "ffn_dim": 48,
        "num_heads": 4,
        "attn_head_dim": 8,
        "text_dim": 16,
    }
    arch = _Arch(cfg)

    assert arch.understanding_expert.cfg.ffn_dim == 96


def test_understanding_expert_rejects_ffn_dim_multiplier(monkeypatch):
    from openwam.model.architectures.tri_system import joint_self_attn as tri_mod
    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture

    class _StubVLM(nn.Module):
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            super().__init__()

        @property
        def hidden_size(self):
            return 20

    class _Arch(TriSystemJointSelfAttnArchitecture):
        def _init_video_backbone(self, cfg):  # noqa: ARG002
            self.video_backbone, _, _ = _make_tiny_trimodal_components(num_layers=1)

    monkeypatch.setattr(tri_mod, "build_vlm_backbone", lambda *a, **k: _StubVLM())

    cfg = {
        "vlm_backbone": {"checkpoint_path": "", "load_pretrained": False},
        "understanding_expert": {
            "dim": 16,
            "ffn_dim_multiplier": 2,
            "vlm_projector_type": "linear",
        },
        "action_dim": 7,
        "dim": 24,
        "ffn_dim": 48,
        "num_heads": 4,
        "attn_head_dim": 8,
        "text_dim": 16,
    }

    with pytest.raises(ValueError, match="understanding_expert.ffn_dim"):
        _Arch(cfg)


def _make_stub_tri_arch(monkeypatch, num_video_layers: int = 3):
    """Build a tri_system arch with a stub VLM + tiny video backbone of N layers."""
    from openwam.model.architectures.tri_system import joint_self_attn as tri_mod
    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture

    class _StubVLM(nn.Module):
        def __init__(self, *args, **kwargs):  # noqa: ARG002
            super().__init__()

        @property
        def hidden_size(self):
            return 20

    class _Arch(TriSystemJointSelfAttnArchitecture):
        def _init_video_backbone(self, cfg):  # noqa: ARG002
            self.video_backbone, _, _ = _make_tiny_trimodal_components(num_layers=num_video_layers)

    monkeypatch.setattr(tri_mod, "build_vlm_backbone", lambda *a, **k: _StubVLM())
    return _Arch


def _tri_arch_min_cfg(extra: dict | None = None) -> dict:
    cfg = {
        "vlm_backbone": {"checkpoint_path": "", "load_pretrained": False},
        "understanding_expert": {"dim": 16, "ffn_dim": 96, "vlm_projector_type": "linear"},
        "action_dim": 7,
        "dim": 24,
        "ffn_dim": 48,
        "num_heads": 4,
        "attn_head_dim": 8,
        "text_dim": 16,
    }
    if extra:
        cfg.update(extra)
    return cfg


def test_tri_system_bridge_layers_default_to_full_range(monkeypatch):
    """When cfg supplies neither bridge_layers nor bridge_interval, every video
    layer participates — this preserves the prior hard-coded default behavior.
    """
    _Arch = _make_stub_tri_arch(monkeypatch, num_video_layers=3)
    arch = _Arch(_tri_arch_min_cfg())
    assert arch.action_backbone.num_layers == 3
    assert arch.action_backbone.bridge_layers == (0, 1, 2)


def test_tri_system_bridge_layers_accepts_explicit_full_list(monkeypatch):
    """Explicit ``bridge_layers`` matching ``vb.num_layers`` works (parity with dual)."""
    _Arch = _make_stub_tri_arch(monkeypatch, num_video_layers=3)
    arch = _Arch(_tri_arch_min_cfg({"bridge_layers": [0, 1, 2]}))
    assert arch.action_backbone.bridge_layers == (0, 1, 2)


def test_tri_system_bridge_layers_accepts_interval_one(monkeypatch):
    """``bridge_interval: 1`` is equivalent to full-range bridge."""
    _Arch = _make_stub_tri_arch(monkeypatch, num_video_layers=4)
    arch = _Arch(_tri_arch_min_cfg({"bridge_interval": 1}))
    assert arch.action_backbone.bridge_layers == (0, 1, 2, 3)


def test_tri_system_sparse_bridge_layers_rejected_by_driver(monkeypatch):
    """Sparse bridges (``len(bl) < vb.num_layers``) are not supported by the
    trimodal MoT driver — must raise at driver construction time. Same constraint
    as dual_system joint_self_attn.
    """
    _Arch = _make_stub_tri_arch(monkeypatch, num_video_layers=3)
    with pytest.raises(ValueError, match="num_layers"):
        _Arch(_tri_arch_min_cfg({"bridge_layers": [0, 2]}))  # missing layer 1


def test_tri_system_yaml_bridge_layers_resolve(monkeypatch):
    """``configs/model/tri_system.yaml`` defaults must resolve to full-range bridge."""
    from openwam.model import resolve_architecture_config

    cfg = OmegaConf.load("configs/model/tri_system.yaml")
    resolved = resolve_architecture_config(cfg)
    # bridge_interval=1 in yaml → resolved param present; bridge_layers null is intentional.
    assert resolved.params["bridge_interval"] == 1
    assert resolved.params["bridge_layers"] is None


def test_tri_system_mot_driver_trimodal_cpu():
    torch.manual_seed(0)
    vb, ab, ub = _make_tiny_trimodal_components()
    vstate = _make_tiny_video_state(vb.dit)
    noisy_actions = torch.randn(2, 4, ab.action_dim)
    timestep = torch.tensor([10.0, 20.0])
    context = torch.randn(2, 3, ab.text_dim)
    context_mask = torch.ones(2, 3, dtype=torch.bool)
    astate = ab.prepare_state(noisy_actions, timestep, context=context, context_mask=context_mask)
    ustate = ub.prepare_state(torch.randn(2, 5, ub.cfg.vlm_input_dim))

    driver = TriSystemMoTDriver(vb, ab, ub)
    vstate, astate, ustate = driver.run_joint_loop(vstate, astate, ustate)

    assert vstate.hidden_states.shape == (2, 6, vb.dim)
    assert astate.payload.x_action.shape == (2, 4, ab.dim)
    assert ustate.und_tokens.shape == (2, 5, ub.cfg.dim)
    assert torch.isfinite(vstate.hidden_states).all()
    assert torch.isfinite(astate.payload.x_action).all()
    assert torch.isfinite(ustate.und_tokens).all()


def test_tri_system_mot_compile_core_matches_eager_cpu():
    torch.manual_seed(0)
    vb, ab, ub = _make_tiny_trimodal_components()
    vb_core = copy.deepcopy(vb)
    ab_core = copy.deepcopy(ab)
    ub_core = copy.deepcopy(ub)

    vstate, astate, ustate = _tri_system_mot_states(vb, ab, ub, seed=123)
    vstate_core, astate_core, ustate_core = _tri_system_mot_states(vb_core, ab_core, ub_core, seed=123)

    eager = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False)
    core = TriSystemMoTDriver(vb_core, ab_core, ub_core, mot_checkpoint_mixed_attn=False)

    with torch.no_grad():
        vstate, astate, ustate = eager.run_joint_loop(vstate, astate, ustate)
        video_tokens_per_frame = core._video_tokens_per_frame(vstate_core)
        attn_mask = core._build_attention_mask(
            s_video=int(vstate_core.grid_frames) * video_tokens_per_frame,
            s_action=core._get_action_tokens(astate_core).shape[1],
            s_understanding=ustate_core.und_tokens.shape[1],
            video_tokens_per_frame=video_tokens_per_frame,
            device=vstate_core.hidden_states.device,
            und_mask=getattr(ustate_core, "und_mask", None),
        )
        vstate_core, astate_core, ustate_core = core.run_joint_loop_for_compile(
            vstate_core,
            astate_core,
            ustate_core,
            attn_mask=attn_mask,
        )

    assert torch.allclose(vstate_core.hidden_states, vstate.hidden_states, atol=1.0e-5, rtol=1.0e-5)
    assert torch.allclose(
        core._get_action_tokens(astate_core),
        eager._get_action_tokens(astate),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(ustate_core.und_tokens, ustate.und_tokens, atol=1.0e-5, rtol=1.0e-5)


def _tri_system_mot_states(vb, ab, ub, seed: int, *, und_mask: torch.Tensor | None = None):
    torch.manual_seed(seed)
    batch = 2
    vstate = _make_tiny_video_state(vb.dit, batch=batch, grid_frames=1, grid_height=2, grid_width=3)
    noisy_actions = torch.randn(batch, 4, ab.action_dim)
    timestep = torch.tensor([10.0, 20.0])
    context = torch.randn(batch, 3, ab.text_dim)
    context_mask = torch.ones(batch, 3, dtype=torch.bool)
    astate = ab.prepare_state(noisy_actions, timestep, context=context, context_mask=context_mask)
    vlm_hidden = torch.randn(batch, 5, ub.cfg.vlm_input_dim)
    ustate = ub.prepare_state(vlm_hidden, vlm_attention_mask=und_mask)
    return vstate, astate, ustate


def _patch_wan_flash_attention_to_sdpa(monkeypatch):
    """Keep CPU-only tests off CUDA-only flash-attn kernels when installed."""
    from openwam.model.video_backbone.wan.models import dit as wan_dit

    def _sdpa(q, k, v, num_heads: int, compatibility_mode=False, attn_mask=None):  # noqa: ARG001
        q = wan_dit.rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = wan_dit.rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = wan_dit.rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return wan_dit.rearrange(out, "b n s d -> b s (n d)", n=num_heads)

    monkeypatch.setattr(wan_dit, "flash_attention", _sdpa)


@pytest.fixture(autouse=True)
def _wan_attention_cpu_fallback(monkeypatch):
    _patch_wan_flash_attention_to_sdpa(monkeypatch)


def test_tri_system_joint_mask_layout():
    torch.manual_seed(0)
    vb, ab, ub = _make_tiny_trimodal_components()
    driver = TriSystemMoTDriver(vb, ab, ub, attention_mask_mode=ACTION_SEES_VIDEO)

    Sv, Sa, Su = 6, 4, 5
    mask = driver._build_attention_mask(  # noqa: SLF001 - targeted mask contract test
        s_video=Sv,
        s_action=Sa,
        s_understanding=Su,
        video_tokens_per_frame=3,
        device=torch.device("cpu"),
    )

    assert mask is not None
    assert mask.shape == (Sv + Sa + Su, Sv + Sa + Su)
    assert mask.dtype == torch.bool
    assert not mask[:Sv, Sv : Sv + Sa].any()
    assert mask[:Sv, Sv + Sa :].all()
    assert mask[Sv : Sv + Sa].all()
    assert not mask[Sv + Sa :, : Sv + Sa].any()
    assert mask[Sv + Sa :, Sv + Sa :].all()


def test_tri_system_joint_mask_blocks_action_indirect_video_coupling():
    """Changing action inputs must not perturb video output under joint mask.

    Understanding rows are isolated from video/action keys, so they cannot carry
    action information forward into later video layers via v->u attention.
    """

    torch.manual_seed(0)
    vb, ab, ub = _make_tiny_trimodal_components()
    ab.eval()
    ub.eval()
    driver = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False, attention_mask_mode=ACTION_SEES_VIDEO)

    batch, s_action = 1, 4
    video_x = torch.randn(batch, 4, vb.dim)
    t_mod = torch.zeros(batch, 6, vb.dim)
    video_context = torch.randn(batch, 4, vb.dim)
    context = torch.randn(batch, 3, ab.text_dim)
    context_mask = torch.ones(batch, 3, dtype=torch.bool)
    vlm_hidden = torch.randn(batch, 5, ub.cfg.vlm_input_dim)

    def _run(action_seed: int) -> torch.Tensor:
        vstate = _make_tiny_video_state(vb.dit, batch=batch, grid_frames=1, grid_height=2, grid_width=2)
        vstate.hidden_states = video_x.clone()
        vstate.time_mod = t_mod.clone()
        vstate.context = video_context.clone()
        actions = torch.randn(batch, s_action, ab.action_dim, generator=torch.Generator().manual_seed(action_seed))
        astate = ab.prepare_state(actions, torch.zeros(batch), context=context, context_mask=context_mask)
        ustate = ub.prepare_state(vlm_hidden)
        with torch.no_grad():
            vstate, _, _ = driver.run_joint_loop(vstate, astate, ustate)
        return vstate.hidden_states

    out_a = _run(action_seed=11)
    out_b = _run(action_seed=22)
    assert torch.allclose(out_a, out_b, atol=1e-6), (
        f"tri_system joint mask leaked action into video; max diff = {(out_a - out_b).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "mode, v_sees_a, a_sees_all_v",
    [
        (MUTUAL, True, True),
        (ACTION_SEES_VIDEO, False, True),
        (VIDEO_SEES_ACTION, True, False),
        (ISOLATED, False, False),
    ],
)
def test_tri_system_cross_modal_modes_with_und_tail(mode, v_sees_a, a_sees_all_v):
    """All four modes drive the trimodal mask; understanding stays a read-only
    tail (everyone sees u, u sees only itself) regardless of the v↔a mode."""
    vb, ab, ub = _make_tiny_trimodal_components()
    driver = TriSystemMoTDriver(vb, ab, ub, attention_mask_mode=mode)

    Sv, Sa, Su, ff = 6, 2, 4, 2
    mask = driver._build_attention_mask(  # noqa: SLF001 - targeted mask contract test
        s_video=Sv,
        s_action=Sa,
        s_understanding=Su,
        video_tokens_per_frame=ff,
        device=torch.device("cpu"),
    )
    u_start = Sv + Sa
    # a↔a full.
    assert mask[Sv:u_start, Sv:u_start].all()
    # understanding read-only tail: everyone sees u; u sees only u.
    assert mask[:u_start, u_start:].all()
    assert mask[u_start:, u_start:].all()
    assert not mask[u_start:, :u_start].any()
    # v→a: first-frame rows never see action; later rows follow the mode.
    assert not mask[:ff, Sv:u_start].any()
    assert mask[ff:Sv, Sv:u_start].all() if v_sees_a else not mask[ff:Sv, Sv:u_start].any()
    # a→v: all video, or first frame only.
    if a_sees_all_v:
        assert mask[Sv:u_start, :Sv].all()
    else:
        assert mask[Sv:u_start, :ff].all()
        assert not mask[Sv:u_start, ff:Sv].any()


def test_tri_system_yaml_mask_settings_resolve_to_driver():
    from openwam.model import resolve_architecture_config

    cfg = OmegaConf.load("configs/model/tri_system.yaml")
    resolved = resolve_architecture_config(cfg)

    assert resolved.registry_name == "tri_system_joint_self_attn"
    assert resolved.params["mot_checkpoint_mixed_attn"] is True

    # The YAML is free to select any mode accepted by the driver.  This test
    # verifies propagation, not one particular training policy.
    attention_mask_mode = str(resolved.params["attention_mask_mode"])
    video_attention_mask_mode = str(resolved.params["video_attention_mask_mode"])

    vb, ab, ub = _make_tiny_trimodal_components()
    driver = TriSystemMoTDriver(
        vb,
        ab,
        ub,
        mot_checkpoint_mixed_attn=bool(resolved.params["mot_checkpoint_mixed_attn"]),
        attention_mask_mode=attention_mask_mode,
        video_attention_mask_mode=video_attention_mask_mode,
    )
    assert driver.mot_checkpoint_mixed_attn is True

    assert driver.attention_mask_mode == attention_mask_mode
    assert vb.video_attention_mask_mode == video_attention_mask_mode

    Sv, Sa, Su, tokens_per_frame = 6, 4, 5, 3
    mask = driver._build_attention_mask(  # noqa: SLF001 - config-to-mask contract test
        s_video=Sv,
        s_action=Sa,
        s_understanding=Su,
        video_tokens_per_frame=tokens_per_frame,
        device=torch.device("cpu"),
    )

    assert mask is not None
    expected = build_cross_modal_attention_mask(
        vb,
        s_video=Sv,
        s_action=Sa,
        video_tokens_per_frame=tokens_per_frame,
        mode=attention_mask_mode,
        device=torch.device("cpu"),
        n_readonly_tail=Su,
    )
    assert torch.equal(mask, expected)


class _FakeQwenModel(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.config = type("Cfg", (), {"text_config": type("TextCfg", (), {"hidden_size": hidden_size})()})()
        self.param = nn.Parameter(torch.zeros(1))
        self.seen = None
        self.model = self

    def forward(self, **kwargs):
        self.seen = kwargs
        input_ids = kwargs["input_ids"]
        bsz, seq_len = input_ids.shape
        hidden = torch.arange(
            bsz * seq_len * self.config.text_config.hidden_size,
            dtype=torch.float32,
            device=self.param.device,
        )
        hidden = hidden.view(bsz, seq_len, self.config.text_config.hidden_size)
        hidden = hidden + self.param
        return type("Out", (), {"hidden_states": (hidden + 1, hidden + 2), "last_hidden_state": hidden + 3})()


def _make_qwen_backbone_for_test():
    backbone = object.__new__(Qwen3VLBackbone)
    nn.Module.__init__(backbone)
    backbone.dtype = torch.float32
    backbone._checkpoint_path = "fake"
    backbone.processor = None
    backbone.vlm_model = _FakeQwenModel()
    backbone.add_module("vlm_model", backbone.vlm_model)
    return backbone


def test_qwen3_vl_extract_features_batched_dict_fake():
    backbone = _make_qwen_backbone_for_test()
    inputs = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "pixel_values": torch.zeros(2, 4),
        "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
    }
    out = backbone.extract_features(inputs)
    assert out.shape == (2, 3, backbone.hidden_size)
    assert out.requires_grad is True
    assert backbone.vlm_model.seen["use_cache"] is False
    assert backbone.vlm_model.seen["pixel_values"].dtype == torch.float32


def test_qwen3_vl_extract_features_list_fake_pads_and_concats():
    backbone = _make_qwen_backbone_for_test()
    backbone.processor = type("Processor", (), {"tokenizer": type("Tokenizer", (), {"pad_token_id": 99})()})()
    inputs = [
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": torch.zeros(1, 4),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        },
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "pixel_values": torch.zeros(1, 4),
            "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        },
    ]
    out = backbone.extract_features(inputs)
    assert out.shape == (2, 4, backbone.hidden_size)
    assert torch.equal(backbone.vlm_model.seen["attention_mask"][0], torch.tensor([1, 1, 0, 0]))
    assert torch.equal(backbone.vlm_model.seen["input_ids"][0], torch.tensor([1, 1, 99, 99]))


def test_qwen3_vl_batch_inputs_rejects_bad_attention_mask_shape():
    backbone = _make_qwen_backbone_for_test()
    with pytest.raises(ValueError, match="attention_mask"):
        backbone.batch_vlm_inputs(
            [
                {
                    "input_ids": torch.ones(1, 2, dtype=torch.long),
                    "attention_mask": torch.ones(1, 3, dtype=torch.long),
                }
            ]
        )


def test_qwen3_vl_pad_token_id_raises_when_missing():
    """`_pad_token_id` must raise (not silently default to 0) when neither processor
    nor model.config provides one. Returning 0 would silently treat token id 0 as
    padding even though id 0 may be a meaningful token in the tokenizer
    (e.g. `<|endoftext|>`), corrupting downstream visual-grid bookkeeping.
    """
    backbone = _make_qwen_backbone_for_test()
    # Fake's processor is None and the fake model.config has no pad_token_id
    # → fallback path must raise.
    with pytest.raises(ValueError, match="pad_token_id"):
        backbone._pad_token_id()  # noqa: SLF001 - direct contract test


def test_tri_system_generate_reuses_cached_vlm_hidden(monkeypatch):
    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture

    class _Arch(TriSystemJointSelfAttnArchitecture):
        def __init__(self):
            nn.Module.__init__(self)
            self.action_backbone = nn.Module()
            self.vlm_backbone = type("VLM", (), {})()
            self.calls = 0

            def _prepare(prompts, images):  # noqa: ARG001
                return {"input_ids": torch.ones(1, 2, dtype=torch.long)}

            def _extract(vlm_inputs):  # noqa: ARG001
                self.calls += 1
                return torch.ones(1, 2, 3)

            self.vlm_backbone.prepare_vlm_inputs = _prepare
            self.vlm_backbone.extract_features = _extract

    captured = {}

    def _fake_base_generate(self, schedule, prompt, *, first_frame_image=None, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return {"video": None, "actions": None}

    monkeypatch.setattr(
        "openwam.model.architectures.base.BaseWAMArchitecture.generate",
        _fake_base_generate,
    )

    arch = _Arch()
    result = arch.generate([(1, 1), (0, 0)], "pick", first_frame_image=[object()])

    assert result == {"video": None, "actions": None}
    assert arch.calls == 1
    assert "vlm_inputs" not in captured
    assert torch.equal(captured["vlm_hidden"], torch.ones(1, 2, 3))


def test_save_checkpoint_excludes_vlm_backbone(tmp_path):
    """``save_checkpoint`` must exclude ``vlm_backbone.*`` params.

    tri_system's Qwen3-VL has tied weights (lm_head ↔ embed_tokens).
    Rather than deduplicating tied storage in safetensors, we exclude the
    entire VLM from the safetensors file — it is saved as a separate
    checkpoint directory by the trainer. This test verifies:
    (a) vlm_backbone params are not in the saved file,
    (b) non-VLM params round-trip correctly,
    (c) load_checkpoint with vlm_backbone present uses strict=False.
    """
    from openwam.model.architectures.base import BaseWAMArchitecture

    class _VLMArch(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self.action_head = nn.Linear(16, 8)
            self.vlm_backbone = nn.Linear(16, 32)

        def forward(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    arch = _VLMArch()
    with torch.no_grad():
        arch.action_head.weight.copy_(torch.randn_like(arch.action_head.weight))

    out = tmp_path / "vlm_ckpt.safetensors"
    arch.save_checkpoint(str(out))
    assert out.exists() and out.stat().st_size > 0

    from safetensors.torch import load_file

    saved = load_file(str(out))
    assert not any(k.startswith("vlm_backbone.") for k in saved), "vlm_backbone params should be excluded"
    assert any(k.startswith("action_head.") for k in saved), "non-VLM params should be saved"

    reloaded = _VLMArch()
    reloaded.load_checkpoint(str(out))
    assert torch.equal(reloaded.action_head.weight, arch.action_head.weight)


def test_save_load_round_trip_vlm_tied_weights_excluded(tmp_path):
    """VLM's tied weights (lm_head <-> embed_tokens) are excluded from safetensors.

    The save/load round-trip must succeed because ``_exclude_vlm_from_state_dict``
    removes all ``vlm_backbone.*`` keys (including the tied pair) before
    ``save_file`` sees them.  Non-VLM params round-trip with value equality.
    """
    from openwam.model.architectures.base import BaseWAMArchitecture

    class _VLMWithTiedWeights(BaseWAMArchitecture):
        def __init__(self):
            super().__init__(cfg=None)
            self.action_head = nn.Linear(16, 8)
            # Simulate Qwen3-VL: vlm_backbone has tied weights internally
            self.vlm_backbone = nn.Module()
            self.vlm_backbone.embed_tokens = nn.Embedding(10, 16)
            self.vlm_backbone.lm_head = nn.Linear(16, 10, bias=False)
            self.vlm_backbone.lm_head.weight = self.vlm_backbone.embed_tokens.weight  # tied

        def forward(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    arch = _VLMWithTiedWeights()
    # VLM tied weights exist
    assert arch.vlm_backbone.embed_tokens.weight.data_ptr() == arch.vlm_backbone.lm_head.weight.data_ptr()

    with torch.no_grad():
        arch.action_head.weight.copy_(torch.randn_like(arch.action_head.weight))

    out = tmp_path / "tied_vlm.safetensors"
    arch.save_checkpoint(str(out))  # must not crash — VLM ties excluded

    from safetensors.torch import load_file

    saved = load_file(str(out))
    assert not any(k.startswith("vlm_backbone.") for k in saved), "vlm_backbone params should be excluded"

    reloaded = _VLMWithTiedWeights()
    reloaded.load_checkpoint(str(out))
    assert torch.equal(reloaded.action_head.weight, arch.action_head.weight)


def test_tri_system_forward_rejects_vlm_hidden_batch_mismatch():
    """Cached vlm_hidden with wrong batch size must raise, not silently miscompute.

    We monkeypatch ``vb.prepare`` to return a pre-built ``vstate`` so the
    test doesn't need a fully-configured video pipeline. The validation
    lives between ``vb.prepare()`` and ``ub.prepare_state()`` in forward().
    """
    torch.manual_seed(0)
    vb, ab, ub = _make_tiny_trimodal_components()
    dit = vb.dit  # noqa: SLF001

    # batch=2 video state
    vstate = _make_tiny_video_state(dit, batch=2)

    # Monkeypatch vb.prepare to return our pre-built vstate
    vb.prepare = lambda **kw: vstate  # noqa: ARG005

    arch = _make_forward_tri_arch(vb, ab, ub)
    # batch=1 vlm_hidden vs batch=2 video state — mismatch.
    # ``context`` and ``context_mask`` go via **pipeline_inputs and are
    # extracted as ``action_context`` inside forward().
    with pytest.raises(ValueError, match="vlm_hidden batch size"):
        arch.forward(
            noisy_actions=torch.randn(2, 4, ab.action_dim),
            action_timestep=torch.tensor([10.0, 20.0]),
            context=torch.randn(2, 3, ab.text_dim),
            context_mask=torch.ones(2, 3, dtype=torch.bool),
            vlm_hidden=torch.randn(1, 5, ub.cfg.vlm_input_dim),
            vlm_attention_mask=torch.ones(1, 5, dtype=torch.bool),
        )


def test_tri_system_forward_applies_vace_hints_not_rejected():
    """tri_system + VACE: a vstate carrying ``vace_hints`` must no longer raise
    (the old ``NotImplementedError("tri_system + VACE not supported")`` guard is
    gone) and the per-video-block hint residual must reach the output via the
    shared ``post_attn_at_layer`` → ``apply_post_block_residuals`` path.
    """
    vb, ab, ub = _make_tiny_trimodal_components()
    dit = vb.dit  # noqa: SLF001

    arch = _make_forward_tri_arch(vb, ab, ub)

    torch.manual_seed(0)
    fwd_kwargs = dict(
        noisy_actions=torch.randn(2, 4, ab.action_dim),
        action_timestep=torch.tensor([10.0, 20.0]),
        context=torch.randn(2, 3, ab.text_dim),
        context_mask=torch.ones(2, 3, dtype=torch.bool),
        vlm_hidden=torch.randn(2, 5, ub.cfg.vlm_input_dim),
        vlm_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    # seq_len = grid 1*2*3 = 6, dim = 32 (tiny components defaults).
    hint = torch.randn(2, 6, 32)

    class _StubVace:
        # Block 0 receives hint[0]; mirrors WanVACE.vace_layers_mapping.
        vace_layers_mapping = {0: 0}

    # The tiny mock DiT head is ``Identity`` (cannot consume the time embedding),
    # so bypass the real head+unpatchify; we only need the post-MoT video hidden
    # states (which carry the VACE residual) to compare base vs vace.
    vb.finalize = lambda state: state.hidden_states  # noqa: ARG005

    def _fresh_state(*, with_vace):
        # Re-seed so the base vstate (hidden_states/time_mod/context/time_embed)
        # is identical across the two runs — only ``vace_hints`` differs.
        torch.manual_seed(1)
        state = _make_tiny_video_state(dit, batch=2)
        if with_vace:
            state.vace_hints = [hint.clone()]
            state.extras["vace"] = _StubVace()
        return state

    vb.prepare = lambda **kw: _fresh_state(with_vace=False)  # noqa: ARG005
    v_base, a_base = arch.forward(**fwd_kwargs)
    assert torch.isfinite(v_base).all() and torch.isfinite(a_base).all()

    vb.prepare = lambda **kw: _fresh_state(with_vace=True)  # noqa: ARG005
    v_vace, a_vace = arch.forward(**fwd_kwargs)  # must NOT raise NotImplementedError
    assert torch.isfinite(v_vace).all() and torch.isfinite(a_vace).all()
    # The VACE residual at block 0 must propagate to the video output (and, via
    # trimodal joint attention, to the action output).
    assert not torch.equal(v_base, v_vace), "vace_hints did not affect the video output — hint not applied"
    assert not torch.equal(a_base, a_vace), "vace_hints did not affect the action output via joint attention"


def test_und_mask_baseline_no_mask_unchanged():
    """Not passing und_mask keeps the 2D [S, S] mask path — output bit-for-bit equal to baseline."""
    torch.manual_seed(0)
    vb, ab, ub = _make_tiny_trimodal_components()
    ab.eval()
    ub.eval()
    driver = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False, attention_mask_mode=ACTION_SEES_VIDEO)

    Sv, Sa, Su, tokens_per_frame = 6, 4, 5, 3
    base_only = driver._build_attention_mask(  # noqa: SLF001 - mask contract test
        s_video=Sv,
        s_action=Sa,
        s_understanding=Su,
        video_tokens_per_frame=tokens_per_frame,
        device=torch.device("cpu"),
    )
    all_valid = driver._build_attention_mask(  # noqa: SLF001 - mask contract test
        s_video=Sv,
        s_action=Sa,
        s_understanding=Su,
        video_tokens_per_frame=tokens_per_frame,
        device=torch.device("cpu"),
        und_mask=torch.ones(2, Su, dtype=torch.bool),
    )
    assert base_only is not None and all_valid is not None
    assert base_only.dim() == 2
    # all-valid mask should fall back to the 2D base (no per-batch expansion)
    assert all_valid.dim() == 2
    assert torch.equal(all_valid, base_only)


def test_und_mask_per_batch_expansion_blocks_padding_keys():
    """Padded und positions become invalid KEYS for video/action/und queries."""
    torch.manual_seed(0)
    vb, ab, ub = _make_tiny_trimodal_components()
    driver = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False, attention_mask_mode=ACTION_SEES_VIDEO)

    Sv, Sa, Su, tokens_per_frame = 6, 4, 5, 3
    und_mask = torch.tensor(
        [
            [True, True, True, True, True],  # batch 0: all valid
            [True, True, True, False, False],  # batch 1: last 2 padded
        ],
        dtype=torch.bool,
    )
    mask = driver._build_attention_mask(  # noqa: SLF001 - mask contract test
        s_video=Sv,
        s_action=Sa,
        s_understanding=Su,
        video_tokens_per_frame=tokens_per_frame,
        device=torch.device("cpu"),
        und_mask=und_mask,
    )
    assert mask is not None
    assert mask.shape == (2, 1, Sv + Sa + Su, Sv + Sa + Su)
    u_start = Sv + Sa
    # Batch 0 == baseline 2D mask
    base_2d = build_cross_modal_attention_mask(
        vb,
        s_video=Sv,
        s_action=Sa,
        video_tokens_per_frame=tokens_per_frame,
        mode=driver.attention_mask_mode,
        device=torch.device("cpu"),
        n_readonly_tail=Su,
    )
    assert torch.equal(mask[0, 0], base_2d)
    # Batch 1: last 2 und KEY columns are False for any query (other than padded und self-row)
    assert not mask[1, 0, :u_start, u_start + 3 :].any()
    # The und→u block: padded query rows are self-only (eye); valid query rows still see only valid und
    assert mask[1, 0, u_start + 3, u_start + 3] is not None
    assert bool(mask[1, 0, u_start + 3, u_start + 3]) is True
    # Padded und QUERY row should NOT attend to any non-self position
    pad_row = mask[1, 0, u_start + 3]
    keep = torch.zeros_like(pad_row)
    keep[u_start + 3] = True
    assert torch.equal(pad_row, keep)
    # Valid und queries still see valid und keys but NOT padded und keys
    assert mask[1, 0, u_start, u_start]
    assert not mask[1, 0, u_start, u_start + 3]
    assert not mask[1, 0, u_start, u_start + 4]


def test_und_mask_blocks_padding_leak_end_to_end():
    """End-to-end: replacing the padded portion of und_tokens must NOT change video/action outputs."""
    torch.manual_seed(0)
    vb, ab, ub = _make_tiny_trimodal_components()
    ab.eval()
    ub.eval()
    driver = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False, attention_mask_mode=ACTION_SEES_VIDEO)

    batch = 2
    s_action = 4
    valid_su = 3
    total_su = 5  # last 2 padded for batch index 1; batch 0 is fully valid

    vlm_hidden_base = torch.randn(batch, total_su, ub.cfg.vlm_input_dim)
    vlm_hidden_perturbed = vlm_hidden_base.clone()
    # Perturb ONLY the padded region of batch 1 — must not affect outputs.
    vlm_hidden_perturbed[1, valid_su:] = torch.randn(total_su - valid_su, ub.cfg.vlm_input_dim) * 100

    actions = torch.randn(batch, s_action, ab.action_dim)
    context = torch.randn(batch, 3, ab.text_dim)
    context_mask = torch.ones(batch, 3, dtype=torch.bool)
    und_mask = torch.tensor(
        [[True] * total_su, [True] * valid_su + [False] * (total_su - valid_su)],
        dtype=torch.bool,
    )

    def _run(vlm_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Deterministic vstate: seed before each construction since
        # _make_tiny_video_state uses torch.randn internally.
        torch.manual_seed(42)
        vstate = _make_tiny_video_state(vb.dit, batch=batch, grid_frames=1, grid_height=2, grid_width=3)
        astate = ab.prepare_state(actions.clone(), torch.zeros(batch), context=context, context_mask=context_mask)
        ustate = ub.prepare_state(vlm_hidden, vlm_attention_mask=und_mask)
        with torch.no_grad():
            vstate, astate, _ = driver.run_joint_loop(vstate, astate, ustate)
        return vstate.hidden_states, astate.payload.x_action

    v0, a0 = _run(vlm_hidden_base)
    v1, a1 = _run(vlm_hidden_perturbed)

    # Batch 0 unchanged (all valid und); padded perturbation in batch 1 must NOT propagate.
    assert torch.allclose(v0, v1, atol=1e-5), (
        f"padded und positions leaked into video output; max diff = {(v0 - v1).abs().max().item()}"
    )
    assert torch.allclose(a0, a1, atol=1e-5), (
        f"padded und positions leaked into action output; max diff = {(a0 - a1).abs().max().item()}"
    )


def test_und_mask_shape_validation():
    _, _, ub = _make_tiny_trimodal_components()
    vlm_hidden = torch.randn(2, 5, ub.cfg.vlm_input_dim)
    # Wrong [B, L] shape
    with pytest.raises(ValueError, match="shape must match"):
        ub.prepare_state(vlm_hidden, vlm_attention_mask=torch.ones(2, 4, dtype=torch.bool))
    # Wrong ndim
    with pytest.raises(ValueError, match="must be 2D"):
        ub.prepare_state(vlm_hidden, vlm_attention_mask=torch.ones(2, 5, 1, dtype=torch.bool))


def test_freeze_modules_wraps_vlm_extract_features_in_no_grad():
    """End-to-end: ``freeze_modules`` on a tri-system arch should make the VLM
    forward output non-grad-tracking — both ``requires_grad=False`` on params
    and ``forward`` wrapped in ``no_grad``, without any backbone-internal check.
    """
    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture

    backbone = _make_qwen_backbone_for_test()

    arch = TriSystemJointSelfAttnArchitecture.__new__(TriSystemJointSelfAttnArchitecture)
    BaseWAMArchitecture.__init__(arch, cfg=None)
    arch.vlm_backbone = backbone
    arch.add_module("vlm_backbone", backbone)

    frozen = arch.freeze_modules(["vlm_backbone.vlm_model"])
    assert frozen == ["vlm_backbone.vlm_model"]
    assert not any(p.requires_grad for p in backbone.vlm_model.parameters())

    inputs = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    out = backbone.extract_features(inputs)
    assert out.requires_grad is False, "frozen VLM forward should not build a backward graph"


def test_freeze_modules_keeps_grad_when_not_frozen():
    """Without ``freeze_modules``, VLM forward keeps grad — no hidden behavior."""
    backbone = _make_qwen_backbone_for_test()
    assert any(p.requires_grad for p in backbone.vlm_model.parameters())
    inputs = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    out = backbone.extract_features(inputs)
    assert out.requires_grad is True


def test_freeze_modules_covers_qwen3vl_inner_model_bypass():
    """Production bypass-path regression: real Qwen3VLForConditionalGeneration has
    ``vlm_model.model`` as a SEPARATE inner ``Qwen3VLModel`` (not aliased to self).
    ``Qwen3VLBackbone.extract_features`` calls ``self.vlm_model.model(...)``
    directly to skip the LM head — so ``freeze_modules(["vlm_backbone.vlm_model"])``
    must recursively wrap the inner ``.model``'s forward in ``no_grad`` too.

    Without recursive wrapping, only ``vlm_model.forward`` is wrapped; the
    actually-called ``vlm_model.model.forward`` stays grad-tracking and the
    activation memory we tried to save is silently never released.
    """
    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture

    class _InnerQwenModel(nn.Module):
        """Mimics Qwen3VLModel — the inner transformer that extract_features actually calls."""

        def __init__(self, hidden_size=6):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
            self._hidden = hidden_size

        def forward(self, **kwargs):
            input_ids = kwargs["input_ids"]
            bsz, seq_len = input_ids.shape
            hidden = (
                torch.arange(bsz * seq_len * self._hidden, dtype=torch.float32, device=self.param.device).view(
                    bsz, seq_len, self._hidden
                )
                + self.param
            )
            return type("Out", (), {"hidden_states": (hidden,), "last_hidden_state": hidden})()

    class _FakeQwenForCondGen(nn.Module):
        """Mimics Qwen3VLForConditionalGeneration — wraps .model + has its own LM head."""

        def __init__(self, hidden_size=6):
            super().__init__()
            self.config = type("Cfg", (), {"text_config": type("TextCfg", (), {"hidden_size": hidden_size})()})()
            self.model = _InnerQwenModel(hidden_size)
            self.lm_head = nn.Linear(hidden_size, hidden_size)

        def forward(self, **kwargs):  # pragma: no cover - extract_features bypasses this
            out = self.model(**kwargs)
            logits = self.lm_head(out.last_hidden_state)
            return type("CondGenOut", (), {"hidden_states": out.hidden_states, "logits": logits})()

    backbone = object.__new__(Qwen3VLBackbone)
    nn.Module.__init__(backbone)
    backbone.dtype = torch.float32
    backbone._checkpoint_path = "fake"
    backbone.processor = None
    backbone.vlm_model = _FakeQwenForCondGen()
    backbone.add_module("vlm_model", backbone.vlm_model)

    arch = TriSystemJointSelfAttnArchitecture.__new__(TriSystemJointSelfAttnArchitecture)
    BaseWAMArchitecture.__init__(arch, cfg=None)
    arch.vlm_backbone = backbone
    arch.add_module("vlm_backbone", backbone)

    arch.freeze_modules(["vlm_backbone.vlm_model"])

    # Inner .model has its own forward, distinct from .forward — verify both are wrapped.
    assert getattr(backbone.vlm_model, "_openwam_no_grad_wrapped", False) is True
    assert getattr(backbone.vlm_model.model, "_openwam_no_grad_wrapped", False) is True

    # The production call path: extract_features → vlm_model.model(...) — bypasses vlm_model.forward.
    inputs = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    out = backbone.extract_features(inputs)
    assert out.requires_grad is False, "frozen VLM inner-model forward built a graph — recursive wrap regressed"


def test_tri_system_freeze_modules_rejects_understanding_expert():
    """``freeze_modules`` must refuse to freeze ``understanding_expert``."""
    from openwam.model.architectures.tri_system.joint_self_attn import TriSystemJointSelfAttnArchitecture

    arch = TriSystemJointSelfAttnArchitecture(cfg=None)
    arch.understanding_expert = nn.Linear(4, 4)

    with pytest.raises(ValueError, match="refusing to freeze"):
        arch.freeze_modules(["understanding_expert"])

    # Freezing other modules should still work
    arch.some_module = nn.Linear(4, 4)
    frozen = arch.freeze_modules(["some_module"])
    assert "some_module" in frozen


# ---------------------------------------------------------------------------
# Backward / gradient-flow regression guards
# ---------------------------------------------------------------------------


def test_tri_system_backward_per_block_modulation_grad():
    """Every video DiT block must receive nonzero modulation gradients.

    The closure-snapshot bug routes ALL gradients to the last block, leaving
    earlier blocks with zero grad.  This test catches it using tiny models
    on CPU — no GPU or real weights needed.
    """
    torch.manual_seed(0)
    num_layers = 3
    vb, ab, ub = _make_tiny_trimodal_components(num_layers=num_layers)
    for p in vb.parameters():
        p.requires_grad_(True)
    for p in ab.parameters():
        p.requires_grad_(True)
    for p in ub.parameters():
        p.requires_grad_(True)

    dit = vb.dit  # noqa: SLF001
    vstate = _make_tiny_video_state(dit, batch=2)
    actions = torch.randn(2, 4, ab.action_dim)
    timestep = torch.tensor([10.0, 20.0])
    context = torch.randn(2, 3, ab.text_dim)
    context_mask = torch.ones(2, 3, dtype=torch.bool)
    astate = ab.prepare_state(actions, timestep, context=context, context_mask=context_mask)
    ustate = ub.prepare_state(torch.randn(2, 5, ub.cfg.vlm_input_dim))

    driver = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False)
    vstate, astate, ustate = driver.run_joint_loop(vstate, astate, ustate)
    pred = ab.extract_prediction(astate)

    loss = vstate.hidden_states.float().sum() + pred.float().sum() + ustate.und_tokens.float().sum()
    loss.backward()

    for i, block in enumerate(dit.blocks):
        grad = block.modulation.grad
        assert grad is not None, f"block {i} modulation has no grad"
        assert grad.abs().sum() > 0, f"block {i} modulation grad is all-zero — closure bug"


def test_tri_system_per_token_tmod_forward():
    """4D t_mod (per-token modulation) path must not crash."""
    torch.manual_seed(0)
    vb, ab, ub = _make_tiny_trimodal_components()
    dim = vb.dim

    dit = vb.dit  # noqa: SLF001
    batch, seq_len = 2, 6
    vstate = _make_tiny_video_state(dit, batch=batch)
    # 4D t_mod: [B, S, 6, dim] — per-token modulation path (wan_backbone.py:575)
    vstate.time_mod = torch.randn(batch, seq_len, 6, dim)

    actions = torch.randn(batch, 4, ab.action_dim)
    timestep = torch.tensor([10.0, 20.0])
    context = torch.randn(batch, 3, ab.text_dim)
    context_mask = torch.ones(batch, 3, dtype=torch.bool)
    astate = ab.prepare_state(actions, timestep, context=context, context_mask=context_mask)
    ustate = ub.prepare_state(torch.randn(batch, 5, ub.cfg.vlm_input_dim))

    driver = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False)
    with torch.no_grad():
        vstate, astate, ustate = driver.run_joint_loop(vstate, astate, ustate)

    assert torch.isfinite(vstate.hidden_states).all()
    assert torch.isfinite(ab.extract_prediction(astate)).all()


def test_tri_system_grad_ckpt_backward_per_block():
    """Gradient checkpointing + backward: every block gets nonzero modulation grad."""
    torch.manual_seed(0)
    num_layers = 2
    vb, ab, ub = _make_tiny_trimodal_components(num_layers=num_layers)
    for p in vb.parameters():
        p.requires_grad_(True)
    for p in ab.parameters():
        p.requires_grad_(True)
    for p in ub.parameters():
        p.requires_grad_(True)
    ab.train()

    dit = vb.dit  # noqa: SLF001
    vstate = _make_tiny_video_state(dit, batch=2)
    actions = torch.randn(2, 4, ab.action_dim)
    timestep = torch.tensor([10.0, 20.0])
    context = torch.randn(2, 3, ab.text_dim)
    context_mask = torch.ones(2, 3, dtype=torch.bool)
    astate = ab.prepare_state(actions, timestep, context=context, context_mask=context_mask)
    ustate = ub.prepare_state(torch.randn(2, 5, ub.cfg.vlm_input_dim))

    driver = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False)
    vstate, astate, ustate = driver.run_joint_loop(
        vstate,
        astate,
        ustate,
        use_gradient_checkpointing=True,
    )
    pred = ab.extract_prediction(astate)

    loss = vstate.hidden_states.float().sum() + pred.float().sum()
    loss.backward()

    for i, block in enumerate(dit.blocks):
        grad = block.modulation.grad
        assert grad is not None, f"block {i} modulation has no grad under ckpt"
        assert grad.abs().sum() > 0, f"block {i} modulation grad is all-zero under ckpt"


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("OPENWAM_RUN_TRI_SYSTEM_GPU_SMOKE") != "1",
    reason="set OPENWAM_RUN_TRI_SYSTEM_GPU_SMOKE=1 to run optional tri_system GPU smoke",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not os.path.isdir(WAN22_TI2V_5B), reason=f"checkpoint not mounted: {WAN22_TI2V_5B}")
def test_tri_system_optional_gpu_smoke():
    """Optional real-device smoke for trimodal attention wiring.

    This intentionally stays small and defaults to skipped; it is for mounted
    model/GPU environments that want a quick dtype/device check.
    """

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    vb, ab, ub = _make_tiny_trimodal_components(num_layers=1, dim=32, num_heads=4)
    vb.to(device=device, dtype=dtype)
    ab.to(device=device, dtype=dtype)
    ub.to(device=device, dtype=dtype)

    vstate = _make_tiny_video_state(vb.dit, batch=1, grid_frames=1, grid_height=2, grid_width=2, dtype=dtype)
    vstate.hidden_states = vstate.hidden_states.to(device=device)
    vstate.time_mod = vstate.time_mod.to(device=device)
    vstate.rope_freqs = vstate.rope_freqs.to(device=device)
    vstate.context = vstate.context.to(device=device)
    vstate.t = vstate.t.to(device=device)

    actions = torch.randn(1, 3, ab.action_dim, device=device, dtype=dtype)
    timestep = torch.zeros(1, device=device, dtype=dtype)
    context = torch.randn(1, 2, ab.text_dim, device=device, dtype=dtype)
    context_mask = torch.ones(1, 2, device=device, dtype=torch.bool)
    astate = ab.prepare_state(actions, timestep, context=context, context_mask=context_mask)
    ustate = ub.prepare_state(torch.randn(1, 4, ub.cfg.vlm_input_dim, device=device, dtype=dtype), dtype=dtype)

    driver = TriSystemMoTDriver(vb, ab, ub, mot_checkpoint_mixed_attn=False)
    with torch.no_grad():
        vstate, astate, ustate = driver.run_joint_loop(vstate, astate, ustate)
        action_pred = ab.extract_prediction(astate)

    assert vstate.hidden_states.device == device
    assert action_pred.shape == (1, 3, ab.action_dim)
    assert torch.isfinite(vstate.hidden_states).all()
    assert torch.isfinite(action_pred).all()
    assert torch.isfinite(ustate.und_tokens).all()
