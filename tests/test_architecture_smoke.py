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
    """DualSystem joint_cross_attn: ActionDiT.forward can consume context."""
    arch = _make_dual_system("joint_cross_attn", detach_bridge=True, dim=64, video_dim=128)
    B, T_action, T_video = 2, 5, 10
    noisy_actions = torch.randn(B, T_action, 7)
    timestep = torch.tensor([0.5, 0.8])

    ab = arch.action_backbone
    bridges = {bid: torch.randn(B, T_video, 128) for bid in ab.bridge_layers}
    context = torch.randn(B, 4, ab.text_dim)
    context_mask = torch.ones(B, 4, dtype=torch.bool)
    pred = ab(noisy_actions, bridges, timestep, context=context, context_mask=context_mask)
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
    context = torch.randn(B, 4, ab.text_dim)
    context_mask = torch.ones(B, 4, dtype=torch.bool)
    astate = ab.prepare_state(noisy_actions, timestep, context=context, context_mask=context_mask)

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


def _build_dispatch_arch_with_fakes(*, kernel: str):
    """Construct a ``DualSystemSelfAttnArchitecture`` with stub backbones and
    a chosen ``attn_kernel`` for dispatch unit tests.

    The architecture's normal ``__init__`` requires a video backbone config
    to build ActionDiT; here we sidestep that by constructing with
    ``cfg=None`` and attaching stub backbones manually, so the test focuses
    on :meth:`build_mot_driver` dispatch logic alone.
    """
    import torch.nn as nn

    from openwam.model.architectures.dual_system.joint_self_attn import (
        DualSystemSelfAttnArchitecture,
    )

    class _StubBackbone(nn.Module):
        def __init__(self, attn_kernel: str):
            super().__init__()
            self._attn_kernel = attn_kernel

        @property
        def num_layers(self) -> int:
            return 2

        @property
        def num_heads(self) -> int:
            return 4

        @property
        def head_dim(self) -> int:
            return 16

        @property
        def attn_kernel(self) -> str:
            return self._attn_kernel

        video_attention_mask_mode = "bidirectional"

    arch = DualSystemSelfAttnArchitecture(cfg=None)
    arch.video_backbone = _StubBackbone(kernel)
    arch.action_backbone = _StubBackbone(kernel)
    arch._mot_driver_kwargs = {
        "mot_checkpoint_mixed_attn": True,
        "attention_mask_mode": "joint",
        "video_attention_mask_mode": "first_frame_causal",
    }
    return arch


def test_build_mot_driver_dispatches_softmax_to_mot_driver():
    """Default kernel routes to :class:`MoTJointDriver` (Wan/Cosmos25 unaffected)."""
    from openwam.model.architectures.dual_system.mot_driver import MoTJointDriver

    arch = _build_dispatch_arch_with_fakes(kernel="softmax")
    driver = arch.build_mot_driver()
    assert type(driver) is MoTJointDriver


def test_build_mot_driver_dispatches_linear_relu_to_sana_driver():
    """``linear_relu`` kernel routes to :class:`SanaMoTJointDriver`."""
    from openwam.model.architectures.dual_system.sana_mot_driver import (
        SanaMoTJointDriver,
    )

    arch = _build_dispatch_arch_with_fakes(kernel="linear_relu")
    driver = arch.build_mot_driver()
    assert isinstance(driver, SanaMoTJointDriver)


def test_build_mot_driver_rejects_unknown_kernel():
    """Unknown kernel names raise rather than silently falling back."""
    import pytest

    arch = _build_dispatch_arch_with_fakes(kernel="flash")
    with pytest.raises(ValueError, match="unsupported video_backbone.attn_kernel"):
        arch.build_mot_driver()


def test_all_architectures_registered():
    """Supported top-level architecture families should be in the registry."""
    from openwam.model import list_supported_architectures

    supported = list_supported_architectures()
    assert "dual_system_cross_attn" in supported
    assert "dual_system_self_attn" in supported
    assert "dual_system_idm" in supported
    assert "shared_backbone_vanilla" in supported
    assert "shared_backbone_moe" in supported


def test_dual_system_self_attn_sana_yaml_resolves_to_linear_relu():
    """``configs/model/dual_system_self_attn_sana.yaml`` must resolve to the
    joint_self_attn registry entry and propagate ``attn_kernel='linear_relu'``
    on both the video and action backbone sides.

    This is the CPU-side gate for Phase 4 — it does not build the architecture
    (which needs the SANA submodule + weights), but it pins the contract that
    the dispatch path in ``DualSystemSelfAttnArchitecture.build_mot_driver``
    relies on (joint_self_attn.py:125-143). If someone renames
    ``video_backbone.attn_kernel`` or drops the action-side override, this
    test will catch it before training touches a GPU.
    """
    from omegaconf import OmegaConf

    from openwam.model import resolve_architecture_config

    cfg = OmegaConf.load("configs/model/dual_system_self_attn_sana.yaml")
    resolved = resolve_architecture_config(cfg)

    assert resolved.registry_name == "dual_system_self_attn"
    assert resolved.canonical.framework == "dual_system"
    assert resolved.canonical.variant == "joint_self_attn"

    # Video-side kernel — drives `build_mot_driver` to SanaMoTJointDriver.
    vb = resolved.params["video_backbone"]
    assert vb["name"] == "sana_video_2b"
    assert vb["attn_kernel"] == "linear_relu"

    # Action-side kernel — must match video to satisfy
    # ``SanaMoTJointDriver.__init__`` cross-modal kernel check.
    assert resolved.params["attn_kernel"] == "linear_relu"

    # FastWAM-Joint mask defaults — same contract as the cosmos25 self-attn yaml.
    assert resolved.params["attention_mask_mode"] == "joint"
    assert resolved.params["video_attention_mask_mode"] == "first_frame_causal"
    assert resolved.params["mot_checkpoint_mixed_attn"] is True
