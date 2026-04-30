"""Detailed load + run tests for the four production architecture variants.

For each of:
    - dual_system_cross_attn
    - dual_system_self_attn
    - shared_backbone_vanilla
    - shared_backbone_moe

verify that:
  1. The architecture builds from a yaml-shaped config (canonical
     ``framework`` + ``variant`` fields) under a Wan2.2-TI2V-5B-shaped mock
     video backbone (30 layers).
  2. ``len(action_backbone.bridge_layers)`` matches what the config specifies
     (explicit list, ``bridge_interval``, or empty for vanilla).
  3. ``compute_loss(...)`` runs end-to-end and returns finite scalar losses.

Action backbone parameter counts are reported via the test's ``-s`` output for
quick inspection (this is informational, not asserted on a magic number).

The video backbone is a lightweight ``_MockVideoBackbone`` (reused from
``test_openwam_trainer``) so the test fits in a CPU CI run; the action backbone
itself is built at meaningful sizes (dim=256, ffn_dim=1024, num_heads=8) so
parameter counts are non-trivial.
"""

from __future__ import annotations

import pytest
import torch

from tests.test_openwam_trainer import (
    _make_fake_loss_inputs,
    _MockVideoBackbone,
)

# Wan2.2-TI2V-5B has 30 DiT blocks; mirror that so bridge_interval math is real.
WAN_NUM_LAYERS = 30
WAN_VIDEO_DIM = 64  # mock dim — real is 3072 but irrelevant for shape tests
ACTION_DIM = 7
T_ACTION = 5


def _build_arch(registry_name: str, cfg: dict, *, num_layers: int = WAN_NUM_LAYERS):
    """Build an architecture under a Wan-shaped mock video backbone.

    The production path constructs the real Wan2.2 video backbone in
    ``BaseWAMArchitecture.__init__`` and then resolves ``video_dim`` /
    ``num_dit_layers`` from it. Tests can't load that real backbone on CPU,
    so we inject those fields directly into the cfg and attach a mock
    backbone afterwards. This exercises exactly the same architecture
    init code path as production for everything that matters here
    (bridge_interval resolution, action_backbone instantiation).
    """
    from openwam.model import build_architecture

    cfg = dict(cfg)
    cfg.setdefault("video_dim", WAN_VIDEO_DIM)
    cfg.setdefault("num_dit_layers", num_layers)
    arch = build_architecture(registry_name, cfg)
    arch.video_backbone = _MockVideoBackbone(dim=WAN_VIDEO_DIM, num_layers=num_layers)
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    return arch


def _count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _run_compute_loss(arch):
    arch.init_training_schedulers(1000)
    actions = torch.randn(1, T_ACTION, ACTION_DIM)
    inputs = _make_fake_loss_inputs(B=1, action_dim=ACTION_DIM, T_action=T_ACTION, video_dim=WAN_VIDEO_DIM)
    out = arch.compute_loss(**inputs, actions=actions, current_step=0)
    return out


# ---------------------------------------------------------------------------
# 1. dual_system_cross_attn
# ---------------------------------------------------------------------------


def test_dual_system_cross_attn_bridge_interval_1():
    """bridge_interval=1: every video DiT layer feeds the bridge → 30 ActionDiT layers."""
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
        "bridge_interval": 1,
        "dim": 256,
        "ffn_dim": 1024,
        "num_heads": 8,
    }
    arch = _build_arch("dual_system_cross_attn", cfg)

    bl = arch.action_backbone.bridge_layers
    assert len(bl) == WAN_NUM_LAYERS, f"expected {WAN_NUM_LAYERS} bridge layers, got {len(bl)}"
    assert bl == tuple(range(WAN_NUM_LAYERS))
    assert arch.action_backbone.num_layers == WAN_NUM_LAYERS, "ActionDiT num_layers must equal len(bridge_layers)"

    n_params = _count_params(arch.action_backbone)
    print(f"\n[dual_system_cross_attn / interval=1] action_backbone params: {n_params:,}")

    out = _run_compute_loss(arch)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["loss_action"])
    assert torch.isfinite(out["loss_video"])


def test_dual_system_cross_attn_bridge_interval_2():
    """bridge_interval=2: every other video DiT layer → 15 ActionDiT layers."""
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
        "bridge_interval": 2,
        "dim": 128,
        "ffn_dim": 512,
        "num_heads": 4,
    }
    arch = _build_arch("dual_system_cross_attn", cfg)

    expected = tuple(range(0, WAN_NUM_LAYERS, 2))
    bl = arch.action_backbone.bridge_layers
    assert len(bl) == len(expected) == 15
    assert bl == expected
    assert arch.action_backbone.num_layers == 15

    n_params = _count_params(arch.action_backbone)
    print(f"\n[dual_system_cross_attn / interval=2] action_backbone params: {n_params:,}")

    out = _run_compute_loss(arch)
    assert torch.isfinite(out["loss"])


def test_dual_system_cross_attn_explicit_bridge_layers():
    """Explicit bridge_layers list overrides interval mode and pins num_layers."""
    explicit = (0, 5, 10, 15, 20, 25, 29)
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": True,
        "action_dim": ACTION_DIM,
        "bridge_layers": list(explicit),
        "dim": 128,
        "ffn_dim": 512,
        "num_heads": 4,
    }
    arch = _build_arch("dual_system_cross_attn", cfg)

    assert arch.action_backbone.bridge_layers == explicit
    assert len(arch.action_backbone.bridge_layers) == 7
    assert arch.action_backbone.num_layers == 7

    n_params = _count_params(arch.action_backbone)
    print(f"\n[dual_system_cross_attn / explicit-7] action_backbone params: {n_params:,}")
    _run_compute_loss(arch)


# ---------------------------------------------------------------------------
# 2. dual_system_self_attn
# ---------------------------------------------------------------------------


def test_dual_system_self_attn_bridge_interval_1():
    """joint_self_attn with bridge_interval=1: 30 JointActionDiTBlocks."""
    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
        "bridge_interval": 1,
        "dim": 128,
        "ffn_dim": 512,
        "num_heads": 4,
    }
    arch = _build_arch("dual_system_self_attn", cfg)

    bl = arch.action_backbone.bridge_layers
    assert len(bl) == WAN_NUM_LAYERS, f"expected {WAN_NUM_LAYERS} joint blocks, got {len(bl)}"
    assert bl == tuple(range(WAN_NUM_LAYERS))
    # joint_self_attn allocates one JointActionDiTBlock per bridge layer.
    assert arch.action_backbone.num_layers == WAN_NUM_LAYERS
    # video_projs / video_back_projs: one per bridge layer.
    assert len(arch.action_backbone.video_projs) == WAN_NUM_LAYERS
    assert len(arch.action_backbone.video_back_projs) == WAN_NUM_LAYERS

    n_params = _count_params(arch.action_backbone)
    print(f"\n[dual_system_self_attn / interval=1] action_backbone params: {n_params:,}")
    _run_compute_loss(arch)


def test_dual_system_self_attn_bridge_interval_3():
    """interval=3 → ceil(30/3)=10 joint blocks."""
    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
        "bridge_interval": 3,
        "dim": 128,
        "ffn_dim": 512,
        "num_heads": 4,
    }
    arch = _build_arch("dual_system_self_attn", cfg)

    expected = tuple(range(0, WAN_NUM_LAYERS, 3))
    assert arch.action_backbone.bridge_layers == expected
    assert len(expected) == 10
    assert arch.action_backbone.num_layers == 10

    n_params = _count_params(arch.action_backbone)
    print(f"\n[dual_system_self_attn / interval=3] action_backbone params: {n_params:,}")
    _run_compute_loss(arch)


# ---------------------------------------------------------------------------
# 3. shared_backbone_vanilla
# ---------------------------------------------------------------------------


def test_shared_backbone_vanilla_loads_and_runs():
    """Vanilla SharedBackbone has no bridge_layers — action rides the video DiT."""
    cfg = {
        "framework": "shared_backbone",
        "variant": "vanilla",
        "action_dim": ACTION_DIM,
        "max_action_len": 64,
    }
    arch = _build_arch("shared_backbone_vanilla", cfg)

    assert arch.action_backbone.bridge_layers == ()
    from openwam.model.base import ExecutionPlan

    assert arch.action_backbone.execution_plan == ExecutionPlan.INTERLEAVED_WHOLE_BLOCK

    n_params = _count_params(arch.action_backbone)
    print(f"\n[shared_backbone_vanilla] action_backbone params: {n_params:,}")
    out = _run_compute_loss(arch)
    assert torch.isfinite(out["loss"])


# ---------------------------------------------------------------------------
# 4. shared_backbone_moe
# ---------------------------------------------------------------------------


def test_shared_backbone_moe_default_all_layers():
    """MoE with ``bridge_layers: null + bridge_interval: 1``: one expert per video DiT layer."""
    # shared_backbone_moe shares ``resolve_bridge_layers`` with dual_system: pass
    # ``bridge_layers: null`` + ``bridge_interval: 1`` to mean "one expert per
    # inferred video DiT layer".
    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": ACTION_DIM,
        "expert_ffn_dim": 1024,
        "bridge_layers": None,
        "bridge_interval": 1,
    }
    arch = _build_arch("shared_backbone_moe", cfg)

    bl = arch.action_backbone.bridge_layers  # MoE exposes expert_layers via bridge_layers
    assert len(bl) == WAN_NUM_LAYERS, f"expected {WAN_NUM_LAYERS} expert layers, got {len(bl)}"
    assert bl == tuple(range(WAN_NUM_LAYERS))
    assert len(arch.action_backbone.expert_blocks) == WAN_NUM_LAYERS

    n_params = _count_params(arch.action_backbone)
    print(f"\n[shared_backbone_moe / default-all-layers] action_backbone params: {n_params:,}")
    out = _run_compute_loss(arch)
    assert torch.isfinite(out["loss"])


def test_shared_backbone_moe_explicit_expert_layers():
    """Explicit bridge_layers controls which video DiT layers carry an expert FFN."""
    expert_layers = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28)
    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": ACTION_DIM,
        "expert_ffn_dim": 1024,
        "bridge_layers": list(expert_layers),
    }
    arch = _build_arch("shared_backbone_moe", cfg)

    assert arch.action_backbone.bridge_layers == expert_layers
    assert len(arch.action_backbone.expert_blocks) == len(expert_layers) == 10

    n_params = _count_params(arch.action_backbone)
    print(f"\n[shared_backbone_moe / explicit-10] action_backbone params: {n_params:,}")
    _run_compute_loss(arch)


# ---------------------------------------------------------------------------
# Summary table — printed once when the whole file is run with ``-s``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "registry_name,cfg,expected_bridge_count",
    [
        (
            "dual_system_cross_attn",
            {
                "framework": "dual_system",
                "variant": "joint_cross_attn",
                "detach_bridge": True,
                "action_dim": ACTION_DIM,
                "bridge_layers": None,
                "bridge_interval": 1,
                "dim": 256,
                "ffn_dim": 1024,
                "num_heads": 8,
            },
            WAN_NUM_LAYERS,
        ),
        (
            "dual_system_self_attn",
            {
                "framework": "dual_system",
                "variant": "joint_self_attn",
                "action_dim": ACTION_DIM,
                "bridge_layers": None,
                "bridge_interval": 2,
                "dim": 128,
                "ffn_dim": 512,
                "num_heads": 4,
            },
            15,
        ),
        (
            "shared_backbone_vanilla",
            {"framework": "shared_backbone", "variant": "vanilla", "action_dim": ACTION_DIM, "max_action_len": 64},
            0,
        ),
        (
            "shared_backbone_moe",
            {
                "framework": "shared_backbone",
                "variant": "moe",
                "action_dim": ACTION_DIM,
                "expert_ffn_dim": 1024,
                "bridge_layers": list(range(0, WAN_NUM_LAYERS, 2)),
            },
            15,
        ),
    ],
    ids=["dual_cross_attn", "dual_self_attn", "shared_vanilla", "shared_moe"],
)
def test_all_variants_load_and_run(registry_name, cfg, expected_bridge_count):
    """Parametric smoke: every variant builds, has expected bridge count, runs loss."""
    arch = _build_arch(registry_name, cfg)

    assert len(arch.action_backbone.bridge_layers) == expected_bridge_count

    n_params = _count_params(arch.action_backbone)
    print(f"\n[{registry_name}] bridge_layers={len(arch.action_backbone.bridge_layers)}, params={n_params:,}")

    out = _run_compute_loss(arch)
    for k in ("loss", "loss_video", "loss_action"):
        assert torch.isfinite(out[k]), f"{registry_name} {k} is not finite: {out[k]}"
