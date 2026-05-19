"""CPU-only tests for the ZeRO-3 external-parameter protocol on the
``BaseWAMArchitecture`` (``openwam/model/base.py``) plus regression coverage
that the new ``self(...)`` call in ``compute_loss`` does not affect non-ZeRO-3
paths.

Scope:
  - ``_register_zero3_externals`` is a no-op when deepspeed is unimportable,
    and a no-op when params lack ``ds_id`` (CPU eager / non-prepared paths).
  - The ``compute_loss`` change from ``self.forward(...)`` to ``self(...)``
    actually routes through ``nn.Module.__call__`` so a ``forward_pre_hook``
    fires.
  - The three PR#48-already-working variants (dual_system_cross_attn,
    shared_backbone_vanilla, shared_backbone_moe) still return well-formed
    ``loss / loss_video / loss_action`` dicts under the new call site.

Numerical equivalence across ZeRO stages is covered by the GPU sandbox
parity harness (``sandbox/d-finite/zero3_self_attn/parity_layer1.py``), not
here.
"""

from __future__ import annotations

import sys
import types

import torch

from tests.test_architecture_variants import (
    ACTION_DIM,
    _build_arch,
    _build_shared_moe_arch,
    _run_compute_loss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_self_attn_arch():
    """Build a minimal ``DualSystemSelfAttnArchitecture`` against the CPU mock.

    Uses the same mock video backbone + heterogeneous num_heads/head_dim
    combo as ``test_architecture_variants.py`` so any change to the override
    surface area is exercised end-to-end.
    """
    cfg = {
        "framework": "dual_system",
        "variant": "joint_self_attn",
        "action_dim": ACTION_DIM,
        "use_proprioception": False,
        "bridge_layers": None,
        "bridge_interval": 1,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "attn_head_dim": 16,
        "text_dim": 64,
        "mot_checkpoint_mixed_attn": False,
        "attention_mask_mode": "joint",
        "video_attention_mask_mode": "first_frame_causal",
    }
    return _build_arch("dual_system_self_attn", cfg)


def _tiny_idm_arch():
    """Build a minimal ``DualSystemIDMArchitecture`` against the CPU mock.

    Mirrors :func:`_tiny_self_attn_arch` but selects ``variant='idm'`` so the
    architecture wires up :class:`IDMMoTDriver`. Used by the IDM-specific
    ZeRO-3 external-parameter tests.
    """
    cfg = {
        "framework": "dual_system",
        "variant": "idm",
        "action_dim": ACTION_DIM,
        "use_proprioception": False,
        "bridge_layers": None,
        "bridge_interval": 1,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "attn_head_dim": 16,
        "text_dim": 64,
        "mot_checkpoint_mixed_attn": False,
        "attention_mask_mode": "joint",
        "video_attention_mask_mode": "first_frame_causal",
        "idm_video_cond_noise_prob": 0.0,
    }
    return _build_arch("dual_system_idm", cfg)


# ---------------------------------------------------------------------------
# 1. Register protocol behavior
# ---------------------------------------------------------------------------


def test_register_zero3_externals_no_deepspeed(monkeypatch):
    """When ``deepspeed.runtime.zero`` is unimportable, register is a permanent no-op."""
    arch = _tiny_self_attn_arch()

    # Force the ImportError branch deterministically: poisoning the entry in
    # ``sys.modules`` with ``None`` makes the import machinery raise
    # ImportError on lookup BEFORE invoking the loader, so the
    # ``from deepspeed.runtime.zero import ...`` line in ``base.py`` lands on
    # the except branch regardless of whether the real package is installed
    # in the test environment.
    monkeypatch.setitem(sys.modules, "deepspeed.runtime.zero", None)

    # First call: lands on ImportError branch and seals the gate.
    arch._register_zero3_externals()
    assert getattr(arch, "_zero3_externals_registered", False) is True

    # Second call short-circuits at the gate (would raise if we re-tried import).
    arch._register_zero3_externals()


def test_register_zero3_externals_unmanaged_params(monkeypatch):
    """When deepspeed is importable but params lack ds_id, register is skipped
    and the gate stays OPEN so a later post-prepare call can retry."""
    arch = _tiny_self_attn_arch()

    # Spy on register_external_parameter — it must NOT be called.
    spy_calls: list[tuple] = []

    def _spy(module, parameter):
        spy_calls.append((module, parameter))

    # Inject a fake deepspeed.runtime.zero exposing the spy. Use a module
    # object so ``from ... import register_external_parameter`` resolves.
    fake_zero = types.ModuleType("deepspeed.runtime.zero")
    fake_zero.register_external_parameter = _spy
    fake_runtime = types.ModuleType("deepspeed.runtime")
    fake_runtime.zero = fake_zero
    fake_deepspeed = types.ModuleType("deepspeed")
    fake_deepspeed.runtime = fake_runtime
    monkeypatch.setitem(sys.modules, "deepspeed", fake_deepspeed)
    monkeypatch.setitem(sys.modules, "deepspeed.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "deepspeed.runtime.zero", fake_zero)

    arch._register_zero3_externals()

    assert spy_calls == [], "register_external_parameter must not fire on params without ds_id"
    assert getattr(arch, "_zero3_externals_registered", False) is False, (
        "gate must stay open so a post-accelerator.prepare retry can succeed"
    )


def test_register_zero3_externals_with_ds_id_calls_spy(monkeypatch):
    """When at least one param has a ds_id, register fires and the gate seals."""
    arch = _tiny_self_attn_arch()

    spy_calls: list[tuple] = []

    def _spy(module, parameter):
        spy_calls.append((module, parameter))

    fake_zero = types.ModuleType("deepspeed.runtime.zero")
    fake_zero.register_external_parameter = _spy
    fake_runtime = types.ModuleType("deepspeed.runtime")
    fake_runtime.zero = fake_zero
    fake_deepspeed = types.ModuleType("deepspeed")
    fake_deepspeed.runtime = fake_runtime
    monkeypatch.setitem(sys.modules, "deepspeed", fake_deepspeed)
    monkeypatch.setitem(sys.modules, "deepspeed.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "deepspeed.runtime.zero", fake_zero)

    # Stamp the action backbone's block modulations with a fake ds_id so the
    # register loop accepts them. (video backbone is the mock without
    # blocks/modulation, so only action contributes here.)
    expected = 0
    for block in arch.action_backbone.blocks:
        if getattr(block, "modulation", None) is not None:
            block.modulation.ds_id = 42
            expected += 1
    assert expected > 0, "test precondition: action backbone must expose modulation leaves"

    arch._register_zero3_externals()

    assert len(spy_calls) == expected, (
        f"expected {expected} registrations (one per action block modulation), got {len(spy_calls)}"
    )
    assert all(mod is arch for mod, _ in spy_calls), "all leaves must register against the architecture"
    assert getattr(arch, "_zero3_externals_registered", False) is True, (
        "gate must seal after a successful register"
    )


# ---------------------------------------------------------------------------
# 2. compute_loss now routes through ``self.__call__`` so forward_pre_hook fires
# ---------------------------------------------------------------------------


def test_compute_loss_fires_architecture_forward_pre_hook():
    """Prove that ``base.compute_loss`` calls ``self(...)``, not ``self.forward(...)``.

    PyTorch's ``register_forward_pre_hook`` only fires under ``__call__``;
    this is exactly the same hook DeepSpeed ZeRO-3 uses to gather external
    parameters. If the hook fires for our spy it will fire for DeepSpeed too.
    """
    arch = _tiny_self_attn_arch()
    fired: list[int] = []

    def _spy(module, inputs):
        fired.append(1)

    handle = arch.register_forward_pre_hook(_spy)
    try:
        _run_compute_loss(arch)
    finally:
        handle.remove()

    assert fired, "architecture-level forward_pre_hook never fired — compute_loss must call self(...)"


# ---------------------------------------------------------------------------
# 3. PR#48 already-working variants stay structurally correct
# ---------------------------------------------------------------------------


def test_dual_system_cross_attn_compute_loss_structural_no_regression():
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
        "bridge_interval": 1,
        "dim": 128,
        "ffn_dim": 512,
        "num_heads": 4,
    }
    arch = _build_arch("dual_system_cross_attn", cfg)
    out = _run_compute_loss(arch)

    assert {"loss", "loss_video", "loss_action"} <= set(out.keys())
    for k in ("loss", "loss_video", "loss_action"):
        assert isinstance(out[k], torch.Tensor), f"{k} must be a tensor"
        assert out[k].dim() == 0, f"{k} must be scalar"
        assert torch.isfinite(out[k]), f"{k} must be finite"


def test_shared_backbone_vanilla_compute_loss_structural_no_regression():
    cfg = {
        "framework": "shared_backbone",
        "variant": "vanilla",
        "action_dim": ACTION_DIM,
        "use_proprioception": False,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
    }
    arch = _build_arch("shared_backbone_vanilla", cfg)
    out = _run_compute_loss(arch)
    assert {"loss", "loss_video", "loss_action"} <= set(out.keys())
    for k in ("loss", "loss_video", "loss_action"):
        assert torch.isfinite(out[k]), f"{k} must be finite"


def test_shared_backbone_moe_compute_loss_structural_no_regression():
    cfg = {
        "framework": "shared_backbone",
        "variant": "moe",
        "action_dim": ACTION_DIM,
        "use_proprioception": False,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
        "expert_layers": None,
        "expert_interval": 1,
    }
    arch = _build_shared_moe_arch(cfg)
    out = _run_compute_loss(arch)
    assert {"loss", "loss_video", "loss_action"} <= set(out.keys())
    for k in ("loss", "loss_video", "loss_action"):
        assert torch.isfinite(out[k]), f"{k} must be finite"


# ---------------------------------------------------------------------------
# 4. _iter_zero3_external_params enumerates the right leaves
# ---------------------------------------------------------------------------


def test_dual_system_self_attn_external_params_enumerate_action_modulation():
    """Action backbone's block.modulation leaves must appear in the iterator.

    On the CPU mock, the video backbone has no ``_dit.blocks`` so the iterator
    only yields the action modulations. We verify the action side here; the
    video side is structurally identical and is covered by the real-backbone
    smoke tests.
    """
    arch = _tiny_self_attn_arch()
    leaves = list(arch._iter_zero3_external_params())
    assert leaves, "joint_self_attn must enumerate at least one external leaf"
    expected = [block.modulation for block in arch.action_backbone.blocks]
    leaf_ids = {id(p) for p in leaves}
    assert leaf_ids == {id(p) for p in expected}, (
        "iterator must yield exactly the action backbone modulation leaves"
    )


def test_dual_system_idm_external_params_enumerate_action_modulation():
    """IDM's ``_iter_zero3_external_params`` must enumerate the same MoT
    raw-access leaves as ``DualSystemSelfAttnArchitecture`` — IDM training
    runs through ``IDMMoTDriver`` which inherits ``MoTJointDriver.step`` and
    reads ``block.modulation`` at the same call sites.
    """
    arch = _tiny_idm_arch()
    leaves = list(arch._iter_zero3_external_params())
    assert leaves, "dual_system_idm must enumerate at least one external leaf"
    expected = [block.modulation for block in arch.action_backbone.blocks]
    leaf_ids = {id(p) for p in leaves}
    assert leaf_ids == {id(p) for p in expected}, (
        "iterator must yield exactly the action backbone modulation leaves"
    )


def test_tri_system_external_params_enumerate_action_and_und_leaves(monkeypatch):
    """Tri-system iterator must yield ``action.block.modulation`` AND
    ``understanding_expert.block.wan_und_qkv``.

    Mirrors ``test_dual_system_self_attn_external_params_enumerate_action_modulation``
    but also covers the understanding-expert branch unique to tri_system. The
    ``wan_und_qkv`` leaf is what the trimodal MoT driver reads outside
    ``UnderstandingExpert.__call__`` (see ``und_expert.py:155``); the gold
    cross-product test ensures a future refactor that drops the ub branch
    would fail loudly here instead of silently regressing under ZeRO-3.

    The tri_system test helper builds a real (tiny) Wan video backbone with
    ``_dit.blocks`` populated, so unlike the dual_self_attn CPU mock the video
    side here also yields leaves — the test pins the full action ∪ und ∪ video
    set.
    """
    from tests.test_tri_system_smoke import _make_stub_tri_arch, _tri_arch_min_cfg

    _Arch = _make_stub_tri_arch(monkeypatch, num_video_layers=2)
    arch = _Arch(_tri_arch_min_cfg())

    leaves = list(arch._iter_zero3_external_params())
    assert leaves, "tri_system_joint_self_attn must enumerate at least one external leaf"

    expected_video = [block.modulation for block in arch.video_backbone._dit.blocks]
    expected_action = [block.modulation for block in arch.action_backbone.blocks]
    expected_und = [block.wan_und_qkv for block in arch.understanding_expert.blocks]
    expected_ids = (
        {id(p) for p in expected_video}
        | {id(p) for p in expected_action}
        | {id(p) for p in expected_und}
    )
    leaf_ids = {id(p) for p in leaves}
    assert leaf_ids == expected_ids, (
        "iterator must yield exactly video.modulation ∪ action.modulation ∪ und.wan_und_qkv leaves"
    )
    # Guard the unique-to-tri branch explicitly so a future "iterate only
    # video+action" regression doesn't slip past the set-equality alone.
    und_ids = {id(p) for p in expected_und}
    assert any(id(p) in und_ids for p in leaves), (
        "understanding_expert.wan_und_qkv leaves must be present"
    )


def test_idm_compute_loss_fires_architecture_forward_pre_hook():
    """Prove that ``DualSystemIDMArchitecture.compute_loss`` routes its
    3-branch forward through ``self.__call__``.

    Same construction as :func:`test_compute_loss_fires_architecture_forward_pre_hook`
    but for the IDM compute_loss override. If the hook fires for our spy it
    fires for DeepSpeed too — covers limitation 1
    (``dual_system_idm × ZeRO-3``).
    """
    from tests.test_dual_system_idm import _CapturePrepareVideoBackbone, _make_idm_with_video

    vb = _CapturePrepareVideoBackbone()
    arch = _make_idm_with_video(vb)
    arch.init_training_schedulers(10)

    fired: list[int] = []

    def _spy(module, inputs):
        fired.append(1)

    handle = arch.register_forward_pre_hook(_spy)
    try:
        arch.compute_loss(
            input_latents=torch.randn(2, 1, 1, 1, 1),
            context=torch.randn(2, 2, arch.action_backbone.text_dim),
            context_mask=torch.ones(2, 2, dtype=torch.bool),
            actions=torch.randn(2, 3, arch.action_backbone.action_dim),
            lambda_video=1.0,
            lambda_action=1.0,
        )
    finally:
        handle.remove()

    assert fired, (
        "architecture-level forward_pre_hook never fired — "
        "IDM.compute_loss must route its 3-branch forward through self(...)"
    )


def test_default_external_params_iterator_is_empty():
    """``BaseWAMArchitecture._iter_zero3_external_params`` returns nothing by default.

    Architectures whose forward paths don't bypass ``module.__call__`` must
    not register anything — verifying the default keeps non-MoT variants
    cheap under ZeRO-3.
    """
    cfg = {
        "framework": "dual_system",
        "variant": "joint_cross_attn",
        "detach_bridge": False,
        "action_dim": ACTION_DIM,
        "bridge_layers": None,
        "bridge_interval": 1,
        "dim": 64,
        "ffn_dim": 128,
        "num_heads": 4,
    }
    arch = _build_arch("dual_system_cross_attn", cfg)
    assert list(arch._iter_zero3_external_params()) == []
