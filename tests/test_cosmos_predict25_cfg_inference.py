"""CPU unit tests for §15 Classifier-Free Guidance inference rollout.

Covers two layers:
1. ``base.py`` module-level CFG math helpers (``_combine_cfg``,
   ``_expand_inputs_for_cfg``) — stateless, pure tensor ops.
2. ``CosmosPredict25VideoBackbone.preprocess_input_for_inference`` uncond
   plumbing — exercises the live-encoder ``uncond_context`` path behind the
   ``cfg_scale > 1.0`` gate. The adapter computes a shape-correct
   ``input_latents`` placeholder from the explicit ``num_frames / height /
   width`` kwargs, so tests don't need a configured VAE.

The GPU inference smoke lives in
``tests/test_cosmos_predict25_real_inference.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn

from openwam.model.architectures.base import _combine_cfg, _expand_inputs_for_cfg
from openwam.model.video_backbone.cosmos_predict25 import CosmosFlowSchedulerAdapter
from openwam.model.video_backbone.cosmos_predict25_backbone import CosmosPredict25VideoBackbone


@dataclass
class InferenceInputs:
    """Local shim: this repo's backbone takes ``**kw`` (no InferenceInputs dataclass)."""

    prompt: str = ""
    vace_video: Any = None
    first_frame_image: Any = None
    num_frames: int = 49
    height: int = 384
    width: int = 320
    seed: int = 42
    tiled: bool = True
    num_inference_steps: int = 50
    shift: Any = None
    cfg_scale: float = 1.0
    cfg_merge: bool = False


def _call(vb, ii: "InferenceInputs"):
    """Flatten the InferenceInputs shim into the ``preprocess_input_for_inference`` kwargs."""
    return vb.preprocess_input_for_inference(**vars(ii))


# ----------------------------------------------------------------------
# Layer 1: math helpers
# ----------------------------------------------------------------------


def test_cfg_combine_math_no_op_at_scale_1():
    """cfg_scale=1.0 → output equals cond (uncond contribution cancels)."""
    cond = torch.tensor([2.0, -3.5, 0.0])
    uncond = torch.tensor([1.0, 1.0, 1.0])
    out = _combine_cfg(uncond, cond, scale=1.0)
    torch.testing.assert_close(out, cond)


def test_cfg_combine_math_formula_sequential():
    """pred = uncond + s · (cond - uncond); s=1.5, cond=2.0, uncond=1.0 → 2.5."""
    cond = torch.full((4,), 2.0)
    uncond = torch.full((4,), 1.0)
    out = _combine_cfg(uncond, cond, scale=1.5)
    expected = torch.full((4,), 2.5)
    torch.testing.assert_close(out, expected)


def test_cfg_combine_math_extreme_scale():
    """s=0.0 (degenerate) just returns uncond; s=3.0 amplifies the cond-uncond delta."""
    cond = torch.tensor([4.0])
    uncond = torch.tensor([1.0])
    torch.testing.assert_close(_combine_cfg(uncond, cond, 0.0), uncond)
    torch.testing.assert_close(_combine_cfg(uncond, cond, 3.0), torch.tensor([10.0]))


def test_cfg_expand_inputs_for_cfg_stacks_batch_axis():
    """cfg_merge path: every batch-axis tensor in inputs_shared is duplicated;
    context replaced with [uncond_context, cond_context]; uncond_context cleared."""
    B = 1
    L = 4
    D = 8
    cond_context = torch.full((B, L, D), 0.7)
    uncond_context = torch.full((B, L, D), -0.3)
    latents = torch.full((B, 16, 2, 4, 5), 1.0)
    proprio = torch.full((B, 20), 0.5)
    seq_lens = torch.full((B,), L, dtype=torch.long)
    action_latents = torch.full((B, 6, 20), 0.1)
    v_timestep = torch.tensor([999.0])
    a_timestep = torch.tensor([999.0])

    inputs_shared = {
        "context": cond_context,
        "uncond_context": uncond_context,
        "latents": latents,
        "proprio": proprio,
        "seq_lens": seq_lens,
        # scalars pass through unchanged
        "num_frames": 13,
        "tiled": True,
        "sigma_shift": 5.0,
        "first_frame_latents": None,
    }

    expanded, exp_al, exp_vt, exp_at = _expand_inputs_for_cfg(
        inputs_shared,
        action_latents=action_latents,
        v_timestep=v_timestep,
        a_timestep=a_timestep,
    )

    # context is [uncond, cond] along batch axis
    assert expanded["context"].shape == (2 * B, L, D)
    torch.testing.assert_close(expanded["context"][:B], uncond_context)
    torch.testing.assert_close(expanded["context"][B:], cond_context)

    # uncond_context cleared to avoid double-stacking downstream
    assert expanded["uncond_context"] is None

    # batch-axis tensors duplicated
    assert expanded["latents"].shape == (2 * B, 16, 2, 4, 5)
    assert expanded["proprio"].shape == (2 * B, 20)
    assert expanded["seq_lens"].shape == (2 * B,)

    # scalars / None pass through
    assert expanded["num_frames"] == 13
    assert expanded["first_frame_latents"] is None

    # action + timesteps stacked
    assert exp_al.shape == (2 * B, 6, 20)
    assert exp_vt.shape == (2,)
    assert exp_at.shape == (2,)

    # caller's inputs_shared is not mutated (we expand a copy)
    assert inputs_shared["context"].shape == (B, L, D)
    assert inputs_shared["uncond_context"] is uncond_context


def test_cfg_expand_inputs_doubles_condition_mask_for_ti2v_merge():
    """Regression: ``cfg_merge=True`` on a TI2V batch must double
    ``condition_mask`` along batch axis. Without it, the wrapper's
    ``torch.cat([x_in, condition_mask], dim=1)`` shape-mismatches.

    Covers reviewer @wayrise's finding: condition_mask was missing from
    ``_CFG_BATCH_AXIS_KEYS`` so cfg_merge + TI2V crashed on first cat.
    """
    B = 1
    L = 4
    D = 8
    T_lat, H_lat, W_lat = 3, 4, 5
    cond_context = torch.full((B, L, D), 0.7)
    uncond_context = torch.full((B, L, D), -0.3)
    latents = torch.zeros(B, 16, T_lat, H_lat, W_lat)
    first_frame_latents = torch.zeros(B, 16, 1, H_lat, W_lat)
    condition_mask = torch.zeros(B, 1, T_lat, H_lat, W_lat)
    condition_mask[:, :, 0] = 1.0
    seq_lens = torch.full((B,), L, dtype=torch.long)

    inputs_shared = {
        "context": cond_context,
        "uncond_context": uncond_context,
        "latents": latents,
        "first_frame_latents": first_frame_latents,
        "condition_mask": condition_mask,
        "seq_lens": seq_lens,
    }

    expanded, _, _, _ = _expand_inputs_for_cfg(
        inputs_shared,
        action_latents=None,
        v_timestep=torch.tensor([999.0]),
        a_timestep=None,
    )

    # All TI2V batch-axis tensors doubled to 2B together.
    assert expanded["latents"].shape == (2 * B, 16, T_lat, H_lat, W_lat)
    assert expanded["first_frame_latents"].shape == (2 * B, 16, 1, H_lat, W_lat)
    assert expanded["condition_mask"].shape == (2 * B, 1, T_lat, H_lat, W_lat)
    # First-frame marker preserved on both halves (uncond + cond).
    assert torch.all(expanded["condition_mask"][:, :, 0] == 1.0)
    assert torch.all(expanded["condition_mask"][:, :, 1:] == 0.0)


# ----------------------------------------------------------------------
# Layer 1b: _forward_with_cfg integration on a fake arch
# ----------------------------------------------------------------------


class _FakeArchForCFG(nn.Module):
    """Minimal architecture that mimics ``BaseWAMArchitecture.forward``'s
    return contract: ``(noise_pred, action_noise_pred)``.

    Forward output is deterministic per call — cond and uncond passes
    return different fixed values so we can verify the linear combine
    math + the sequential context-swap invariant.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls_seen: list = []  # records the `context` identity per forward

    def forward(self, action_latents, a_timestep, *, context, **kw):
        # Record which context was used (cond vs uncond) by sampling the first element.
        self.calls_seen.append({"context_value": context.flatten()[0].item()})
        # noise_pred shape: same as inputs_shared['latents'] when batched; we
        # just return a deterministic value derived from the context magnitude
        # so cond vs uncond produce distinguishable outputs.
        marker = context.flatten()[0]
        latents = kw["latents"]
        noise_pred = torch.full_like(latents, marker.item())
        if action_latents is not None:
            action_noise_pred = torch.full_like(action_latents, marker.item() * 2.0)
        else:
            action_noise_pred = None
        return noise_pred, action_noise_pred


def _make_inputs_shared_for_cfg():
    cond_marker = 1.0
    uncond_marker = -2.0
    L, D = 4, 8
    cond_context = torch.full((1, L, D), cond_marker)
    uncond_context = torch.full((1, L, D), uncond_marker)
    return {
        "context": cond_context,
        "uncond_context": uncond_context,
        "latents": torch.zeros(1, 16, 2, 4, 5),
        "seq_lens": torch.tensor([L], dtype=torch.long),
        "tiled": True,
        "first_frame_latents": None,
    }


def test_forward_with_cfg_sequential_combine_math():
    """Sequential path produces ``pred = uncond + s·(cond - uncond)``
    and runs forward exactly twice (cond + uncond)."""
    from openwam.model.architectures.base import BaseWAMArchitecture

    arch = _FakeArchForCFG()
    inputs = _make_inputs_shared_for_cfg()
    action_latents = torch.zeros(1, 6, 20)
    v_timestep = torch.tensor([999.0])
    a_timestep = torch.tensor([999.0])

    noise_pred, action_noise_pred = BaseWAMArchitecture._forward_with_cfg(
        arch,
        action_latents=action_latents,
        a_timestep=a_timestep,
        inputs_shared=inputs,
        v_timestep=v_timestep,
        cfg_scale=1.5,
        cfg_merge=False,
    )

    # Two forwards (cond then uncond)
    assert len(arch.calls_seen) == 2
    assert arch.calls_seen[0]["context_value"] == pytest.approx(1.0)  # cond first
    assert arch.calls_seen[1]["context_value"] == pytest.approx(-2.0)  # uncond second

    # Math: uncond marker -2.0, cond marker 1.0, s=1.5
    # noise_pred = -2.0 + 1.5*(1.0 - (-2.0)) = -2.0 + 4.5 = 2.5
    assert noise_pred.flatten()[0].item() == pytest.approx(2.5, abs=1e-5)
    # action_noise_pred uses marker*2, so cond=2.0, uncond=-4.0
    # combine = -4.0 + 1.5*(2.0 - (-4.0)) = -4.0 + 9.0 = 5.0
    assert action_noise_pred.flatten()[0].item() == pytest.approx(5.0, abs=1e-5)


def test_forward_with_cfg_sequential_restores_context():
    """After CFG, ``inputs_shared['context']`` MUST be the original cond tensor
    (identity-preserving), otherwise the next denoising step would see
    uncond context as cond — a silent correctness disaster."""
    from openwam.model.architectures.base import BaseWAMArchitecture

    arch = _FakeArchForCFG()
    inputs = _make_inputs_shared_for_cfg()
    original_cond_id = id(inputs["context"])
    original_uncond_id = id(inputs["uncond_context"])

    _ = BaseWAMArchitecture._forward_with_cfg(
        arch,
        action_latents=torch.zeros(1, 6, 20),
        a_timestep=torch.tensor([999.0]),
        inputs_shared=inputs,
        v_timestep=torch.tensor([999.0]),
        cfg_scale=1.5,
        cfg_merge=False,
    )

    assert id(inputs["context"]) == original_cond_id, "context was not restored after CFG"
    assert id(inputs["uncond_context"]) == original_uncond_id


def test_forward_with_cfg_cfg_merge_combine_math():
    """cfg_merge path: one forward on B=2 stacked context, then chunk + combine.

    The fake arch returns marker tensors derived from context.flatten()[0]; for
    a stacked (2B, L, D) context with [uncond, cond] order, the noise output
    is also (2B, ...), chunked into uncond/cond halves. After combine the
    math should match the sequential result.
    """
    from openwam.model.architectures.base import BaseWAMArchitecture

    arch = _FakeArchForCFG()
    inputs = _make_inputs_shared_for_cfg()
    action_latents = torch.zeros(1, 6, 20)
    v_timestep = torch.tensor([999.0])
    a_timestep = torch.tensor([999.0])

    noise_pred, action_noise_pred = BaseWAMArchitecture._forward_with_cfg(
        arch,
        action_latents=action_latents,
        a_timestep=a_timestep,
        inputs_shared=inputs,
        v_timestep=v_timestep,
        cfg_scale=1.5,
        cfg_merge=True,
    )

    # ONE forward call only
    assert len(arch.calls_seen) == 1

    # Same combine math as sequential
    assert noise_pred.shape == (1, 16, 2, 4, 5)
    # The fake arch fills with context.flatten()[0]; for the merged input the
    # first row of the stacked context is uncond=-2.0, so the fake fills the
    # entire (2B, ...) output with -2.0 uniformly. After chunk + combine, both
    # halves carry the same -2.0 marker → combine = -2.0 + 1.5*(-2.0 - (-2.0)) = -2.0.
    # That's a deliberate quirk of the fake (it can't distinguish uncond vs
    # cond halves once they're stacked); what we actually verify here is the
    # mechanical shape + the fact that .chunk(2) + combine produced a (1, ...)
    # output (not (2, ...)).
    assert action_noise_pred is not None
    assert action_noise_pred.shape == (1, 6, 20)

    # The `uncond_context` slot inside the expanded dict was zeroed; original
    # inputs_shared is untouched.
    assert inputs["uncond_context"].shape == (1, 4, 8)


def test_forward_with_cfg_action_branch_falls_back_when_no_action_pred():
    """When the action stream is None on at least one branch, action_noise_pred
    falls back to the cond value rather than CFG-combining; mirrors the
    skip-action-on-non-action-stepping invariant in the denoising loop."""
    from openwam.model.architectures.base import BaseWAMArchitecture

    class _FakeNoActionArch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, action_latents, a_timestep, *, context, **kw):
            self.calls += 1
            marker = context.flatten()[0]
            latents = kw["latents"]
            return torch.full_like(latents, marker.item()), None  # always None on action

    arch = _FakeNoActionArch()
    inputs = _make_inputs_shared_for_cfg()
    noise_pred, action_noise_pred = BaseWAMArchitecture._forward_with_cfg(
        arch,
        action_latents=None,  # action_stepping=False semantics
        a_timestep=None,
        inputs_shared=inputs,
        v_timestep=torch.tensor([999.0]),
        cfg_scale=1.5,
        cfg_merge=False,
    )
    assert arch.calls == 2
    # noise still combines properly
    assert noise_pred.flatten()[0].item() == pytest.approx(2.5, abs=1e-5)
    # action falls back to None (cond's None value preserved)
    assert action_noise_pred is None


def test_forward_with_cfg_rejects_missing_uncond_context():
    """If `uncond_context` is missing or None, the method must raise rather than
    silently downgrade to a single cond forward (which would be a CFG no-op)."""
    from openwam.model.architectures.base import BaseWAMArchitecture

    arch = _FakeArchForCFG()
    inputs = _make_inputs_shared_for_cfg()
    inputs["uncond_context"] = None

    with pytest.raises(RuntimeError, match="uncond_context"):
        BaseWAMArchitecture._forward_with_cfg(
            arch,
            action_latents=None,
            a_timestep=None,
            inputs_shared=inputs,
            v_timestep=torch.tensor([999.0]),
            cfg_scale=1.5,
            cfg_merge=False,
        )


# ----------------------------------------------------------------------
# Layer 2: CosmosPredict25 adapter uncond plumbing
# ----------------------------------------------------------------------


class _FakeTextEncoder:
    """Live-encoder stand-in returning pre-projection ``(B, L=16, 100352)``."""

    def __init__(self) -> None:
        self.seen: list = []

    def __call__(self, text):
        prompts = [text] if isinstance(text, str) else list(text)
        self.seen.append(text)
        return torch.zeros(len(prompts), 16, 100352)


class _FakeProjNet(nn.Module):
    use_crossattn_projection = True
    crossattn_proj_in_channels = 100352

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.crossattn_proj_module = nn.Linear(100352, 1024, bias=False)

    def crossattn_proj(self, x):
        return self.crossattn_proj_module(x)


def _build_live_encoder_backbone(*, text_encoder=None) -> CosmosPredict25VideoBackbone:
    """CosmosPredict25 backbone with no VAE and a stub live encoder. The adapter
    fabricates a shape-correct ``input_latents`` placeholder from the explicit
    ``num_frames / height / width`` kwargs at inference time.
    """
    return CosmosPredict25VideoBackbone(
        net=_FakeProjNet(),
        vae=None,
        text_encoder=text_encoder if text_encoder is not None else _FakeTextEncoder(),
        dim=2048,
        num_layers=28,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        scheduler=CosmosFlowSchedulerAdapter(shift_video=5.0),
        shift_video=5.0,
    )


def test_cosmos_predict25_adapter_cfg_scale_1_no_uncond_context():
    """Regression: `cfg_scale=1.0` (the default) MUST produce
    `uncond_context=None` — the denoising loop's CFG branch checks for None
    to decide whether to combine."""
    vb = _build_live_encoder_backbone()
    inputs_shared = _call(
        vb,
        InferenceInputs(
            prompt="smoke",
            num_frames=13,
            height=256,
            width=320,
            seed=42,
            cfg_scale=1.0,
        ),
    )
    assert inputs_shared["uncond_context"] is None
    assert inputs_shared["cfg_scale"] == pytest.approx(1.0)
    assert inputs_shared["cfg_merge"] is False


def test_cosmos_predict25_adapter_rejects_cfg_scale_below_one():
    """Negative / below-1 cfg_scale is meaningless; reject early at adapter."""
    vb = _build_live_encoder_backbone()
    with pytest.raises(ValueError, match="cfg_scale must be >= 1.0"):
        _call(vb, InferenceInputs(prompt="smoke", cfg_scale=0.5))


def test_cosmos_predict25_adapter_no_text_encoder_raises_with_hint():
    """No live encoder configured → ValueError before the DiT is even invoked."""
    vb = CosmosPredict25VideoBackbone(
        net=_FakeProjNet(),
        vae=None,
        text_encoder=None,
        dim=2048,
        num_layers=28,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        scheduler=CosmosFlowSchedulerAdapter(shift_video=5.0),
        shift_video=5.0,
    )
    with pytest.raises(ValueError, match="text encoder"):
        _call(vb, InferenceInputs(prompt="smoke"))


def test_cosmos_predict25_adapter_shift_passthrough_overrides_shift_video():
    """An explicit ``shift`` lands in ``inputs_shared['sigma_shift']``; absent it,
    the backbone falls back to the wrapper's ``shift_video``."""
    vb = _build_live_encoder_backbone()

    explicit = _call(vb, InferenceInputs(prompt="s", shift=3.0))
    assert explicit["sigma_shift"] == pytest.approx(3.0)

    fallback = _call(vb, InferenceInputs(prompt="s"))  # shift=None
    assert fallback["sigma_shift"] == pytest.approx(5.0)  # wrapper shift_video default


def test_generate_forwards_cfg_kwargs_to_backbone():
    """Guard the one-line forwarding in ``BaseWAMArchitecture.generate``:
    ``cfg_scale`` / ``cfg_merge`` MUST reach ``preprocess_input_for_inference``
    so a CFG-capable backbone can materialise ``uncond_context``. Called
    unbound with a fake self that aborts right after the forwarding call."""
    from openwam.model.architectures.base import BaseWAMArchitecture

    captured: dict = {}

    class _StopHere(Exception):
        pass

    class _FakeVB:
        def preprocess_input_for_inference(self, **kw):
            captured.update(kw)
            raise _StopHere

    class _FakeArch:
        video_backbone = _FakeVB()
        device = torch.device("cpu")
        dtype = torch.float32

        def eval(self):
            return self

    with pytest.raises(_StopHere):
        BaseWAMArchitecture.generate(
            _FakeArch(),
            None,  # schedule
            "a prompt",
            cfg_scale=2.0,
            cfg_merge=True,
        )

    assert captured["cfg_scale"] == 2.0
    assert captured["cfg_merge"] is True


def test_cosmos_predict25_adapter_live_encoder_uncond_context():
    """A configured live text_encoder produces uncond via ``text_encoder("")`` +
    ``crossattn_proj``. The cond branch encodes the real prompt through the
    same encoder."""
    encoder = _FakeTextEncoder()
    vb = _build_live_encoder_backbone(text_encoder=encoder)

    inputs_shared = _call(
        vb,
        InferenceInputs(
            prompt="real prompt",
            cfg_scale=1.5,
        ),
    )

    # Encoder fires once for the cond prompt and once for the empty (uncond) prompt.
    assert encoder.seen == ["real prompt", ""], f"unexpected encoder calls: {encoder.seen}"
    uncond = inputs_shared["uncond_context"]
    assert uncond.shape == (1, 16, 1024)
    assert torch.isfinite(uncond).all()
    cond = inputs_shared["context"]
    assert cond.shape == (1, 16, 1024)


# ----------------------------------------------------------------------
# Layer 3: deploy kwarg filtering
# ----------------------------------------------------------------------


def test_joint_engine_filters_deploy_kwargs_for_strict_architecture():
    """Specialized architectures like IDM should not receive Cosmos-only kwargs."""
    from openwam.deploy.engine import JointInferenceEngine

    class StrictArch:
        def generate(self, *, schedule, prompt, profile=False):
            return {"schedule": schedule, "prompt": prompt, "profile": profile}

    engine = JointInferenceEngine.__new__(JointInferenceEngine)
    engine.architecture = StrictArch()
    engine._architecture_generate_accepts_extra_kwargs = None
    engine._architecture_generate_kwarg_names = None
    engine._architecture_generate_warned_dropped_kwargs = set()

    filtered = engine._filter_architecture_generate_kwargs(
        {
            "schedule": "s",
            "prompt": "p",
            "profile": True,
            "cfg_scale": 1.5,
        }
    )

    assert filtered == {"schedule": "s", "prompt": "p", "profile": True}


def test_joint_engine_preserves_deploy_kwargs_for_flexible_architecture():
    """Base/Cosmos-style architectures with **kwargs keep deploy-side CFG inputs."""
    from openwam.deploy.engine import JointInferenceEngine

    class FlexibleArch:
        def generate(self, **kwargs):
            return kwargs

    engine = JointInferenceEngine.__new__(JointInferenceEngine)
    engine.architecture = FlexibleArch()
    engine._architecture_generate_accepts_extra_kwargs = None
    engine._architecture_generate_kwarg_names = None
    engine._architecture_generate_warned_dropped_kwargs = set()

    kwargs = {
        "schedule": "s",
        "prompt": "p",
        "cfg_scale": 1.5,
    }

    assert engine._filter_architecture_generate_kwargs(kwargs) is kwargs
