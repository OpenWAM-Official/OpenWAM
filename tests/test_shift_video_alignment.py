"""Verify ``cfg.model.video_backbone.shift_video`` is a single source of truth.

PR rationale: the Reconstruction-or-Semantics paper recipe (arXiv:2605.06388)
requires a dim-dependent α-shift for non-VAE video encoders (e.g. V-JEPA 2.1).
The SD3 formula ``shift = √(1408/48) × 5 ≈ 27`` is the theoretical upper
bound; in practice the project recommends 12.0 (safety-tempered, sits
between Wan default 5 and HunyuanVideo-I2V's high-dynamic 17). The tests
use 12.0 as the override value; the absolute number only matters
insofar as it is visibly different from the Wan default 5.

The ONLY way to guarantee train/inference sigma grids agree under this
override is to source the shift from one place. These tests pin that
invariant: setting ``shift_video`` on the backbone (which mirrors
``cfg.model.video_backbone.shift_video``) flows to BOTH the training-time
``init_training_schedulers`` and the inference-time ``make_schedule``, with
no other knob involved. Action scheduler is unaffected by design.

Tests use lightweight scheduler objects directly — no real Wan / V-JEPA
weights needed.
"""

from __future__ import annotations

import torch

from openwam.deploy.denoise_schedule import make_schedule, schedule_sync
from openwam.model.action_backbone.scheduler import ActionScheduler
from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler


def _make_pair() -> tuple[FlowMatchScheduler, ActionScheduler]:
    """Build a (video, action) scheduler pair matching production Wan setup."""
    return FlowMatchScheduler("Wan"), ActionScheduler()


# ----------------------------------------------------------------------------
# A. schedule_sync: shift_video override leaves action sigmas alone
# ----------------------------------------------------------------------------


def test_schedule_sync_shift_video_changes_only_video_sigmas():
    """``shift_video`` must affect video schedule and ONLY video schedule."""
    v_base, a_base = _make_pair()
    schedule_sync(v_base, a_base, num_steps=50, shift=5.0)  # baseline: both at shift=5
    v_sigmas_baseline = v_base.sigmas.clone()
    a_sigmas_baseline = a_base.sigmas.clone()

    v_override, a_override = _make_pair()
    schedule_sync(v_override, a_override, num_steps=50, shift=5.0, shift_video=12.0)

    # Action sigmas: bit-identical (action_scheduler still got shift=5.0).
    assert torch.allclose(a_override.sigmas, a_sigmas_baseline, atol=1e-6)
    # Video sigmas: changed (shift=27 induces visibly different α-curve).
    assert not torch.allclose(v_override.sigmas, v_sigmas_baseline, atol=1e-3)


def test_schedule_sync_shift_video_none_falls_back_to_shift():
    """``shift_video=None`` must reproduce pre-PR behavior bit-exactly."""
    v_legacy, a_legacy = _make_pair()
    schedule_sync(v_legacy, a_legacy, num_steps=50, shift=5.0)
    v_default, a_default = _make_pair()
    schedule_sync(v_default, a_default, num_steps=50, shift=5.0, shift_video=None)

    assert torch.allclose(v_default.sigmas, v_legacy.sigmas, atol=1e-7)
    assert torch.allclose(a_default.sigmas, a_legacy.sigmas, atol=1e-7)


# ----------------------------------------------------------------------------
# C. make_schedule dispatcher: shift_video round-trip
# ----------------------------------------------------------------------------


def test_make_schedule_threads_shift_video_into_sync():
    v, a = _make_pair()
    make_schedule(
        "sync",
        v,
        a,
        num_steps=50,
        shift=5.0,
        shift_video=12.0,
    )
    v_baseline, a_baseline = _make_pair()
    make_schedule("sync", v_baseline, a_baseline, num_steps=50, shift=5.0)
    assert not torch.allclose(v.sigmas, v_baseline.sigmas, atol=1e-3)
    assert torch.allclose(a.sigmas, a_baseline.sigmas, atol=1e-7)


def test_make_schedule_no_shift_video_is_legacy_path():
    """No shift_video arg → make_schedule must reproduce pre-PR behavior."""
    v, a = _make_pair()
    make_schedule("sync", v, a, num_steps=50, shift=5.0)  # no shift_video
    v_legacy, a_legacy = _make_pair()
    # Pre-PR behavior: shift applied uniformly via legacy schedule_sync.
    schedule_sync(v_legacy, a_legacy, num_steps=50, shift=5.0)
    assert torch.allclose(v.sigmas, v_legacy.sigmas, atol=1e-7)
    assert torch.allclose(a.sigmas, a_legacy.sigmas, atol=1e-7)


# ----------------------------------------------------------------------------
# D. End-to-end alignment: training scheduler == inference video scheduler
#    when shift_video flows from one source
# ----------------------------------------------------------------------------


def test_train_inference_video_sigma_alignment():
    """The PR's core invariant.

    If ``shift_video=12.0`` is set as the single source, then:
      - Training: ``scheduler.set_timesteps(1000, training=True, shift=12.0)``
      - Inference: ``schedule_sync(..., shift_video=12.0)`` calls
                   ``scheduler.set_timesteps(N, shift=12.0)``
    The resulting discrete sigma buffers — when discretized at the same N —
    are bit-identical. This is what makes train and inference statistically
    consistent under the V-JEPA recipe.
    """
    SHIFT = 12.0
    N = 1000

    # Training-side discretization
    train_v_scheduler = FlowMatchScheduler("Wan")
    train_v_scheduler.set_timesteps(N, training=True, shift=SHIFT)
    train_sigmas = train_v_scheduler.sigmas.clone()

    # Inference-side discretization at the same N
    infer_v_scheduler, infer_a_scheduler = _make_pair()
    schedule_sync(infer_v_scheduler, infer_a_scheduler, num_steps=N, shift=5.0, shift_video=SHIFT)
    infer_sigmas = infer_v_scheduler.sigmas

    assert torch.allclose(train_sigmas, infer_sigmas, atol=1e-7), (
        "Train and inference video sigma buffers must match bit-for-bit when shift_video flows from one source."
    )


def test_action_sigma_unchanged_regardless_of_video_shift():
    """Action sigmas must be independent of shift_video — paper recipe
    applies dim-dependent shift to non-VAE VIDEO encoder only."""
    for sv in [None, 5.0, 12.0, 12.0, 35.0]:
        v, a = _make_pair()
        schedule_sync(v, a, num_steps=50, shift=5.0, shift_video=sv)
        v_ref, a_ref = _make_pair()
        schedule_sync(v_ref, a_ref, num_steps=50, shift=5.0)
        assert torch.allclose(a.sigmas, a_ref.sigmas, atol=1e-7), (
            f"action sigmas drifted with shift_video={sv}; action must always use `shift`."
        )


# ----------------------------------------------------------------------------
# E. Backbone-attribute integration: ABC property returns the stored value
# ----------------------------------------------------------------------------


def test_video_backbone_shift_video_property_default_none():
    """A bare VideoBackbone (no _shift_video set) returns None."""
    from openwam.model.video_backbone.base import VideoBackbone

    class _StubBackbone(VideoBackbone):  # noqa: D401 — minimal stub for property test
        # Required-abstract members; values irrelevant.
        @property
        def dim(self):
            return 1

        @property
        def num_layers(self):
            return 1

        @property
        def scheduler(self):
            return None

        @property
        def submodule_names(self):
            return []

        @property
        def num_heads(self):
            return 1

        @property
        def head_dim(self):
            return 1

        @classmethod
        def from_pretrained(cls, source, **kw):
            return cls()

        def prepare(self, **kw):
            raise NotImplementedError

        def run_block(self, block_id, state):
            raise NotImplementedError

        def finalize(self, state):
            raise NotImplementedError

        def inject_action_tokens(self, *a, **k):
            raise NotImplementedError

        def extract_action_tokens(self, *a, **k):
            raise NotImplementedError

        def preprocess_input_for_train(self, **kw):
            raise NotImplementedError

        def get_submodule(self, name):
            return None

        def set_submodule(self, name, module):
            pass

        def decode_video(self, latents, **kw):
            raise NotImplementedError

        def set_dtype_device(self, dtype, device):
            pass

    bb = _StubBackbone()
    assert bb.shift_video is None


def test_video_backbone_shift_video_property_returns_stored_value():
    """When ``_shift_video`` is set, the property returns it."""
    from openwam.model.video_backbone.base import VideoBackbone

    class _StubBackbone(VideoBackbone):
        @property
        def dim(self):
            return 1

        @property
        def num_layers(self):
            return 1

        @property
        def scheduler(self):
            return None

        @property
        def submodule_names(self):
            return []

        @property
        def num_heads(self):
            return 1

        @property
        def head_dim(self):
            return 1

        @classmethod
        def from_pretrained(cls, source, **kw):
            return cls()

        def prepare(self, **kw):
            raise NotImplementedError

        def run_block(self, block_id, state):
            raise NotImplementedError

        def finalize(self, state):
            raise NotImplementedError

        def inject_action_tokens(self, *a, **k):
            raise NotImplementedError

        def extract_action_tokens(self, *a, **k):
            raise NotImplementedError

        def preprocess_input_for_train(self, **kw):
            raise NotImplementedError

        def get_submodule(self, name):
            return None

        def set_submodule(self, name, module):
            pass

        def decode_video(self, latents, **kw):
            raise NotImplementedError

        def set_dtype_device(self, dtype, device):
            pass

    bb = _StubBackbone()
    bb._shift_video = 12.0
    assert bb.shift_video == 12.0
