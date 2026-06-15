"""``prepare_inputs`` collects per-sample ``proprio_mask`` alongside ``proprio``.

The full ``prepare_inputs`` flow runs through ``video_backbone.preprocess_input_for_train``
(real VAE / text encoder), which we don't want to spin up in a unit test.
We test the proprio_mask collection by stubbing ``preprocess`` and the
``_pipeline_transform_instance``, keeping the rest of the path real.
"""

from __future__ import annotations

import torch

from openwam.model.architectures.dual_system.joint_self_attn import (
    DualSystemSelfAttnArchitecture,
)


class _IdentityTransform:
    def apply(self, sample):
        return sample


class _StubArch(DualSystemSelfAttnArchitecture):
    """Architecture stub: just enough state for ``prepare_inputs`` proprio/mask paths."""

    def __init__(self, *, state_dim: int = 20, text_dim: int = 16):
        super().__init__(None)
        self._device = torch.device("cpu")
        self._dtype = torch.float32
        self._init_proprio_context(
            {
                "use_proprioception": True,
                "state_dim": state_dim,
                "text_dim": text_dim,
            },
            text_dim=text_dim,
        )
        self._use_gradient_checkpointing = False
        self._use_gradient_checkpointing_offload = False
        self._max_timestep_boundary = 1.0
        self._min_timestep_boundary = 0.0
        self._pipeline_transform_instance = _IdentityTransform()

    # Bypass the real video preprocess; tests only care about proprio_mask wiring.
    def preprocess(self, **kwargs) -> dict:
        return {}


def _make_sample(*, proprio=None, proprio_mask=None, state_dim: int = 20):
    if proprio is None:
        proprio = torch.zeros(1, state_dim, dtype=torch.float32)
    sample = {
        "video": [],
        "prompt": "",
        "action": torch.zeros(2, state_dim, dtype=torch.float32),
        # Omit action_mask / video_mask: those code paths reach into video_backbone
        # (needs_first_frame_skip) which we don't stub here.
        "action_mask": None,
        "video_mask": None,
        "proprio": proprio,
    }
    if proprio_mask is not None:
        sample["proprio_mask"] = proprio_mask
    return sample


def test_collect_proprio_mask_from_samples():
    """Every sample carries proprio_mask -> inputs['proprio_mask'] shape (B, 1)."""
    arch = _StubArch(state_dim=20, text_dim=16)
    batch = [
        _make_sample(proprio_mask=torch.ones(1, dtype=torch.bool)),
        _make_sample(proprio_mask=torch.zeros(1, dtype=torch.bool)),
        _make_sample(proprio_mask=torch.ones(1, dtype=torch.bool)),
    ]
    inputs = arch.prepare_inputs(batch)
    assert "proprio_mask" in inputs
    assert inputs["proprio_mask"].shape == (3, 1)
    assert inputs["proprio_mask"].dtype == torch.bool
    assert inputs["proprio_mask"].squeeze(-1).tolist() == [True, False, True]


def test_collect_proprio_mask_missing_fallback():
    """Sample missing proprio_mask field -> auto-fills True."""
    arch = _StubArch(state_dim=20, text_dim=16)
    batch = [_make_sample(proprio_mask=None), _make_sample(proprio_mask=None)]
    inputs = arch.prepare_inputs(batch)
    assert inputs["proprio_mask"].shape == (2, 1)
    assert inputs["proprio_mask"].all().item() is True


def test_collect_proprio_mask_mixed_present_missing():
    """Mixed batch: missing entries fall back to True, stack succeeds."""
    arch = _StubArch(state_dim=20, text_dim=16)
    batch = [
        _make_sample(proprio_mask=torch.zeros(1, dtype=torch.bool)),
        _make_sample(proprio_mask=None),
        _make_sample(proprio_mask=torch.ones(1, dtype=torch.bool)),
    ]
    inputs = arch.prepare_inputs(batch)
    assert inputs["proprio_mask"].shape == (3, 1)
    assert inputs["proprio_mask"].squeeze(-1).tolist() == [False, True, True]
