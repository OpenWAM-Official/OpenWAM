"""Tests for the temporal_compression / causal_temporal plumbing (A4).

The flow under test:

    configs/model/*.yaml
        (video_backbone.temporal_compression / causal_temporal)
            │
            ▼
    openwam.model.architectures.architecture_base.BaseWAMArchitecture._init_video_backbone
        (cross-check: yaml values match constructed encoder.properties — fail-fast)
            │
            ▼
    openwam.train.utils.temporal_contract.apply_temporal_contract_bridge
        (invoked by scripts/train.py)
        (forward into cfg.dataloader before build_dataset runs)
            │
            ▼
    openwam.dataloader.robotwin_dataset
        (branching divisibility check: causal=(N-1)%tc, non-causal=N%tc)

The tests below exercise each hop in isolation — no full DiT / pipeline
build, no real episode HDF5 files.
"""

from __future__ import annotations

import logging

import pytest
import torch.nn as nn
from omegaconf import OmegaConf

from openwam.model.video_backbone.encoder import VideoEncoder, VideoEncoderProperties

# ---------------------------------------------------------------------------
# Local mock encoder — minimal concrete VideoEncoder, parameterizable spec.
# ---------------------------------------------------------------------------


class _MockEncoderBase(VideoEncoder):
    _SPEC_KWARGS: dict = {
        "z_dim": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
        "causal_temporal": True,
    }

    def __init__(self):
        super().__init__()
        self._spec = VideoEncoderProperties(**self._SPEC_KWARGS)

    @property
    def properties(self) -> VideoEncoderProperties:
        return self._spec

    def preprocess_video(self, frames):
        import torch

        return torch.zeros(1, 3, 4, 8, 8)

    def batch_encode(self, video):
        import torch

        s = self.properties
        B, _, T, H, W = video.shape
        return torch.zeros(
            B, s.z_dim, T // s.temporal_compression, H // s.spatial_compression, W // s.spatial_compression
        )

    @classmethod
    def from_pretrained(cls, model_path: str, **kw):
        return cls()


def _make_mock_encoder(**spec_overrides) -> VideoEncoder:
    class _Encoder(_MockEncoderBase):
        _SPEC_KWARGS = {**_MockEncoderBase._SPEC_KWARGS, **spec_overrides}

    return _Encoder()


# ---------------------------------------------------------------------------
# _init_video_backbone stub harness (mirrors tests/test_external_encoder.py).
# ---------------------------------------------------------------------------


class _StubArchitecture:
    video_backbone = None

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)


def _run_init_video_backbone(model_cfg):
    from openwam.model.architectures.architecture_base import BaseWAMArchitecture

    stub = _StubArchitecture()
    BaseWAMArchitecture._init_video_backbone(stub, model_cfg)
    return stub


def _patch_backbone_builders(monkeypatch, encoder_factory):
    """Patch build_video_encoder / build_video_backbone so the cross-check
    runs without standing up a real Wan pipeline.

    The fake backbone exposes ``temporal_compression`` / ``causal_temporal``
    sourced from the external encoder spec (or Wan-native defaults) so the
    new cross-check in ``_init_video_backbone`` — which reads from the
    constructed backbone, not directly from the encoder spec — gets a sane
    value to compare against.
    """

    def fake_build_encoder(enc_cfg):
        return encoder_factory()

    def fake_build_backbone(name, cfg, **kw):
        enc = kw.get("external_encoder", None)
        bb = nn.Module()
        bb._pipe = None
        bb.video_encoder = enc
        if enc is not None:
            bb.temporal_compression = int(enc.properties.temporal_compression)
            bb.causal_temporal = bool(enc.properties.causal_temporal)
        else:
            bb.temporal_compression = 4
            bb.causal_temporal = True
        return bb

    import openwam.model.video_backbone as vb_pkg
    from openwam.model.video_backbone import encoder as enc_pkg

    monkeypatch.setattr(enc_pkg, "build_video_encoder", fake_build_encoder)
    monkeypatch.setattr(vb_pkg, "build_video_backbone", fake_build_backbone)


# ===========================================================================
# C3 — bridge propagation: model-side (tc=2) shows up on cfg.dataloader
# ===========================================================================


def test_C3_bridge_propagates_model_side_contract():
    """``apply_temporal_contract_bridge`` copies model-side fields onto
    ``cfg.dataloader`` so the dataset's divisibility check sees the right
    rule even when the original dataloader yaml never mentioned it."""
    from openwam.train.utils.temporal_contract import apply_temporal_contract_bridge

    cfg = OmegaConf.create(
        {
            "model": {
                "video_backbone": {
                    "temporal_compression": 2,
                    "causal_temporal": True,
                }
            },
            "dataloader": {
                "name": "robotwin",
                # no temporal_compression here — legacy yaml shape
            },
        }
    )
    apply_temporal_contract_bridge(cfg)
    assert cfg.dataloader.temporal_compression == 2
    assert cfg.dataloader.causal_temporal is True


# ===========================================================================
# C4 — bridge mismatch warning: dataloader (4) vs model (2) → warning fires
# ===========================================================================


def test_C4_bridge_mismatch_emits_warning(caplog):
    """If the dataloader yaml already carries an out-of-sync
    temporal_compression value, the bridge overwrites it but logs a warning
    so the user notices the conflict instead of silently winning."""
    from openwam.train.utils import temporal_contract as tc_mod
    from openwam.train.utils.temporal_contract import apply_temporal_contract_bridge

    cfg = OmegaConf.create(
        {
            "model": {
                "video_backbone": {
                    "temporal_compression": 2,
                    "causal_temporal": True,
                }
            },
            "dataloader": {
                "name": "robotwin",
                "temporal_compression": 4,
                "causal_temporal": True,
            },
        }
    )
    caplog.set_level(logging.WARNING, logger=tc_mod.__name__)
    apply_temporal_contract_bridge(cfg)
    assert any("Overriding cfg.dataloader.temporal_compression" in rec.message for rec in caplog.records), (
        "expected an override warning when dataloader-side value disagrees with model-side"
    )
    # Model wins.
    assert cfg.dataloader.temporal_compression == 2


# ===========================================================================
# C4b — bridge mismatch warning is symmetric: causal_temporal disagreement
#       fires the same kind of override warning, not silently overwritten
# ===========================================================================


def test_C4b_bridge_causal_mismatch_emits_warning(caplog):
    """Mirror of C4 for ``causal_temporal``: if the dataloader yaml already
    declares ``causal_temporal`` and it disagrees with the model-side value,
    the bridge must warn (not silently overwrite) — same UX contract as the
    ``temporal_compression`` branch above."""
    from openwam.train.utils import temporal_contract as tc_mod
    from openwam.train.utils.temporal_contract import apply_temporal_contract_bridge

    cfg = OmegaConf.create(
        {
            "model": {
                "video_backbone": {
                    "temporal_compression": 2,
                    "causal_temporal": False,
                }
            },
            "dataloader": {
                "name": "robotwin",
                "temporal_compression": 2,
                "causal_temporal": True,
            },
        }
    )
    caplog.set_level(logging.WARNING, logger=tc_mod.__name__)
    apply_temporal_contract_bridge(cfg)
    assert any("Overriding cfg.dataloader.causal_temporal" in rec.message for rec in caplog.records), (
        "expected an override warning when dataloader-side causal_temporal disagrees with model-side"
    )
    # Model wins.
    assert cfg.dataloader.causal_temporal is False


# ===========================================================================
# C6 — non-causal divisibility: tc=2, causal=False, N=10 passes, N=11 fails
# ===========================================================================


def test_C6_non_causal_divisibility():
    from openwam.dataloader.robotwin_dataset import _check_temporal_divisibility

    # N=10, tc=2, non-causal → 10 % 2 == 0 → ok
    _check_temporal_divisibility(num_video_frames=10, temporal_compression=2, causal_temporal=False)

    # N=11, tc=2, non-causal → 11 % 2 == 1 → fail
    with pytest.raises(ValueError, match=r"N % 2 == 0"):
        _check_temporal_divisibility(num_video_frames=11, temporal_compression=2, causal_temporal=False)


# ===========================================================================
# C7 — causal divisibility: tc=2, causal=True, N=9 passes, N=10 fails
# ===========================================================================


def test_C7_causal_divisibility():
    from openwam.dataloader.robotwin_dataset import _check_temporal_divisibility

    # N=9, tc=2, causal → (9-1) % 2 == 0 → ok
    _check_temporal_divisibility(num_video_frames=9, temporal_compression=2, causal_temporal=True)

    # N=10, tc=2, causal → (10-1) % 2 == 1 → fail
    with pytest.raises(ValueError, match=r"\(N-1\) % 2 == 0"):
        _check_temporal_divisibility(num_video_frames=10, temporal_compression=2, causal_temporal=True)


# ===========================================================================
# C8 — mask downsampler honors temporal_factor=2 (parametric contract test)
# ===========================================================================


def test_C8_mask_downsampler_temporal_factor_2():
    """A 9-frame video with ``temporal_factor=2 / causal=True`` collapses
    into ``1 + 8/2 = 5`` latent frames; with ``skip_first=True`` the
    loss-side tail mask must be length ``4`` (not the default-tc=4 Wan
    tail length ``2``).

    Locks the contract that A4's ``base.py`` now plumbs through:
    ``downsample_video_mask_to_latent(..., temporal_factor=2)``. The
    factor is exercised parametrically rather than tied to any specific
    encoder's effective tc; the actual V-JEPA 2 / 2.1 path now emulates
    tc=4 via ViT tubelet=2 + a post-tubelet pool, but the mask plumbing
    must still support non-default factors for other downstream encoders.
    """
    import torch

    from openwam.utils import downsample_video_mask_to_latent

    # All-False = "no padding"; the shape change is the point of the test.
    video_is_pad = torch.zeros((1, 9), dtype=torch.bool)

    # Wan VAE default (tc=4): produces length 2.
    out_legacy = downsample_video_mask_to_latent(video_is_pad, temporal_factor=4, skip_first=True)
    assert out_legacy.shape == (1, 2), f"Wan-legacy default expects length 2, got {tuple(out_legacy.shape)}"

    # Non-default factor (tc=2): must produce length 4 — proves the
    # plumbing carries the spec value verbatim.
    out_tc2 = downsample_video_mask_to_latent(video_is_pad, temporal_factor=2, skip_first=True)
    assert out_tc2.shape == (1, 4), f"temporal_factor=2 expects length 4, got {tuple(out_tc2.shape)}"


# ===========================================================================
# C9 — BaseWAMArchitecture.prepare_inputs plumbs backbone.temporal_compression
#      into downsample_video_mask_to_latent (the actual A4 wiring under test)
# ===========================================================================


def test_C9_prepare_inputs_passes_backbone_temporal_factor(monkeypatch):
    """End-to-end check that A4's wiring reaches the mask downsampler:

    Patches ``downsample_video_mask_to_latent`` to record the
    ``temporal_factor`` it was called with; runs the relevant slice of
    ``BaseWAMArchitecture.prepare_inputs`` with a backbone whose
    ``temporal_compression`` is 2; asserts the recorded value is 2 (not the
    module-level Wan default of 4).
    """
    import torch

    import openwam.model.architectures.architecture_base as base_mod

    captured = {}

    def fake_downsample(video_is_pad, *, temporal_factor=4, skip_first=True):
        captured["temporal_factor"] = temporal_factor
        captured["skip_first"] = skip_first
        # Mimic the real return shape: (..., T_latent_tail) for skip_first=True.
        T = video_is_pad.shape[-1]
        T_tail = max(T - 1, 0)
        T_lat_tail = (T_tail + temporal_factor - 1) // temporal_factor
        return torch.zeros((*video_is_pad.shape[:-1], T_lat_tail), dtype=torch.bool)

    monkeypatch.setattr(base_mod, "downsample_video_mask_to_latent", fake_downsample, raising=False)
    # Also patch the import-from-openwam.utils alias used inside prepare_inputs.
    import openwam.utils as utils_mod

    monkeypatch.setattr(utils_mod, "downsample_video_mask_to_latent", fake_downsample)

    class _FakeBackbone:
        temporal_compression = 2
        needs_first_frame_skip = False

    class _ConcreteArch(base_mod.BaseWAMArchitecture):
        def forward(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    arch = _ConcreteArch(cfg=None)
    arch.video_backbone = _FakeBackbone()
    arch._device = torch.device("cpu")
    arch._dtype = torch.float32
    arch._use_proprioception_context = False

    # Bypass preprocess by stubbing it out — we only need the mask path.
    arch.preprocess = lambda **kw: {"input_latents": torch.zeros(1, 4, 1, 8, 8)}

    sample = {
        "video": [],
        "prompt": "",
        "video_mask": torch.tensor([True] * 9, dtype=torch.bool),  # 9-frame video, all valid
        "action": None,
    }

    arch.prepare_inputs([sample])
    assert captured.get("temporal_factor") == 2, (
        f"BaseWAMArchitecture.prepare_inputs must forward backbone.temporal_compression "
        f"into downsample_video_mask_to_latent; got {captured.get('temporal_factor')}"
    )
