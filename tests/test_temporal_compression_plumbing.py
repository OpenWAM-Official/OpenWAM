"""Tests for the temporal_compression mask-downsampling plumbing.

``temporal_compression`` (from the encoder spec / model yaml) is read by
``BaseWAMArchitecture.prepare_inputs`` and forwarded into
``downsample_video_mask_to_latent`` to build the latent-level video mask.
These tests lock that wiring; no full DiT / pipeline build needed.
"""

from __future__ import annotations

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

    from openwam.model.architectures.utils.common import downsample_video_mask_to_latent

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

    import openwam.model.architectures.base as base_mod

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
    # prepare_inputs does a local ``from ...utils.common import
    # downsample_video_mask_to_latent`` each call, so patch the common module
    # (the actual source the local import resolves against).
    import openwam.model.architectures.utils.common as common_mod

    monkeypatch.setattr(common_mod, "downsample_video_mask_to_latent", fake_downsample)

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
