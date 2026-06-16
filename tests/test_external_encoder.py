"""External video encoder tests.

Layered across the PR's commit sequence:

- A1-A8 (commit 2): ABC + registry + spec + default hooks (this file's first half).
- B1-B4 (commit 3): WanVideoVAEEncoder reference implementation.
- C1-C13 (commit 4): WanVideoBackbone external_encoder injection.
- D1-D5 (commit 5): base.py gate + yaml whitelist + generate(decode_video=True) guard.
- E1-E3 (commit 6): framework yaml encoder-block presence.

Each block depends only on the code introduced up to its commit. Mock
fixtures grow as later commits land — early cases reuse them.
"""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

from openwam.model.video_backbone.encoder import (
    _VIDEO_ENCODER_REGISTRY,
    VideoEncoder,
    VideoEncoderSpec,
    build_video_encoder,
    register_video_encoder,
)
from openwam.model.video_backbone.videobackbone_base import VideoBackbone

# ---------------------------------------------------------------------------
# Mock encoders for the ABC/registry/spec layer (no real weights needed).
# ---------------------------------------------------------------------------


class _MockEncoderBase(VideoEncoder):
    """Minimal concrete VideoEncoder for ABC-level tests. Parameterizable spec.

    Subclasses set ``_SPEC_KWARGS`` at the class level so the spec is fixed
    per-class — frozen dataclass forbids per-instance mutation anyway.
    """

    _SPEC_KWARGS: dict = {
        "z_dim": 16,
        "spatial_compression": 8,
        "temporal_compression": 4,
        "causal_temporal": True,
    }

    def __init__(self):
        super().__init__()
        self._spec = VideoEncoderSpec(**self._SPEC_KWARGS)

    @property
    def spec(self) -> VideoEncoderSpec:
        return self._spec

    def preprocess_video(self, frames):
        return torch.zeros(1, 3, 4, 8, 8)

    def batch_encode(self, video: Tensor) -> Tensor:
        s = self.spec
        B, _, T, H, W = video.shape
        return torch.zeros(
            B, s.z_dim, T // s.temporal_compression, H // s.spatial_compression, W // s.spatial_compression
        )

    @classmethod
    def from_pretrained(cls, model_path: str, **kw):
        return cls()


def _make_mock_encoder(**spec_overrides) -> VideoEncoder:
    """Create a one-off mock encoder with arbitrary spec for parametrized tests."""

    class _Encoder(_MockEncoderBase):
        _SPEC_KWARGS = {**_MockEncoderBase._SPEC_KWARGS, **spec_overrides}

    return _Encoder()


# ===========================================================================
# Commit 2: A1-A8 — ABC + registry + spec defaults + hook defaults
# ===========================================================================


def test_A1_encoder_registry_round_trip():
    """register_video_encoder → registry contains class → build_video_encoder
    dispatches to from_pretrained."""

    @register_video_encoder("_test_a1_enc")
    class _A1Encoder(_MockEncoderBase):
        pass

    try:
        assert "_test_a1_enc" in _VIDEO_ENCODER_REGISTRY
        assert _VIDEO_ENCODER_REGISTRY["_test_a1_enc"] is _A1Encoder
        built = build_video_encoder({"name": "_test_a1_enc", "model_path": "/nonexistent"})
        assert isinstance(built, _A1Encoder)
    finally:
        _VIDEO_ENCODER_REGISTRY.pop("_test_a1_enc", None)


def test_A2_build_video_encoder_rejects_unknown_name():
    with pytest.raises(KeyError, match="_definitely_not_registered_"):
        build_video_encoder({"name": "_definitely_not_registered_", "model_path": "."})


def test_A3_register_rejects_non_subclass():
    with pytest.raises(TypeError, match="VideoEncoder subclass"):

        @register_video_encoder("_test_a3_notencoder")
        class _NotAnEncoder:  # noqa: D401 — intentional non-subclass
            pass

    # Negative case must NOT have registered anything.
    assert "_test_a3_notencoder" not in _VIDEO_ENCODER_REGISTRY


def test_A4_default_decode_raises_with_contract_aware_msg():
    enc = _make_mock_encoder(is_reversible=False)
    with pytest.raises(NotImplementedError) as exc:
        enc.decode(torch.zeros(1, 16, 4, 8, 8))
    assert "is_reversible=False" in str(exc.value)


def test_A5_default_to_frames_raises_with_contract_aware_msg():
    enc = _make_mock_encoder(is_reversible=False)
    with pytest.raises(NotImplementedError) as exc:
        enc.to_frames(torch.zeros(1, 3, 4, 8, 8))
    assert "is_reversible=False" in str(exc.value)


def test_A6_default_hooks_produce_wan_structure_for_z_dim_16():
    enc = _make_mock_encoder(z_dim=16, dit_patch_size=(1, 2, 2))
    inp = enc.build_dit_input_proj(dit_dim=1536)
    out = enc.build_dit_output_proj(dit_dim=1536)
    assert isinstance(inp, nn.Conv3d)
    assert inp.in_channels == 16
    assert inp.out_channels == 1536
    assert inp.kernel_size == (1, 2, 2)
    assert inp.stride == (1, 2, 2)
    assert isinstance(out, nn.Linear)
    assert out.in_features == 1536
    # z_dim * prod(patch_size) = 16 * 1 * 2 * 2 = 64
    assert out.out_features == 16 * math.prod((1, 2, 2))


def test_A7_custom_dit_patch_size_propagates_to_hook_kernel():
    """For ViT-style encoders that pre-patchify, dit_patch_size=(1,1,1) makes
    the DiT's first conv a pure channel projection."""
    enc = _make_mock_encoder(z_dim=1024, dit_patch_size=(1, 1, 1))
    inp = enc.build_dit_input_proj(dit_dim=1536)
    out = enc.build_dit_output_proj(dit_dim=1536)
    assert inp.in_channels == 1024
    assert inp.kernel_size == (1, 1, 1)
    assert inp.stride == (1, 1, 1)
    assert out.out_features == 1024 * 1  # prod((1,1,1)) = 1


def test_A8_validate_encoder_spec_field_by_field():
    want = VideoEncoderSpec(z_dim=16, spatial_compression=8, temporal_compression=4, causal_temporal=True)

    # Identical spec passes.
    VideoBackbone.validate_encoder_spec(want, want)

    # pixel_range / is_reversible / dit_patch_size are NOT in the required set
    # (excluded by VideoBackbone._ENCODER_SPEC_REQUIRED_FIELDS).
    relaxed = VideoEncoderSpec(
        z_dim=16,
        spatial_compression=8,
        temporal_compression=4,
        causal_temporal=True,
        pixel_range=(0.0, 1.0),
        is_reversible=False,
        dit_patch_size=(1, 1, 1),
    )
    VideoBackbone.validate_encoder_spec(relaxed, want)

    # z_dim mismatch raises with a readable message.
    bad_z = VideoEncoderSpec(z_dim=48, spatial_compression=8, temporal_compression=4, causal_temporal=True)
    with pytest.raises(ValueError, match=r"z_dim"):
        VideoBackbone.validate_encoder_spec(bad_z, want)

    # want=None is a graceful no-op (e.g. backbone already released pipe.vae).
    VideoBackbone.validate_encoder_spec(bad_z, None)


# ===========================================================================
# Commit 3: B1-B4 — WanVideoVAEEncoder reference implementation
# ===========================================================================


class _FakeWanVAEModule(nn.Module):
    """Stand-in for the real ``WanVideoVAE`` / ``WanVideoVAE38`` module.

    Mirrors the duck-typed surface the encoder wrapper depends on:
    ``z_dim``, ``upsampling_factor``, ``batch_encode``, ``decode``.
    """

    def __init__(self, z_dim: int = 16, upsampling_factor: int = 8):
        super().__init__()
        self.z_dim = z_dim
        self.upsampling_factor = upsampling_factor
        # A real parameter so ``next(self.parameters()).device`` works.
        self.proj = nn.Linear(z_dim, z_dim)

    def batch_encode(self, videos: Tensor, device) -> Tensor:
        B, _, T, H, W = videos.shape
        return torch.zeros(B, self.z_dim, (T + 3) // 4, H // self.upsampling_factor, W // self.upsampling_factor)

    def decode(self, latents: Tensor, device, tiled: bool = False) -> Tensor:
        B, _, T, H, W = latents.shape
        return torch.zeros(B, 3, T * 4, H * self.upsampling_factor, W * self.upsampling_factor)


def test_B1_wan_vae_registered():
    """Importing the encoder package registers wan_vae under that name."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder

    assert "wan_vae" in _VIDEO_ENCODER_REGISTRY
    assert _VIDEO_ENCODER_REGISTRY["wan_vae"] is WanVideoVAEEncoder


def test_B2_wan_vae_default_hooks_match_wan21():
    """Wan2.1 family: z_dim=16, upsampling_factor=8 → Conv3d(16, dit, (1,2,2), (1,2,2))."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder

    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    assert enc.spec.z_dim == 16
    assert enc.spec.spatial_compression == 8
    assert enc.spec.temporal_compression == 4
    assert enc.spec.causal_temporal is True
    inp = enc.build_dit_input_proj(dit_dim=1536)
    assert isinstance(inp, nn.Conv3d)
    assert (inp.in_channels, inp.out_channels) == (16, 1536)
    assert inp.kernel_size == (1, 2, 2) and inp.stride == (1, 2, 2)
    out = enc.build_dit_output_proj(dit_dim=1536)
    assert isinstance(out, nn.Linear)
    assert (out.in_features, out.out_features) == (1536, 16 * 4)  # z_dim * prod((1,2,2))


def test_B3_wan_vae_default_hooks_match_wan22():
    """Wan2.2 family: z_dim=48, upsampling_factor=16. Same kernel layout, only z_dim differs."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder

    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=48, upsampling_factor=16))
    assert enc.spec.z_dim == 48
    assert enc.spec.spatial_compression == 16
    inp = enc.build_dit_input_proj(dit_dim=1536)
    assert (inp.in_channels, inp.out_channels) == (48, 1536)
    assert inp.kernel_size == (1, 2, 2)
    out = enc.build_dit_output_proj(dit_dim=1536)
    assert out.out_features == 48 * 4


def test_B4_wan_vae_is_reversible_true_by_default():
    """Wan VAE has a real pixel decoder → spec.is_reversible inherits the dataclass
    default ``True``. Confirms WanVideoVAEEncoder doesn't accidentally flip it."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder

    enc = WanVideoVAEEncoder(_FakeWanVAEModule())
    assert enc.spec.is_reversible is True
    assert enc.spec.dit_patch_size == (1, 2, 2)


# ===========================================================================
# Commit 4: C1-C13 — WanVideoBackbone external_encoder injection
# ===========================================================================


class _FakeDiT(nn.Module):
    """Minimal DiT stand-in carrying the attributes wan_adapter / reinit
    inspect. Notably has ``patch_embedding`` and ``head.head`` for the
    rebuild path, and ``has_image_input`` for the I2V fail-fast probe.
    """

    class _FakeHead(nn.Module):
        def __init__(self, dim: int, out_dim: int):
            super().__init__()
            # The real Head module has .head (Linear), .norm, .modulation,
            # and .patch_size (used as the unpatchify hint). Only .head and
            # .patch_size are exercised by the rebuild path; .norm /
            # .modulation belong to the reset_parameters loop.
            self.head = nn.Linear(dim, out_dim * 4)  # prod((1,2,2)) = 4
            self.patch_size = (1, 2, 2)

    def __init__(self, dim: int = 1536, in_dim: int = 16, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.patch_size = (1, 2, 2)
        self.has_image_input = has_image_input
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.head = self._FakeHead(dim, in_dim)
        # Empty blocks list — not exercised in these adapter-level tests.
        self.blocks = nn.ModuleList([])
        self.freq_dim = 256


class _FakePipe(nn.Module):
    """Minimal pipe stand-in that wan_adapter.from_pretrained mutates.

    Inherits ``nn.Module`` so that ``state_dict()`` on the surrounding
    backbone recurses into ``self._pipe.vae.*`` keys — matching how the
    real ``WanVideoPipeline`` (a ``BasePipeline``/``nn.Module``) behaves.
    """

    def __init__(
        self,
        *,
        has_image_input: bool = False,
        vae_z_dim: int = 16,
        vae_upsample: int = 8,
        has_vace: bool = False,
    ):
        super().__init__()
        self.dit = _FakeDiT(in_dim=vae_z_dim, has_image_input=has_image_input)
        self.vae = _FakeWanVAEModule(z_dim=vae_z_dim, upsampling_factor=vae_upsample)
        # ``vace`` is a sibling module on ``WanVideoPipeline`` when the
        # backbone is from the VACE family (``wan21_vace_1_3b`` /
        # ``wan_vace_14b``). The actual module is a ``VaceWanModel``; for
        # the wan_adapter fail-fast probe (which only does
        # ``getattr(pipe, "vace", None) is not None``) a plain placeholder
        # is sufficient.
        self.vace = nn.Module() if has_vace else None
        self.height_division_factor = 0
        self.width_division_factor = 0
        # preprocess_video / vae_output_to_video are called only on the
        # default path; route them through the VAE module to keep mocks lean.
        self.device = "cpu"

    def preprocess_video(self, frames):
        return torch.zeros(1, 3, 4, 64, 64)

    def vae_output_to_video(self, t):
        return ["frame0", "frame1"]


def test_C1_default_path_state_dict_keys_contain_pipe_vae():
    """Without external_encoder, native vae.* keys are present (phase-3c attribute-ization) and _encoder.* is None."""
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe()
    backbone = WanVideoBackbone(pipe)  # default path, no external_encoder
    assert backbone._uses_external_encoder is False
    sd = backbone.state_dict()
    assert any(k.startswith("vae.") for k in sd), "default path should expose vae.* keys"
    assert not any(k.startswith("_encoder.") for k in sd), "default path must not have _encoder.* keys"


def test_C2_default_path_pipe_vae_call_sites_preserved():
    """5 IO entries route through pipe.vae (and pipe.preprocess_video /
    pipe.vae_output_to_video) on the default path."""
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe()
    backbone = WanVideoBackbone(pipe)
    # WanVideoBackbone.__init__ hard-codes ``self._device = torch.device("cuda")``
    # (training-time default). Tests run under CPU-only CI, so override to CPU
    # before exercising _decode_latents (which does ``latents.to(self.device)``).
    backbone._device = torch.device("cpu")
    # _preprocess_video → pipe.preprocess_video
    _ = backbone._preprocess_video([None])
    # _encode_video → pipe.vae.batch_encode
    enc = backbone._encode_video(torch.zeros(1, 3, 4, 64, 64))
    assert enc.shape[1] == 16  # VAE z_dim
    # _decode_latents → pipe.vae.decode; _latents_to_frames → pipe.vae_output_to_video
    dec = backbone._decode_latents(torch.zeros(1, 16, 4, 8, 8))
    assert dec.shape == (1, 3, 16, 64, 64)
    frames = backbone._latents_to_frames(dec)
    assert frames == ["frame0", "frame1"]


def test_C3_external_path_releases_pipe_vae_in_from_pretrained():
    """from_pretrained must set pipe.vae=None when an external_encoder is wired."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    assert backbone._uses_external_encoder is True
    assert backbone._pipe.vae is None


def test_C4_external_path_state_dict_keys_swap():
    """External path: _encoder.* keys present, native vae.* absent."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    sd = backbone.state_dict()
    assert any(k.startswith("_encoder.") for k in sd), "external path should expose _encoder.* keys"
    assert not any(k.startswith("vae.") for k in sd), "external path must release the native vae.*"


def test_C5_submodule_names_vae_alias_in_both_paths():
    """submodule_names always contains 'vae' — alias resolves differently
    depending on whether external_encoder is set."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe_default = _FakePipe()
    backbone_default = WanVideoBackbone(pipe_default)
    assert "vae" in backbone_default.submodule_names

    pipe_external = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone_external = WanVideoBackbone.from_pretrained(pipe_external, external_encoder=enc)
    assert "vae" in backbone_external.submodule_names


def test_C6_get_submodule_vae_routes_to_encoder_on_external_path():
    """get_submodule('vae') returns the encoder on external path, pipe.vae on default."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe_default = _FakePipe()
    backbone_default = WanVideoBackbone(pipe_default)
    assert backbone_default.get_submodule("vae") is pipe_default.vae

    pipe_external = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone_external = WanVideoBackbone.from_pretrained(pipe_external, external_encoder=enc)
    assert backbone_external.get_submodule("vae") is enc


def test_C7_set_dtype_device_moves_external_encoder():
    """The encoder is a submodule, so it follows submodule_names through set_dtype_device."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    # Track moves via a side-effect: read the encoder's proj dtype after .to().
    backbone.set_dtype_device(torch.float32, torch.device("cpu"))
    assert next(enc._m.parameters()).dtype == torch.float32


def test_C8_i2v_backbone_rejects_external_encoder_at_construction():
    """has_image_input=True is the I2V signature. An external encoder is
    rejected at construction time (rather than at runtime in _build_i2v_y)."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe(has_image_input=True)
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    with pytest.raises(ValueError, match="I2V backbones cannot use external encoders"):
        WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)


def test_C8b_vace_backbone_rejects_external_encoder_at_construction():
    """``pipe.vace is not None`` is the VACE backbone signature
    (``wan21_vace_1_3b`` / ``wan_vace_14b``). This PR's scope is
    Wan2.2-TI2V-5B only; VACE has two hard incompatibilities the
    adapter does not yet rebuild:

      (a) ``VaceWanModel.vace_patch_embedding`` hardcodes
          ``vace_in_dim = 2 * z_dim + 64 = 96`` for native Wan VAE z_dim=16;
          ``reinit_dit_from_scratch`` does not rebuild it.
      (b) The vendored ``WanVideoUnit_VACE.process`` calls
          ``pipe.vae.encode(...)`` which AttributeErrors on the
          external-encoder path (pipe.vae is None).

    Fail-fast at construction makes the scope explicit instead of
    letting users discover it only at training shape-mismatch or
    deploy AttributeError. Mirrors test_C8 for I2V.
    """
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe(has_vace=True)
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    with pytest.raises(ValueError, match="VACE backbones cannot use external encoders"):
        WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)


def test_C9_spec_validation_strict_when_reversible():
    """Reversible encoder with mismatched z_dim is rejected immediately."""
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    bad_encoder = WanVideoVAEEncoderStub(spec_z_dim=48, is_reversible=True)
    with pytest.raises(ValueError, match=r"z_dim"):
        WanVideoBackbone.from_pretrained(pipe, external_encoder=bad_encoder)


def test_C10_spec_validation_fully_skipped_when_irreversible():
    """Irreversible encoder declares its own latent geometry; the backbone
    skips validation entirely (z_dim, spatial/temporal compression, causal
    are all encoder-owned). This case exercises a matching spatial=8 so it
    only verifies the z_dim mismatch is tolerated; C10b covers the harder
    case where spatial also differs from the backbone's native VAE."""
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False)
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    assert backbone._uses_external_encoder is True


def test_C10b_spec_validation_skipped_when_spatial_temporal_differ_for_irreversible():
    """The real motivating case: a DINOv3-style encoder declares
    spatial_compression=16 / temporal_compression=1 / causal=False, all of
    which differ from Wan2.1's native VAE (spatial=8, temporal=4,
    causal=True). Pre-fix this raised ValueError on the
    validate_encoder_spec call. Post-fix the call is skipped entirely for
    irreversible encoders and from_pretrained completes successfully."""
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)

    # Direct VideoEncoder subclass with all-different spec dimensions; we
    # don't reuse WanVideoVAEEncoderStub because its spatial_compression is
    # hardcoded to 8 (so it would mask the bug C10b is specifically guarding).
    class _DinoLikeEncoder(VideoEncoder):
        def __init__(self):
            super().__init__()
            self._proj = nn.Conv3d(1024, 1024, kernel_size=1)
            self._spec = VideoEncoderSpec(
                z_dim=1024,
                spatial_compression=16,
                temporal_compression=1,
                causal_temporal=False,
                is_reversible=False,
                dit_patch_size=(1, 1, 1),
            )

        @property
        def spec(self):
            return self._spec

        def preprocess_video(self, frames):
            return torch.zeros(1, 3, 4, 256, 256)

        def batch_encode(self, video):
            return torch.zeros(video.shape[0], 1024, video.shape[2], video.shape[3] // 16, video.shape[4] // 16)

        @classmethod
        def from_pretrained(cls, model_path, **kw):
            return cls()

    enc = _DinoLikeEncoder()
    # Pre-fix this raised:
    #   ValueError: encoder spec mismatch on ['spatial_compression', 'temporal_compression', 'causal_temporal']: ...
    # Post-fix the call below is expected to succeed.
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    assert backbone._uses_external_encoder is True
    # And the division factor honors the encoder's dit_patch_size=(1,1,1):
    # spatial(16) * dit_patch_size[1or2](1) = 16, NOT spatial * 2.
    assert pipe.height_division_factor == 16
    assert pipe.width_division_factor == 16


def test_C11_dit_patch_size_drives_height_width_division_factor():
    """spec.dit_patch_size=(1,1,1) — height/width_division_factor equals
    spatial_compression (no extra *2). Verifies the hardcoded *2 is gone."""
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False, dit_patch_size=(1, 1, 1))
    WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    assert pipe.height_division_factor == 8  # spatial_compression * 1
    assert pipe.width_division_factor == 8

    # And the default (1,2,2) still works the same as before.
    pipe2 = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc2 = WanVideoVAEEncoderStub(spec_z_dim=16, is_reversible=True, dit_patch_size=(1, 2, 2))
    WanVideoBackbone.from_pretrained(pipe2, external_encoder=enc2)
    assert pipe2.height_division_factor == 16  # 8 * 2
    assert pipe2.width_division_factor == 16


def test_C12_decode_video_blocks_irreversible_encoder():
    """decode_video must raise NotImplementedError when the encoder is irreversible.
    The error message must reference the contract violation, not be a generic AttributeError."""
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False)
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    with pytest.raises(NotImplementedError, match="irreversible"):
        backbone.decode_video(torch.zeros(1, 1024, 4, 8, 8))


def test_C13a_reinit_with_external_encoder_rebuilds_modules():
    """reinit_dit_from_scratch(pipe, external_encoder=enc) rebuilds
    patch_embedding and head.head at the encoder's z_dim, and syncs in_dim."""
    from openwam.model.video_backbone.wan_videobackbone import reinit_dit_from_scratch

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False)
    reinit_dit_from_scratch(
        pipe,
        external_encoder=enc,
        dit_patch_size=enc.spec.dit_patch_size,
        verbose=False,
    )
    assert pipe.dit.patch_embedding.in_channels == 1024
    assert pipe.dit.head.head.out_features == 1024 * 4  # z_dim * prod((1,2,2))
    assert pipe.dit.in_dim == 1024


def test_C13b_reinit_without_external_encoder_is_backwards_compat():
    """reinit_dit_from_scratch(pipe) WITHOUT external_encoder kwarg must behave
    exactly as before (no shape change). Guards the 17 existing from_scratch
    test cases in test_video_backbone_from_scratch.py."""
    from openwam.model.video_backbone.wan_videobackbone import reinit_dit_from_scratch

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    original_in_channels = pipe.dit.patch_embedding.in_channels
    original_out_features = pipe.dit.head.head.out_features
    original_in_dim = pipe.dit.in_dim
    original_patch_size = pipe.dit.patch_size
    original_head_patch_size = pipe.dit.head.patch_size

    reinit_dit_from_scratch(pipe, verbose=False)
    assert pipe.dit.patch_embedding.in_channels == original_in_channels
    assert pipe.dit.head.head.out_features == original_out_features
    assert pipe.dit.in_dim == original_in_dim
    assert pipe.dit.patch_size == original_patch_size
    assert pipe.dit.head.patch_size == original_head_patch_size


def test_C13c_reinit_syncs_patch_size_for_non_default_encoder():
    """Regression for the severe S1 bug: ``reinit_dit_from_scratch`` must
    also sync ``dit.patch_size`` (used by ``WanModel.unpatchify``'s einops
    rearrange) and ``dit.head.patch_size`` whenever the encoder declares a
    non-default ``spec.dit_patch_size``. Pre-fix, the patch_embedding and
    head.head Linear were rebuilt at the new shape but the unpatchify hint
    stayed at ``(1, 2, 2)`` — any encoder with ``dit_patch_size=(1,1,1)``
    (DINOv3 / V-JEPA2 patch-at-16) would shape-mismatch on the first
    forward.
    """
    from openwam.model.video_backbone.wan_videobackbone import reinit_dit_from_scratch

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False, dit_patch_size=(1, 1, 1))
    reinit_dit_from_scratch(
        pipe,
        external_encoder=enc,
        dit_patch_size=enc.spec.dit_patch_size,
        verbose=False,
    )

    # Hooks rebuilt the in/out projections at the new geometry.
    assert pipe.dit.patch_embedding.in_channels == 1024
    assert pipe.dit.patch_embedding.kernel_size == (1, 1, 1)
    assert pipe.dit.patch_embedding.stride == (1, 1, 1)
    # head.head out_features = z_dim * prod(dit_patch_size) = 1024 * 1 = 1024
    # (would be 1024 * 4 = 4096 if patch_size stayed at the default (1,2,2)).
    assert pipe.dit.head.head.out_features == 1024
    # And the patch_size *metadata* was synced — without this, unpatchify
    # would still expect z_dim * prod((1,2,2)) = 4096 features per token.
    assert pipe.dit.patch_size == (1, 1, 1)
    assert pipe.dit.head.patch_size == (1, 1, 1)

    # Forward-shape sanity check on the rebuilt projections in isolation
    # (we don't run the full WanModel forward because _FakeDiT has empty
    # blocks). ``patch_embedding`` consumes the encoder latent grid;
    # ``head.head`` produces a per-token vector whose width must equal
    # z_dim * prod(patch_size) for ``unpatchify`` to reconstruct the latent
    # shape. Asserting both line up at 1024 catches the original mismatch
    # at the same boundary the real DiT would hit on its first forward.
    z = torch.zeros(1, 1024, 4, 16, 16)
    tokens = pipe.dit.patch_embedding(z)
    assert tokens.shape == (1, pipe.dit.dim, 4, 16, 16)  # stride=(1,1,1) preserves grid
    flat = torch.zeros(1, 8, pipe.dit.dim)
    out = pipe.dit.head.head(flat)
    assert out.shape[-1] == enc.spec.z_dim * math.prod(enc.spec.dit_patch_size)


def test_C13e_adapt_dit_to_external_encoder_no_reset():
    """``adapt_dit_to_external_encoder`` (deploy path) reshapes
    ``patch_embedding`` / ``head.head`` / ``patch_size`` / ``in_dim``
    without touching the rest of the DiT.

    This is the deploy-only variant: training calls
    ``reinit_dit_from_scratch`` (which internally reshapes AND resets
    every learnable param); deploy must reshape only so the subsequent
    strict ``load_checkpoint`` can populate the rebuilt modules.

    Regression: pre-fix, deploy was stuck with Wan-native
    ``patch_embedding.in_channels == 48`` because the reshape lived
    inside the reset path which is gated on training (source is None).
    """
    from openwam.model.video_backbone.wan_videobackbone import adapt_dit_to_external_encoder

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    # Attach a sentinel sub-module that the adapt path must NOT touch.
    # After adapt, its weight must remain the stamped value — proof that
    # adapt did not run a global reset_parameters like reinit does.
    sentinel = nn.Linear(4, 4)
    sentinel.weight.data.fill_(0.1234)
    sentinel_snapshot = sentinel.weight.detach().clone()
    pipe.dit.add_module("sentinel_check", sentinel)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False)

    adapt_dit_to_external_encoder(pipe, enc, enc.spec.dit_patch_size)

    # Shapes adapted to the encoder.
    assert pipe.dit.patch_embedding.in_channels == 1024
    assert pipe.dit.head.head.out_features == 1024 * 4
    assert pipe.dit.in_dim == 1024
    assert pipe.dit.patch_size == (1, 2, 2)
    assert pipe.dit.head.patch_size == (1, 2, 2)
    # Sentinel preserved — adapt did not reset non-rebuilt sub-modules.
    assert torch.equal(pipe.dit.sentinel_check.weight, sentinel_snapshot)


def test_C13f_adapt_dit_to_external_encoder_requires_patch_size():
    """``adapt_dit_to_external_encoder`` mirrors
    ``reinit_dit_from_scratch``'s single-source-of-truth invariant —
    refuses ``dit_patch_size=None`` so callers source it from the
    backbone rather than the encoder spec.
    """
    from openwam.model.video_backbone.wan_videobackbone import adapt_dit_to_external_encoder

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False)
    with pytest.raises(ValueError, match=r"dit_patch_size is required"):
        adapt_dit_to_external_encoder(pipe, enc, None)


def test_C13d_reinit_with_external_encoder_requires_dit_patch_size():
    """Single-source-of-truth guard: ``reinit_dit_from_scratch`` must refuse
    to silently fall back to ``external_encoder.spec.dit_patch_size`` when
    ``dit_patch_size`` is omitted. The backbone owns this geometry — callers
    must source it from ``self.video_backbone.dit_patch_size`` so the DiT
    rebuild reads the same value as the dataloader bridge and the cross-check
    in :meth:`BaseWAMArchitecture._init_video_backbone`."""
    from openwam.model.video_backbone.wan_videobackbone import reinit_dit_from_scratch

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False)
    with pytest.raises(ValueError, match=r"dit_patch_size is required"):
        reinit_dit_from_scratch(pipe, external_encoder=enc, verbose=False)


def test_C14_wan_copy_deploy_artifacts_forwards_to_external_encoder(tmp_path):
    """``WanVideoBackbone.copy_deploy_artifacts`` must call the external
    encoder's hook so V-JEPA's ``manifest.json`` lands in the checkpoint dir.

    The Wan tokenizer copy step inside the same method is a no-op here
    because we pass a plain ``dict`` cfg — ``copy_video_backbone_tokenizer``
    reaches into ``cfg.model.video_backbone.model_path`` via attribute access
    (DictConfig-style), so a dict cfg raises AttributeError which the
    helper catches and logs. That keeps this test isolated to the
    encoder-forwarding behavior we actually want to verify, without us
    having to stand up a real Wan model directory on disk.
    """
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    class _RecordingEncoder(_MockEncoderBase):
        def __init__(self):
            super().__init__()
            self.calls: list = []

        def copy_deploy_artifacts(self, output_dir, cfg):
            self.calls.append((output_dir, cfg))

    pipe = _FakePipe()
    enc = _RecordingEncoder()
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    cfg = {"model": {"video_backbone": {"model_path": "/nonexistent"}}}
    backbone.copy_deploy_artifacts(str(tmp_path), cfg)

    assert enc.calls == [(str(tmp_path), cfg)]


def test_C15_wan_copy_deploy_artifacts_no_op_without_external_encoder(tmp_path):
    """Without an external encoder, ``WanVideoBackbone.copy_deploy_artifacts``
    only invokes the tokenizer copy — no encoder hook call, no crash on the
    default path. Regression guard against accidentally routing the encoder
    branch into the default path.
    """
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe()
    backbone = WanVideoBackbone(pipe)
    assert backbone._encoder is None
    backbone.copy_deploy_artifacts(str(tmp_path), {"model": {"video_backbone": {"model_path": str(tmp_path)}}})
    # No exception, no spurious files.
    # (Tokenizer copy is itself a warning-on-missing path; we don't assert
    # its side effects here — they are covered by Wan-side unit tests.)


# Tiny helper for C9-C13 — declares a custom spec without going through Wan VAE loading.


class WanVideoVAEEncoderStub(VideoEncoder):
    """Custom-spec encoder used by C9-C13 to control z_dim / is_reversible /
    dit_patch_size without touching real Wan VAE weights."""

    def __init__(self, *, spec_z_dim: int, is_reversible: bool, dit_patch_size=(1, 2, 2)):
        super().__init__()
        # A tiny conv so state_dict has something to enumerate (test C4).
        self._proj = nn.Conv3d(spec_z_dim, spec_z_dim, kernel_size=1)
        self._spec = VideoEncoderSpec(
            z_dim=spec_z_dim,
            spatial_compression=8,
            temporal_compression=4,
            causal_temporal=True,
            is_reversible=is_reversible,
            dit_patch_size=dit_patch_size,
        )

    @property
    def spec(self) -> VideoEncoderSpec:
        return self._spec

    def preprocess_video(self, frames):
        return torch.zeros(1, 3, 4, 64, 64)

    def batch_encode(self, video: Tensor) -> Tensor:
        return torch.zeros(
            video.shape[0], self._spec.z_dim, video.shape[2] // 4, video.shape[3] // 8, video.shape[4] // 8
        )

    @classmethod
    def from_pretrained(cls, model_path: str, **kw):
        return cls(spec_z_dim=16, is_reversible=True)


# ===========================================================================
# Commit 5: D1-D5 — base.py gate + yaml whitelist + generate(decode_video) guard
# ===========================================================================
#
# These tests exercise _init_video_backbone gate logic via direct invocation
# on a lightweight stub architecture; we don't go through the full Hydra +
# build_training_pipeline stack to keep CPU runtime tiny.


class _StubArchitecture:
    """Minimal stand-in for BaseWAMArchitecture that exposes only the bits
    _init_video_backbone touches."""

    video_backbone = None

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)


def _run_init_video_backbone(model_cfg):
    """Drive BaseWAMArchitecture._init_video_backbone in isolation by binding
    the method onto a stub. Returns (stub, raised_or_none)."""
    from openwam.model.base import BaseWAMArchitecture

    stub = _StubArchitecture()
    BaseWAMArchitecture._init_video_backbone(stub, model_cfg)
    return stub


def test_D1_encoder_yaml_rejects_extra_fields(monkeypatch):
    """Yaml whitelist allows only {name, model_path}; extras must raise."""
    # Patch build_video_encoder to a no-op stub so the gate check is reached
    # before any real encoder loading. We expect a ValueError BEFORE that
    # call happens (whitelist runs first).
    cfg = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": True,
            "encoder": {"name": "wan_vae", "model_path": "/dummy", "z_dim": 16},  # z_dim is extra
        }
    }
    with pytest.raises(ValueError, match=r"allows only"):
        _run_init_video_backbone(cfg)


def test_D1a_encoder_yaml_whitelist_extends_per_optional_yaml_keys(monkeypatch):
    """An optional yaml field is allowed only on encoders that opt in via
    :meth:`VideoEncoder.optional_yaml_keys`. The same field on a different
    encoder (which did NOT opt in) is rejected.

    Concretely: ``vjepa2_1_forward`` is in V-JEPA 2.1's optional set, so it
    is accepted on the vjepa2_1 encoder block; the same field is NOT in
    wan_vae's optional set, so it is rejected on a wan_vae encoder block.
    Guards against a yaml typo (``vjepa2_1_forward`` on wan_vae) silently
    being ignored.
    """
    # vjepa2_1 + vjepa2_1_forward: build_video_encoder is the only thing
    # the gate actually invokes after the whitelist passes; patch it to a
    # no-op stub so the test does not need real weights.
    from openwam.model.video_backbone import encoder as encoder_mod

    monkeypatch.setattr(encoder_mod, "build_video_encoder", lambda cfg: object())
    monkeypatch.setattr(
        "openwam.model.video_backbone.build_video_backbone",
        lambda *a, **kw: nn.Module(),
    )

    ok_cfg = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": True,
            "temporal_compression": 4,
            "causal_temporal": True,
            "encoder": {
                "name": "vjepa2_1",
                "model_path": "/dummy",
                "vjepa2_1_forward": "video",
            },
        }
    }
    # Whitelist must allow vjepa2_1_forward on the vjepa2_1 encoder. The
    # temporal_compression cross-check would fire AFTER the whitelist;
    # since build_video_backbone is stubbed to a bare Module (no
    # temporal_compression attribute), that read raises AttributeError —
    # which the try/except below catches. We only care that the whitelist
    # ValueError did NOT fire.
    try:
        _run_init_video_backbone(ok_cfg)
    except ValueError as e:
        if "allows only" in str(e):
            raise AssertionError(
                f"vjepa2_1_forward should be allowed on vjepa2_1 encoder; got whitelist error: {e}"
            ) from e
    except AttributeError:
        pass  # downstream temporal_compression read fails — fine, whitelist already passed

    # Same field on wan_vae must trip the whitelist.
    bad_cfg = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": True,
            "encoder": {
                "name": "wan_vae",
                "model_path": "/dummy",
                "vjepa2_1_forward": "video",
            },
        }
    }
    with pytest.raises(ValueError, match=r"allows only"):
        _run_init_video_backbone(bad_cfg)


def test_D1c_encoder_yaml_whitelist_ignores_yaml_null_fields(monkeypatch):
    """Regression: a yaml-``null`` encoder field must be treated as absent.

    An inline ``encoder:`` block (or a Hydra group merge) can leave a field
    set to ``null`` that the active encoder does not declare in
    ``optional_yaml_keys()``. The whitelist check at
    ``BaseWAMArchitecture._init_video_backbone`` must treat yaml-null as
    "field absent" so it does not trip a ValueError — otherwise every
    from_scratch=true run carrying a stray null field would be blocked.

    An explicit non-null value on the wrong encoder still raises (covered by
    test_D1a)."""
    from openwam.model.video_backbone import encoder as encoder_mod

    monkeypatch.setattr(encoder_mod, "build_video_encoder", lambda cfg: object())
    monkeypatch.setattr(
        "openwam.model.video_backbone.build_video_backbone",
        lambda *a, **kw: nn.Module(),
    )

    cfg = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": True,
            "temporal_compression": 4,
            "causal_temporal": True,
            "encoder": {
                "name": "vjepa2_1",
                "model_path": "/dummy",
                "vjepa2_1_forward": "video",
                # A stray null field the active encoder doesn't declare —
                # must NOT trip the whitelist.
                "unknown_encoder_knob": None,
            },
        }
    }
    # Whitelist must pass. The downstream backbone build may then fail on
    # the temporal_compression / build_video_backbone stub (see test_D1a's
    # comment) — we tolerate that because the assertion here is "no
    # ValueError about extra fields was raised".
    try:
        _run_init_video_backbone(cfg)
    except ValueError as e:
        if "allows only" in str(e):
            raise AssertionError(
                f"a yaml-null field on the vjepa2_1 encoder must not trip the whitelist; got: {e}"
            ) from e
    except AttributeError:
        pass  # downstream temporal_compression read on stubbed backbone — fine

    # Sibling guard: an explicit non-null unknown field on vjepa2_1 must
    # STILL fail (the field is wrong-encoder, not just an inline leftover).
    cfg_explicit = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": True,
            "encoder": {
                "name": "vjepa2_1",
                "model_path": "/dummy",
                "unknown_encoder_knob": True,  # explicit → operator mistake
            },
        }
    }
    with pytest.raises(ValueError, match=r"allows only"):
        _run_init_video_backbone(cfg_explicit)


def test_D2_encoder_block_with_from_scratch_false_silently_ignored(monkeypatch, caplog):
    """The encoder block is silently ignored (no error, encoder NOT built)
    when from_scratch=false. The default yaml ships with an encoder: block
    for documentation discoverability — fail-fast would break the default
    training command. We log INFO instead so users can still find the answer
    when debugging "why isn't my encoder being used?"."""
    encoder_built: list = []
    fake_kw: dict = {}

    def fake_build_encoder(enc_cfg):
        encoder_built.append(enc_cfg)
        raise RuntimeError("should not be reached when from_scratch=false")

    def fake_build_backbone(name, cfg, **kw):
        fake_kw.update(kw)
        bb = nn.Module()
        bb._pipe = None
        bb.temporal_compression = 4
        bb.causal_temporal = True
        return bb

    import openwam.model.video_backbone as vb_pkg
    from openwam.model.video_backbone import encoder as enc_pkg

    monkeypatch.setattr(enc_pkg, "build_video_encoder", fake_build_encoder)
    monkeypatch.setattr(vb_pkg, "build_video_backbone", fake_build_backbone)

    import logging

    caplog.set_level(logging.INFO, logger="openwam.model.base")

    cfg = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": False,
            "encoder": {"name": "wan_vae", "model_path": "/dummy"},
        }
    }
    _run_init_video_backbone(cfg)
    assert encoder_built == [], "build_video_encoder must NOT be called when from_scratch=false"
    assert fake_kw.get("external_encoder") is None, "no external_encoder should be passed to backbone"
    assert any("encoder block IGNORED" in rec.message for rec in caplog.records), (
        "expected an INFO log explaining that the encoder block was ignored"
    )


def test_D2b_deploy_with_encoder_block_and_from_scratch_false_keeps_native_vae(monkeypatch):
    """Backward-compat regression: a ``from_scratch=false`` checkpoint
    saved by current code still carries the framework yaml's inline
    ``encoder:`` block in its config.yaml (PR #60 commit 6588044 added it
    for discoverability). The state_dict topology is ``_pipe.vae.*`` —
    the encoder block must NOT trigger external-encoder-skeleton
    construction on the deploy path; otherwise strict checkpoint load
    would mismatch ``_pipe.vae.*`` vs ``_encoder._m.*``.

    Same gate as training (test_D2): encoder honored ONLY when
    ``from_scratch=true``. Deploy path stays quiet (no log spam) —
    seeing the inline block at from_scratch=false is the expected
    common case, not a user mistake.
    """
    skeleton_calls: list = []
    backbone_kwargs: dict = {}

    def fake_skeleton(enc_cfg, source):
        skeleton_calls.append((enc_cfg, source))
        raise RuntimeError("should not be reached when from_scratch=false on deploy")

    def fake_build_backbone(name, cfg, **kw):
        backbone_kwargs.update(kw)
        bb = nn.Module()
        bb._pipe = None
        bb.temporal_compression = 4
        bb.causal_temporal = True
        return bb

    import openwam.model.video_backbone as vb_pkg

    monkeypatch.setattr(vb_pkg, "build_video_backbone", fake_build_backbone)
    # Patch on the class so the bound-method dispatch in _init_video_backbone
    # picks it up.
    from openwam.model.base import BaseWAMArchitecture

    monkeypatch.setattr(BaseWAMArchitecture, "_build_external_encoder_skeleton", staticmethod(fake_skeleton))

    cfg = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": False,
            "encoder": {"name": "wan_vae", "model_path": "/dummy"},
            "_source": {"components": [{"attr": "vae", "model_class": "x", "extra_kwargs": {}}]},
        }
    }
    _run_init_video_backbone(cfg)
    assert skeleton_calls == [], (
        "_build_external_encoder_skeleton must NOT be called on the deploy path when "
        "from_scratch=false; that would force external-encoder state_dict topology "
        "on a checkpoint saved under the native pipe.vae path"
    )
    assert backbone_kwargs.get("external_encoder") is None, (
        "deploy build_video_backbone must receive external_encoder=None on the "
        "from_scratch=false path, regardless of yaml encoder block"
    )


def test_D3_encoder_built_when_from_scratch_true_and_encoder_set(monkeypatch):
    """from_scratch=true + encoder set → build_video_encoder is called and
    its product is propagated to build_video_backbone."""
    encoder_built: list = []
    backbone_kwargs: dict = {}

    def fake_build_encoder(enc_cfg):
        encoder_built.append(enc_cfg)
        return WanVideoVAEEncoderStub(spec_z_dim=16, is_reversible=True)

    def fake_build_backbone(name, cfg, **kw):
        backbone_kwargs.update(kw)
        bb = nn.Module()
        bb._pipe = None  # triggers the "skipping" warning branch
        bb.temporal_compression = 4
        bb.causal_temporal = True
        return bb

    # _init_video_backbone re-imports both symbols inside the function body,
    # so monkeypatching them on their defining modules covers every call.
    import openwam.model.video_backbone as vb_pkg
    from openwam.model.video_backbone import encoder as enc_pkg

    monkeypatch.setattr(enc_pkg, "build_video_encoder", fake_build_encoder)
    monkeypatch.setattr(vb_pkg, "build_video_backbone", fake_build_backbone)

    cfg = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": True,
            "encoder": {"name": "wan_vae", "model_path": "/dummy"},
        }
    }
    _run_init_video_backbone(cfg)
    assert len(encoder_built) == 1, "build_video_encoder should be called exactly once"
    assert "external_encoder" in backbone_kwargs, "external_encoder must be passed to build_video_backbone"
    assert isinstance(backbone_kwargs["external_encoder"], WanVideoVAEEncoderStub)


def test_D4_encoder_not_built_when_no_encoder_block(monkeypatch):
    """from_scratch=true without an encoder block keeps the historical
    reset-weights-only path: build_video_encoder is NOT called and no
    external_encoder is forwarded to build_video_backbone."""
    encoder_built: list = []
    fake_kw: dict = {}

    def fake_build_encoder(enc_cfg):
        encoder_built.append(enc_cfg)
        raise RuntimeError("should not be reached")

    def fake_build_backbone(name, cfg, **kw):
        fake_kw.update(kw)
        bb = nn.Module()
        bb._pipe = None
        bb.temporal_compression = 4
        bb.causal_temporal = True
        return bb

    import openwam.model.video_backbone as vb_pkg
    from openwam.model.video_backbone import encoder as enc_pkg

    monkeypatch.setattr(enc_pkg, "build_video_encoder", fake_build_encoder)
    monkeypatch.setattr(vb_pkg, "build_video_backbone", fake_build_backbone)

    cfg = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": True,
            # No encoder block — preserves the current main behavior.
        }
    }
    _run_init_video_backbone(cfg)
    assert encoder_built == [], "build_video_encoder should not be called without an encoder block"
    assert fake_kw.get("external_encoder") is None, "no external_encoder should be passed"


def test_D5_generate_decode_video_true_blocks_irreversible_encoder():
    """``_assert_decode_video_supported`` raises when ``vb._encoder`` is
    irreversible — the same helper :meth:`BaseWAMArchitecture.generate`
    calls just before invoking ``vb.decode_video`` when ``decode_video=True``.

    Standing up the full ``generate`` denoising loop in a unit test would
    require the entire scheduler/pipeline stack; instead we extract the
    guard as ``_assert_decode_video_supported`` (base.py) and exercise it
    directly with a stub backbone, so a regression that renames
    ``vb._encoder`` or flips the polarity is caught here.
    """
    from openwam.model.base import _assert_decode_video_supported

    class _StubBackbone:
        pass

    # Irreversible → fail-fast.
    vb_irrev = _StubBackbone()
    vb_irrev._encoder = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False)
    with pytest.raises(ValueError, match=r"irreversible"):
        _assert_decode_video_supported(vb_irrev)

    # Reversible → no-op.
    vb_rev = _StubBackbone()
    vb_rev._encoder = WanVideoVAEEncoderStub(spec_z_dim=16, is_reversible=True)
    _assert_decode_video_supported(vb_rev)

    # Native VAE path (no _encoder attribute at all) → no-op.
    vb_native = _StubBackbone()
    _assert_decode_video_supported(vb_native)

    # Native VAE path (_encoder is None — what WanVideoBackbone sets when
    # external_encoder is not provided) → no-op.
    vb_native_none = _StubBackbone()
    vb_native_none._encoder = None
    _assert_decode_video_supported(vb_native_none)


def test_D6_freeze_modules_resolves_encoder_dotted_path_on_external_path():
    """Regression for the severe S2 bug: when ``video_backbone.from_scratch=true``
    routes through an external encoder, ``pipe.vae`` is None and the
    historical ``freeze: [..., video_backbone._pipe.vae, ...]`` entry is
    silently skipped — leaving the encoder's pretrained weights trainable.
    The fix lists ``video_backbone._encoder`` in the freeze yamls; this
    test verifies the dotted path actually resolves via
    ``nn.Module.get_submodule`` and that ``requires_grad_(False)`` then
    propagates to every encoder parameter.
    """
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False)
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)

    # Mount the backbone on an architecture-shaped container, just like the
    # real BaseWAMArchitecture does — freeze_modules resolves dotted paths
    # rooted at ``self``.
    class _ArchContainer(nn.Module):
        def __init__(self, vb):
            super().__init__()
            self.video_backbone = vb

    arch = _ArchContainer(backbone)

    # get_submodule routes through the WanVideoBackbone's _modules dict;
    # ``_encoder = external_encoder`` in __init__ registers it there.
    resolved = arch.get_submodule("video_backbone._encoder")
    assert resolved is enc

    # Encoder parameters are trainable by default. Pre-fix: the freeze yaml
    # entry (_pipe.vae) was silently skipped because pipe.vae=None, so the
    # encoder stayed trainable. Post-fix: the new yaml entry (_encoder)
    # resolves and the call below disables grad on every encoder param.
    assert any(p.requires_grad for p in enc.parameters())
    resolved.requires_grad_(False)
    assert all(not p.requires_grad for p in enc.parameters())

    # Sibling guards: native VAE path → _encoder is None and not in _modules,
    # so freeze_modules's get_submodule call must raise AttributeError so
    # the framework can silently skip it (yaml lists both paths).
    pipe2 = _FakePipe(vae_z_dim=16, vae_upsample=8)
    backbone2 = WanVideoBackbone.from_pretrained(pipe2)  # no external_encoder
    arch2 = _ArchContainer(backbone2)
    with pytest.raises(AttributeError):
        arch2.get_submodule("video_backbone._encoder")


def test_M3a_filter_native_vae_configs_drops_vae_entries():
    """``_filter_native_vae_configs`` removes any ModelConfig whose path or
    origin_file_pattern matches a Wan VAE weight file (case-insensitive
    'vae' basename), and leaves DiT/T5/CLIP entries untouched. The
    irreversible external-encoder path uses this so the ~1.5GB native VAE
    is never materialized on CPU only to be released seconds later.
    """
    from openwam.model.video_backbone.wan.pipeline_builder import _filter_native_vae_configs
    from openwam.model.video_backbone.wan.shared.core.loader import ModelConfig

    configs = [
        ModelConfig(path="/m/Wan2.2_VAE.safetensors"),
        ModelConfig(path="/m/Wan2.1_VAE.pth"),
        ModelConfig(path=["/m/dit-00001-of-00002.safetensors", "/m/dit-00002-of-00002.safetensors"]),
        ModelConfig(path="/m/models_t5_umt5-xxl-enc-bf16.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="vae/Wan2.1_VAE.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_clip.safetensors"),
    ]
    kept = _filter_native_vae_configs(configs)
    kept_descr = [c.path if c.path else c.origin_file_pattern for c in kept]
    assert configs[0] not in kept, "Wan2.2_VAE.safetensors must be dropped"
    assert configs[1] not in kept, "Wan2.1_VAE.pth must be dropped"
    assert configs[2] in kept, "DiT shard list must be kept"
    assert configs[3] in kept, "T5 must be kept"
    assert configs[4] not in kept, "origin_file_pattern with 'vae' basename must be dropped"
    assert configs[5] in kept, "CLIP must be kept"
    # No surprise drops or duplications.
    assert len(kept) == 3, f"expected 3 surviving configs, got {len(kept)}: {kept_descr}"


def test_M3b_from_pretrained_routes_skip_native_vae():
    """``WanVideoBackbone.from_pretrained`` decides ``skip_native_vae`` via:

      - training (``DictConfig`` source) + irreversible encoder → True
      - training + reversible encoder → False (validation needs native VAE)
      - deploy (``dict`` / ``str`` source) + any external encoder → True
        (state_dict topology is ``_encoder._m.*``, not ``_pipe.vae.*``;
        deploy must not materialize the empty native VAE slot)
      - no external encoder → False on both paths

    Monkeypatches ``_build_pipe_from_model_path`` to capture the kwarg
    rather than stand up a real pipeline. Training-path uses a real
    ``DictConfig`` to exercise the ``isinstance(source, DictConfig)``
    dispatch added for the deploy fix.
    """
    from omegaconf import OmegaConf

    from openwam.model.video_backbone import wan_videobackbone as wan_adapter
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    captured: dict = {}

    def _spy_model_path(model_path, device="cpu", *, skip_native_vae=False):
        captured["skip"] = skip_native_vae
        return _FakePipe(vae_z_dim=16, vae_upsample=8)

    def _spy_training(cfg, *, skip_native_vae=False):
        captured["skip"] = skip_native_vae
        return _FakePipe(vae_z_dim=16, vae_upsample=8)

    original = WanVideoBackbone._build_pipe_from_model_path
    wan_adapter.WanVideoBackbone._build_pipe_from_model_path = staticmethod(_spy_model_path)
    import openwam.model.video_backbone.wan.pipeline_builder as pb_mod

    original_btp = pb_mod.build_training_pipeline
    pb_mod.build_training_pipeline = _spy_training
    try:
        # --- Training path: DictConfig source ---
        train_cfg = OmegaConf.create({"video_backbone": {"model_path": "/dummy"}})

        # Training + irreversible → skip=True.
        captured.clear()
        WanVideoBackbone.from_pretrained(
            train_cfg,
            external_encoder=WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False),
        )
        assert captured["skip"] is True

        # Training + reversible → skip=False (need native VAE for spec validation).
        captured.clear()
        WanVideoBackbone.from_pretrained(
            train_cfg,
            external_encoder=WanVideoVAEEncoderStub(spec_z_dim=16, is_reversible=True),
        )
        assert captured["skip"] is False

        # Training + no encoder → skip=False.
        captured.clear()
        WanVideoBackbone.from_pretrained(train_cfg)
        assert captured["skip"] is False

        # --- Deploy path: dict source (state_dict topology is _encoder._m.*) ---
        deploy_cfg = {"video_backbone": {"model_path": "/dummy"}}

        # Deploy + reversible → skip=True (no native VAE slot to materialize).
        captured.clear()
        WanVideoBackbone.from_pretrained(
            deploy_cfg,
            external_encoder=WanVideoVAEEncoderStub(spec_z_dim=16, is_reversible=True),
        )
        assert captured["skip"] is True

        # Deploy + irreversible → skip=True.
        captured.clear()
        WanVideoBackbone.from_pretrained(
            deploy_cfg,
            external_encoder=WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False),
        )
        assert captured["skip"] is True

        # Deploy + no encoder → skip=False (no encoder means use native VAE).
        captured.clear()
        WanVideoBackbone.from_pretrained(deploy_cfg)
        assert captured["skip"] is False
    finally:
        wan_adapter.WanVideoBackbone._build_pipe_from_model_path = original
        pb_mod.build_training_pipeline = original_btp


def test_M3c_wan_vae_encoder_from_skeleton_matches_from_pretrained_topology():
    """``WanVideoVAEEncoder.from_skeleton(entry)`` (deploy-time, zero
    weights) must produce a state_dict with the EXACT same key set as
    ``WanVideoVAEEncoder(loaded_vae)`` (training-time). Otherwise the
    architecture's strict checkpoint load would mismatch on either path.
    """
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder

    # Training path: real loaded module → encoder wrapping.
    vae_loaded = _FakeWanVAEModule(z_dim=16, upsampling_factor=8)
    enc_train = WanVideoVAEEncoder(vae_loaded)

    # Deploy path: skeleton from components entry. ``_FakeWanVAEModule``
    # lives at this dotted path; in production it'd be
    # ``openwam.model.video_backbone.wan.vae.WanVideoVAE38`` etc.
    components_entry = {
        "attr": "vae",
        "model_class": "tests.test_external_encoder._FakeWanVAEModule",
        "extra_kwargs": {"z_dim": 16, "upsampling_factor": 8},
    }
    enc_deploy = WanVideoVAEEncoder.from_skeleton(components_entry)

    train_keys = set(enc_train.state_dict().keys())
    deploy_keys = set(enc_deploy.state_dict().keys())
    missing_on_deploy = train_keys - deploy_keys
    extra_on_deploy = deploy_keys - train_keys
    assert not missing_on_deploy and not extra_on_deploy, (
        f"state_dict topology mismatch:\n"
        f"  missing on deploy: {sorted(missing_on_deploy)}\n"
        f"  extra on deploy:   {sorted(extra_on_deploy)}"
    )
    # Spec is derived from loaded weights — must agree across paths.
    assert enc_train.spec == enc_deploy.spec


def test_M3d_build_external_encoder_skeleton_picks_vae_entry_from_source():
    """``BaseWAMArchitecture._build_external_encoder_skeleton`` reaches into
    the deploy ``source`` dict for ``components`` and finds the
    ``attr=='vae'`` entry, then dispatches to the registered encoder's
    ``from_skeleton``.
    """
    from openwam.model.base import BaseWAMArchitecture
    from openwam.model.video_backbone.encoder import (
        _VIDEO_ENCODER_REGISTRY,
        WanVideoVAEEncoder,
        register_video_encoder,
    )

    # Use the real wan_vae registry entry — it owns the from_skeleton
    # implementation that the deploy path will route through in
    # production.
    assert _VIDEO_ENCODER_REGISTRY["wan_vae"] is WanVideoVAEEncoder

    enc_cfg = {"name": "wan_vae", "model_path": "/unused-on-deploy-path"}
    source = {
        "components": [
            {
                "attr": "dit",
                "model_class": "openwam.model.video_backbone.wan.dit.WanModel",
                "extra_kwargs": {},
            },
            {
                "attr": "vae",
                "model_class": "tests.test_external_encoder._FakeWanVAEModule",
                "extra_kwargs": {"z_dim": 16, "upsampling_factor": 8},
            },
        ],
    }
    enc = BaseWAMArchitecture._build_external_encoder_skeleton(enc_cfg, source)
    assert isinstance(enc, WanVideoVAEEncoder)
    assert enc.spec.z_dim == 16
    assert enc.spec.spatial_compression == 8

    # Missing components → loud error (no silent fallback to native VAE).
    with pytest.raises(RuntimeError, match=r"no video_backbone\.components|components"):
        BaseWAMArchitecture._build_external_encoder_skeleton(enc_cfg, {"components": []})

    # Extra yaml field → same whitelist as training path.
    with pytest.raises(ValueError, match=r"allows only"):
        BaseWAMArchitecture._build_external_encoder_skeleton(
            {"name": "wan_vae", "model_path": "/x", "extra_field": 1}, source
        )

    # Unknown encoder name.
    register_video_encoder  # noqa: F841 — ensure registry is imported
    with pytest.raises(KeyError, match=r"Unknown video encoder"):
        BaseWAMArchitecture._build_external_encoder_skeleton({"name": "not_a_real_encoder", "model_path": "/x"}, source)


def test_M3e_deploy_path_does_not_reinit_dit_when_from_scratch_true():
    """Deploy-side regression: even when ``cfg.video_backbone.from_scratch=true``
    (because the config was saved from a from-scratch training run), the
    deploy path MUST NOT call ``reinit_dit_from_scratch`` — DiT weights
    come from the checkpoint via ``load_checkpoint`` strict load right
    after architecture construction; reinit would silently wipe them.

    We probe by spying on the module-level function and asserting it
    isn't called when ``source is not None`` in the cfg.
    """
    import openwam.model.video_backbone.wan_videobackbone as wan_adapter_mod
    from openwam.model.base import BaseWAMArchitecture
    from openwam.model.video_backbone import wan_videobackbone as wan_adapter

    reinit_calls = []
    original_reinit = wan_adapter_mod.reinit_dit_from_scratch

    def _spy_reinit(*a, **kw):
        # Record-only stub: don't invoke the real reinit because the
        # stub backbone's _FakePipe has empty dit.blocks and would
        # IndexError. We only care whether reinit was called at all.
        reinit_calls.append((a, kw))

    wan_adapter_mod.reinit_dit_from_scratch = _spy_reinit
    # ``base.py`` imports it locally inside the function, so the patch
    # must target the wan_adapter module attribute.
    wan_adapter.reinit_dit_from_scratch = _spy_reinit

    # Make build_video_backbone hand back a stub backbone so we can drive
    # _init_video_backbone end-to-end without a real pipeline build.
    import openwam.model.video_backbone as vb_pkg

    original_build = vb_pkg.build_video_backbone

    def _stub_build(name, cfg, **kw):
        bb = nn.Module()
        bb._pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
        bb._uses_external_encoder = False
        bb.temporal_compression = 4
        bb.causal_temporal = True
        bb.dit_patch_size = (1, 2, 2)
        return bb

    vb_pkg.build_video_backbone = _stub_build

    try:
        stub = _StubArchitecture()

        # Deploy cfg: has _source (signaling deploy), AND from_scratch=true
        # (carried over from the training run that produced this checkpoint).
        deploy_cfg = {
            "video_backbone": {
                "name": "wan22_ti2v_5b",
                "_source": {"components": []},
                "from_scratch": True,
            }
        }
        BaseWAMArchitecture._init_video_backbone(stub, deploy_cfg)
        assert reinit_calls == [], (
            f"reinit_dit_from_scratch was called {len(reinit_calls)}x on deploy path; "
            "DiT weights would be wiped before load_checkpoint fills them in"
        )

        # Sanity: training path with from_scratch=true DOES call reinit.
        reinit_calls.clear()
        train_cfg = {
            "video_backbone": {
                "name": "wan22_ti2v_5b",
                "from_scratch": True,
            }
        }
        BaseWAMArchitecture._init_video_backbone(stub, train_cfg)
        assert len(reinit_calls) == 1, (
            f"training path with from_scratch=true should reinit exactly once, got {len(reinit_calls)}"
        )
    finally:
        wan_adapter_mod.reinit_dit_from_scratch = original_reinit
        wan_adapter.reinit_dit_from_scratch = original_reinit
        vb_pkg.build_video_backbone = original_build


def test_M3g_from_pretrained_attaches_pipe_latent_spec_on_external_path():
    """``WanVideoBackbone.from_pretrained`` must attach
    ``pipe.latent_spec`` (= ``external_encoder.spec``) so vendored
    inference units (``WanVideoUnit_NoiseInitializer``) can read latent
    shape metadata without falling back to ``pipe.vae`` (which is None
    on this path).

    Native VAE path leaves ``pipe.latent_spec`` absent — the vendored
    unit's fallback branch then reads ``pipe.vae`` as before, preserving
    bit-exact behavior for old checkpoints.
    """
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    # --- External encoder path: pipe.latent_spec is set ---
    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=16, is_reversible=True)
    WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    assert hasattr(pipe, "latent_spec"), "pipe.latent_spec missing on external-encoder path"
    assert pipe.latent_spec is enc.spec, "pipe.latent_spec must reference encoder.spec verbatim"
    assert pipe.vae is None, "pipe.vae must be released on external-encoder path"

    # --- Native VAE path: pipe.latent_spec is absent ---
    pipe2 = _FakePipe(vae_z_dim=16, vae_upsample=8)
    WanVideoBackbone.from_pretrained(pipe2)  # no external_encoder
    assert not hasattr(pipe2, "latent_spec"), (
        "pipe.latent_spec must not be attached on the native VAE path "
        "— the vendored unit's pipe.vae fallback must remain authoritative"
    )


def test_M3h_noise_initializer_reads_latent_spec_when_present():
    """``WanVideoUnit_NoiseInitializer.process`` must:

    - Read ``pipe.latent_spec`` when present (external-encoder path)
      and honor ``temporal_compression`` + ``causal_temporal`` —
      non-Wan-VAE encoders (DINOv3 temporal=1, causal=False) get the
      right length.
    - Fall back to ``pipe.vae`` when ``pipe.latent_spec`` is absent
      (native VAE path) using the historical
      ``(num_frames-1)//4+1`` formula bit-exactly.
    """
    from openwam.model.video_backbone.wan.pipeline import WanVideoUnit_NoiseInitializer

    unit = WanVideoUnit_NoiseInitializer()

    class _StubPipe:
        @staticmethod
        def generate_noise(shape, *, seed, rand_device):
            return torch.zeros(shape)

    # --- Path 1: external encoder spec (DINOv3-style) ---
    pipe = _StubPipe()
    pipe.latent_spec = VideoEncoderSpec(
        z_dim=1024,
        spatial_compression=16,
        temporal_compression=1,
        causal_temporal=False,
        is_reversible=False,
        dit_patch_size=(1, 1, 1),
    )
    out = unit.process(
        pipe, height=256, width=256, num_frames=49, seed=42, rand_device="cpu", vace_reference_image=None
    )
    # length = (49-1)//1 + 0 = 48
    # shape = (1, 1024, 48, 256/16, 256/16) = (1, 1024, 48, 16, 16)
    assert out["noise"].shape == (1, 1024, 48, 16, 16)

    # --- Path 2: external encoder spec (Wan VAE-style, causal) ---
    pipe = _StubPipe()
    pipe.latent_spec = VideoEncoderSpec(
        z_dim=48,
        spatial_compression=16,
        temporal_compression=4,
        causal_temporal=True,
    )
    out = unit.process(
        pipe, height=480, width=832, num_frames=49, seed=42, rand_device="cpu", vace_reference_image=None
    )
    # length = (49-1)//4 + 1 = 13
    # shape = (1, 48, 13, 480/16, 832/16) = (1, 48, 13, 30, 52)
    assert out["noise"].shape == (1, 48, 13, 30, 52)

    # --- Path 3: fallback to pipe.vae (native VAE path, old behavior) ---
    pipe = _StubPipe()  # no latent_spec attribute

    class _NativeVae:
        upsampling_factor = 8

        class model:  # noqa: N801 — vendored attribute access path
            z_dim = 16

    pipe.vae = _NativeVae()
    out = unit.process(
        pipe, height=480, width=832, num_frames=49, seed=42, rand_device="cpu", vace_reference_image=None
    )
    # length = (49-1)//4 + 1 = 13  (historical hardcoded formula)
    # shape = (1, 16, 13, 480/8, 832/8) = (1, 16, 13, 60, 104)
    assert out["noise"].shape == (1, 16, 13, 60, 104)


def test_M3f_train_save_deploy_state_dict_topology_matches():
    """End-to-end closure for the deploy fix: state_dict produced on the
    training side (real ``WanVideoVAEEncoder`` wrapping a loaded VAE
    inside a ``WanVideoBackbone`` with ``external_encoder=enc``) and on
    the deploy side (encoder built via ``from_skeleton`` from a saved
    components entry) must share the EXACT same key set.

    Without this, ``architecture.load_checkpoint(path)`` strict load on
    deploy raises ``RuntimeError: Strict load failed`` — exactly the
    failure mode that surfaced after PR #60's first deploy attempt.
    """
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone

    # --- "Training" side ---
    train_pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    train_enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    train_bb = WanVideoBackbone.from_pretrained(train_pipe, external_encoder=train_enc)
    train_keys = set(train_bb.state_dict().keys())
    # Must use the external-encoder slot, not the native vae.*.
    assert any(k.startswith("_encoder.") for k in train_keys), (
        "training-time backbone state_dict missing _encoder.* keys"
    )
    assert not any(k.startswith("vae.") for k in train_keys), (
        "training-time backbone state_dict has vae.* — native VAE wasn't released"
    )

    # --- "Deploy" side: reconstruct from saved components entry ---
    components_entry = {
        "attr": "vae",
        "model_class": "tests.test_external_encoder._FakeWanVAEModule",
        "extra_kwargs": {"z_dim": 16, "upsampling_factor": 8},
    }
    deploy_enc = WanVideoVAEEncoder.from_skeleton(components_entry)
    # In production this comes from cls.from_pretrained(source=dict-with-components),
    # which routes through _build_pipe_from_components(skip_native_vae=True). For this
    # closure check the WanVideoBackbone construction path is the same as training,
    # just with a fresh pipe sans the loaded VAE — using _FakePipe directly is
    # equivalent because skip_native_vae=True on deploy zeroes pipe.vae anyway.
    deploy_pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    deploy_bb = WanVideoBackbone.from_pretrained(deploy_pipe, external_encoder=deploy_enc)
    deploy_keys = set(deploy_bb.state_dict().keys())

    # The crux: bit-exact key topology.
    missing = train_keys - deploy_keys
    extra = deploy_keys - train_keys
    assert not missing and not extra, (
        f"deploy state_dict topology mismatch (would break strict load):\n"
        f"  on train but not on deploy: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}\n"
        f"  on deploy but not on train: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}"
    )

    # Strict load smoke test: instantiate two backbones, save one,
    # load into the other — must succeed.
    sd = train_bb.state_dict()
    missing_keys, unexpected = deploy_bb.load_state_dict(sd, strict=False)
    assert not missing_keys and not unexpected, (
        f"deploy load_state_dict found missing={missing_keys[:3]}, unexpected={unexpected[:3]}"
    )


def test_D8_wan_vae_path_end_to_end_freeze_excludes_encoder_params_from_optimizer():
    """End-to-end check for the ``from_scratch=true + encoder.name=wan_vae``
    path: starting from the yaml freeze list, walk the actual production
    code (``BaseWAMArchitecture.freeze_modules`` →
    ``_pipe_named_parameters``) and verify ZERO encoder parameters survive
    into the optimizer.

    Uses the real ``WanVideoVAEEncoder`` class (not a stub) wrapped around
    a ``_FakeWanVAEModule`` so the test runs without GPU/real-weight
    dependencies but exercises the same nn.Module nesting layout the
    production encoder has: ``WanVideoVAEEncoder._m = vae`` with the VAE
    holding the bulk of the parameters.
    """
    import pathlib

    from omegaconf import OmegaConf

    from openwam.model.base import BaseWAMArchitecture
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_videobackbone import WanVideoBackbone
    from openwam.train.utils.optimizer_groups import _pipe_named_parameters

    # 1) Real encoder + real backbone.
    vae_module = _FakeWanVAEModule(z_dim=16, upsampling_factor=8)
    enc = WanVideoVAEEncoder(vae_module)
    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    assert backbone._uses_external_encoder is True
    assert backbone._encoder is enc

    # 2) Stand-in architecture: only the bits freeze_modules /
    #    _pipe_named_parameters touch. Cannot subclass BaseWAMArchitecture
    #    directly because abstract methods (forward, generate, ...) demand
    #    full pipeline machinery. An ``nn.Module`` container with the
    #    right child name is enough — freeze_modules uses self.get_submodule,
    #    and _pipe_named_parameters uses arch.named_children().
    class _ArchContainer(nn.Module):
        def __init__(self, vb):
            super().__init__()
            self.video_backbone = vb

        def get_trainable_modules(self, freeze_list=()):
            return BaseWAMArchitecture.get_trainable_modules(self, freeze_list)

    arch = _ArchContainer(backbone)

    # 3) Real yaml freeze list (no hand-curation — read the file that ships).
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    yaml_cfg = OmegaConf.load(repo_root / "configs/training_strategy/joint.yaml")
    freeze_list = list(yaml_cfg.freeze)
    assert "video_backbone._encoder" in freeze_list, (
        "joint.yaml must list video_backbone._encoder for this test to be meaningful"
    )

    # 4) Run the exact production freeze_modules call.
    frozen = BaseWAMArchitecture.freeze_modules(arch, freeze_list)
    assert "video_backbone._encoder" in frozen, f"_encoder failed to freeze; freeze_modules returned: {frozen}"

    # 5) Sanity: encoder has parameters at all, and every single one is now
    #    requires_grad=False.
    enc_params = list(enc.parameters())
    assert len(enc_params) > 0, "WanVideoVAEEncoder must expose VAE parameters via _m"
    assert all(not p.requires_grad for p in enc_params), "freeze did not propagate to encoder._m.* parameters"

    # 6) Run the exact production optimizer-param collection path
    #    (``_pipe_named_parameters``) and confirm ZERO encoder parameters
    #    leak into the optimizer.
    class _ModelStub:
        architecture = arch
        lambda_action = 0  # excluded; not relevant here

    pairs = _pipe_named_parameters(_ModelStub())
    # Encoder parameter names appear as ``video_backbone._encoder._m.*`` in
    # the production output (mod_name="video_backbone" + named_parameters
    # path).
    leaked = [name for name, _ in pairs if "_encoder." in name]
    assert leaked == [], f"Encoder parameters leaked into _pipe_named_parameters: {leaked[:5]}..."

    # 7) Sibling guard: encoder params are reachable from the backbone via
    #    backbone.named_parameters() (so the test isn't trivially passing
    #    because they were hidden), they're just filtered by requires_grad.
    by_name = dict(backbone.named_parameters())
    enc_param_keys = [k for k in by_name if k.startswith("_encoder.")]
    assert len(enc_param_keys) > 0, (
        "encoder parameters must be reachable via backbone.named_parameters otherwise the leak check above is trivial"
    )
    assert all(not by_name[k].requires_grad for k in enc_param_keys)


@pytest.mark.parametrize(
    "yaml_path",
    [
        "configs/training_strategy/joint.yaml",
        "configs/training_strategy/video_only.yaml",
    ],
)
def test_D7_training_strategy_yaml_freezes_encoder(yaml_path):
    """The two training_strategy yamls that ship a ``freeze:`` list MUST
    enumerate ``video_backbone._encoder``. Pre-fix only ``_pipe.vae`` was
    listed and the external-encoder path silently bypassed freeze."""
    import pathlib

    from omegaconf import OmegaConf

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    cfg = OmegaConf.load(repo_root / yaml_path)
    freeze = list(cfg.get("freeze", []) or [])
    assert "video_backbone._encoder" in freeze, (
        f"{yaml_path} freeze list missing video_backbone._encoder; "
        f"external-encoder path will leave the encoder trainable. Got: {freeze}"
    )
    # The native-path entry must remain so default training stays bit-exact.
    assert "video_backbone._pipe.vae" in freeze


# ===========================================================================
# Commit 6: E1-E3 — framework yamls carry the encoder block (inline)
# ===========================================================================
#
# These tests verify the yaml ships with the documented defaults so that
# `from_scratch=false` users see the encoder field but it stays inert,
# and `from_scratch=true` users only need to flip one switch.


@pytest.mark.parametrize(
    "yaml_path",
    [
        # dual_system.yaml no longer ships an inline `video_backbone:` block —
        # it composes from the Hydra `backbone` group instead (default
        # `backbone/wan.yaml`). See test_E_dual_system_composed_encoder_block
        # below for the composed-default verification.
        "configs/model/shared_backbone.yaml",
        "configs/model/tri_system.yaml",
    ],
)
def test_E_framework_yaml_has_inline_encoder_block(yaml_path):
    """Each framework yaml's video_backbone block contains an inline
    encoder: {name, model_path} sub-block with `wan_vae` as the default."""
    import pathlib

    from omegaconf import OmegaConf

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    cfg = OmegaConf.load(repo_root / yaml_path)

    vb = cfg.video_backbone
    assert vb is not None, f"{yaml_path} missing video_backbone block"
    enc = vb.get("encoder")
    assert enc is not None, f"{yaml_path} missing video_backbone.encoder block"
    assert enc.name == "wan_vae", f"{yaml_path} default encoder.name should be wan_vae, got {enc.name!r}"
    assert "model_path" in enc, f"{yaml_path} encoder block missing model_path"
    # Yaml whitelist enforces {name, model_path} only; ensure no extra fields
    # have crept in.
    extras = set(enc.keys()) - {"name", "model_path"}
    assert extras == set(), f"{yaml_path} encoder block has extra fields {extras}, will trip the gate's whitelist"


def test_E_dual_system_composed_encoder_block():
    """dual_system.yaml composes its video_backbone from the Hydra `backbone`
    group (default `backbone/wan.yaml`). The composed config must still expose
    a `video_backbone.encoder: {name=wan_vae, model_path=...}` block so the
    encoder-gate path stays identical to shared_backbone / tri_system."""
    import os
    import pathlib

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    config_dir = os.path.abspath(repo_root / "configs")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["model=dual_system"])

    vb = cfg.model.video_backbone
    enc = vb.get("encoder")
    assert enc is not None, "dual_system + default backbone is missing video_backbone.encoder"
    assert enc.name == "wan_vae"
    assert "model_path" in enc
    # Keep the allowed encoder-block fields in this regression test in sync
    # with ``BaseWAMArchitecture._init_video_backbone`` /
    # ``_build_external_encoder_skeleton``.
    extras = set(enc.keys()) - {"name", "model_path"}
    assert extras == set(), f"encoder block has extra fields {extras}, will trip the gate's whitelist"


# ======================================================================
# A3: V-JEPA 2.1 encoder (V1-V8)
# ======================================================================


class _MockVJEPAViT(nn.Module):
    """Tiny CPU stand-in for the V-JEPA 2.1 ViT.

    Reproduces only what ``VJEPA21VideoEncoder._encode_image`` /
    ``_encode_video_tubelet`` consume: ``(B, C, T, H, W) -> (B, L, D)`` with
    ``L`` matching the post-patchify token count for tubelet=1 (T==1 branch)
    and tubelet=2 (T>1 branch). Has at least one parameter so ``next(
    self.parameters())`` yields a device/dtype anchor.
    """

    def __init__(self, embed_dim: int = 8, patch: int = 16, tubelet: int = 2):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch = int(patch)
        self.tubelet = int(tubelet)
        self.img_temporal_dim_size = 1
        self._proj = nn.Linear(1, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, T, H, W = x.shape
        h = H // self.patch
        w = W // self.patch
        if T == 1:
            L = h * w
        else:
            assert T % self.tubelet == 0
            L = (T // self.tubelet) * h * w
        # Use the parameter so requires_grad propagates and the dtype is real.
        seed = torch.zeros(B, L, 1, device=x.device, dtype=x.dtype)
        return self._proj(seed)


def _build_vjepa_encoder(embed_dim: int = 8):
    """Construct a ``VJEPA21VideoEncoder`` around the mock ViT, no weights load."""
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    vit = _MockVJEPAViT(embed_dim=embed_dim)
    return VJEPA21VideoEncoder(vit, embed_dim=embed_dim, variant="mock")


def test_V1_vjepa21_registration_round_trip():
    """``register_video_encoder("vjepa2_1")`` exposes the class via the registry."""
    from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder  # noqa: F401

    assert "vjepa2_1" in _VIDEO_ENCODER_REGISTRY
    assert _VIDEO_ENCODER_REGISTRY["vjepa2_1"] is VJEPA21VideoEncoder


def test_V2_vjepa21_from_pretrained_missing_manifest(tmp_path):
    """``from_pretrained`` on a dir without ``manifest.json`` raises FileNotFoundError."""
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        VJEPA21VideoEncoder.from_pretrained(str(tmp_path))


def test_V3_vjepa21_spec_invariants():
    """spec fields are nailed down: irreversible, causal, (1,2,2) DiT patch,
    temporal_compression=4 (ViT tubelet=2 + extra avg-pool stride=2, Wan VAE
    parity), z_dim wired from manifest."""
    enc = _build_vjepa_encoder(embed_dim=1408)
    spec = enc.spec
    assert spec.is_reversible is False
    assert spec.causal_temporal is True
    assert spec.dit_patch_size == (1, 2, 2)
    assert spec.z_dim == 1408
    assert spec.spatial_compression == 16
    assert spec.temporal_compression == 4
    assert spec.pixel_range == (-1.0, 1.0)


def test_V4_vjepa21_preprocess_imagenet_normalize():
    """preprocess_video ImageNet-normalizes — uniform 0.5-gray frames land near zero."""
    enc = _build_vjepa_encoder()
    frames = [Image.new("RGB", (32, 32), color=(128, 128, 128)) for _ in range(3)]
    video = enc.preprocess_video(frames)
    assert video.shape == (1, 3, 3, 32, 32)
    # 0.5 input - ImageNet mean (~0.45) / std (~0.22) ≈ small non-zero;
    # the std is what matters: properly normalized data has unit-ish channel std.
    flat = video.reshape(3, -1)
    assert flat.mean(dim=1).abs().max() < 1.0  # not absurdly far from 0
    assert flat.std(dim=1).max() < 1.0  # constant input -> per-channel std == 0


def test_V5_vjepa21_batch_encode_t_lat_shapes():
    """batch_encode T_pixel-to-T_lat dispatch:
    T_pixel == 1 -> T_lat == 1; T_pixel == 5 -> T_lat == 2; T_pixel == 9
    -> T_lat == 3 (== 1 + (T_pixel - 1) / 4).

    Independent of ``vjepa2_1_forward`` — both modes preserve the
    (1 cond + N_target/4 target) layout the host backbone consumes. The
    /4 factor comes from ViT tubelet=2 followed by the encoder-side
    avg-pool over time with stride=2 (Wan VAE parity). T_pixel=5 is the
    smallest non-trivial pool case: T_target_raw=2 -> 1 pooled target,
    exercising the pool reshape's boundary (B, D, 1, 2, h, w).
    """
    enc = _build_vjepa_encoder(embed_dim=8)
    # T_pixel == 1: cond pass only, no target stream → no pooling needed.
    v1 = torch.randn(1, 3, 1, 32, 32)
    z1 = enc.batch_encode(v1)
    assert z1.shape == (1, 8, 1, 2, 2)  # (B, D, T_lat=1, H/16, W/16)

    # T_pixel == 5: 1 cond + 2 raw target (tubelet=2 over (2 dup + 4
    # target), first slice dropped) → 1 pooled target == 2 latent frames.
    v5 = torch.randn(1, 3, 5, 32, 32)
    z5 = enc.batch_encode(v5)
    assert z5.shape == (1, 8, 2, 2, 2)

    # T_pixel == 9: 1 cond + 4 raw target (tubelet=2 over the (2 dup + 8
    # target) prepended clip, first slice dropped) → 2 pooled target
    # (avg-pool over time stride=2) == 3 latent frames total.
    v9 = torch.randn(1, 3, 9, 32, 32)
    z9 = enc.batch_encode(v9)
    assert z9.shape == (1, 8, 3, 2, 2)


def test_V5b_vjepa21_pool_target_temporal_is_mean():
    """``_pool_target_temporal`` is an arithmetic mean over consecutive
    pairs — NOT slice-keep-first ([:, :, ::2]) or slice-keep-second
    ([:, :, 1::2]). A 1, 2, 3, 4 sequence per channel must average to
    1.5, 3.5. Guards against a future "optimization" that silently
    swaps mean for stride-2 indexing — every other test in this file
    would still pass because shapes match.
    """
    enc = _build_vjepa_encoder(embed_dim=2)
    # Build a controlled target latent: B=1, D=2, T=4, h=w=1 so the
    # per-channel values are easy to eyeball.
    z_target = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 4, 1, 1).expand(1, 2, 4, 1, 1).contiguous()
    pooled = enc._pool_target_temporal(z_target)
    assert pooled.shape == (1, 2, 2, 1, 1)
    # Expected: mean(1, 2) = 1.5; mean(3, 4) = 3.5.
    assert torch.allclose(pooled[0, 0, 0, 0, 0], torch.tensor(1.5))
    assert torch.allclose(pooled[0, 0, 1, 0, 0], torch.tensor(3.5))


def test_V6_vjepa21_decode_raises():
    """Irreversible encoder: decode/to_frames raise NotImplementedError."""
    enc = _build_vjepa_encoder()
    with pytest.raises(NotImplementedError, match="irreversible"):
        enc.decode(torch.zeros(1, 8, 1, 2, 2))
    with pytest.raises(NotImplementedError, match="irreversible"):
        enc.to_frames(torch.zeros(1, 3, 1, 32, 32))


class _SpyVJEPAViT(nn.Module):
    """Like ``_MockVJEPAViT`` but records every forward-call shape so tests
    can verify which V-JEPA branch (image vs video) was hit and what the
    target-pass prepend produced.
    """

    def __init__(self, embed_dim: int = 8, patch: int = 16, tubelet: int = 2):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch = int(patch)
        self.tubelet = int(tubelet)
        self.img_temporal_dim_size = 1
        self.call_log: list[dict] = []
        self._proj = nn.Linear(1, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, T, H, W = x.shape
        self.call_log.append({"T": int(T), "H": int(H), "W": int(W)})
        h = H // self.patch
        w = W // self.patch
        if T == 1:
            L = h * w
        else:
            assert T % self.tubelet == 0
            L = (T // self.tubelet) * h * w
        seed = torch.zeros(B, L, 1, device=x.device, dtype=x.dtype)
        return self._proj(seed)


def _build_vjepa_spy_encoder(*, embed_dim: int = 8, vjepa2_1_forward: str = "video"):
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    vit = _SpyVJEPAViT(embed_dim=embed_dim)
    enc = VJEPA21VideoEncoder(vit, embed_dim=embed_dim, variant="spy", vjepa2_1_forward=vjepa2_1_forward)
    return enc, vit


def test_V6a_vjepa21_optional_yaml_keys_exposes_forward_knob():
    """``optional_yaml_keys`` returns exactly the yaml fields this encoder
    consumes beyond ``{name, model_path}``: the ``vjepa2_1_forward`` knob plus
    the optional S-VAE reducer wiring (``svae_path`` / ``svae_target_dim``).
    The base-side whitelist (``BaseWAMArchitecture._compute_encoder_yaml_whitelist``)
    reads this method, so an empty / wrong set here is what gates a typo
    being silently accepted from yaml.
    """
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    assert VJEPA21VideoEncoder.optional_yaml_keys() == {"vjepa2_1_forward", "svae_path", "svae_target_dim"}


def test_V6b_vjepa21_forward_default_is_video():
    """No-kwarg construction picks ``vjepa2_1_forward="video"`` — the
    intended default after this PR (so cond and target both come from the
    V-JEPA video branch).
    """
    enc = _build_vjepa_encoder()
    assert enc.vjepa2_1_forward == "video"


def test_V6c_vjepa21_invalid_forward_raises():
    """Constructing with an unknown ``vjepa2_1_forward`` value fails fast
    at __init__ rather than producing a confusing branch-routing error
    inside ``batch_encode``.
    """
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    with pytest.raises(ValueError, match="vjepa2_1_forward must be one of"):
        VJEPA21VideoEncoder(
            _MockVJEPAViT(embed_dim=8),
            embed_dim=8,
            variant="mock",
            vjepa2_1_forward="image",  # type: ignore[arg-type]
        )


def test_V6d_vjepa21_video_mode_cond_dups_frame_zero():
    """``vjepa2_1_forward="video"``: cond pass dups frame 0 and routes the
    2-frame clip through the video branch (T=2). Target pass prepends the
    same dup'd pair to N target frames (T=2+N). Two video-branch forwards
    total — no image-branch call. The encoder-side avg-pool is invisible
    in the call_log (it happens after the forwards complete).
    """
    enc, vit = _build_vjepa_spy_encoder(vjepa2_1_forward="video")
    # T_pixel=9 → N_target=8, so target-pass clip has T=2+8=10.
    z = enc.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert z.shape == (1, 8, 3, 2, 2)  # 1 cond + 2 pooled target = 3 latent slices
    ts = [c["T"] for c in vit.call_log]
    assert ts == [2, 10], f"expected [2, 10] for video mode, got {ts}"


def test_V6e_vjepa21_mixed_mode_cond_uses_image_branch():
    """``vjepa2_1_forward="mixed"``: cond pass routes frame 0 through the
    image branch (T=1). Target pass is unchanged — still the prepend-and-
    drop-then-avg-pool path (T=2+N). One image-branch + one video-branch
    forward.
    """
    enc, vit = _build_vjepa_spy_encoder(vjepa2_1_forward="mixed")
    z = enc.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert z.shape == (1, 8, 3, 2, 2)
    ts = [c["T"] for c in vit.call_log]
    assert ts == [1, 10], f"expected [1, 10] for mixed mode, got {ts}"


def test_V6f_vjepa21_target_pass_drops_prepended_slice():
    """Verify the target-pass output drops exactly the FIRST temporal slice
    of the video-branch forward — the one produced by the prepended
    ``[f0, f0]`` pair under tubelet=2 — AND then halves the remaining
    target latents via avg-pool stride=2.

    The shape check (``T_lat == 1 + N/4``) verifies both steps: without
    the drop we'd have ``1 + 1 + N/2 = 2 + N/2`` raw latents, then ``(2 +
    N/2) / 2`` after pool. With the drop, ``1 + N/4`` (N=8 → 1 + 2 = 3).
    """
    enc, vit = _build_vjepa_spy_encoder(vjepa2_1_forward="video")
    z = enc.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert z.shape[2] == 3  # 1 cond + 2 pooled target; both steps verified
    # Second call is the target pass with 2 prepended + 8 targets.
    assert vit.call_log[1]["T"] == 10


def test_V6g_vjepa21_t_pixel_1_honours_forward_mode():
    """TI2V single-frame fast path (T_pixel==1) routes through the cond
    pass only. The forward mode still selects the branch: ``video`` dups,
    ``mixed`` goes single-frame.
    """
    enc_v, vit_v = _build_vjepa_spy_encoder(vjepa2_1_forward="video")
    enc_m, vit_m = _build_vjepa_spy_encoder(vjepa2_1_forward="mixed")
    z_v = enc_v.batch_encode(torch.randn(1, 3, 1, 32, 32))
    z_m = enc_m.batch_encode(torch.randn(1, 3, 1, 32, 32))
    assert z_v.shape == z_m.shape == (1, 8, 1, 2, 2)
    assert [c["T"] for c in vit_v.call_log] == [2]
    assert [c["T"] for c in vit_m.call_log] == [1]


@pytest.mark.parametrize("Tp", [8, 3, 7])
def test_V6h_vjepa21_t_pixel_not_div4_minus1_rejected(Tp):
    """``(T_pixel - 1)`` must be divisible by ``2 * pool_stride = 4`` (ViT
    tubelet=2 needs an even target count; the post-tubelet avg-pool needs
    that count even too). Rejected values include:

    - ``T_pixel=8``: (8-1)=7 — odd target count → tubelet=2 already fails.
    - ``T_pixel=3``: (3-1)=2 — divisible by 2 (old check passed) but NOT
      by 4 (new check fails) → guards the new constraint.
    - ``T_pixel=7``: (7-1)=6 — same case as T_pixel=3 (passes %2, fails %4).

    Fail-fast happens before any encoder forward runs. The regex uses
    ``\\d+`` instead of hardcoded ``4`` so the assertion tracks the
    encoder's ``_TARGET_TEMPORAL_POOL_STRIDE`` constant if it's ever bumped.
    """
    enc = _build_vjepa_encoder()
    with pytest.raises(ValueError, match=r"\(T_pixel - 1\) % \d+ == 0"):
        enc.batch_encode(torch.randn(1, 3, Tp, 32, 32))


class _NaNPropagatingVJEPAViT(nn.Module):
    """Mock ViT whose per-token output is a function of the corresponding
    input region — any NaN in the input region propagates to the output
    token. Lets tests verify that the cond pass does NOT see target frames
    by poisoning the target inputs with NaN and checking the cond latent
    stays finite while target latents become NaN.

    Mirrors the (B, C, T, H, W) → (B, L, D) shape contract of the real
    V-JEPA ViT, with avg_pool3d standing in for patch_embed + tubelet.
    """

    def __init__(self, embed_dim: int = 8, patch: int = 16, tubelet: int = 2):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.patch = int(patch)
        self.tubelet = int(tubelet)
        self.img_temporal_dim_size = 1
        self._proj = nn.Linear(1, embed_dim, bias=False)
        # Identity-ish init so the projection preserves NaN propagation.
        with torch.no_grad():
            self._proj.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _B, _C, T, _H, _W = x.shape
        if T == 1:
            kernel = (1, self.patch, self.patch)
        else:
            assert T % self.tubelet == 0
            kernel = (self.tubelet, self.patch, self.patch)
        pooled = nn.functional.avg_pool3d(x, kernel_size=kernel)  # (B, C, T_lat, h, w)
        scalar = pooled.mean(dim=1, keepdim=False)  # (B, T_lat, h, w)
        flat = scalar.flatten(1).unsqueeze(-1)  # (B, T_lat*h*w, 1)
        return self._proj(flat)


def _build_vjepa_nan_propagating_encoder(*, embed_dim: int = 8, vjepa2_1_forward: str = "video"):
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    vit = _NaNPropagatingVJEPAViT(embed_dim=embed_dim)
    return VJEPA21VideoEncoder(vit, embed_dim=embed_dim, variant="nan-prop", vjepa2_1_forward=vjepa2_1_forward)


@pytest.mark.parametrize("mode", ["video", "mixed"])
def test_V6j_vjepa21_cond_does_not_leak_target_pixels(mode):
    """Stronger independence proof than the spy log: poison the target
    pixel frames with NaN, run ``batch_encode``, and verify the cond
    latent slice (output[:, :, 0:1]) is NaN-free while the target slices
    (output[:, :, 1:]) carry the poisoned signal. Tightens the contract
    the spy test only proves at the call-shape level — a NaN reaching
    the cond latent would mean some target pixel was read by the cond
    forward, which would break the deploy/train identity for the cond
    latent. NaN propagates through the avg-pool (mean of any NaN-tainted
    group is NaN), so the post-pool target slices stay NaN-tainted too.
    """
    enc = _build_vjepa_nan_propagating_encoder(vjepa2_1_forward=mode)
    video = torch.zeros(1, 3, 9, 32, 32)
    video[:, :, 1:] = float("nan")  # target frames poisoned; frame 0 still clean
    z = enc.batch_encode(video)
    assert z.shape == (1, 8, 3, 2, 2)
    assert not torch.isnan(z[:, :, 0:1]).any(), (
        f"cond latent contains NaN under vjepa2_1_forward={mode!r} — target frames are leaking into the cond pass"
    )
    # ``.all()`` is the right strength here: every target pixel frame is
    # NaN, the tubelet=2 pool groups each contain at least one NaN frame
    # (hence NaN out), the prepend-drop discards the one clean tubelet
    # group, the avg-pool stride=2 over NaN-tainted raw target latents
    # stays NaN, and LayerNorm propagates NaN through mean/var. Anything
    # weaker than ``.all()`` would let a regression where pooling reads
    # only ``[::2]`` (skipping poisoned frames) silently slip through.
    assert torch.isnan(z[:, :, 1:]).all(), (
        "every target latent slice should be NaN under fully-poisoned target "
        "inputs; a partially-finite target slice means the target pass is "
        "reading clean frames it shouldn't be (e.g. stride-2 indexing instead "
        "of mean pooling)."
    )


def test_V6i_vjepa21_from_pretrained_forwards_yaml_field(tmp_path, monkeypatch):
    """End-to-end yaml plumbing: ``build_video_encoder`` packs
    ``vjepa2_1_forward`` from cfg into ``from_pretrained(...)`` kwargs, and
    the constructed encoder reflects the chosen mode. Uses the fake-imports
    helper so no actual V-JEPA weights are needed.
    """
    import json as _json

    from openwam.model.video_backbone.encoder import build_video_encoder
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    # Wire fake vjepa2 modules; route the arch wrapper to our spy ViT.
    # ``vit_kwargs`` from ``_build_vit_from_manifest`` does NOT include
    # ``embed_dim`` (it's used by the encoder wrapper, not the upstream
    # ViT factory) — so we hardcode the spy's embed_dim to match the
    # manifest's value (8) the encoder will read.
    def _wrapper(**kwargs):
        return _SpyVJEPAViT(embed_dim=8)

    _install_fake_vjepa_modules(monkeypatch, _wrapper)
    # Patch the weight loader; the ViT is zero-weight already and the
    # spy doesn't have the matching state_dict shape, so we skip load.
    monkeypatch.setattr(VJEPA21VideoEncoder, "_load_vit_weights", lambda *a, **kw: None)

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 8,
        "variant": "mock-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "img_temporal_dim_size": 1,
        "interpolate_rope": True,
        "checkpoint_file": "fake.pt",
        "checkpoint_key": "target_encoder",
    }
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))

    enc = build_video_encoder(
        {
            "name": "vjepa2_1",
            "model_path": str(tmp_path),
            "vjepa2_1_forward": "mixed",
        }
    )
    assert isinstance(enc, VJEPA21VideoEncoder)
    assert enc.vjepa2_1_forward == "mixed"


def test_V7_vjepa21_default_dit_input_proj_shape():
    """The default ``build_dit_input_proj`` at dit_patch_size=(1,2,2) produces
    a Conv3d(z_dim, dit_dim, (1,2,2), (1,2,2)) — matches Wan VAE's DiT-side
    patch layout so the per-frame token grid lines up with the native VAE
    path. Tokens-per-frame is (H/16/2) × (W/16/2) — 4× fewer than the prior
    lossless (1,1,1) layout, in exchange for Wan VAE token-count parity.
    """
    enc = _build_vjepa_encoder(embed_dim=1408)
    conv = enc.build_dit_input_proj(dit_dim=1024)
    assert isinstance(conv, nn.Conv3d)
    assert conv.in_channels == 1408
    assert conv.out_channels == 1024
    assert tuple(conv.kernel_size) == (1, 2, 2)
    assert tuple(conv.stride) == (1, 2, 2)


@pytest.mark.parametrize(
    "patch, tubelet",
    [(14, 2), (16, 1), (8, 4)],
)
def test_V8_vjepa21_from_pretrained_rejects_manifest_geometry_mismatch(tmp_path, patch, tubelet):
    """``from_pretrained`` fails fast when ``manifest.patch`` / ``manifest.tubelet``
    differ from the (16, 2) values the spec + reshape paths are hard-wired against.

    Without this guard, a (patch=14) manifest would build a ViT with the wrong
    grid and only fail later inside ``batch_encode`` at the ``H // 16`` reshape
    with a generic shape-mismatch RuntimeError. We want the load-time error to
    name the offending fields instead.
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-rope-256",
        "patch": patch,
        "img_size": 256,
        "training_num_frames": 64,
        "tubelet": tubelet,
        "use_rope": True,
        "img_temporal_dim_size": 1,
        "interpolate_rope": True,
        "checkpoint_file": "fake.pt",
        "checkpoint_key": "target_encoder",
    }
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))
    with pytest.raises(ValueError, match="patch/tubelet must be"):
        VJEPA21VideoEncoder.from_pretrained(str(tmp_path))


def test_V9_vjepa21_load_vit_no_double_use_rope_on_rope_arch(monkeypatch):
    """``_load_vit`` must not pass ``use_rope`` to ``*_rope`` arch wrappers.

    Upstream ``vit_giant_xformers_rope`` (and its siblings) hardcode
    ``use_rope=True`` inside the wrapper and forward ``**kwargs`` to
    ``VisionTransformer`` — handing them a second ``use_rope=...`` from the
    OpenWAM call site raises ``TypeError: got multiple values for keyword
    argument 'use_rope'`` at train start. Regression guard for that exact
    crash, exercised against the canonical manifest the production checkpoint
    ships with.
    """
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    captured_kwargs: dict = {}

    class _StopAfterConstruct(Exception):
        pass

    def _fake_wrapper(**kwargs):
        if "use_rope" in kwargs:
            raise TypeError("got multiple values for keyword argument 'use_rope'")
        captured_kwargs.update(kwargs)
        raise _StopAfterConstruct()

    fake_module = types.SimpleNamespace(
        __dict__={"vit_giant_xformers_rope": _fake_wrapper},
        vit_giant_xformers_rope=_fake_wrapper,
    )
    fake_vjepa_modules = types.SimpleNamespace(
        rotate_queries_or_keys=lambda x, pos, n_registers, has_cls_first: x,
    )
    fake_app = types.ModuleType("app")
    fake_app_vjepa = types.ModuleType("app.vjepa_2_1")
    fake_app_vjepa_models = types.ModuleType("app.vjepa_2_1.models")
    fake_app_vjepa_models.vision_transformer = fake_module
    fake_app_vjepa_models_utils = types.ModuleType("app.vjepa_2_1.models.utils")
    fake_app_vjepa_models_utils.modules = fake_vjepa_modules
    monkeypatch.setitem(sys.modules, "app", fake_app)
    monkeypatch.setitem(sys.modules, "app.vjepa_2_1", fake_app_vjepa)
    monkeypatch.setitem(sys.modules, "app.vjepa_2_1.models", fake_app_vjepa_models)
    monkeypatch.setitem(sys.modules, "app.vjepa_2_1.models.vision_transformer", fake_module)
    monkeypatch.setitem(sys.modules, "app.vjepa_2_1.models.utils", fake_app_vjepa_models_utils)
    monkeypatch.setitem(sys.modules, "app.vjepa_2_1.models.utils.modules", fake_vjepa_modules)

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-rope-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "img_temporal_dim_size": 1,
        "interpolate_rope": True,
        "checkpoint_file": "fake.pt",
        "checkpoint_key": "target_encoder",
    }
    with pytest.raises(_StopAfterConstruct):
        VJEPA21VideoEncoder._load_vit("/unused", manifest)
    assert "use_rope" not in captured_kwargs
    assert captured_kwargs["patch_size"] == 16
    assert captured_kwargs["interpolate_rope"] is True


def test_V10_vjepa21_load_vit_rope_arch_with_use_rope_false_fails_fast():
    """Manifest with ``arch_name=*_rope`` and ``use_rope=False`` is contradictory —
    we raise a ``ValueError`` at load time instead of silently overriding."""
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-rope-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": False,
        "img_temporal_dim_size": 1,
        "interpolate_rope": True,
        "checkpoint_file": "fake.pt",
        "checkpoint_key": "target_encoder",
    }
    with pytest.raises(ValueError, match="hardcodes use_rope=True"):
        VJEPA21VideoEncoder._load_vit("/unused", manifest)


# ----------------------------------------------------------------------
# V11-V14: VJEPA21VideoEncoder.from_skeleton (deploy path)
# ----------------------------------------------------------------------


def _install_fake_vjepa_modules(monkeypatch, wrapper_factory):
    """Wire fake ``app.vjepa_2_1.*`` modules so VJEPA imports resolve to test
    fixtures. ``wrapper_factory`` is a callable used for every arch lookup —
    the test decides what to inspect / what to return.
    """

    class _Module:
        def __init__(self, **attrs):
            self.__dict__.update(attrs)

    vision_transformer = _Module()
    # The encoder code does ``vit_encoder.__dict__[arch_name](**kwargs)``.
    # Make every arch name route to ``wrapper_factory``.
    for arch in (
        "vit_giant_xformers",
        "vit_giant_xformers_rope",
    ):
        setattr(vision_transformer, arch, wrapper_factory)
    fake_vjepa_modules = types.SimpleNamespace(
        rotate_queries_or_keys=lambda x, pos, n_registers, has_cls_first: x,
    )
    fake_app = types.ModuleType("app")
    fake_app_vjepa = types.ModuleType("app.vjepa_2_1")
    fake_app_vjepa_models = types.ModuleType("app.vjepa_2_1.models")
    fake_app_vjepa_models.vision_transformer = vision_transformer
    fake_app_vjepa_models_utils = types.ModuleType("app.vjepa_2_1.models.utils")
    fake_app_vjepa_models_utils.modules = fake_vjepa_modules
    monkeypatch.setitem(sys.modules, "app", fake_app)
    monkeypatch.setitem(sys.modules, "app.vjepa_2_1", fake_app_vjepa)
    monkeypatch.setitem(sys.modules, "app.vjepa_2_1.models", fake_app_vjepa_models)
    monkeypatch.setitem(sys.modules, "app.vjepa_2_1.models.vision_transformer", vision_transformer)
    monkeypatch.setitem(sys.modules, "app.vjepa_2_1.models.utils", fake_app_vjepa_models_utils)
    monkeypatch.setitem(sys.modules, "app.vjepa_2_1.models.utils.modules", fake_vjepa_modules)


def test_V11_vjepa21_from_skeleton_happy_path(tmp_path, monkeypatch):
    """``from_skeleton`` reads manifest from encoder_cfg.model_path, builds a
    zero-weight ViT shell, and skips torch.load entirely — even though
    ``manifest['checkpoint_file']`` would point at a non-existent file.
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-rope-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "img_temporal_dim_size": 1,
        "interpolate_rope": True,
        "checkpoint_file": "DOES-NOT-EXIST.pt",
        "checkpoint_key": "target_encoder",
    }
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))

    captured: dict = {}

    def _fake_wrapper(**kwargs):
        captured.update(kwargs)
        if "use_rope" in kwargs:
            raise TypeError("got multiple values for keyword argument 'use_rope'")
        return _MockVJEPAViT(embed_dim=1408)

    _install_fake_vjepa_modules(monkeypatch, _fake_wrapper)

    enc = VJEPA21VideoEncoder.from_skeleton(
        components_entry={"attr": "vae", "model_class": "ignored", "extra_kwargs": {}},
        encoder_cfg={"name": "vjepa2_1", "model_path": str(tmp_path)},
    )
    assert isinstance(enc, VJEPA21VideoEncoder)
    assert enc.spec.z_dim == 1408
    assert enc.variant == "vitg-rope-384"
    # _rope wrapper must NOT receive use_rope (PR #83 invariant)
    assert "use_rope" not in captured
    assert captured["patch_size"] == 16
    assert captured["img_size"] == (384, 384)
    assert captured["tubelet_size"] == 2


def test_V11b_vjepa21_from_skeleton_propagates_vjepa2_1_forward(tmp_path, monkeypatch, caplog):
    """Deploy-side knob plumbing — positive path. Pair to ``test_V11`` which
    omits the field (default branch) and ``test_W9`` / ``test_W9b`` which
    cover the rejection side on V-JEPA 2:

    - ``encoder_cfg={"vjepa2_1_forward": "mixed", ...}`` → the built
      encoder reports ``vjepa2_1_forward == "mixed"`` and the migration
      warning is silent (the field is present, so this is NOT a pre-PR
      checkpoint).
    - ``encoder_cfg`` without the field → resolved mode is the default
      AND the migration warning fires once, so an operator who deploys
      a pre-PR checkpoint without hand-adding ``vjepa2_1_forward: mixed``
      sees a noisy signal instead of a silently-divergent cond latent.
    """
    import json as _json
    import logging

    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-rope-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "img_temporal_dim_size": 1,
        "interpolate_rope": True,
        "checkpoint_file": "DOES-NOT-EXIST.pt",
        "checkpoint_key": "target_encoder",
    }
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))

    def _fake_wrapper(**kwargs):
        return _MockVJEPAViT(embed_dim=1408)

    _install_fake_vjepa_modules(monkeypatch, _fake_wrapper)

    # --- Path A: field explicit → no warning ---
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="openwam.model.video_backbone.encoder.vjepa2_1"):
        enc_mixed = VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "ignored", "extra_kwargs": {}},
            encoder_cfg={"name": "vjepa2_1", "model_path": str(tmp_path), "vjepa2_1_forward": "mixed"},
        )
    assert enc_mixed.vjepa2_1_forward == "mixed"
    # Render via ``getMessage()`` (not ``r.message``) so the assertion compares
    # against the formatted log line — robust to %-substitutions and parity
    # with ``test_W17``'s path B style.
    assert not any("vjepa2_1_forward" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records), (
        "explicit vjepa2_1_forward must NOT trip the pre-PR-checkpoint warning"
    )

    # --- Path B: field absent → default + warning ---
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="openwam.model.video_backbone.encoder.vjepa2_1"):
        enc_default = VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "ignored", "extra_kwargs": {}},
            encoder_cfg={"name": "vjepa2_1", "model_path": str(tmp_path)},
        )
    assert enc_default.vjepa2_1_forward == "video"  # current _VJEPA21_FORWARD_DEFAULT
    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("vjepa2_1_forward" in m and "mixed" in m for m in warning_msgs), (
        f"expected migration warning naming the field and the legacy ``mixed`` value; got: {warning_msgs}"
    )


def test_V12_vjepa21_from_skeleton_requires_some_manifest_source():
    """``from_skeleton`` without ``encoder_cfg`` AND without ``ckpt_dir`` raises
    a single FileNotFoundError naming both attempted paths (None / None).
    components_entry alone doesn't carry ViT geometry (it's Wan VAE class).
    """
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    with pytest.raises(FileNotFoundError, match=r"ckpt_dir.*encoder\.model_path"):
        VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
        )


def test_V13_vjepa21_from_skeleton_requires_model_path(tmp_path):
    """``encoder_cfg`` without ``model_path`` (or with empty string) fails fast
    via the same dual-path FileNotFoundError so the operator sees we tried
    both sources before giving up.
    """
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    with pytest.raises(FileNotFoundError, match=r"ckpt_dir.*encoder\.model_path"):
        VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
            encoder_cfg={"name": "vjepa2_1"},
        )
    with pytest.raises(FileNotFoundError, match=r"ckpt_dir.*encoder\.model_path"):
        VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
            encoder_cfg={"name": "vjepa2_1", "model_path": ""},
        )


def test_V14_vjepa21_from_skeleton_missing_manifest(tmp_path):
    """``encoder_cfg.model_path`` that has no ``manifest.json`` → FileNotFoundError."""
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
            encoder_cfg={"name": "vjepa2_1", "model_path": str(tmp_path)},
        )


# ----------------------------------------------------------------------
# V15-V19: deploy self-containment — ckpt_dir manifest takes priority
# ----------------------------------------------------------------------


def _build_vjepa_manifest_payload() -> dict:
    """Manifest dict matching what V-JEPA 2.1 vit_giant_xformers_rope writes."""
    return {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-rope-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "img_temporal_dim_size": 1,
        "interpolate_rope": True,
        "checkpoint_file": "DOES-NOT-EXIST.pt",
        "checkpoint_key": "target_encoder",
    }


def test_V15_vjepa21_from_skeleton_prefers_ckpt_dir_manifest(tmp_path, monkeypatch):
    """When ``<ckpt_dir>/manifest.json`` exists, ``from_skeleton`` reads it
    and does NOT touch ``encoder_cfg.model_path`` — proving deploy is self-
    contained on machines where the training-time encoder path is unmounted.
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "manifest.json").write_text(_json.dumps(_build_vjepa_manifest_payload()))

    def _fake_wrapper(**kwargs):
        return _MockVJEPAViT(embed_dim=1408)

    _install_fake_vjepa_modules(monkeypatch, _fake_wrapper)

    # Deliberately point encoder_cfg.model_path at a NON-EXISTENT directory.
    # If from_skeleton's priority order is wrong it'll try this path and
    # raise FileNotFoundError; the test verifies it never gets there.
    enc = VJEPA21VideoEncoder.from_skeleton(
        components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
        encoder_cfg={"name": "vjepa2_1", "model_path": "/nonexistent/unmounted/path"},
        ckpt_dir=str(ckpt_dir),
    )
    assert isinstance(enc, VJEPA21VideoEncoder)
    assert enc.spec.z_dim == 1408
    assert enc.variant == "vitg-rope-384"


def test_V16_vjepa21_from_skeleton_falls_back_to_encoder_cfg_when_ckpt_dir_lacks_manifest(tmp_path, monkeypatch):
    """Old checkpoints saved before self-containment have no
    ``<ckpt_dir>/manifest.json`` — ``from_skeleton`` must fall back to the
    yaml's ``encoder.model_path`` so those checkpoints keep deploying.
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()  # no manifest.json here
    encoder_src = tmp_path / "vjepa-weights"
    encoder_src.mkdir()
    (encoder_src / "manifest.json").write_text(_json.dumps(_build_vjepa_manifest_payload()))

    def _fake_wrapper(**kwargs):
        return _MockVJEPAViT(embed_dim=1408)

    _install_fake_vjepa_modules(monkeypatch, _fake_wrapper)

    enc = VJEPA21VideoEncoder.from_skeleton(
        components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
        encoder_cfg={"name": "vjepa2_1", "model_path": str(encoder_src)},
        ckpt_dir=str(ckpt_dir),
    )
    assert isinstance(enc, VJEPA21VideoEncoder)
    assert enc.spec.z_dim == 1408


def test_V17_vjepa21_from_skeleton_no_manifest_anywhere(tmp_path):
    """Neither ``<ckpt_dir>/manifest.json`` nor ``encoder.model_path`` works:
    fail with a single error that names BOTH paths verbatim so the operator
    can see both locations without reading the source.
    """
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()  # empty
    encoder_src = tmp_path / "vjepa-weights"
    encoder_src.mkdir()  # empty too

    with pytest.raises(FileNotFoundError) as exc:
        VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
            encoder_cfg={"name": "vjepa2_1", "model_path": str(encoder_src)},
            ckpt_dir=str(ckpt_dir),
        )
    # Lock the operator-facing UX: both attempted paths appear in the message.
    msg = str(exc.value)
    assert str(ckpt_dir) in msg, f"ckpt_dir missing from error: {msg}"
    assert str(encoder_src) in msg, f"encoder.model_path missing from error: {msg}"


def test_V18_vjepa21_copy_deploy_artifacts_copies_manifest(tmp_path, monkeypatch):
    """Training-side hook copies ``<encoder.model_path>/manifest.json`` into
    ``<output_dir>/manifest.json``. New checkpoints saved after this change
    are self-contained.
    """
    import json as _json

    from omegaconf import OmegaConf

    encoder_src = tmp_path / "vjepa-weights"
    encoder_src.mkdir()
    manifest_payload = _build_vjepa_manifest_payload()
    (encoder_src / "manifest.json").write_text(_json.dumps(manifest_payload))

    output_dir = tmp_path / "ckpt-out"
    output_dir.mkdir()

    # Build an encoder instance without going through from_pretrained
    # (the test doesn't need real ViT weights — we only exercise the
    # copy hook, which is a method on the encoder *instance*).
    enc = _build_vjepa_encoder(embed_dim=1408)
    cfg = OmegaConf.create(
        {"model": {"video_backbone": {"encoder": {"name": "vjepa2_1", "model_path": str(encoder_src)}}}}
    )
    enc.copy_deploy_artifacts(str(output_dir), cfg)

    dst = output_dir / "manifest.json"
    assert dst.exists()
    assert _json.loads(dst.read_text()) == manifest_payload


def test_V19_vjepa21_copy_deploy_artifacts_missing_cfg_is_warning_not_raise(tmp_path, caplog):
    """The hook must NEVER raise on missing source — a copy failure must
    not crash an otherwise-good training run. Missing cfg / missing source
    file log a warning and return; deploy then falls back to
    ``encoder.model_path``.
    """
    import logging

    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder  # noqa: F401

    enc = _build_vjepa_encoder(embed_dim=1408)

    output_dir = tmp_path / "ckpt-out"
    output_dir.mkdir()

    # cfg without model.video_backbone.encoder → warning + no-op.
    with caplog.at_level(logging.WARNING):
        enc.copy_deploy_artifacts(str(output_dir), cfg={})
    assert not (output_dir / "manifest.json").exists()
    assert any("model_path" in r.message for r in caplog.records)

    caplog.clear()
    # cfg points at a directory with no manifest.json → warning + no-op.
    from omegaconf import OmegaConf

    empty_src = tmp_path / "empty"
    empty_src.mkdir()
    cfg = OmegaConf.create(
        {"model": {"video_backbone": {"encoder": {"name": "vjepa2_1", "model_path": str(empty_src)}}}}
    )
    with caplog.at_level(logging.WARNING):
        enc.copy_deploy_artifacts(str(output_dir), cfg)
    assert not (output_dir / "manifest.json").exists()
    assert any("manifest.json" in r.message for r in caplog.records)


def test_V20_vjepa21_copy_deploy_artifacts_io_error_does_not_crash(tmp_path, caplog, monkeypatch):
    """``copy_deploy_artifacts`` must NEVER raise on IO failure either —
    permission denied / disk full / disappearing mount must collapse to a
    warning + return so the trainer's safetensors save isn't lost.

    Reproduces wayrise #1: read-only fs / PermissionError / ENOSPC.
    """
    import json as _json
    import logging
    import shutil

    from omegaconf import OmegaConf

    encoder_src = tmp_path / "vjepa-weights"
    encoder_src.mkdir()
    (encoder_src / "manifest.json").write_text(_json.dumps(_build_vjepa_manifest_payload()))
    output_dir = tmp_path / "ckpt-out"
    output_dir.mkdir()
    enc = _build_vjepa_encoder(embed_dim=1408)
    cfg = OmegaConf.create(
        {"model": {"video_backbone": {"encoder": {"name": "vjepa2_1", "model_path": str(encoder_src)}}}}
    )

    def _boom(*args, **kwargs):
        raise PermissionError("simulated read-only filesystem")

    monkeypatch.setattr(shutil, "copyfile", _boom)
    with caplog.at_level(logging.WARNING):
        # Must NOT raise — assertion is "we got here".
        enc.copy_deploy_artifacts(str(output_dir), cfg)
    assert not (output_dir / "manifest.json").exists()
    formatted = [r.getMessage() for r in caplog.records]
    assert any("failed" in m and "simulated" in m for m in formatted), (
        f"expected warning naming the copy failure; got: {formatted}"
    )


def test_V21_vjepa21_feature_norm_keys_present_in_state_dict():
    """``self.feature_norm`` must live directly on the encoder (NOT inside
    ``self._m``) so the freeze yaml's ``video_backbone._encoder`` line
    recursively covers it AND the safetensors carries it under
    ``video_backbone._encoder.feature_norm.*``. If a future refactor
    moves the LN into ``self._m``, deploy round-trip would still pass
    (state_dict key sets remain consistent) but the freeze granularity
    would silently change. Pinning the location here surfaces that as a
    test break."""
    enc = _build_vjepa_encoder(embed_dim=8)
    keys = set(enc.state_dict().keys())
    assert "feature_norm.weight" in keys, (
        "feature_norm.weight is missing from V-JEPA encoder state_dict. "
        "It must live on the encoder (``self.feature_norm``), not inside ``self._m``."
    )
    assert "feature_norm.bias" in keys, "feature_norm.bias is missing from V-JEPA encoder state_dict."


# ======================================================================
# W1-W12: V-JEPA 2 encoder (vjepa2). The upstream ViT (``src.models``) has
# no image branch — every forward routes through the tubelet=2 video
# branch. No ``vjepa2_1_forward`` knob.
# ======================================================================


def _build_vjepa2_encoder(embed_dim: int = 8):
    """Construct a ``VJEPA2VideoEncoder`` around the shared mock ViT."""
    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder

    vit = _MockVJEPAViT(embed_dim=embed_dim)
    return VJEPA2VideoEncoder(vit, embed_dim=embed_dim, variant="mock-v2")


def _build_vjepa2_spy_encoder(*, embed_dim: int = 8):
    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder

    vit = _SpyVJEPAViT(embed_dim=embed_dim)
    return VJEPA2VideoEncoder(vit, embed_dim=embed_dim, variant="spy-v2"), vit


def _install_fake_vjepa2_modules(monkeypatch, wrapper_factory):
    """Wire fake ``src.models.*`` modules so V-JEPA 2 imports resolve to test
    fixtures. Counterpart to ``_install_fake_vjepa_modules`` (which targets
    ``app.vjepa_2_1.models.*`` for V-JEPA 2.1).
    """

    class _Module:
        def __init__(self, **attrs):
            self.__dict__.update(attrs)

    vision_transformer = _Module()
    for arch in ("vit_giant_xformers", "vit_giant_xformers_rope"):
        setattr(vision_transformer, arch, wrapper_factory)
    # Match V-JEPA 2 upstream signature: ``rotate_queries_or_keys(x, pos)``.
    # V-JEPA 2.1 added ``n_registers`` / ``has_cls_first`` (see
    # ``third_party/vjepa2/app/vjepa_2_1/models/utils/modules.py``); the
    # V-JEPA 2 fake must mirror its own upstream to avoid suggesting the
    # signatures are interchangeable.
    fake_vjepa_modules = types.SimpleNamespace(
        rotate_queries_or_keys=lambda x, pos: x,
    )
    fake_src = types.ModuleType("src")
    fake_src_models = types.ModuleType("src.models")
    fake_src_models.vision_transformer = vision_transformer
    fake_src_models_utils = types.ModuleType("src.models.utils")
    fake_src_models_utils.modules = fake_vjepa_modules
    monkeypatch.setitem(sys.modules, "src", fake_src)
    monkeypatch.setitem(sys.modules, "src.models", fake_src_models)
    monkeypatch.setitem(sys.modules, "src.models.vision_transformer", vision_transformer)
    monkeypatch.setitem(sys.modules, "src.models.utils", fake_src_models_utils)
    monkeypatch.setitem(sys.modules, "src.models.utils.modules", fake_vjepa_modules)


def test_W1_vjepa2_registration_round_trip():
    """``register_video_encoder("vjepa2")`` exposes the class via the registry."""
    from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY
    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder  # noqa: F401

    assert "vjepa2" in _VIDEO_ENCODER_REGISTRY
    assert _VIDEO_ENCODER_REGISTRY["vjepa2"] is VJEPA2VideoEncoder


def test_W2_vjepa2_optional_yaml_keys_is_empty():
    """V-JEPA 2 exposes no yaml knobs beyond ``{name, model_path}``.

    The ``vjepa2_1_forward`` field is V-JEPA 2.1-only; the V-JEPA 2 ViT
    has no image branch, so the field has no meaning here. Returning an
    empty set means the base-side whitelist rejects ``vjepa2_1_forward``
    on a vjepa2 encoder block — that's the contract that lets a yaml
    typo (carrying the 2.1 knob over) fail fast.
    """
    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder

    assert VJEPA2VideoEncoder.optional_yaml_keys() == set()


def test_W3_vjepa2_spec_invariants():
    """Spec is identical to V-JEPA 2.1: irreversible, causal, (1,2,2) DiT
    patch, temporal_compression=4, z_dim from manifest. Locks the contract
    so a future refactor of either encoder cannot silently drift them apart.
    """
    enc = _build_vjepa2_encoder(embed_dim=1408)
    spec = enc.spec
    assert spec.is_reversible is False
    assert spec.causal_temporal is True
    assert spec.dit_patch_size == (1, 2, 2)
    assert spec.z_dim == 1408
    assert spec.spatial_compression == 16
    assert spec.temporal_compression == 4
    assert spec.pixel_range == (-1.0, 1.0)


def test_W4_vjepa2_batch_encode_t_lat_shapes():
    """batch_encode T_pixel-to-T_lat dispatch matches V-JEPA 2.1's contract:
    T_pixel == 1 -> T_lat == 1; T_pixel == 5 -> T_lat == 2; T_pixel == 9
    -> T_lat == 3. Same downstream shapes so the host backbone consumes
    either encoder interchangeably.
    """
    enc = _build_vjepa2_encoder(embed_dim=8)
    v1 = torch.randn(1, 3, 1, 32, 32)
    z1 = enc.batch_encode(v1)
    assert z1.shape == (1, 8, 1, 2, 2)

    # T_pixel=5 — smallest non-trivial pool case (T_target_raw=2 → 1).
    v5 = torch.randn(1, 3, 5, 32, 32)
    z5 = enc.batch_encode(v5)
    assert z5.shape == (1, 8, 2, 2, 2)

    v9 = torch.randn(1, 3, 9, 32, 32)
    z9 = enc.batch_encode(v9)
    assert z9.shape == (1, 8, 3, 2, 2)


def test_W4b_vjepa2_pool_target_temporal_is_mean():
    """V-JEPA 2 mirror of test_V5b: ``_pool_target_temporal`` must be an
    actual mean, not stride-2 indexing. Guards against silent regression.
    """
    enc = _build_vjepa2_encoder(embed_dim=2)
    z_target = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 4, 1, 1).expand(1, 2, 4, 1, 1).contiguous()
    pooled = enc._pool_target_temporal(z_target)
    assert pooled.shape == (1, 2, 2, 1, 1)
    assert torch.allclose(pooled[0, 0, 0, 0, 0], torch.tensor(1.5))
    assert torch.allclose(pooled[0, 0, 1, 0, 0], torch.tensor(3.5))


def test_W5_vjepa2_routes_every_forward_through_video_branch():
    """``vjepa2`` always uses the dup+video forward for the cond pass (no
    image branch exists upstream), so the spy log shows T=2 for the cond
    and T=2+N for the target. Never T=1 — the image-branch fast path
    that V-JEPA 2.1 supports is not available here.
    """
    enc, vit = _build_vjepa2_spy_encoder()
    # T_pixel=9 → cond pass T=2, target pass T=2+8=10.
    z = enc.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert z.shape == (1, 8, 3, 2, 2)  # 1 cond + 2 pooled target
    ts = [c["T"] for c in vit.call_log]
    assert ts == [2, 10], f"vjepa2 should never call image branch (T=1); got {ts}"

    vit.call_log.clear()
    # Single-frame fast path: dup → T=2 video forward, still no T=1.
    z1 = enc.batch_encode(torch.randn(1, 3, 1, 32, 32))
    assert z1.shape == (1, 8, 1, 2, 2)
    ts1 = [c["T"] for c in vit.call_log]
    assert ts1 == [2], f"single-frame should dup to T=2; got {ts1}"


def test_W6_vjepa2_target_pass_drops_prepended_slice():
    """Verify the target-pass drops exactly the first temporal slice AND
    halves the remaining latents via avg-pool stride=2. Shape check:
    T_lat == 1 + N/4 (without the drop+pool we'd have 2 + N/2 raw or
    1 + N/2 drop-only).
    """
    enc, vit = _build_vjepa2_spy_encoder()
    z = enc.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert z.shape[2] == 3  # 1 cond + 2 pooled target; drop+pool verified
    assert vit.call_log[1]["T"] == 10


@pytest.mark.parametrize("Tp", [8, 3, 7])
def test_W7_vjepa2_t_pixel_not_div4_minus1_rejected(Tp):
    """``(T_pixel - 1) % 4 == 0`` is needed (ViT tubelet=2 + post-tubelet
    avg-pool stride=2). T_pixel ∈ {8, 3, 7} all fail; fail-fast happens
    before any encoder forward runs. T_pixel=3 / T_pixel=7 specifically
    guard against a regression that only checks %2 (they pass the old
    constraint but fail the new one). Loose ``\\d+`` regex tracks the
    encoder's pool-stride constant.
    """
    enc = _build_vjepa2_encoder()
    with pytest.raises(ValueError, match=r"\(T_pixel - 1\) % \d+ == 0"):
        enc.batch_encode(torch.randn(1, 3, Tp, 32, 32))


def test_W8_vjepa2_from_pretrained_rejects_vjepa2_1_forward(tmp_path):
    """``vjepa2_1_forward`` is V-JEPA 2.1-only; passing it to
    ``vjepa2.from_pretrained`` must raise rather than silently
    ignore (which would let a yaml typo go unflagged).
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "checkpoint_file": "fake.pt",
        "checkpoint_key": "target_encoder",
    }
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))
    with pytest.raises(ValueError, match="vjepa2_1_forward is V-JEPA 2.1-only"):
        VJEPA2VideoEncoder.from_pretrained(str(tmp_path), vjepa2_1_forward="video")


def test_W9_vjepa2_from_skeleton_rejects_vjepa2_1_forward(tmp_path, monkeypatch):
    """Same rejection at deploy time: ``from_skeleton`` reads encoder_cfg;
    if ``vjepa2_1_forward`` is present in the saved yaml encoder block,
    raise rather than silently ignore.
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "checkpoint_file": "fake.pt",
        "checkpoint_key": "target_encoder",
    }
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))
    encoder_cfg = {"name": "vjepa2", "model_path": str(tmp_path), "vjepa2_1_forward": "mixed"}
    with pytest.raises(ValueError, match="vjepa2_1_forward is V-JEPA 2.1-only"):
        VJEPA2VideoEncoder.from_skeleton(
            {"attr": "vae", "model_class": "X", "extra_kwargs": {}},
            ckpt_dir=str(tmp_path),
            encoder_cfg=encoder_cfg,
        )


def test_W9b_vjepa2_from_skeleton_rejects_vjepa2_1_forward_null(tmp_path):
    """Tighter version of W9: ``vjepa2_1_forward: null`` (yaml null) on a
    vjepa2 encoder block must also raise — silently accepting null
    diverges from the training-side semantic where the kwarg being
    present at all is the rejection trigger. Same operator-mistake
    pattern as W9, just with the yaml-null spelling instead of a value.
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "checkpoint_file": "fake.pt",
        "checkpoint_key": "target_encoder",
    }
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))
    encoder_cfg = {"name": "vjepa2", "model_path": str(tmp_path), "vjepa2_1_forward": None}
    with pytest.raises(ValueError, match="vjepa2_1_forward is V-JEPA 2.1-only"):
        VJEPA2VideoEncoder.from_skeleton(
            {"attr": "vae", "model_class": "X", "extra_kwargs": {}},
            ckpt_dir=str(tmp_path),
            encoder_cfg=encoder_cfg,
        )


def test_W10_vjepa2_decode_raises():
    """Irreversible encoder: decode/to_frames raise NotImplementedError."""
    enc = _build_vjepa2_encoder()
    with pytest.raises(NotImplementedError, match="irreversible"):
        enc.decode(torch.zeros(1, 8, 1, 2, 2))
    with pytest.raises(NotImplementedError, match="irreversible"):
        enc.to_frames(torch.zeros(1, 3, 1, 32, 32))


def test_W11_vjepa2_default_dit_input_proj_shape():
    """V-JEPA 2 uses ``dit_patch_size=(1,2,2)`` (Wan VAE parity) — the
    default ``build_dit_input_proj`` produces Conv3d(z_dim, dit_dim,
    (1,2,2), (1,2,2)). Mirrors V-JEPA 2.1 (test_V7).
    """
    enc = _build_vjepa2_encoder(embed_dim=1408)
    conv = enc.build_dit_input_proj(dit_dim=1024)
    assert isinstance(conv, nn.Conv3d)
    assert conv.in_channels == 1408
    assert conv.out_channels == 1024
    assert tuple(conv.kernel_size) == (1, 2, 2)
    assert tuple(conv.stride) == (1, 2, 2)


def test_W12_vjepa2_from_pretrained_end_to_end(tmp_path, monkeypatch):
    """End-to-end yaml plumbing: ``build_video_encoder({"name": "vjepa2",
    "model_path": ...})`` constructs the encoder via ``from_pretrained``,
    routing through the V-JEPA 2 src.models imports.
    """
    import json as _json

    from openwam.model.video_backbone.encoder import build_video_encoder
    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder

    def _wrapper(**kwargs):
        # vit_kwargs do not include embed_dim — hardcode to match manifest.
        return _SpyVJEPAViT(embed_dim=8)

    _install_fake_vjepa2_modules(monkeypatch, _wrapper)
    monkeypatch.setattr(VJEPA2VideoEncoder, "_load_vit_weights", lambda *a, **kw: None)

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 8,
        "variant": "mock-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "checkpoint_file": "fake.pt",
        "checkpoint_key": "target_encoder",
    }
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))
    enc = build_video_encoder({"name": "vjepa2", "model_path": str(tmp_path)})
    assert isinstance(enc, VJEPA2VideoEncoder)
    # Sanity: a forward goes through the dup+video path (T=2).
    z = enc.batch_encode(torch.randn(1, 3, 1, 32, 32))
    assert z.shape == (1, 8, 1, 2, 2)


def test_W12b_vjepa2_cond_does_not_leak_target_pixels():
    """V-JEPA 2 mirror of test_V6j: poison target inputs with NaN and verify
    cond latent stays finite (target frames must not leak into the cond
    pass). NaN propagates through the avg-pool, so the post-pool target
    slices stay NaN-tainted.
    """
    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder

    vit = _NaNPropagatingVJEPAViT(embed_dim=8)
    enc = VJEPA2VideoEncoder(vit, embed_dim=8, variant="nan-prop-v2")
    video = torch.zeros(1, 3, 9, 32, 32)
    video[:, :, 1:] = float("nan")
    z = enc.batch_encode(video)
    assert z.shape == (1, 8, 3, 2, 2)
    assert not torch.isnan(z[:, :, 0:1]).any(), (
        "vjepa2 cond latent contains NaN — target frames are leaking into the cond pass"
    )
    # ``.all()`` mirrors test_V6j — see that test for the propagation
    # argument; the V-JEPA 2 path runs the exact same prepend-drop + pool
    # pipeline so the same all-NaN-target invariant holds here.
    assert torch.isnan(z[:, :, 1:]).all(), (
        "every vjepa2 target latent slice should be NaN under fully-poisoned "
        "target inputs; partial finiteness implies the pool reads clean "
        "frames it shouldn't be."
    )


@pytest.mark.parametrize("field", ["img_temporal_dim_size", "interpolate_rope"])
def test_W12c_vjepa2_rejects_vjepa2_1_manifest_fields(tmp_path, field):
    """V-JEPA 2 manifest validation rejects V-JEPA 2.1-only fields
    (``img_temporal_dim_size`` / ``interpolate_rope``). These come from
    the V-JEPA 2.1 ViT wrapper (image branch + RoPE interpolation) and
    have no analog in the upstream V-JEPA 2 ViT. Without this guard, a
    copy-pasted V-JEPA 2.1 manifest in a vjepa2 weight dir would slip
    past and surface later as an opaque state_dict mismatch.
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "checkpoint_file": "fake.pt",
        "checkpoint_key": "target_encoder",
        field: 1 if field == "img_temporal_dim_size" else True,
    }
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))
    with pytest.raises(ValueError, match="V-JEPA 2.1-only fields"):
        VJEPA2VideoEncoder.from_pretrained(str(tmp_path))


def test_W13_vjepa2_manifest_patch_tubelet_mismatch_rejected(tmp_path):
    """Manifest with patch/tubelet != (16, 2) is rejected at load time —
    the reshape paths and spec block are hard-wired against those values.
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder

    manifest = {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-384",
        "patch": 14,  # wrong
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "checkpoint_file": "fake.pt",
        "checkpoint_key": "target_encoder",
    }
    (tmp_path / "manifest.json").write_text(_json.dumps(manifest))
    with pytest.raises(ValueError, match="patch/tubelet must be"):
        VJEPA2VideoEncoder.from_pretrained(str(tmp_path))


def test_W14_vjepa2_feature_norm_keys_present_in_state_dict():
    """Mirror of test_V21 for V-JEPA 2: ``self.feature_norm`` must live
    directly on the encoder (NOT inside ``self._m``) so the freeze yaml's
    ``video_backbone._encoder`` line recursively covers it AND the
    safetensors carries it under ``video_backbone._encoder.feature_norm.*``.
    Without an explicit pin here, a future refactor moving the LN into
    ``self._m`` would still round-trip fine but silently change the freeze
    granularity for V-JEPA 2 (deploy round-trip is invariant, but a
    user's freeze yaml that targets ``_encoder.feature_norm`` no longer
    matches). The V-JEPA 2.1 counterpart is ``test_V21``."""
    enc = _build_vjepa2_encoder(embed_dim=8)
    keys = set(enc.state_dict().keys())
    assert "feature_norm.weight" in keys, (
        "feature_norm.weight is missing from V-JEPA 2 encoder state_dict. "
        "It must live on the encoder (``self.feature_norm``), not inside ``self._m``."
    )
    assert "feature_norm.bias" in keys, "feature_norm.bias is missing from V-JEPA 2 encoder state_dict."


def _build_vjepa2_manifest_payload() -> dict:
    """V-JEPA 2 manifest dict (no V-JEPA 2.1-only fields)."""
    return {
        "arch_name": "vit_giant_xformers_rope",
        "embed_dim": 1408,
        "variant": "vitg-384",
        "patch": 16,
        "img_size": 384,
        "training_num_frames": 64,
        "tubelet": 2,
        "use_rope": True,
        "checkpoint_file": "DOES-NOT-EXIST.pt",
        "checkpoint_key": "target_encoder",
    }


def test_W15_vjepa2_copy_deploy_artifacts_copies_manifest(tmp_path):
    """V-JEPA 2 mirror of test_V18: training-side hook copies
    ``<encoder.model_path>/manifest.json`` into ``<output_dir>/manifest.json``
    so new checkpoints are self-contained on deploy."""
    import json as _json

    from omegaconf import OmegaConf

    encoder_src = tmp_path / "vjepa2-weights"
    encoder_src.mkdir()
    manifest_payload = _build_vjepa2_manifest_payload()
    (encoder_src / "manifest.json").write_text(_json.dumps(manifest_payload))

    output_dir = tmp_path / "ckpt-out"
    output_dir.mkdir()

    enc = _build_vjepa2_encoder(embed_dim=1408)
    cfg = OmegaConf.create(
        {"model": {"video_backbone": {"encoder": {"name": "vjepa2", "model_path": str(encoder_src)}}}}
    )
    enc.copy_deploy_artifacts(str(output_dir), cfg)

    dst = output_dir / "manifest.json"
    assert dst.exists()
    assert _json.loads(dst.read_text()) == manifest_payload


def test_W16_vjepa2_copy_deploy_artifacts_missing_cfg_is_warning_not_raise(tmp_path, caplog):
    """V-JEPA 2 mirror of test_V19: missing cfg / missing source file must
    log a warning and skip the copy rather than raising. A copy hiccup
    cannot crash an otherwise-good training run; deploy then falls back
    to ``encoder.model_path``."""
    import logging

    from omegaconf import OmegaConf

    enc = _build_vjepa2_encoder(embed_dim=1408)
    output_dir = tmp_path / "ckpt-out"
    output_dir.mkdir()

    # cfg without model.video_backbone.encoder → warning + no-op.
    with caplog.at_level(logging.WARNING):
        enc.copy_deploy_artifacts(str(output_dir), cfg={})
    assert not (output_dir / "manifest.json").exists()
    assert any("model_path" in r.message for r in caplog.records)

    caplog.clear()
    # cfg points at a directory with no manifest.json → warning + no-op.
    empty_src = tmp_path / "empty"
    empty_src.mkdir()
    cfg = OmegaConf.create({"model": {"video_backbone": {"encoder": {"name": "vjepa2", "model_path": str(empty_src)}}}})
    with caplog.at_level(logging.WARNING):
        enc.copy_deploy_artifacts(str(output_dir), cfg)
    assert not (output_dir / "manifest.json").exists()
    assert any("manifest.json" in r.message for r in caplog.records)


def test_W17_vjepa2_copy_deploy_artifacts_io_error_does_not_crash(tmp_path, caplog, monkeypatch):
    """V-JEPA 2 mirror of test_V20: PermissionError / ENOSPC / disappearing-
    mount OSError during the manifest copy must collapse to a warning +
    return so the trainer's safetensors save isn't lost."""
    import json as _json
    import logging
    import shutil

    from omegaconf import OmegaConf

    encoder_src = tmp_path / "vjepa2-weights"
    encoder_src.mkdir()
    (encoder_src / "manifest.json").write_text(_json.dumps(_build_vjepa2_manifest_payload()))
    output_dir = tmp_path / "ckpt-out"
    output_dir.mkdir()
    enc = _build_vjepa2_encoder(embed_dim=1408)
    cfg = OmegaConf.create(
        {"model": {"video_backbone": {"encoder": {"name": "vjepa2", "model_path": str(encoder_src)}}}}
    )

    def _boom(*args, **kwargs):
        raise PermissionError("simulated read-only filesystem")

    monkeypatch.setattr(shutil, "copyfile", _boom)
    with caplog.at_level(logging.WARNING):
        # Must NOT raise — assertion is "we got here".
        enc.copy_deploy_artifacts(str(output_dir), cfg)
    assert not (output_dir / "manifest.json").exists()
    formatted = [r.getMessage() for r in caplog.records]
    assert any("failed" in m and "simulated" in m for m in formatted), (
        f"expected warning naming the copy failure; got: {formatted}"
    )


def test_W18_vjepa_target_temporal_pool_stride_constant_pinned():
    """The encoder's ``_TARGET_TEMPORAL_POOL_STRIDE = 2`` is what makes
    ``temporal_compression == 4`` (ViT tubelet=2 × encoder pool stride=2)
    and gives Wan VAE causal-group token-count parity. Tests
    ``test_V6h`` / ``test_W7`` deliberately use a loose ``\\d+`` regex in
    the rejection message so they keep working if the constant is bumped
    — but a silent bump would still break production token-count parity.
    Pinning the value here surfaces an undocumented constant change as a
    noisy test failure without disturbing the loose-regex design.
    """
    from openwam.model.video_backbone.encoder.vjepa2 import VJEPA2VideoEncoder
    from openwam.model.video_backbone.encoder.vjepa2_1 import VJEPA21VideoEncoder

    assert VJEPA21VideoEncoder._TARGET_TEMPORAL_POOL_STRIDE == 2
    assert VJEPA2VideoEncoder._TARGET_TEMPORAL_POOL_STRIDE == 2
