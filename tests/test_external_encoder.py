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

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from openwam.model.video_backbone.adapter import VideoBackbone
from openwam.model.video_backbone.encoder import (
    _VIDEO_ENCODER_REGISTRY,
    VideoEncoder,
    VideoEncoderSpec,
    build_video_encoder,
    register_video_encoder,
)

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
    """Without external_encoder, _pipe.vae.* keys are present and _encoder.* is None."""
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

    pipe = _FakePipe()
    backbone = WanVideoBackbone(pipe)  # default path, no external_encoder
    assert backbone._uses_external_encoder is False
    sd = backbone.state_dict()
    assert any(k.startswith("_pipe.vae.") for k in sd), "default path should expose _pipe.vae.* keys"
    assert not any(k.startswith("_encoder.") for k in sd), "default path must not have _encoder.* keys"


def test_C2_default_path_pipe_vae_call_sites_preserved():
    """5 IO entries route through pipe.vae (and pipe.preprocess_video /
    pipe.vae_output_to_video) on the default path."""
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

    pipe = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    assert backbone._uses_external_encoder is True
    assert backbone._pipe.vae is None


def test_C4_external_path_state_dict_keys_swap():
    """External path: _encoder.* keys present, _pipe.vae.* absent."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

    pipe = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    sd = backbone.state_dict()
    assert any(k.startswith("_encoder.") for k in sd), "external path should expose _encoder.* keys"
    assert not any(k.startswith("_pipe.vae.") for k in sd), "external path must release _pipe.vae"


def test_C5_submodule_names_vae_alias_in_both_paths():
    """submodule_names always contains 'vae' — alias resolves differently
    depending on whether external_encoder is set."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

    pipe = _FakePipe(has_vace=True)
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    with pytest.raises(ValueError, match="VACE backbones cannot use external encoders"):
        WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)


def test_C9_spec_validation_strict_when_reversible():
    """Reversible encoder with mismatched z_dim is rejected immediately."""
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False)
    backbone = WanVideoBackbone.from_pretrained(pipe, external_encoder=enc)
    with pytest.raises(NotImplementedError, match="irreversible"):
        backbone.decode_video(torch.zeros(1, 1024, 4, 8, 8))


def test_C13a_reinit_with_external_encoder_rebuilds_modules():
    """reinit_dit_from_scratch(pipe, external_encoder=enc) rebuilds
    patch_embedding and head.head at the encoder's z_dim, and syncs in_dim."""
    from openwam.model.video_backbone.wan_adapter import reinit_dit_from_scratch

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False)
    reinit_dit_from_scratch(pipe, external_encoder=enc, verbose=False)
    assert pipe.dit.patch_embedding.in_channels == 1024
    assert pipe.dit.head.head.out_features == 1024 * 4  # z_dim * prod((1,2,2))
    assert pipe.dit.in_dim == 1024


def test_C13b_reinit_without_external_encoder_is_backwards_compat():
    """reinit_dit_from_scratch(pipe) WITHOUT external_encoder kwarg must behave
    exactly as before (no shape change). Guards the 17 existing from_scratch
    test cases in test_video_backbone_from_scratch.py."""
    from openwam.model.video_backbone.wan_adapter import reinit_dit_from_scratch

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
    'Public implementation.'
    from openwam.model.video_backbone.wan_adapter import reinit_dit_from_scratch

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, is_reversible=False, dit_patch_size=(1, 1, 1))
    reinit_dit_from_scratch(pipe, external_encoder=enc, verbose=False)

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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

    from openwam.model.video_backbone import wan_adapter
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    import openwam.model.video_backbone.wan_adapter as wan_adapter_mod
    from openwam.model.base import BaseWAMArchitecture
    from openwam.model.video_backbone import wan_adapter

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

    # --- "Training" side ---
    train_pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    train_enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    train_bb = WanVideoBackbone.from_pretrained(train_pipe, external_encoder=train_enc)
    train_keys = set(train_bb.state_dict().keys())
    # Must use the external-encoder slot, not _pipe.vae.
    assert any(k.startswith("_encoder.") for k in train_keys), (
        "training-time backbone state_dict missing _encoder.* keys"
    )
    assert not any(k.startswith("_pipe.vae.") for k in train_keys), (
        "training-time backbone state_dict has _pipe.vae.* — native VAE wasn't released"
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
    from openwam.model.video_backbone.wan_adapter import WanVideoBackbone
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
        "configs/model/dual_system.yaml",
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
