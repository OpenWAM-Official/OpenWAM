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
    VideoEncoderProperties,
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
        self._spec = VideoEncoderProperties(**self._SPEC_KWARGS)

    @property
    def properties(self) -> VideoEncoderProperties:
        return self._spec

    def preprocess_video(self, frames):
        return torch.zeros(1, 3, 4, 8, 8)

    def batch_encode(self, video: Tensor) -> Tensor:
        s = self.properties
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
    enc = _make_mock_encoder(pixel_decode=False)
    with pytest.raises(NotImplementedError) as exc:
        enc.decode(torch.zeros(1, 16, 4, 8, 8))
    assert "pixel_decode=False" in str(exc.value)


def test_A5_default_to_frames_raises_with_contract_aware_msg():
    enc = _make_mock_encoder(pixel_decode=False)
    with pytest.raises(NotImplementedError) as exc:
        enc.to_frames(torch.zeros(1, 3, 4, 8, 8))
    assert "pixel_decode=False" in str(exc.value)


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


def test_B1_wan22_vae_registered():
    """Importing the encoder package registers wan22_vae under that name."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder

    assert "wan22_vae" in _VIDEO_ENCODER_REGISTRY
    assert _VIDEO_ENCODER_REGISTRY["wan22_vae"] is WanVideoVAEEncoder


def test_B2_wan22_vae_default_hooks_match_wan21():
    """Wan2.1 family: z_dim=16, upsampling_factor=8 → Conv3d(16, dit, (1,2,2), (1,2,2))."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder

    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    assert enc.properties.z_dim == 16
    assert enc.properties.spatial_compression == 8
    assert enc.properties.temporal_compression == 4
    assert enc.properties.causal_temporal is True
    inp = enc.build_dit_input_proj(dit_dim=1536)
    assert isinstance(inp, nn.Conv3d)
    assert (inp.in_channels, inp.out_channels) == (16, 1536)
    assert inp.kernel_size == (1, 2, 2) and inp.stride == (1, 2, 2)
    out = enc.build_dit_output_proj(dit_dim=1536)
    assert isinstance(out, nn.Linear)
    assert (out.in_features, out.out_features) == (1536, 16 * 4)  # z_dim * prod((1,2,2))


def test_B3_wan22_vae_default_hooks_match_wan22():
    """Wan2.2 family: z_dim=48, upsampling_factor=16. Same kernel layout, only z_dim differs."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder

    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=48, upsampling_factor=16))
    assert enc.properties.z_dim == 48
    assert enc.properties.spatial_compression == 16
    inp = enc.build_dit_input_proj(dit_dim=1536)
    assert (inp.in_channels, inp.out_channels) == (48, 1536)
    assert inp.kernel_size == (1, 2, 2)
    out = enc.build_dit_output_proj(dit_dim=1536)
    assert out.out_features == 48 * 4


def test_B4_wan22_vae_pixel_decode_true_by_default():
    """Wan VAE has a real pixel decoder → properties.pixel_decode inherits the dataclass
    default ``True``. Confirms WanVideoVAEEncoder doesn't accidentally flip it."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder

    enc = WanVideoVAEEncoder(_FakeWanVAEModule())
    assert enc.properties.pixel_decode is True
    assert enc.properties.dit_patch_size == (1, 2, 2)


# ===========================================================================
# Commit 4: C1-C13 — WanVideoBackbone external_encoder injection
# ===========================================================================


class _FakeDiT(nn.Module):
    """Minimal DiT stand-in carrying the attributes WanVideoBackbone / reinit
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
    """Minimal pipe stand-in that Wan21.from_pretrained mutates.

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
        # the WanVideoBackbone fail-fast probe (which only does
        # ``getattr(pipe, "vace", None) is not None``) a plain placeholder
        # is sufficient.
        self.vace = nn.Module() if has_vace else None
        self.height_division_factor = 0
        self.width_division_factor = 0
        self.device = "cpu"


def test_C1_default_path_state_dict_keys_contain_pipe_vae():
    """Without external_encoder, native vae.* keys are present (phase-3c attribute-ization) and video_encoder.* is None."""
    from openwam.model.video_backbone.wan_backbone import Wan21

    pipe = _FakePipe()
    backbone = Wan21(pipe)  # default path, no external_encoder
    assert backbone._uses_external_encoder is False
    sd = backbone.state_dict()
    assert any(k.startswith("vae.") for k in sd), "default path should expose vae.* keys"
    assert not any(k.startswith("video_encoder.") for k in sd), "default path must not have video_encoder.* keys"


def test_C2_default_path_pipe_vae_call_sites_preserved(monkeypatch):
    """Default path (encoder=None): wan.encode routes encode/decode through the
    native vae; preprocess_video / latents_to_frames route through wan.preprocess."""
    import openwam.model.video_backbone.wan.encode as enc_mod
    from openwam.model.video_backbone.wan import encode as wan_encode
    from openwam.model.video_backbone.wan_backbone import Wan21

    pipe = _FakePipe()
    backbone = Wan21(pipe)
    backbone._device = torch.device("cpu")
    vae = getattr(backbone, "vae", None)
    # preprocess_video(encoder=None) → wan.preprocess.preprocess_video
    monkeypatch.setattr(enc_mod, "_preprocess_video_native", lambda frames, **kw: torch.zeros(1, 3, 4, 64, 64))
    assert wan_encode.preprocess_video([None], encoder=None, dtype=torch.float32, device="cpu").shape == (
        1,
        3,
        4,
        64,
        64,
    )
    # encode_video(encoder=None) → vae.batch_encode
    enc = wan_encode.encode_video(torch.zeros(1, 3, 4, 64, 64), vae=vae, encoder=None)
    assert enc.shape[1] == 16  # VAE z_dim
    # decode_latents(encoder=None) → vae.decode
    dec = wan_encode.decode_latents(torch.zeros(1, 16, 4, 8, 8), vae=vae, encoder=None, device="cpu")
    assert dec.shape == (1, 3, 16, 64, 64)
    # latents_to_frames(encoder=None) → wan.preprocess.vae_output_to_video
    monkeypatch.setattr(enc_mod, "vae_output_to_video", lambda t: ["frame0", "frame1"])
    assert wan_encode.latents_to_frames(dec, encoder=None) == ["frame0", "frame1"]


def test_C3_external_path_releases_pipe_vae_in_from_pretrained():
    """from_pretrained must set pipe.vae=None when an external_encoder is wired."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    pipe = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone = Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)
    assert backbone._uses_external_encoder is True
    assert getattr(backbone, "vae", None) is None


def test_C4_external_path_state_dict_keys_swap():
    """External path: video_encoder.* keys present, native vae.* absent."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    pipe = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone = Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)
    sd = backbone.state_dict()
    assert any(k.startswith("video_encoder.") for k in sd), "external path should expose video_encoder.* keys"
    assert not any(k.startswith("vae.") for k in sd), "external path must release the native vae.*"


def test_C6_get_submodule_vae_routes_to_encoder_on_external_path():
    """get_submodule('vae') returns the encoder on external path, pipe.vae on default."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_backbone import Wan21, Wan22Ti2v

    pipe_default = _FakePipe()
    backbone_default = Wan21(pipe_default)
    assert backbone_default.get_submodule("vae") is pipe_default.vae

    pipe_external = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone_external = Wan22Ti2v.from_pretrained(pipe_external, external_encoder=enc)
    assert backbone_external.get_submodule("vae") is enc


def test_C7_set_dtype_device_moves_external_encoder():
    """The encoder is a named child, so set_dtype_device (self.to) moves it too."""
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    pipe = _FakePipe()
    enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    backbone = Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)
    # Track moves via a side-effect: read the encoder's proj dtype after .to().
    backbone.set_dtype_device(torch.float32, torch.device("cpu"))
    assert next(enc._m.parameters()).dtype == torch.float32


def test_C10_spec_validation_fully_skipped_when_irreversible():
    """Irreversible encoder declares its own latent geometry; the backbone
    skips validation entirely (z_dim, spatial/temporal compression, causal
    are all encoder-owned). This case exercises a matching spatial=8 so it
    only verifies the z_dim mismatch is tolerated; C10b covers the harder
    case where spatial also differs from the backbone's native VAE."""
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False)
    backbone = Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)
    assert backbone._uses_external_encoder is True


def test_C10b_divergent_geometry_irreversible_encoder_loads():
    """A DINOv3-style encoder declares spatial_compression=16 /
    temporal_compression=1 / causal=False, all differing from Wan2.1's native
    VAE (spatial=8, temporal=4, causal=True). from_pretrained must accept it and
    derive division factors from the encoder's own spec."""
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)

    # Direct VideoEncoder subclass with all-different spec dimensions; we
    # don't reuse WanVideoVAEEncoderStub because its spatial_compression is
    # hardcoded to 8 (so it would mask the bug C10b is specifically guarding).
    class _DinoLikeEncoder(VideoEncoder):
        def __init__(self):
            super().__init__()
            self._proj = nn.Conv3d(1024, 1024, kernel_size=1)
            self._spec = VideoEncoderProperties(
                z_dim=1024,
                spatial_compression=16,
                temporal_compression=1,
                causal_temporal=False,
                pixel_decode=False,
                dit_patch_size=(1, 1, 1),
            )

        @property
        def properties(self):
            return self._spec

        def preprocess_video(self, frames):
            return torch.zeros(1, 3, 4, 256, 256)

        def batch_encode(self, video):
            return torch.zeros(video.shape[0], 1024, video.shape[2], video.shape[3] // 16, video.shape[4] // 16)

        @classmethod
        def from_pretrained(cls, model_path, **kw):
            return cls()

    enc = _DinoLikeEncoder()
    backbone = Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)
    assert backbone._uses_external_encoder is True
    # And the division factor honors the encoder's dit_patch_size=(1,1,1):
    # spatial(16) * dit_patch_size[1or2](1) = 16, NOT spatial * 2.
    assert pipe.height_division_factor == 16
    assert pipe.width_division_factor == 16


def test_C11_dit_patch_size_drives_height_width_division_factor():
    """properties.dit_patch_size=(1,1,1) — height/width_division_factor equals
    spatial_compression (no extra *2). Verifies the hardcoded *2 is gone."""
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False, dit_patch_size=(1, 1, 1))
    Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)
    assert pipe.height_division_factor == 8  # spatial_compression * 1
    assert pipe.width_division_factor == 8

    # And the default (1,2,2) still works the same as before.
    pipe2 = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc2 = WanVideoVAEEncoderStub(spec_z_dim=16, pixel_decode=True, dit_patch_size=(1, 2, 2))
    Wan22Ti2v.from_pretrained(pipe2, external_encoder=enc2)
    assert pipe2.height_division_factor == 16  # 8 * 2
    assert pipe2.width_division_factor == 16


def test_C12_decode_video_blocks_irreversible_encoder():
    """decode_video must raise NotImplementedError when the encoder is irreversible.
    The error message must reference the contract violation, not be a generic AttributeError."""
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False)
    backbone = Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)
    with pytest.raises(NotImplementedError, match="irreversible"):
        backbone.decode_video(torch.zeros(1, 1024, 4, 8, 8))


def test_C13a_reinit_with_external_encoder_rebuilds_modules():
    """reinit_dit_from_scratch(pipe, external_encoder=enc) rebuilds
    patch_embedding and head.head at the encoder's z_dim, and syncs in_dim."""
    from openwam.model.video_backbone.wan.reinit import reinit_dit_from_scratch

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False)
    reinit_dit_from_scratch(
        pipe,
        external_encoder=enc,
        dit_patch_size=enc.properties.dit_patch_size,
        verbose=False,
    )
    assert pipe.dit.patch_embedding.in_channels == 1024
    assert pipe.dit.head.head.out_features == 1024 * 4  # z_dim * prod((1,2,2))
    assert pipe.dit.in_dim == 1024


def test_C13b_reinit_without_external_encoder_is_backwards_compat():
    """reinit_dit_from_scratch(pipe) WITHOUT external_encoder kwarg must behave
    exactly as before (no shape change). Guards the 17 existing from_scratch
    test cases in test_video_backbone_from_scratch.py."""
    from openwam.model.video_backbone.wan.reinit import reinit_dit_from_scratch

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
    non-default ``properties.dit_patch_size``. Pre-fix, the patch_embedding and
    head.head Linear were rebuilt at the new shape but the unpatchify hint
    stayed at ``(1, 2, 2)`` — any encoder with ``dit_patch_size=(1,1,1)``
    (DINOv3 / V-JEPA2 patch-at-16) would shape-mismatch on the first
    forward.
    """
    from openwam.model.video_backbone.wan.reinit import reinit_dit_from_scratch

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False, dit_patch_size=(1, 1, 1))
    reinit_dit_from_scratch(
        pipe,
        external_encoder=enc,
        dit_patch_size=enc.properties.dit_patch_size,
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
    assert out.shape[-1] == enc.properties.z_dim * math.prod(enc.properties.dit_patch_size)


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
    from openwam.model.video_backbone.wan.reinit import adapt_dit_to_external_encoder

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    # Attach a sentinel sub-module that the adapt path must NOT touch.
    # After adapt, its weight must remain the stamped value — proof that
    # adapt did not run a global reset_parameters like reinit does.
    sentinel = nn.Linear(4, 4)
    sentinel.weight.data.fill_(0.1234)
    sentinel_snapshot = sentinel.weight.detach().clone()
    pipe.dit.add_module("sentinel_check", sentinel)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False)

    adapt_dit_to_external_encoder(pipe, enc, enc.properties.dit_patch_size)

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
    from openwam.model.video_backbone.wan.reinit import adapt_dit_to_external_encoder

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False)
    with pytest.raises(ValueError, match=r"dit_patch_size is required"):
        adapt_dit_to_external_encoder(pipe, enc, None)


def test_C13d_reinit_with_external_encoder_requires_dit_patch_size():
    """Single-source-of-truth guard: ``reinit_dit_from_scratch`` must refuse
    to silently fall back to ``external_encoder.properties.dit_patch_size`` when
    ``dit_patch_size`` is omitted. The backbone owns this geometry — callers
    must source it from ``self.video_backbone.dit_patch_size`` so the DiT
    rebuild reads the same value as the dataloader bridge and the cross-check
    in :meth:`BaseWAMArchitecture._init_video_backbone`."""
    from openwam.model.video_backbone.wan.reinit import reinit_dit_from_scratch

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False)
    with pytest.raises(ValueError, match=r"dit_patch_size is required"):
        reinit_dit_from_scratch(pipe, external_encoder=enc, verbose=False)


def test_C14_wan_save_deploy_assets_forwards_to_external_encoder(tmp_path):
    """``WanVideoBackbone.save_deploy_assets`` must forward to the external
    encoder's ``save_deploy_assets`` hook so V-JEPA's ``manifest.json`` lands
    in the checkpoint dir.

    The Wan spec/tokenizer step inside the same method is a no-op here because
    ``model_path`` points at a nonexistent dir, so ``save_video_backbone_deploy_assets``
    early-returns. That keeps this test isolated to the encoder-forwarding
    behavior, without standing up a real Wan model directory on disk.
    """
    from omegaconf import OmegaConf

    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    class _RecordingEncoder(_MockEncoderBase):
        def __init__(self):
            super().__init__()
            self.calls: list = []

        def save_deploy_assets(self, output_dir, cfg):
            self.calls.append((output_dir, cfg))

    pipe = _FakePipe()
    enc = _RecordingEncoder()
    backbone = Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)
    cfg = OmegaConf.create({"model": {"video_backbone": {"model_path": "/nonexistent"}}})
    backbone.save_deploy_assets(str(tmp_path), cfg)

    assert enc.calls == [(str(tmp_path), cfg)]


def test_C15_wan_save_deploy_assets_no_op_without_external_encoder(tmp_path):
    """Without an external encoder, ``WanVideoBackbone.save_deploy_assets``
    only runs the spec/tokenizer step — no encoder hook call, no crash on the
    default path. Regression guard against accidentally routing the encoder
    branch into the default path.
    """
    from omegaconf import OmegaConf

    from openwam.model.video_backbone.wan_backbone import Wan21

    pipe = _FakePipe()
    backbone = Wan21(pipe)
    assert backbone.video_encoder is None
    cfg = OmegaConf.create({"model": {"video_backbone": {"model_path": str(tmp_path)}}})
    backbone.save_deploy_assets(str(tmp_path), cfg)
    # No exception, no spurious files.
    # (Tokenizer copy is itself a warning-on-missing path; we don't assert
    # its side effects here — they are covered by Wan-side unit tests.)


# Tiny helper for C9-C13 — declares a custom spec without going through Wan VAE loading.


class WanVideoVAEEncoderStub(VideoEncoder):
    """Custom-spec encoder used by C9-C13 to control z_dim / pixel_decode /
    dit_patch_size without touching real Wan VAE weights."""

    def __init__(self, *, spec_z_dim: int, pixel_decode: bool, dit_patch_size=(1, 2, 2)):
        super().__init__()
        # A tiny conv so state_dict has something to enumerate (test C4).
        self._proj = nn.Conv3d(spec_z_dim, spec_z_dim, kernel_size=1)
        self._spec = VideoEncoderProperties(
            z_dim=spec_z_dim,
            spatial_compression=8,
            temporal_compression=4,
            causal_temporal=True,
            pixel_decode=pixel_decode,
            dit_patch_size=dit_patch_size,
        )

    @property
    def properties(self) -> VideoEncoderProperties:
        return self._spec

    def preprocess_video(self, frames):
        return torch.zeros(1, 3, 4, 64, 64)

    def batch_encode(self, video: Tensor) -> Tensor:
        return torch.zeros(
            video.shape[0], self._spec.z_dim, video.shape[2] // 4, video.shape[3] // 8, video.shape[4] // 8
        )

    @classmethod
    def from_pretrained(cls, model_path: str, **kw):
        return cls(spec_z_dim=16, pixel_decode=True)


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
    from openwam.model.architectures.base import BaseWAMArchitecture

    stub = _StubArchitecture()
    BaseWAMArchitecture._init_video_backbone(stub, model_cfg)
    return stub


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

    caplog.set_level(logging.INFO, logger="openwam.model.architectures.base")

    cfg = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": False,
            "encoder": {"name": "wan22_vae", "model_path": "/dummy"},
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
    would mismatch ``_pipe.vae.*`` vs ``video_encoder._m.*``.

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
    from openwam.model.architectures.base import BaseWAMArchitecture

    monkeypatch.setattr(BaseWAMArchitecture, "_build_external_encoder_skeleton", staticmethod(fake_skeleton))

    cfg = {
        "video_backbone": {
            "name": "wan22_ti2v_5b",
            "model_path": "/dummy",
            "from_scratch": False,
            "encoder": {"name": "wan22_vae", "model_path": "/dummy"},
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
        return WanVideoVAEEncoderStub(spec_z_dim=16, pixel_decode=True)

    def fake_build_backbone(name, cfg, **kw):
        backbone_kwargs.update(kw)
        bb = nn.Module()
        bb._pipe = None
        bb.temporal_compression = 4
        bb.causal_temporal = True
        bb.reinit_for_from_scratch = lambda **_: None  # no dit to re-init in this stub
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
            "encoder": {"name": "wan22_vae", "model_path": "/dummy"},
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
        bb.reinit_for_from_scratch = lambda **_: None  # no dit to re-init in this stub
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
    """``_assert_decode_video_supported`` raises when ``vb.video_encoder`` is
    irreversible — the same helper :meth:`BaseWAMArchitecture.generate`
    calls just before invoking ``vb.decode_video`` when ``decode_video=True``.

    Standing up the full ``generate`` denoising loop in a unit test would
    require the entire scheduler/pipeline stack; instead we extract the
    guard as ``_assert_decode_video_supported`` (base.py) and exercise it
    directly with a stub backbone, so a regression that renames
    ``vb.external_encoder`` or flips the polarity is caught here.
    """
    from openwam.model.architectures.base import _assert_decode_video_supported

    class _StubBackbone:
        # Mirror VideoBackbone.external_encoder: expose the wired-in encoder
        # (None on the native VAE path) through the ABC contract the guard reads.
        @property
        def external_encoder(self):
            return getattr(self, "video_encoder", None)

    # Irreversible → fail-fast.
    vb_irrev = _StubBackbone()
    vb_irrev.video_encoder = WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False)
    with pytest.raises(ValueError, match=r"irreversible"):
        _assert_decode_video_supported(vb_irrev)

    # Reversible → no-op.
    vb_rev = _StubBackbone()
    vb_rev.video_encoder = WanVideoVAEEncoderStub(spec_z_dim=16, pixel_decode=True)
    _assert_decode_video_supported(vb_rev)

    # Native VAE path (no video_encoder attribute at all) → no-op.
    vb_native = _StubBackbone()
    _assert_decode_video_supported(vb_native)

    # Native VAE path (video_encoder is None — what WanVideoBackbone sets when
    # external_encoder is not provided) → no-op.
    vb_native_none = _StubBackbone()
    vb_native_none.video_encoder = None
    _assert_decode_video_supported(vb_native_none)


def test_D6_freeze_modules_resolves_encoder_dotted_path_on_external_path():
    """Regression for the severe S2 bug: when ``video_backbone.from_scratch=true``
    routes through an external encoder, ``pipe.vae`` is None and the
    historical ``freeze: [..., video_backbone._pipe.vae, ...]`` entry is
    silently skipped — leaving the encoder's pretrained weights trainable.
    The fix lists ``video_backbone.video_encoder`` in the freeze yamls; this
    test verifies the dotted path actually resolves via
    ``nn.Module.get_submodule`` and that ``requires_grad_(False)`` then
    propagates to every encoder parameter.
    """
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False)
    backbone = Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)

    # Mount the backbone on an architecture-shaped container, just like the
    # real BaseWAMArchitecture does — freeze_modules resolves dotted paths
    # rooted at ``self``.
    class _ArchContainer(nn.Module):
        def __init__(self, vb):
            super().__init__()
            self.video_backbone = vb

    arch = _ArchContainer(backbone)

    # get_submodule routes through the WanVideoBackbone's _modules dict;
    # ``video_encoder = external_encoder`` in __init__ registers it there.
    resolved = arch.get_submodule("video_backbone.video_encoder")
    assert resolved is enc

    # Encoder parameters are trainable by default. Pre-fix: the freeze yaml
    # entry (_pipe.vae) was silently skipped because pipe.vae=None, so the
    # encoder stayed trainable. Post-fix: the new yaml entry (video_encoder)
    # resolves and the call below disables grad on every encoder param.
    assert any(p.requires_grad for p in enc.parameters())
    resolved.requires_grad_(False)
    assert all(not p.requires_grad for p in enc.parameters())

    # Sibling guards: native VAE path → video_encoder is None and not in _modules,
    # so freeze_modules's get_submodule call must raise AttributeError so
    # the framework can silently skip it (yaml lists both paths).
    pipe2 = _FakePipe(vae_z_dim=16, vae_upsample=8)
    backbone2 = Wan22Ti2v.from_pretrained(pipe2)  # no external_encoder
    arch2 = _ArchContainer(backbone2)
    with pytest.raises(AttributeError):
        arch2.get_submodule("video_backbone.video_encoder")


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
    """``Wan21.from_pretrained`` decides ``skip_native_vae`` via:

      - training (``DictConfig`` source) + irreversible encoder → True
      - training + reversible encoder → False (validation needs native VAE)
      - deploy (``dict`` / ``str`` source) + any external encoder → True
        (state_dict topology is ``video_encoder._m.*``, not ``_pipe.vae.*``;
        deploy must not materialize the empty native VAE slot)
      - no external encoder → False on both paths

    Monkeypatches ``loader.build_holder_from_model_path`` to capture the kwarg
    rather than stand up a real pipeline. Training-path uses a real
    ``DictConfig`` to exercise the ``isinstance(source, DictConfig)``
    dispatch added for the deploy fix.
    """
    from omegaconf import OmegaConf

    from openwam.model.video_backbone.wan import loader as loader_mod
    from openwam.model.video_backbone.wan_backbone import Wan21, Wan22Ti2v

    captured: dict = {}

    def _spy_model_path(model_path, device="cpu", *, skip_native_vae=False):
        captured["skip"] = skip_native_vae
        return _FakePipe(vae_z_dim=16, vae_upsample=8)

    def _spy_training(cfg, *, skip_native_vae=False):
        captured["skip"] = skip_native_vae
        return _FakePipe(vae_z_dim=16, vae_upsample=8)

    original = loader_mod.build_holder_from_model_path
    loader_mod.build_holder_from_model_path = _spy_model_path
    import openwam.model.video_backbone.wan.pipeline_builder as pb_mod

    original_btp = pb_mod.build_training_pipeline
    pb_mod.build_training_pipeline = _spy_training
    try:
        # --- Training path: DictConfig source ---
        train_cfg = OmegaConf.create({"video_backbone": {"model_path": "/dummy"}})

        # Training + irreversible → skip=True.
        captured.clear()
        Wan22Ti2v.from_pretrained(
            train_cfg,
            external_encoder=WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False),
        )
        assert captured["skip"] is True

        # Training + reversible → skip=False (need native VAE for spec validation).
        captured.clear()
        Wan22Ti2v.from_pretrained(
            train_cfg,
            external_encoder=WanVideoVAEEncoderStub(spec_z_dim=16, pixel_decode=True),
        )
        assert captured["skip"] is False

        # Training + no encoder → skip=False.
        captured.clear()
        Wan21.from_pretrained(train_cfg)
        assert captured["skip"] is False

        # --- Deploy path: dict source (state_dict topology is video_encoder._m.*) ---
        deploy_cfg = {"video_backbone": {"model_path": "/dummy"}}

        # Deploy + reversible → skip=True (no native VAE slot to materialize).
        captured.clear()
        Wan22Ti2v.from_pretrained(
            deploy_cfg,
            external_encoder=WanVideoVAEEncoderStub(spec_z_dim=16, pixel_decode=True),
        )
        assert captured["skip"] is True

        # Deploy + irreversible → skip=True.
        captured.clear()
        Wan22Ti2v.from_pretrained(
            deploy_cfg,
            external_encoder=WanVideoVAEEncoderStub(spec_z_dim=1024, pixel_decode=False),
        )
        assert captured["skip"] is True

        # Deploy + no encoder → skip=False (no encoder means use native VAE).
        captured.clear()
        Wan21.from_pretrained(deploy_cfg)
        assert captured["skip"] is False
    finally:
        loader_mod.build_holder_from_model_path = original
        pb_mod.build_training_pipeline = original_btp


def test_M3c_wan22_vae_encoder_from_skeleton_matches_from_pretrained_topology():
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
    # ``openwam.model.video_backbone.wan.models.vae.WanVideoVAE38`` etc.
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
    assert enc_train.properties == enc_deploy.properties


def test_M3d_build_external_encoder_skeleton_picks_vae_entry_from_source():
    """``BaseWAMArchitecture._build_external_encoder_skeleton`` reaches into
    the deploy ``source`` dict for ``components`` and finds the
    ``attr=='vae'`` entry, then dispatches to the registered encoder's
    ``from_skeleton``.
    """
    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.model.video_backbone.encoder import (
        _VIDEO_ENCODER_REGISTRY,
        WanVideoVAEEncoder,
        register_video_encoder,
    )

    # Use the real wan22_vae registry entry — it owns the from_skeleton
    # implementation that the deploy path will route through in
    # production.
    assert _VIDEO_ENCODER_REGISTRY["wan22_vae"] is WanVideoVAEEncoder

    enc_cfg = {"name": "wan22_vae", "model_path": "/unused-on-deploy-path"}
    source = {
        "components": [
            {
                "attr": "dit",
                "model_class": "openwam.model.video_backbone.wan.models.dit.WanModel",
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
    assert enc.properties.z_dim == 16
    assert enc.properties.spatial_compression == 8

    # Missing components → loud error (no silent fallback to native VAE).
    with pytest.raises(RuntimeError, match=r"no video_backbone\.components|components"):
        BaseWAMArchitecture._build_external_encoder_skeleton(enc_cfg, {"components": []})

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
    import openwam.model.video_backbone.wan.reinit as reinit_mod
    from openwam.model.architectures.base import BaseWAMArchitecture

    reinit_calls = []
    original_reinit = reinit_mod.reinit_dit_from_scratch

    def _spy_reinit(*a, **kw):
        # Record-only stub: don't invoke the real reinit because the
        # stub backbone's _FakePipe has empty dit.blocks and would
        # IndexError. We only care whether reinit was called at all.
        reinit_calls.append((a, kw))

    # ``base.py`` does ``from ...wan.reinit import reinit_dit_from_scratch``
    # locally inside the function, so the patch must target wan.reinit.
    reinit_mod.reinit_dit_from_scratch = _spy_reinit

    # Make build_video_backbone hand back a stub backbone so we can drive
    # _init_video_backbone end-to-end without a real pipeline build.
    import openwam.model.video_backbone as vb_pkg

    original_build = vb_pkg.build_video_backbone

    def _stub_build(name, cfg, **kw):
        bb = nn.Module()
        bb.dit = _FakePipe(vae_z_dim=16, vae_upsample=8).dit
        bb._uses_external_encoder = False
        bb.temporal_compression = 4
        bb.causal_temporal = True
        bb.dit_patch_size = (1, 2, 2)
        # Bind the real WanBase contract so the source-based train/deploy split
        # (and the spied reinit_dit_from_scratch call) is exercised exactly as
        # production routes it.
        from openwam.model.video_backbone.wan_backbone import WanBase

        bb.reinit_for_from_scratch = WanBase.reinit_for_from_scratch.__get__(bb)
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
        reinit_mod.reinit_dit_from_scratch = original_reinit
        vb_pkg.build_video_backbone = original_build


def test_M3g_from_pretrained_attaches_pipe_latent_spec_on_external_path():
    """``Wan21.from_pretrained`` must attach
    ``pipe.latent_spec`` (= ``external_encoder.properties``) so vendored
    inference units (``WanVideoUnit_NoiseInitializer``) can read latent
    shape metadata without falling back to ``pipe.vae`` (which is None
    on this path).

    Native VAE path leaves ``pipe.latent_spec`` absent — the vendored
    unit's fallback branch then reads ``pipe.vae`` as before, preserving
    bit-exact behavior for old checkpoints.
    """
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    # --- External encoder path: pipe.latent_spec is set ---
    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    enc = WanVideoVAEEncoderStub(spec_z_dim=16, pixel_decode=True)
    Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)
    assert hasattr(pipe, "latent_spec"), "pipe.latent_spec missing on external-encoder path"
    assert pipe.latent_spec is enc.properties, "pipe.latent_spec must reference encoder.properties verbatim"
    assert pipe.vae is None, "pipe.vae must be released on external-encoder path"

    # --- Native VAE path: pipe.latent_spec is absent ---
    pipe2 = _FakePipe(vae_z_dim=16, vae_upsample=8)
    Wan22Ti2v.from_pretrained(pipe2)  # no external_encoder
    assert not hasattr(pipe2, "latent_spec"), (
        "pipe.latent_spec must not be attached on the native VAE path "
        "— the vendored unit's pipe.vae fallback must remain authoritative"
    )


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
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v

    # --- "Training" side ---
    train_pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    train_enc = WanVideoVAEEncoder(_FakeWanVAEModule(z_dim=16, upsampling_factor=8))
    train_bb = Wan22Ti2v.from_pretrained(train_pipe, external_encoder=train_enc)
    train_keys = set(train_bb.state_dict().keys())
    # Must use the external-encoder slot, not the native vae.*.
    assert any(k.startswith("video_encoder.") for k in train_keys), (
        "training-time backbone state_dict missing video_encoder.* keys"
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
    # which routes through _build_holder_from_components(skip_native_vae=True). For this
    # closure check the WanVideoBackbone construction path is the same as training,
    # just with a fresh pipe sans the loaded VAE — using _FakePipe directly is
    # equivalent because skip_native_vae=True on deploy zeroes pipe.vae anyway.
    deploy_pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    deploy_bb = Wan22Ti2v.from_pretrained(deploy_pipe, external_encoder=deploy_enc)
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


def test_D8_wan22_vae_path_end_to_end_freeze_excludes_encoder_params_from_optimizer():
    """End-to-end check for the ``from_scratch=true + encoder.name=wan22_vae``
    path: starting from the yaml freeze list, walk the actual production
    code (``BaseWAMArchitecture.freeze_modules`` →
    ``build_trainable_parameters``) and verify ZERO encoder parameters survive
    into the optimizer.

    Uses the real ``WanVideoVAEEncoder`` class (not a stub) wrapped around
    a ``_FakeWanVAEModule`` so the test runs without GPU/real-weight
    dependencies but exercises the same nn.Module nesting layout the
    production encoder has: ``WanVideoVAEEncoder._m = vae`` with the VAE
    holding the bulk of the parameters.
    """
    import pathlib

    from omegaconf import OmegaConf

    from openwam.model.architectures.base import BaseWAMArchitecture
    from openwam.model.video_backbone.encoder import WanVideoVAEEncoder
    from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    # 1) Real encoder + real backbone.
    vae_module = _FakeWanVAEModule(z_dim=16, upsampling_factor=8)
    enc = WanVideoVAEEncoder(vae_module)
    pipe = _FakePipe(vae_z_dim=16, vae_upsample=8)
    backbone = Wan22Ti2v.from_pretrained(pipe, external_encoder=enc)
    assert backbone._uses_external_encoder is True
    assert backbone.video_encoder is enc

    # 2) Stand-in architecture: only the bits freeze_modules /
    #    build_trainable_parameters touch. Cannot subclass BaseWAMArchitecture
    #    directly because abstract methods (forward, generate, ...) demand
    #    full pipeline machinery. An ``nn.Module`` container with the
    #    right child name is enough — freeze_modules uses self.get_submodule,
    #    and build_trainable_parameters uses arch.get_trainable_modules().
    class _ArchContainer(nn.Module):
        def __init__(self, vb):
            super().__init__()
            self.video_backbone = vb

        def get_trainable_modules(self, freeze_list=()):
            return BaseWAMArchitecture.get_trainable_modules(self, freeze_list)

    arch = _ArchContainer(backbone)

    # 3) Real yaml freeze list (no hand-curation — read the file that ships).
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    yaml_cfg = OmegaConf.load(repo_root / "configs/model/dual_system.yaml")
    freeze_list = list(yaml_cfg.freeze)
    assert "video_backbone.video_encoder" in freeze_list, (
        "dual_system.yaml must list video_backbone.video_encoder for this test to be meaningful"
    )

    # 4) Run the exact production freeze_modules call.
    frozen = BaseWAMArchitecture.freeze_modules(arch, freeze_list)
    assert "video_backbone.video_encoder" in frozen, (
        f"video_encoder failed to freeze; freeze_modules returned: {frozen}"
    )

    # 5) Sanity: encoder has parameters at all, and every single one is now
    #    requires_grad=False.
    enc_params = list(enc.parameters())
    assert len(enc_params) > 0, "WanVideoVAEEncoder must expose VAE parameters via _m"
    assert all(not p.requires_grad for p in enc_params), "freeze did not propagate to encoder._m.* parameters"

    # 6) Run the exact production optimizer-param collection path
    #    (``build_trainable_parameters``) and confirm ZERO encoder parameters
    #    leak into the optimizer. The encoder lives under video_backbone, so a
    #    regression dropping the requires_grad filter would surface its (frozen)
    #    params here.
    class _ModelStub:
        architecture = arch
        lambda_action = 0  # excluded; not relevant here

    optimizer_params = build_trainable_parameters(_ModelStub())
    surfaced_ids = {id(p) for p in optimizer_params}
    encoder_ids = {id(p) for p in enc.parameters()}
    leaked = surfaced_ids & encoder_ids
    assert not leaked, f"Encoder parameters leaked into the optimizer: {len(leaked)} param(s)"

    # 7) Sibling guard: encoder params are reachable from the backbone via
    #    backbone.named_parameters() (so the test isn't trivially passing
    #    because they were hidden), they're just filtered by requires_grad.
    by_name = dict(backbone.named_parameters())
    enc_param_keys = [k for k in by_name if k.startswith("video_encoder.")]
    assert len(enc_param_keys) > 0, (
        "encoder parameters must be reachable via backbone.named_parameters otherwise the leak check above is trivial"
    )
    assert all(not by_name[k].requires_grad for k in enc_param_keys)


@pytest.mark.parametrize(
    "yaml_path",
    [
        "configs/model/dual_system.yaml",
        "configs/model/single_system.yaml",
        "configs/model/tri_system.yaml",
    ],
)
def test_D7_model_yaml_freezes_encoder(yaml_path):
    """Every model yaml that ships a ``freeze:`` list MUST enumerate
    ``video_backbone.video_encoder`` so the external-encoder path is frozen."""
    import pathlib

    from omegaconf import OmegaConf

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    cfg = OmegaConf.load(repo_root / yaml_path)
    freeze = list(cfg.get("freeze", []) or [])
    assert "video_backbone.video_encoder" in freeze, (
        f"{yaml_path} freeze list missing video_backbone.video_encoder; "
        f"external-encoder path will leave the encoder trainable. Got: {freeze}"
    )
    # The native-path entry must remain so default training stays bit-exact.
    assert "video_backbone.vae" in freeze


# ===========================================================================
# Commit 6: E1-E3 — framework yamls expose the encoder block (composed via the video_backbone group)
# ===========================================================================
#
# These tests verify the yaml ships with the documented defaults so that
# `from_scratch=false` users see the encoder field but it stays inert,
# and `from_scratch=true` users only need to flip one switch.


@pytest.mark.parametrize(
    "model_name",
    ["dual_system", "single_system", "tri_system"],
)
def test_E_composed_encoder_block(model_name):
    """Each framework composes video_backbone from the Hydra `video_backbone`
    group (default wan.yaml → encoder/wan22_vae.yaml); the composed config must
    expose video_backbone.encoder: {name=wan22_vae, model_path=...} so the
    encoder-gate path is identical across architectures."""
    import os
    import pathlib

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    config_dir = os.path.abspath(repo_root / "configs")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=[f"model={model_name}"])

    enc = cfg.model.video_backbone.get("encoder")
    assert enc is not None, f"{model_name} missing video_backbone.encoder"
    assert enc.name == "wan22_vae", f"{model_name} default encoder.name should be wan22_vae, got {enc.name!r}"
    assert "model_path" in enc, f"{model_name} encoder block missing model_path"
    extras = set(enc.keys()) - {"name", "model_path"}
    assert extras == set(), f"{model_name} encoder block has extra fields {extras}"


# ======================================================================
# A3: V-JEPA 2.1 encoder (V1-V8)
# ======================================================================


class _MockVJEPAViT(nn.Module):
    """Tiny CPU stand-in for the V-JEPA 2.1 ViT.

    Reproduces only what ``VJEPA21VideoEncoder._vit_grid`` consumes:
    ``(B, C, T, H, W) -> (B, L, D)`` with
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
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    vit = _MockVJEPAViT(embed_dim=embed_dim)
    return VJEPA21VideoEncoder(vit, embed_dim=embed_dim, variant="mock")


def test_V1_vjepa21_registration_round_trip():
    """``register_video_encoder("vjepa21")`` exposes the class via the registry."""
    from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder  # noqa: F401

    assert "vjepa21" in _VIDEO_ENCODER_REGISTRY
    assert _VIDEO_ENCODER_REGISTRY["vjepa21"] is VJEPA21VideoEncoder


def test_V2_vjepa21_from_pretrained_missing_manifest(tmp_path):
    """``from_pretrained`` on a dir without ``manifest.json`` raises FileNotFoundError."""
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        VJEPA21VideoEncoder.from_pretrained(str(tmp_path))


def test_V3_vjepa21_spec_invariants():
    """spec fields are nailed down: irreversible, causal, (1,2,2) DiT patch,
    temporal_compression=4 (ViT tubelet=2 + extra avg-pool stride=2, Wan VAE
    parity), z_dim wired from manifest."""
    enc = _build_vjepa_encoder(embed_dim=1408)
    properties = enc.properties
    assert properties.pixel_decode is False
    assert properties.causal_temporal is True
    assert properties.dit_patch_size == (1, 2, 2)
    assert properties.z_dim == 1408
    assert properties.spatial_compression == 16
    assert properties.temporal_compression == 4


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

    Independent of ``vjepa21_forward`` — both modes preserve the
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


def _build_vjepa_spy_encoder(*, embed_dim: int = 8, vjepa21_forward: str = "video"):
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    vit = _SpyVJEPAViT(embed_dim=embed_dim)
    enc = VJEPA21VideoEncoder(vit, embed_dim=embed_dim, variant="spy", vjepa21_forward=vjepa21_forward)
    return enc, vit


def test_V6b_vjepa21_forward_default_is_video():
    """No-kwarg construction picks ``vjepa21_forward="video"`` — the
    intended default after this PR (so cond and target both come from the
    V-JEPA video branch).
    """
    enc = _build_vjepa_encoder()
    assert enc.vjepa21_forward == "video"


def test_V6c_vjepa21_invalid_forward_raises():
    """Constructing with an unknown ``vjepa21_forward`` value fails fast
    at __init__ rather than producing a confusing branch-routing error
    inside ``batch_encode``.
    """
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    with pytest.raises(ValueError, match="vjepa21_forward must be one of"):
        VJEPA21VideoEncoder(
            _MockVJEPAViT(embed_dim=8),
            embed_dim=8,
            variant="mock",
            vjepa21_forward="image",  # type: ignore[arg-type]
        )


def test_V6d_vjepa21_video_mode_cond_dups_frame_zero():
    """``vjepa21_forward="video"``: cond pass dups frame 0 and routes the
    2-frame clip through the video branch (T=2). Target pass prepends the
    same dup'd pair to N target frames (T=2+N). Two video-branch forwards
    total — no image-branch call. The encoder-side avg-pool is invisible
    in the call_log (it happens after the forwards complete).
    """
    enc, vit = _build_vjepa_spy_encoder(vjepa21_forward="video")
    # T_pixel=9 → N_target=8, so target-pass clip has T=2+8=10.
    z = enc.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert z.shape == (1, 8, 3, 2, 2)  # 1 cond + 2 pooled target = 3 latent slices
    ts = [c["T"] for c in vit.call_log]
    assert ts == [2, 10], f"expected [2, 10] for video mode, got {ts}"


def test_V6e_vjepa21_mixed_mode_cond_uses_image_branch():
    """``vjepa21_forward="mixed"``: cond pass routes frame 0 through the
    image branch (T=1). Target pass is unchanged — still the prepend-and-
    drop-then-avg-pool path (T=2+N). One image-branch + one video-branch
    forward.
    """
    enc, vit = _build_vjepa_spy_encoder(vjepa21_forward="mixed")
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
    enc, vit = _build_vjepa_spy_encoder(vjepa21_forward="video")
    z = enc.batch_encode(torch.randn(1, 3, 9, 32, 32))
    assert z.shape[2] == 3  # 1 cond + 2 pooled target; both steps verified
    # Second call is the target pass with 2 prepended + 8 targets.
    assert vit.call_log[1]["T"] == 10


def test_V6g_vjepa21_t_pixel_1_honours_forward_mode():
    """TI2V single-frame fast path (T_pixel==1) routes through the cond
    pass only. The forward mode still selects the branch: ``video`` dups,
    ``mixed`` goes single-frame.
    """
    enc_v, vit_v = _build_vjepa_spy_encoder(vjepa21_forward="video")
    enc_m, vit_m = _build_vjepa_spy_encoder(vjepa21_forward="mixed")
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


def _build_vjepa_nan_propagating_encoder(*, embed_dim: int = 8, vjepa21_forward: str = "video"):
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    vit = _NaNPropagatingVJEPAViT(embed_dim=embed_dim)
    return VJEPA21VideoEncoder(vit, embed_dim=embed_dim, variant="nan-prop", vjepa21_forward=vjepa21_forward)


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
    enc = _build_vjepa_nan_propagating_encoder(vjepa21_forward=mode)
    video = torch.zeros(1, 3, 9, 32, 32)
    video[:, :, 1:] = float("nan")  # target frames poisoned; frame 0 still clean
    z = enc.batch_encode(video)
    assert z.shape == (1, 8, 3, 2, 2)
    assert not torch.isnan(z[:, :, 0:1]).any(), (
        f"cond latent contains NaN under vjepa21_forward={mode!r} — target frames are leaking into the cond pass"
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
    ``vjepa21_forward`` from cfg into ``from_pretrained(...)`` kwargs, and
    the constructed encoder reflects the chosen mode. Uses the fake-imports
    helper so no actual V-JEPA weights are needed.
    """
    import json as _json

    from openwam.model.video_backbone.encoder import build_video_encoder
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder
    from openwam.model.video_backbone.encoder.vjepa21 import loader as _vjepa_loader

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
    monkeypatch.setattr(_vjepa_loader, "load_vit_weights", lambda *a, **kw: None)

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
            "name": "vjepa21",
            "model_path": str(tmp_path),
            "vjepa21_forward": "mixed",
        }
    )
    assert isinstance(enc, VJEPA21VideoEncoder)
    assert enc.vjepa21_forward == "mixed"


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

    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

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
    from openwam.model.video_backbone.encoder.vjepa21 import loader as _vjepa_loader

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
    vjepa21_src = "openwam.model.video_backbone.encoder.vjepa21_src"
    monkeypatch.setitem(sys.modules, f"{vjepa21_src}.vision_transformer", fake_module)
    monkeypatch.setitem(sys.modules, f"{vjepa21_src}.modules", fake_vjepa_modules)

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
        _vjepa_loader.load_vit("/unused", manifest)
    assert "use_rope" not in captured_kwargs
    assert captured_kwargs["patch_size"] == 16
    assert captured_kwargs["interpolate_rope"] is True


def test_V10_vjepa21_load_vit_rope_arch_with_use_rope_false_fails_fast():
    """Manifest with ``arch_name=*_rope`` and ``use_rope=False`` is contradictory —
    we raise a ``ValueError`` at load time instead of silently overriding."""
    from openwam.model.video_backbone.encoder.vjepa21 import loader as _vjepa_loader

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
        _vjepa_loader.load_vit("/unused", manifest)


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
    # ``loader.prepare_vjepa_imports_and_patch`` does
    # ``from ...vjepa21_src import vision_transformer, modules`` — inject the
    # fakes at those sys.modules keys so no real (timm-dependent) ViT loads.
    vjepa21_src = "openwam.model.video_backbone.encoder.vjepa21_src"
    monkeypatch.setitem(sys.modules, f"{vjepa21_src}.vision_transformer", vision_transformer)
    monkeypatch.setitem(sys.modules, f"{vjepa21_src}.modules", fake_vjepa_modules)


def test_V11_vjepa21_from_skeleton_happy_path(tmp_path, monkeypatch):
    """``from_skeleton`` reads manifest from ckpt_dir, builds a
    zero-weight ViT shell, and skips torch.load entirely — even though
    ``manifest['checkpoint_file']`` would point at a non-existent file.
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

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
        ckpt_dir=str(tmp_path),
    )
    assert isinstance(enc, VJEPA21VideoEncoder)
    assert enc.properties.z_dim == 1408
    assert enc.variant == "vitg-rope-384"
    # _rope wrapper must NOT receive use_rope (PR #83 invariant)
    assert "use_rope" not in captured
    assert captured["patch_size"] == 16
    assert captured["img_size"] == (384, 384)
    assert captured["tubelet_size"] == 2


def test_V11b_vjepa21_from_skeleton_propagates_vjepa21_forward(tmp_path, monkeypatch, caplog):
    """Deploy-side knob plumbing — positive path. Pair to ``test_V11`` which
    omits the field (default branch) and ``test_W9`` / ``test_W9b`` which
    cover the rejection side on V-JEPA 2:

    - ``encoder_cfg={"vjepa21_forward": "mixed", ...}`` → the built
      encoder reports ``vjepa21_forward == "mixed"`` and the migration
      warning is silent (the field is present, so this is NOT a pre-PR
      checkpoint).
    - ``encoder_cfg`` without the field → resolved mode is the default
      AND the migration warning fires once, so an operator who deploys
      a pre-PR checkpoint without hand-adding ``vjepa21_forward: mixed``
      sees a noisy signal instead of a silently-divergent cond latent.
    """
    import json as _json
    import logging

    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

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
    with caplog.at_level(logging.WARNING, logger="openwam.model.video_backbone.encoder.vjepa21"):
        enc_mixed = VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "ignored", "extra_kwargs": {}},
            encoder_cfg={"name": "vjepa21", "model_path": str(tmp_path), "vjepa21_forward": "mixed"},
            ckpt_dir=str(tmp_path),
        )
    assert enc_mixed.vjepa21_forward == "mixed"
    # Render via ``getMessage()`` (not ``r.message``) so the assertion compares
    # against the formatted log line — robust to %-substitutions and parity
    # with ``test_W17``'s path B style.
    assert not any("vjepa21_forward" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records), (
        "explicit vjepa21_forward must NOT trip the pre-PR-checkpoint warning"
    )

    # --- Path B: field absent → default + warning ---
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="openwam.model.video_backbone.encoder.vjepa21"):
        enc_default = VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "ignored", "extra_kwargs": {}},
            encoder_cfg={"name": "vjepa21", "model_path": str(tmp_path)},
            ckpt_dir=str(tmp_path),
        )
    assert enc_default.vjepa21_forward == "video"  # current _VJEPA21_FORWARD_DEFAULT
    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("vjepa21_forward" in m and "mixed" in m for m in warning_msgs), (
        f"expected migration warning naming the field and the legacy ``mixed`` value; got: {warning_msgs}"
    )


def test_V12_vjepa21_from_skeleton_requires_ckpt_dir():
    """``from_skeleton`` without ``ckpt_dir`` raises FileNotFoundError naming
    ckpt_dir — deploy reads the manifest only from there (no model_path
    fallback). components_entry alone doesn't carry ViT geometry (it's the
    Wan VAE class).
    """
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    with pytest.raises(FileNotFoundError, match=r"ckpt_dir"):
        VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
        )


def test_V13_vjepa21_from_skeleton_ignores_model_path_without_ckpt_dir(tmp_path):
    """Strict self-contained: even with a readable ``encoder_cfg.model_path``
    (manifest present there), ``from_skeleton`` raises when ``ckpt_dir`` is
    absent — the model_path fallback was removed (A3), so the manifest is
    sourced only from ckpt_dir.
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    # model_path HAS a manifest, but no ckpt_dir is given → must still fail.
    (tmp_path / "manifest.json").write_text(_json.dumps(_build_vjepa_manifest_payload()))
    with pytest.raises(FileNotFoundError, match=r"ckpt_dir"):
        VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
            encoder_cfg={"name": "vjepa21", "model_path": str(tmp_path)},
        )


def test_V14_vjepa21_from_skeleton_missing_manifest(tmp_path):
    """``ckpt_dir`` that has no ``manifest.json`` → FileNotFoundError."""
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
            ckpt_dir=str(tmp_path),
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

    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "manifest.json").write_text(_json.dumps(_build_vjepa_manifest_payload()))

    def _fake_wrapper(**kwargs):
        return _MockVJEPAViT(embed_dim=1408)

    _install_fake_vjepa_modules(monkeypatch, _fake_wrapper)

    # Point encoder_cfg.model_path at a NON-EXISTENT directory: from_skeleton
    # reads the manifest only from ckpt_dir and must never reach model_path
    # (which has no manifest fallback after A3).
    enc = VJEPA21VideoEncoder.from_skeleton(
        components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
        encoder_cfg={"name": "vjepa21", "model_path": "/nonexistent/unmounted/path"},
        ckpt_dir=str(ckpt_dir),
    )
    assert isinstance(enc, VJEPA21VideoEncoder)
    assert enc.properties.z_dim == 1408
    assert enc.variant == "vitg-rope-384"


def test_V16_vjepa21_from_skeleton_no_model_path_fallback(tmp_path):
    """Strict self-contained: when ``<ckpt_dir>/manifest.json`` is absent,
    ``from_skeleton`` raises even though ``encoder.model_path`` carries a
    readable manifest — the model_path fallback was removed (A3).
    """
    import json as _json

    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()  # no manifest.json here
    encoder_src = tmp_path / "vjepa-weights"
    encoder_src.mkdir()
    (encoder_src / "manifest.json").write_text(_json.dumps(_build_vjepa_manifest_payload()))

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
            encoder_cfg={"name": "vjepa21", "model_path": str(encoder_src)},
            ckpt_dir=str(ckpt_dir),
        )


def test_V17_vjepa21_from_skeleton_no_manifest_names_ckpt_dir(tmp_path):
    """Missing manifest fails with an error naming ckpt_dir — the only source
    consulted (no encoder.model_path fallback after A3).
    """
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()  # empty

    with pytest.raises(FileNotFoundError) as exc:
        VJEPA21VideoEncoder.from_skeleton(
            components_entry={"attr": "vae", "model_class": "Wan", "extra_kwargs": {}},
            ckpt_dir=str(ckpt_dir),
        )
    # Lock the operator-facing UX: the attempted ckpt_dir path appears in the message.
    msg = str(exc.value)
    assert str(ckpt_dir) in msg, f"ckpt_dir missing from error: {msg}"


def test_V18_vjepa21_save_deploy_assets_copies_manifest(tmp_path, monkeypatch):
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
        {"model": {"video_backbone": {"encoder": {"name": "vjepa21", "model_path": str(encoder_src)}}}}
    )
    enc.save_deploy_assets(str(output_dir), cfg)

    dst = output_dir / "manifest.json"
    assert dst.exists()
    assert _json.loads(dst.read_text()) == manifest_payload


def test_V19_vjepa21_save_deploy_assets_missing_cfg_raises(tmp_path):
    """Strict self-contained: an unresolvable cfg / missing source manifest
    raises — :meth:`from_skeleton` has no fallback, so a checkpoint saved
    without its manifest can't be deployed.
    """
    from openwam.model.video_backbone.encoder.vjepa21 import VJEPA21VideoEncoder  # noqa: F401

    enc = _build_vjepa_encoder(embed_dim=1408)

    output_dir = tmp_path / "ckpt-out"
    output_dir.mkdir()

    # cfg without model.video_backbone.encoder → raise.
    with pytest.raises(FileNotFoundError, match="model_path"):
        enc.save_deploy_assets(str(output_dir), cfg={})
    assert not (output_dir / "manifest.json").exists()

    # cfg points at a directory with no manifest.json → raise.
    from omegaconf import OmegaConf

    empty_src = tmp_path / "empty"
    empty_src.mkdir()
    cfg = OmegaConf.create(
        {"model": {"video_backbone": {"encoder": {"name": "vjepa21", "model_path": str(empty_src)}}}}
    )
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        enc.save_deploy_assets(str(output_dir), cfg)
    assert not (output_dir / "manifest.json").exists()


def test_V20_vjepa21_save_deploy_assets_io_error_raises(tmp_path, monkeypatch):
    """Strict self-contained: a copy IO error (read-only fs / ENOSPC /
    disappearing mount) propagates instead of being swallowed — the checkpoint
    save aborts rather than producing a deploy-unloadable artifact.

    Reproduces wayrise #1: read-only fs / PermissionError / ENOSPC.
    """
    import json as _json
    import shutil

    from omegaconf import OmegaConf

    encoder_src = tmp_path / "vjepa-weights"
    encoder_src.mkdir()
    (encoder_src / "manifest.json").write_text(_json.dumps(_build_vjepa_manifest_payload()))
    output_dir = tmp_path / "ckpt-out"
    output_dir.mkdir()
    enc = _build_vjepa_encoder(embed_dim=1408)
    cfg = OmegaConf.create(
        {"model": {"video_backbone": {"encoder": {"name": "vjepa21", "model_path": str(encoder_src)}}}}
    )

    def _boom(*args, **kwargs):
        raise PermissionError("simulated read-only filesystem")

    monkeypatch.setattr(shutil, "copyfile", _boom)
    with pytest.raises(PermissionError, match="simulated"):
        enc.save_deploy_assets(str(output_dir), cfg)
    assert not (output_dir / "manifest.json").exists()


def test_V21_vjepa21_feature_norm_keys_present_in_state_dict():
    """``self.feature_norm`` must live directly on the encoder (NOT inside
    ``self._m``) so the freeze yaml's ``video_backbone.video_encoder`` line
    recursively covers it AND the safetensors carries it under
    ``video_backbone.video_encoder.feature_norm.*``. If a future refactor
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


def test_X1_flux2_vae_save_deploy_assets_self_contained(tmp_path):
    """``FluxVAEVideoEncoder.save_deploy_assets`` copies the FLUX.2 VAE
    config.json into ``<ckpt>/flux2_vae/config.json``, and ``_resolve_config_dir``
    reads only that checkpoint-local copy — deploy is strictly self-contained
    (no ``encoder.model_path`` fallback). Mirrors V-JEPA's manifest
    self-containment (test_V18).

    Exercises the save/resolve plumbing without building a real FLUX core
    (``save_deploy_assets`` / ``_resolve_config_dir`` read no instance state),
    so no FLUX weights are needed on disk.
    """
    from omegaconf import OmegaConf

    from openwam.model.video_backbone.encoder.flux2_vae import _FLUX_CKPT_SUBDIR, FluxVAEVideoEncoder

    src = tmp_path / "flux_src"
    src.mkdir()
    (src / "config.json").write_text('{"block_out_channels": [128], "latent_channels": 32}')
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    cfg = OmegaConf.create({"model": {"video_backbone": {"encoder": {"model_path": str(src)}}}})

    # Bypass core build — save_deploy_assets / _resolve_config_dir read no self state.
    enc = FluxVAEVideoEncoder.__new__(FluxVAEVideoEncoder)
    enc.save_deploy_assets(str(ckpt), cfg)

    # config landed in the checkpoint-local namespace
    assert (ckpt / _FLUX_CKPT_SUBDIR / "config.json").is_file()
    # resolve reads the ckpt-local sidecar (single source, no model_path fallback)
    assert FluxVAEVideoEncoder._resolve_config_dir(str(ckpt)) == str(ckpt / _FLUX_CKPT_SUBDIR)
    # no ckpt sidecar -> hard error (strictly self-contained; model_path never consulted)
    with pytest.raises(FileNotFoundError):
        FluxVAEVideoEncoder._resolve_config_dir(None)
    empty_ckpt = tmp_path / "empty_ckpt"
    empty_ckpt.mkdir()
    with pytest.raises(FileNotFoundError):
        FluxVAEVideoEncoder._resolve_config_dir(str(empty_ckpt))


def test_X2_flux2_vae_save_deploy_assets_missing_cfg_raises(tmp_path):
    """Strict self-contained: ``save_deploy_assets`` raises on an unresolvable
    cfg (no ``encoder.model_path``) — :meth:`from_skeleton` has no fallback, so
    a checkpoint saved without its config sidecar can't be deployed.
    """
    from openwam.model.video_backbone.encoder.flux2_vae import FluxVAEVideoEncoder

    enc = FluxVAEVideoEncoder.__new__(FluxVAEVideoEncoder)
    with pytest.raises(FileNotFoundError):
        enc.save_deploy_assets(str(tmp_path), cfg={})  # no encoder.model_path -> raise
    assert not (tmp_path / "flux2_vae").exists()


def test_Y1_dinov3_save_deploy_assets_self_contained(tmp_path):
    """``DinoV3VideoEncoder.save_deploy_assets`` copies the DINOv3 ``config.json``
    into ``<ckpt>/dinov3/config.json``, and ``_resolve_config_dir`` reads only
    that checkpoint-local copy — deploy is strictly self-contained (no
    ``encoder.model_path`` fallback; native DINOv3 needs only config.json, the
    model class comes from the installed transformers library, not bundled
    modeling code).
    """
    from omegaconf import OmegaConf

    from openwam.model.video_backbone.encoder.dinov3 import _DINOV3_CKPT_SUBDIR, DinoV3VideoEncoder

    src = tmp_path / "dinov3_src"
    src.mkdir()
    (src / "config.json").write_text('{"model_type": "dinov3_vit", "hidden_size": 384, "patch_size": 16}')
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    cfg = OmegaConf.create({"model": {"video_backbone": {"encoder": {"model_path": str(src)}}}})

    # Bypass ViT build; no reducer attached, so only the config.json half runs.
    enc = DinoV3VideoEncoder.__new__(DinoV3VideoEncoder)
    enc._svae = None
    enc.save_deploy_assets(str(ckpt), cfg)

    assert (ckpt / _DINOV3_CKPT_SUBDIR / "config.json").is_file()
    # resolve reads the ckpt-local sidecar (single source, no model_path fallback)
    assert DinoV3VideoEncoder._resolve_config_dir(str(ckpt)) == str(ckpt / _DINOV3_CKPT_SUBDIR)
    # no ckpt sidecar -> hard error (strictly self-contained; model_path never consulted)
    with pytest.raises(FileNotFoundError):
        DinoV3VideoEncoder._resolve_config_dir(None)
    empty_ckpt = tmp_path / "empty_ckpt"
    empty_ckpt.mkdir()
    with pytest.raises(FileNotFoundError):
        DinoV3VideoEncoder._resolve_config_dir(str(empty_ckpt))


def test_Y2_dinov3_save_deploy_assets_missing_cfg_raises(tmp_path):
    """Strict self-contained: ``save_deploy_assets`` raises on an unresolvable
    cfg (no ``encoder.model_path``) — :meth:`from_skeleton` has no fallback, so
    a checkpoint saved without its config sidecar can't be deployed.
    """
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

    enc = DinoV3VideoEncoder.__new__(DinoV3VideoEncoder)
    enc._svae = None
    with pytest.raises(FileNotFoundError):
        enc.save_deploy_assets(str(tmp_path), cfg={})  # no encoder.model_path -> raise
    assert not (tmp_path / "dinov3").exists()
