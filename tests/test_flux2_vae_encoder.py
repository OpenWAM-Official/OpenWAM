"""FLUX.2 VAE video encoder tests.

Covers ``openwam/model/video_backbone/encoder/flux2_vae.py`` and the
diffusers->BFL state-dict converter in ``flux/flux2_vae_src.py``:

- F1   registry round-trip (``register_video_encoder("flux2_vae")``)
- F2   spec invariants (z_dim/spatial from the loaded core, fixed temporal=4,
       causal=True, dit_patch_size=(1,2,2), pixel_decode=False)
- F3   preprocess shape + [-1, 1] range (matches Wan VAE, not ImageNet)
- F4   ``batch_encode`` shape contract on T_pixel ∈ {1, 5, 9}
- F5   causal mean pool semantics through the wrapper (frame 0 verbatim,
       rest 4-grouped -> mean) — the NO-extra-LayerNorm path
- F6   fail-fasts: T ≢ 1 mod 4, H not divisible by spatial_compression
- F7   ``decode`` / ``to_frames`` raise NotImplementedError (irreversible)
- F8   converter round-trip: a synthetic diffusers state-dict converts to a
       dict that strict-loads into ``FluxVaeEncoderCore`` (key-mapping +
       attention Linear->Conv2d reshape locked against drift)
- F9   default DiT hooks produce the Wan-parity Conv3d/Linear shapes
- F10  core rejects a non-square pixel-shuffle pack (spatial_compression scalar)
- F11  converter raises a friendly error when BatchNorm whitening stats are absent

All tests use a tiny CPU mock core (wrapper tests) or a tiny real core
(converter test) — no checkpoint load — so they run on the lint CPU image.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

# ---------------------------------------------------------------------------
# Mock encoder core (wrapper tests)
# ---------------------------------------------------------------------------


class _MockFluxCore(nn.Module):
    """Tiny stand-in for :class:`FluxVaeEncoderCore`.

    ``encode`` returns a per-sample-constant latent equal to the frame's mean
    pixel value, broadcast over ``(z_dim, H/sc, W/sc)``. That lets the wrapper's
    causal pool be eye-checked: a video whose frame ``t`` is the constant ``t+1``
    yields ``encode`` value ``t+1`` per frame, so pooled[0]==1 and
    pooled[1]==mean(2,3,4,5)==3.5.
    """

    def __init__(self, z_dim: int = 128, spatial_compression: int = 16):
        super().__init__()
        self.z_dim = int(z_dim)
        self.spatial_compression = int(spatial_compression)
        self._anchor = nn.Parameter(torch.zeros(1))

    def encode(self, x: Tensor) -> Tensor:
        bt, _c, h, w = x.shape
        sc = self.spatial_compression
        val = x.mean(dim=(1, 2, 3)).view(bt, 1, 1, 1)
        return val.expand(bt, self.z_dim, h // sc, w // sc).contiguous()


def _build_encoder(*, z_dim: int = 128, spatial_compression: int = 16):
    from openwam.model.video_backbone.encoder.flux2_vae import FluxVAEVideoEncoder

    return FluxVAEVideoEncoder(_MockFluxCore(z_dim=z_dim, spatial_compression=spatial_compression))


# ---------------------------------------------------------------------------
# F1: registration
# ---------------------------------------------------------------------------


def test_F1_flux2_vae_registration_round_trip():
    from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY
    from openwam.model.video_backbone.encoder.flux2_vae import FluxVAEVideoEncoder

    assert "flux2_vae" in _VIDEO_ENCODER_REGISTRY
    assert _VIDEO_ENCODER_REGISTRY["flux2_vae"] is FluxVAEVideoEncoder


# ---------------------------------------------------------------------------
# F2: spec invariants
# ---------------------------------------------------------------------------


def test_F2_flux2_vae_spec_invariants():
    """Token-count parity with Wan VAE depends on temporal=4, causal=True,
    dit_patch_size=(1,2,2); z_dim/spatial come from the loaded core."""
    enc = _build_encoder(z_dim=128, spatial_compression=16)
    properties = enc.properties
    assert properties.z_dim == 128
    assert properties.spatial_compression == 16
    assert properties.temporal_compression == 4
    assert properties.causal_temporal is True
    assert properties.dit_patch_size == (1, 2, 2)
    assert properties.pixel_decode is False


# ---------------------------------------------------------------------------
# F3: preprocess
# ---------------------------------------------------------------------------


def test_F3_flux2_vae_preprocess_minus_one_to_one():
    """``preprocess_video`` rescales [0,255] -> [-1,1] (Wan VAE range, not
    ImageNet): black -> -1, white -> +1, gray 128 -> ~0."""
    enc = _build_encoder()
    # PIL size is (W, H); np.array -> (H, W, C), so a (W=32, H=48) image
    # preprocesses to (1, 3, T, H=48, W=32).
    frames = [Image.new("RGB", (32, 48), color=c) for c in ((0, 0, 0), (128, 128, 128), (255, 255, 255))]
    video = enc.preprocess_video(frames)
    assert video.shape == (1, 3, 3, 48, 32)
    assert torch.allclose(video[0, :, 0], torch.full((3, 48, 32), -1.0))
    assert pytest.approx(video[0, 0, 1, 0, 0].item(), abs=1e-3) == 128 * 2 / 255 - 1
    assert torch.allclose(video[0, :, 2], torch.full((3, 48, 32), 1.0))


# ---------------------------------------------------------------------------
# F4: batch_encode shape contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T_pixel, T_lat", [(1, 1), (5, 2), (9, 3)])
def test_F4_flux2_vae_batch_encode_t_lat_shapes(T_pixel, T_lat):
    """``(B, 3, T, H, W) → (B, z_dim, 1 + (T-1)/4, H/16, W/16)``."""
    enc = _build_encoder(z_dim=128, spatial_compression=16)
    v = torch.randn(2, 3, T_pixel, 32, 48)
    z = enc.batch_encode(v)
    assert z.shape == (2, 128, T_lat, 32 // 16, 48 // 16)


# ---------------------------------------------------------------------------
# F5: causal pool semantics through the wrapper (no extra LayerNorm)
# ---------------------------------------------------------------------------


def test_F5_flux2_vae_causal_pool_keeps_frame0_and_means_rest():
    """Frame ``t`` constant == ``t+1`` -> pooled[0]==1, pooled[1]==mean(2,3,4,5)==3.5.
    The latent passes through with no extra normalization (the VAE's bn lives
    inside ``core.encode``, mocked away here), so values survive verbatim."""
    enc = _build_encoder(z_dim=4, spatial_compression=16)
    # video[:, :, t] is the constant (t+1) across C,H,W.
    frames = torch.arange(1, 6, dtype=torch.float32).view(1, 1, 5, 1, 1)
    v = frames.expand(1, 3, 5, 32, 32).contiguous()
    z = enc.batch_encode(v)
    assert z.shape == (1, 4, 2, 2, 2)
    assert torch.allclose(z[0, :, 0], torch.ones(4, 2, 2))
    assert torch.allclose(z[0, :, 1], torch.full((4, 2, 2), 3.5))


# ---------------------------------------------------------------------------
# F6: fail-fasts
# ---------------------------------------------------------------------------


def test_F6a_flux2_vae_T_not_1_mod_4_raises():
    enc = _build_encoder()
    with pytest.raises(ValueError, match=r"T ≡ 1 \(mod 4\)"):
        enc.batch_encode(torch.randn(1, 3, 4, 32, 32))  # (4-1)%4 = 3 ≠ 0


def test_F6b_flux2_vae_H_not_divisible_by_spatial_raises():
    enc = _build_encoder(spatial_compression=16)
    with pytest.raises(ValueError, match=r"divisible by spatial_compression"):
        enc.batch_encode(torch.randn(1, 3, 5, 40, 32))  # 40 % 16 != 0


# ---------------------------------------------------------------------------
# F7: irreversible
# ---------------------------------------------------------------------------


def test_F7_flux2_vae_decode_and_to_frames_raise():
    enc = _build_encoder()
    with pytest.raises(NotImplementedError):
        enc.decode(torch.randn(1, 128, 1, 2, 2))
    with pytest.raises(NotImplementedError):
        enc.to_frames(torch.randn(1, 3, 1, 16, 16))


# ---------------------------------------------------------------------------
# F8: converter round-trip (key mapping + attention reshape)
# ---------------------------------------------------------------------------


def _bfl_key_to_diffusers(key: str) -> str:
    """Inverse of the converter's name map (independent spelling, for the test)."""
    if key.startswith("bn."):
        return key
    if key == "encoder.norm_out.weight":
        return "encoder.conv_norm_out.weight"
    if key == "encoder.norm_out.bias":
        return "encoder.conv_norm_out.bias"
    if key.startswith("encoder.quant_conv."):
        return key[len("encoder.") :]  # top-level quant_conv.*
    k = key
    k = k.replace("encoder.mid.block_1", "encoder.mid_block.resnets.0")
    k = k.replace("encoder.mid.block_2", "encoder.mid_block.resnets.1")
    k = k.replace("encoder.mid.attn_1.norm", "encoder.mid_block.attentions.0.group_norm")
    k = k.replace("encoder.mid.attn_1.proj_out", "encoder.mid_block.attentions.0.to_out.0")
    k = k.replace("encoder.mid.attn_1.q", "encoder.mid_block.attentions.0.to_q")
    k = k.replace("encoder.mid.attn_1.k", "encoder.mid_block.attentions.0.to_k")
    k = k.replace("encoder.mid.attn_1.v", "encoder.mid_block.attentions.0.to_v")
    import re

    k = re.sub(r"encoder\.down\.(\d+)\.block\.(\d+)", r"encoder.down_blocks.\1.resnets.\2", k)
    k = re.sub(r"encoder\.down\.(\d+)\.downsample\.conv", r"encoder.down_blocks.\1.downsamplers.0.conv", k)
    k = k.replace("nin_shortcut", "conv_shortcut")
    return k


def test_F8_converter_round_trips_into_core():
    """A synthetic diffusers state-dict (built by inverse-mapping a tiny real
    core's keys, with attention weights squeezed to 2D) must convert back to a
    dict that strict-loads into the core."""
    from openwam.model.video_backbone.encoder.flux2_vae_src import (
        FluxVaeEncoderCore,
        convert_diffusers_encoder_sd,
    )

    # ch must be a multiple of 32 (GroupNorm-32). Tiny 2-resolution config.
    core = FluxVaeEncoderCore(ch=32, ch_mult=[1, 2], z_channels=4, num_res_blocks=1)
    core_sd = core.state_dict()

    diff_sd: dict = {}
    for k, v in core_sd.items():
        dk = _bfl_key_to_diffusers(k)
        # diffusers VAE attention uses Linear ([C,C]); BFL uses Conv2d ([C,C,1,1]).
        if dk.startswith("encoder.mid_block.attentions.0.to_") and dk.endswith(".weight"):
            v = v.squeeze(-1).squeeze(-1)
        diff_sd[dk] = v

    converted = convert_diffusers_encoder_sd(diff_sd)
    assert set(converted) == set(core_sd), (
        f"missing={set(core_sd) - set(converted)} extra={set(converted) - set(core_sd)}"
    )
    # strict load proves shapes (incl. attention re-expanded to 4D) are right.
    fresh = FluxVaeEncoderCore(ch=32, ch_mult=[1, 2], z_channels=4, num_res_blocks=1)
    fresh.load_state_dict(converted, strict=True)


# ---------------------------------------------------------------------------
# F9: default DiT hooks produce Wan-parity shapes
# ---------------------------------------------------------------------------


def test_F9_flux2_vae_default_hooks_wan_parity():
    enc = _build_encoder(z_dim=128, spatial_compression=16)
    inp = enc.build_dit_input_proj(dit_dim=1536)
    assert isinstance(inp, nn.Conv3d)
    assert (inp.in_channels, inp.out_channels) == (128, 1536)
    assert inp.kernel_size == (1, 2, 2)
    assert inp.stride == (1, 2, 2)

    out = enc.build_dit_output_proj(dit_dim=1536)
    assert isinstance(out, nn.Linear)
    assert out.in_features == 1536
    assert out.out_features == 128 * 4  # z_dim * prod(dit_patch_size)


# ---------------------------------------------------------------------------
# F10/F11: core + converter input-validation fail-fasts
# ---------------------------------------------------------------------------


def test_F10_core_rejects_non_square_pack():
    """spatial_compression is a single scalar, so a non-square pack (ps[0] !=
    ps[1]) would silently describe only one axis — the core must reject it."""
    from openwam.model.video_backbone.encoder.flux2_vae_src import FluxVaeEncoderCore

    with pytest.raises(ValueError, match=r"square pixel-shuffle pack"):
        FluxVaeEncoderCore(ch=32, ch_mult=[1, 2], z_channels=4, num_res_blocks=1, ps=(2, 4))


def test_F11_converter_rejects_missing_bn_stats():
    """A checkpoint without the FLUX.2 BatchNorm whitening stats is not an
    AutoencoderKLFlux2; the converter must say so instead of a bare KeyError."""
    from openwam.model.video_backbone.encoder.flux2_vae_src import (
        FluxVaeEncoderCore,
        convert_diffusers_encoder_sd,
    )

    core = FluxVaeEncoderCore(ch=32, ch_mult=[1, 2], z_channels=4, num_res_blocks=1)
    diff_sd = {}
    for k, v in core.state_dict().items():
        dk = _bfl_key_to_diffusers(k)
        if dk.startswith("encoder.mid_block.attentions.0.to_") and dk.endswith(".weight"):
            v = v.squeeze(-1).squeeze(-1)
        diff_sd[dk] = v
    # Drop the whitening stats — e.g. a vanilla KL-VAE checkpoint.
    for buf in ("running_mean", "running_var", "num_batches_tracked"):
        diff_sd.pop(f"bn.{buf}")

    with pytest.raises(KeyError, match=r"AutoencoderKLFlux2"):
        convert_diffusers_encoder_sd(diff_sd)
