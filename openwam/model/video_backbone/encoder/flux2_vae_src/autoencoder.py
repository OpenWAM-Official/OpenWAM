# Source: https://github.com/black-forest-labs/flux2/blob/main/src/flux2/autoencoder.py
# Upstream revision: UNKNOWN (the original internal import did not record a commit SHA).
# License: Apache-2.0; see the repository-level LICENSE (Apache-2.0).
# Modified by OpenWAM contributors: decoder functionality was removed and
# OpenWAM checkpoint conversion/encoder integration was added.

"""FLUX.2 VAE encoder — extracted, decoder-free.

Only the *encoder* half of the FLUX.2-dev VAE is vendored here, lifted verbatim
(module-for-module) from ``references/flux2/src/flux2/autoencoder.py`` so the
weight tensors load by name. The pixel ``Decoder`` / ``Upsample`` are
deliberately omitted — this subsystem consumes the VAE purely as a frozen
image→latent feature extractor (``properties.pixel_decode=False``), so carrying the
decoder would only burn memory.

The on-disk checkpoint at the configured ``model_path`` is in **diffusers
``AutoencoderKLFlux2`` layout** (``encoder.down_blocks.*.resnets.*`` etc.), while
the vendored class uses the BFL-native layout (``encoder.down.*.block.*``). The
two are the *same* network (block_out_channels [128,256,512,512], z_channels=32,
2×2 pixel-shuffle pack, GroupNorm-32, mid self-attention, a final non-affine
BatchNorm whitening) — only the parameter names differ.
:func:`convert_diffusers_encoder_sd` bridges them.

``FluxVaeEncoderCore.encode`` reproduces ``AutoEncoder.encode``'s deterministic
branch: take the distribution mean, pack 2×2 spatial into channels (→ z_dim=128,
spatial_compression=16), then apply the VAE's built-in per-channel BatchNorm
whitening. That BatchNorm is the FLUX.2 analogue of Wan VAE's per-channel
z-score ``(mu - mean) / std`` (see ``wan/vae.py`` ``WanVideoVAE.encode``); it is
why this encoder needs no extra LayerNorm on top (contrast ``dinov3.py``, whose
ViT features have no such built-in whitening).
"""

from __future__ import annotations

import math
import re

import torch
from einops import rearrange
from torch import Tensor, nn

# ----------------------------------------------------------------------
# Vendored FLUX.2 VAE *encoder* modules (BFL-native layout).
# Lifted from references/flux2/src/flux2/autoencoder.py without behavioural
# change so the converted state-dict loads strictly by name.
# ----------------------------------------------------------------------


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class FluxVaeEncoderCore(nn.Module):
    """FLUX.2 VAE encode path (decoder-free), reproducing ``AutoEncoder.encode``.

    Holds the vendored :class:`Encoder` plus the non-affine BatchNorm whitening
    that ``references/flux2/autoencoder.py`` keeps on the ``AutoEncoder`` and the
    2×2 pixel-shuffle pack factor ``ps``. ``encode`` returns the deterministic
    (mode) latent, z-score-whitened per channel — z_dim = ``z_channels *
    prod(ps)`` = 128, spatial_compression = ``2**(len(ch_mult)-1) * ps`` = 16.
    """

    def __init__(
        self,
        *,
        resolution: int = 256,
        in_channels: int = 3,
        ch: int = 128,
        ch_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        z_channels: int = 32,
        ps: tuple[int, int] = (2, 2),
        bn_eps: float = 1e-4,
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        if ch_mult is None:
            ch_mult = [1, 2, 4, 4]
        self.encoder = Encoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        self.ps = list(ps)
        # spatial_compression is a single scalar (VideoEncoderProperties), so the pack
        # must be square — otherwise H and W would compress by different factors
        # and the scalar would silently describe only one axis.
        if self.ps[0] != self.ps[1]:
            raise ValueError(f"FluxVaeEncoderCore assumes a square pixel-shuffle pack; got ps={tuple(self.ps)}.")
        self.z_channels = z_channels
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * z_channels,
            eps=bn_eps,
            momentum=bn_momentum,
            affine=False,
            track_running_stats=True,
        )
        # Derived latent geometry — consumed by the wrapper's spec.
        self.z_dim = math.prod(self.ps) * z_channels
        self.spatial_compression = (2 ** (len(ch_mult) - 1)) * self.ps[0]

    def normalize(self, z: Tensor) -> Tensor:
        # The whitening BatchNorm is a *fixed* per-channel z-score: affine=False,
        # running stats loaded from the VAE checkpoint. Pin it to eval() on every
        # call so it always whitens with those frozen stats and never updates them
        # or uses batch stats — even when the parent module is in train() mode.
        # This is deliberate (frozen calibration, not a trainable layer).
        self.bn.eval()
        return self.bn(z)

    def encode(self, x: Tensor) -> Tensor:
        """``(B, 3, H, W) -> (B, z_dim, H/spatial, W/spatial)`` whitened latent."""
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z


# ----------------------------------------------------------------------
# diffusers AutoencoderKLFlux2 -> vendored BFL Encoder state-dict converter.
# ----------------------------------------------------------------------


def _reshape_linear_to_conv1x1(t: Tensor) -> Tensor:
    """diffusers ``Attention`` uses ``Linear`` (``[C, C]``) where the BFL block
    uses ``Conv2d(.., 1)`` (``[C, C, 1, 1]``). Biases (``[C]``) are unchanged."""
    if t.dim() == 2:
        return t[:, :, None, None]
    return t


def convert_resnet(sd: dict, src: str, dst: str, out: dict) -> None:
    """Map a diffusers resnet (``src.``) onto a BFL ResnetBlock (``dst.``)."""
    for a, b in (("norm1", "norm1"), ("conv1", "conv1"), ("norm2", "norm2"), ("conv2", "conv2")):
        for suffix in ("weight", "bias"):
            out[f"{dst}.{b}.{suffix}"] = sd[f"{src}.{a}.{suffix}"]
    # diffusers ``conv_shortcut`` == BFL ``nin_shortcut`` (only on channel-change blocks).
    if f"{src}.conv_shortcut.weight" in sd:
        for suffix in ("weight", "bias"):
            out[f"{dst}.nin_shortcut.{suffix}"] = sd[f"{src}.conv_shortcut.{suffix}"]


def convert_attn(sd: dict, src: str, dst: str, out: dict) -> None:
    """Map a diffusers VAE ``Attention`` (``src.``) onto a BFL AttnBlock (``dst.``)."""
    for suffix in ("weight", "bias"):
        out[f"{dst}.norm.{suffix}"] = sd[f"{src}.group_norm.{suffix}"]
    for a, b in (("to_q", "q"), ("to_k", "k"), ("to_v", "v"), ("to_out.0", "proj_out")):
        out[f"{dst}.{b}.weight"] = _reshape_linear_to_conv1x1(sd[f"{src}.{a}.weight"])
        out[f"{dst}.{b}.bias"] = sd[f"{src}.{a}.bias"]


def _infer_encoder_shape(sd: dict) -> tuple[int, int]:
    """Read ``(num_resolutions, num_res_blocks)`` straight off the diffusers sd.

    Deriving the structure from the keys actually present (rather than trusting
    caller-supplied counts) means a mismatched count can't silently skip — and
    thus drop — whole down-blocks / resnets during conversion.
    """
    levels = {int(m.group(1)) for k in sd if (m := re.match(r"encoder\.down_blocks\.(\d+)\.", k))}
    if not levels:
        raise KeyError("no 'encoder.down_blocks.*' keys: this is not an AutoencoderKLFlux2 encoder state-dict.")
    blocks = {int(m.group(1)) for k in sd if (m := re.match(r"encoder\.down_blocks\.0\.resnets\.(\d+)\.", k))}
    return len(levels), len(blocks)


def convert_diffusers_encoder_sd(sd: dict) -> dict:
    """diffusers ``AutoencoderKLFlux2`` state-dict -> vendored encoder+bn state-dict.

    Returns only the keys consumed by :class:`FluxVaeEncoderCore`
    (``encoder.*`` and ``bn.*``); ``decoder.*`` / ``post_quant_conv.*`` are
    dropped. Pass the *full* checkpoint dict — extra keys are ignored. The
    ``down_blocks`` / ``resnets`` fan-out is inferred from ``sd`` itself, so the
    mapping always matches the checkpoint's actual depth.
    """
    num_resolutions, num_res_blocks = _infer_encoder_shape(sd)
    out: dict = {}

    # conv_in / conv_out / final norm.
    for suffix in ("weight", "bias"):
        out[f"encoder.conv_in.{suffix}"] = sd[f"encoder.conv_in.{suffix}"]
        out[f"encoder.conv_out.{suffix}"] = sd[f"encoder.conv_out.{suffix}"]
        out[f"encoder.norm_out.{suffix}"] = sd[f"encoder.conv_norm_out.{suffix}"]

    # down blocks: diffusers down_blocks[i] == BFL down[i] (same order).
    for i in range(num_resolutions):
        for j in range(num_res_blocks):
            convert_resnet(
                sd,
                f"encoder.down_blocks.{i}.resnets.{j}",
                f"encoder.down.{i}.block.{j}",
                out,
            )
        # downsampler present on all but the last resolution.
        if f"encoder.down_blocks.{i}.downsamplers.0.conv.weight" in sd:
            for suffix in ("weight", "bias"):
                out[f"encoder.down.{i}.downsample.conv.{suffix}"] = sd[
                    f"encoder.down_blocks.{i}.downsamplers.0.conv.{suffix}"
                ]

    # mid block: resnets.0/1 -> block_1/block_2, attentions.0 -> attn_1.
    convert_resnet(sd, "encoder.mid_block.resnets.0", "encoder.mid.block_1", out)
    convert_resnet(sd, "encoder.mid_block.resnets.1", "encoder.mid.block_2", out)
    convert_attn(sd, "encoder.mid_block.attentions.0", "encoder.mid.attn_1", out)

    # quant_conv: top-level in diffusers, inside Encoder in BFL.
    for suffix in ("weight", "bias"):
        out[f"encoder.quant_conv.{suffix}"] = sd[f"quant_conv.{suffix}"]

    # BatchNorm whitening buffers (non-affine -> only running stats + counter).
    # These are the FLUX.2 VAE's per-channel whitening calibration; their absence
    # means the checkpoint isn't an AutoencoderKLFlux2 (e.g. a vanilla KL-VAE).
    for buf in ("running_mean", "running_var", "num_batches_tracked"):
        key = f"bn.{buf}"
        if key not in sd:
            raise KeyError(
                f"missing '{key}': this is not an AutoencoderKLFlux2 checkpoint (no BatchNorm whitening stats)."
            )
        out[key] = sd[key]

    return out


__all__ = [
    "Encoder",
    "FluxVaeEncoderCore",
    "convert_diffusers_encoder_sd",
    "convert_resnet",
    "convert_attn",
]
