"""Wan2.2 VAE encode (``VideoVAE38_``) over the whole clip in one pass (``OPENWAM_VAE_FULLCLIP=1``).

``VideoVAE38_.encode`` runs the causal encoder chunk by chunk (the first
frame, then 4-frame chunks), carrying the last frames of every causal conv's input
in ``feat_cache`` and concatenating them in front of the next chunk. For a clip of
``1 + 4k`` frames (``k >= 1``) that is the same function as running every layer
once on the whole clip. A single frame stays on the original chunked encoder:
its temporal downsamplers pass that frame through without running ``time_conv``.
For multi-frame clips:

- ``CausalConv3d`` (temporal kernel 3): chunk ``i`` sees ``[cache (last 2 input
  frames, or 1 after the 1-frame first chunk, zero-padded to 2), chunk]``, i.e. each
  output frame reads input frames ``t-2, t-1, t`` with zeros before frame 0: the
  whole-clip causal conv with 2 zero frames in front (its own padding);
- ``Resample38`` ``downsample3d``: the first chunk's frame passes through unchanged
  and every later chunk runs the stride-2 ``time_conv`` on ``[last frame of the
  previous chunk, chunk]``: windows starting at frames 0, 2, 4, ... of the whole clip,
  i.e. ``cat([x[:, :, :1], time_conv(x)])``;
- ``AvgDown3D`` with temporal factor 2 pads one zero frame in front of a 1-frame
  chunk and none in front of the even-length chunks: the same pairs ``(0, x0)``,
  ``(x1, x2)``, ... as padding the whole ``1 + 2m``-frame clip once;
- the norms, SiLU, the per-frame 2-D convs and the per-frame attention are per frame.

So the result is the same real-valued function; the convolutions run with other
shapes (all frames at once), so cuDNN may pick other kernels and the rounding can
differ. ``OPENWAM_COMPILE_VAE=1`` also runs it through ``torch.compile`` (Inductor
fuses the RMS-norm / SiLU / padding chains between the cuDNN convolutions; no
gradient, frozen weights). ``OPENWAM_VAE_CL=1`` (with the compiled encode) runs the
convolutions channels-last, see below.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from einops import rearrange

logger = logging.getLogger(__name__)

_FULL = os.environ.get("OPENWAM_VAE_FULLCLIP", "0") == "1"
_COMPILE = os.environ.get("OPENWAM_COMPILE_VAE", "0") == "1"
_COMPILED = {}
# OPENWAM_VAE_CL=1 (with the compiled whole-clip path): run the convolutions channels-last. cuDNN's 3-D convolutions
# here are NDHWC kernels; on NCDHW tensors every call converts its input (nchwToNhwc) and its output (nhwcToNchw),
# and Inductor's 3-D convolution lowering keeps NCDHW (its layout optimization covers 2-D convolutions only). The
# convolutions are therefore wrapped as opaque custom ops that take and return channels-last tensors (Inductor keeps
# a custom op's input strides, so the fused norm/SiLU/padding kernel in front writes channels-last directly and the
# next one reads the channels-last output). Same convolutions on the same values; layouts only.
_CL = os.environ.get("OPENWAM_VAE_CL", "0") == "1"


def _out_len(n, k, s):
    return (n - k) // s + 1


@torch.library.custom_op("openwam::vae_conv3d_cl", mutates_args=())
def _conv3d_cl(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], stride: list[int]) -> torch.Tensor:
    import torch.nn.functional as F

    cl = torch.channels_last_3d
    out = F.conv3d(x.contiguous(memory_format=cl), weight.contiguous(memory_format=cl), bias, stride)
    return out.contiguous(memory_format=cl)


@_conv3d_cl.register_fake
def _(x, weight, bias, stride):
    n, _, d, h, w = x.shape
    o, _, kd, kh, kw = weight.shape
    shape = (n, o, _out_len(d, kd, stride[0]), _out_len(h, kh, stride[1]), _out_len(w, kw, stride[2]))
    return torch.empty(shape, dtype=x.dtype, device=x.device, memory_format=torch.channels_last_3d)


@torch.library.custom_op("openwam::vae_conv2d_cl", mutates_args=())
def _conv2d_cl(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], stride: list[int]) -> torch.Tensor:
    import torch.nn.functional as F

    cl = torch.channels_last
    out = F.conv2d(x.contiguous(memory_format=cl), weight.contiguous(memory_format=cl), bias, stride)
    return out.contiguous(memory_format=cl)


@_conv2d_cl.register_fake
def _(x, weight, bias, stride):
    n, _, h, w = x.shape
    o, _, kh, kw = weight.shape
    shape = (n, o, _out_len(h, kh, stride[0]), _out_len(w, kw, stride[1]))
    return torch.empty(shape, dtype=x.dtype, device=x.device, memory_format=torch.channels_last)


def _cconv(conv, x):
    """``CausalConv3d.forward(x)`` without a cache: causal zero padding, then the convolution."""
    if not _CL:
        return conv(x)
    import torch.nn.functional as F

    pad = list(conv._padding)
    xp = F.pad(x, pad) if any(pad) else x
    xp = xp.contiguous(memory_format=torch.channels_last_3d)
    y = torch.ops.openwam.vae_conv3d_cl(xp, conv.weight, None, list(conv.stride))
    # bias added outside the op (as Inductor does for its own convolutions) so it fuses into the next kernel
    return y if conv.bias is None else y + conv.bias.view(1, -1, 1, 1, 1)


def _seq(seq, x):
    from openwam.model.video_backbone.wan.models.vae import CausalConv3d

    for layer in seq:
        x = _cconv(layer, x) if isinstance(layer, CausalConv3d) else layer(x)
    return x


def _resblock(rb, x):
    from openwam.model.video_backbone.wan.models.vae import CausalConv3d

    h = _cconv(rb.shortcut, x) if isinstance(rb.shortcut, CausalConv3d) else rb.shortcut(x)
    x = _seq(rb.residual, x)
    return x + h


def _spatial_down(rs, y):
    """``rs.resample`` (ZeroPad2d((0, 1, 0, 1)) + stride-2 Conv2d) on ``(b t) c h w``."""
    if not _CL:
        return rs.resample(y)
    import torch.nn.functional as F

    pad, conv = rs.resample[0], rs.resample[1]
    yp = F.pad(y, list(pad.padding)).contiguous(memory_format=torch.channels_last)
    out = torch.ops.openwam.vae_conv2d_cl(yp, conv.weight, None, list(conv.stride))
    return out if conv.bias is None else out + conv.bias.view(1, -1, 1, 1)


def _resample(rs, x):
    t = x.shape[2]
    y = rearrange(x, "b c t h w -> (b t) c h w")
    y = _spatial_down(rs, y)
    y = rearrange(y, "(b t) c h w -> b c t h w", t=t)
    if rs.mode == "downsample3d":
        y = torch.cat([y[:, :, :1], _cconv(rs.time_conv, y)], dim=2)
    elif rs.mode != "downsample2d":
        raise NotImplementedError(rs.mode)
    return y


def _down_block(db, x):
    from openwam.model.video_backbone.wan.models.vae import ResidualBlock

    x_copy = x
    for m in db.downsamples:
        x = _resblock(m, x) if isinstance(m, ResidualBlock) else _resample(m, x)
    return x + db.avg_shortcut(x_copy)


def _encoder_full(enc, x):
    from openwam.model.video_backbone.wan.models.vae import ResidualBlock

    x = _cconv(enc.conv1, x)
    for db in enc.downsamples:
        x = _down_block(db, x)
    for layer in enc.middle:
        x = _resblock(layer, x) if isinstance(layer, ResidualBlock) else layer(x)
    return _seq(enc.head, x)


def _encode_full(model, x, mean, inv_std):
    from openwam.model.video_backbone.wan.models.vae import patchify

    x = patchify(x, patch_size=2)
    out = _encoder_full(model.encoder, x)
    mu, _log_var = _cconv(model.conv1, out).chunk(2, dim=1)
    return (mu - mean.view(1, model.z_dim, 1, 1, 1)) * inv_std.view(1, model.z_dim, 1, 1, 1)


def supported(model, x) -> bool:
    from openwam.model.video_backbone.wan.models.vae import Encoder3d_38, VideoVAE38_

    return (
        _FULL
        and isinstance(model, VideoVAE38_)
        and isinstance(model.encoder, Encoder3d_38)
        and x.dim() == 5
        and x.shape[2] > 1
        and (x.shape[2] - 1) % 4 == 0
        and not torch.is_grad_enabled()
    )


def encode(model, x, scale):
    """``model.encode(x, scale)`` (``VideoVAE38_``) over the whole clip; see the module doc."""
    from openwam.model.video_backbone.wan.models.vae import _scale_on_device

    mean, inv_std = _scale_on_device(model, scale, x.dtype, x.device)
    fn = _encode_full
    if _COMPILE:
        fn = _COMPILED.get("encode")
        if fn is None:
            fn = torch.compile(_encode_full, dynamic=False)
            _COMPILED["encode"] = fn
    # The compiled graph may hand back the latents in another memory layout (channels-last from the convolutions).
    # Return them in the standard contiguous layout of VideoVAE38_.encode: torch.randn_like(input_latents) follows
    # the memory layout, so another layout would assign the same random draws to different latent elements.
    return fn(model, x, mean, inv_std).contiguous()
