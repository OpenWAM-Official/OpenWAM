"""FLUX.2 VAE video feature encoder.

The FLUX.2-dev image VAE (2D, no temporal axis) applied frame-by-frame, followed
by a parameter-free causal mean pool along time to mirror Wan VAE's
``T_lat = 1 + (T - 1) / 4`` semantics. Token counts match the Wan VAE path on the
same input frames because the FLUX.2 VAE compresses space by 16
(``spatial_compression=16``, identical to Wan2.2-TI2V-5B) and this encoder reuses
the Wan DiT's native ``dit_patch_size=(1, 2, 2)``.

Shape contract (RoboTwin default, H=384, W=320):

    pixel  (B, 3, T, H, W),   T ≡ 1 (mod 4)
      ↓ [-1, 1] rescale, per-frame FLUX.2 VAE encode
    grid   (B, 128, T, H/16, W/16)
      ↓ causal mean pool over T
    latent (B, 128, 1 + (T-1)/4, H/16, W/16)

Unlike :mod:`dinov3`, this encoder adds **no** trailing LayerNorm: the FLUX.2 VAE
already ends its encode with a non-affine BatchNorm that whitens the latent
per-channel (see :class:`FluxVaeEncoderCore`), which is the direct analogue of
Wan VAE's per-channel z-score ``(mu - mean) / std``
(``openwam/model/video_backbone/wan/vae.py`` ``WanVideoVAE.encode``). DINOv3's
ViT features have no such built-in whitening, which is why it needs the extra
LayerNorm and this encoder does not — matching the user's "logically aligned with
Wan VAE" requirement without bolting on a redundant second normalizer.

The DiT's ``patch_embedding`` / ``head.head`` are rebuilt from the default hooks
on :class:`VideoEncoder` (``Conv3d(128, dit_dim, (1,2,2), (1,2,2))`` and
``Linear(dit_dim, 128 * 4)``), giving a token count identical to the Wan VAE path.

See [docs/external_video_encoder.md] for the framework contract.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from openwam.model.video_backbone.encoder.flux import (
    FluxVaeEncoderCore,
    convert_diffusers_encoder_sd,
)
from openwam.model.video_backbone.encoder.registry import register_video_encoder
from openwam.model.video_backbone.encoder.videoencoder_base import VideoEncoder, VideoEncoderSpec

logger = logging.getLogger(__name__)


def _causal_temporal_pool(x: Tensor) -> Tensor:
    """Wan-style causal time pool: keep frame 0 as-is, mean-pool the rest in 4s.

    ``(B, D, T, H, W)`` with ``T ≡ 1 (mod 4)`` -> ``(B, D, 1 + (T - 1) // 4, H, W)``.
    A few trivial lines kept local to this encoder (mirrors :mod:`dinov3`) rather
    than shared, so neither per-frame encoder reaches into the other's internals.
    """
    first = x[:, :, :1]
    rest = rearrange(x[:, :, 1:], "B D (k g) H W -> B D k g H W", g=4)
    return torch.cat([first, rest.mean(dim=3)], dim=2)


def _preprocess_image(image: Image.Image, *, dtype, device) -> Tensor:
    """PIL image -> ``(1, 3, H, W)`` tensor in ``[-1, 1]``.

    Mirrors :func:`openwam.model.video_backbone.encoder.wan_vae._preprocess_image`
    (FLUX.2 VAE expects the same ``[-1, 1]`` pixel range as Wan VAE; see
    ``references/flux2/sampling.py`` ``default_images_prep`` = ``2 * x - 1``).
    """
    arr = torch.tensor(np.array(image, dtype=np.float32), dtype=dtype, device=device)
    arr = arr * (2.0 / 255.0) - 1.0
    return repeat(arr, "H W C -> B C H W", B=1)


def _read_flux_vae_config(model_path: str) -> dict:
    """Read the diffusers ``AutoencoderKLFlux2`` ``config.json`` structural fields.

    Returns the kwargs :class:`FluxVaeEncoderCore` needs. ``block_out_channels``
    pins ``ch`` (first entry) and ``ch_mult`` (entries / first); ``latent_channels``
    is ``z_channels``; ``layers_per_block`` is ``num_res_blocks``; ``patch_size``
    is the 2×2 pixel-shuffle pack; ``batch_norm_eps`` the whitening eps.
    """
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"FLUX.2 VAE config.json not found at {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    block_out = cfg["block_out_channels"]
    ch = int(block_out[0])
    ch_mult = [int(b) // ch for b in block_out]
    ps = cfg.get("patch_size", [2, 2])
    return {
        "ch": ch,
        "ch_mult": ch_mult,
        "num_res_blocks": int(cfg.get("layers_per_block", 2)),
        "z_channels": int(cfg.get("latent_channels", 32)),
        "ps": (int(ps[0]), int(ps[1])),
        "bn_eps": float(cfg.get("batch_norm_eps", 1e-4)),
        "bn_momentum": float(cfg.get("batch_norm_momentum", 0.1)),
    }


def _find_vae_weights(model_path: str) -> str:
    """Locate the single diffusers VAE safetensors inside ``model_path``."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"encoder model_path is not a directory: {model_path}")
    cand = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
    if os.path.isfile(cand):
        return cand
    import glob

    hits = glob.glob(os.path.join(model_path, "*.safetensors"))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"No *.safetensors VAE weights found in {model_path}")
    raise RuntimeError(f"Multiple safetensors candidates in {model_path}: {hits}")


@register_video_encoder("flux_vae")
class FluxVAEVideoEncoder(VideoEncoder):
    """:class:`VideoEncoder` wrapping the FLUX.2-dev image VAE encoder.

    Per-frame 2D VAE encode + parameter-free causal mean pool along T to align
    with Wan VAE's temporal semantics. ``z_dim`` / ``spatial_compression`` are
    read from the loaded core (which sized itself from the on-disk
    ``config.json``). The DiT-side projections reuse :class:`VideoEncoder`'s
    default hooks so flux_vae shares the exact Wan VAE token layout.

    No ``decode`` / ``to_frames`` (``spec.is_reversible=False``) — only the
    encoder half of the VAE is loaded, so the ABC defaults raise
    ``NotImplementedError`` with a contract-aware message.
    """

    def __init__(self, core: FluxVaeEncoderCore):
        super().__init__()
        self._core = core
        self._spec = VideoEncoderSpec(
            z_dim=int(core.z_dim),
            spatial_compression=int(core.spatial_compression),
            temporal_compression=4,
            causal_temporal=True,
            pixel_range=(-1.0, 1.0),
            is_reversible=False,
            dit_patch_size=(1, 2, 2),
        )

    @property
    def spec(self) -> VideoEncoderSpec:
        return self._spec

    def preprocess_video(self, frames) -> Tensor:
        dtype = next(self._core.parameters()).dtype
        device = next(self._core.parameters()).device
        images = [_preprocess_image(img, dtype=dtype, device=device) for img in frames]
        return torch.stack(images, dim=2)

    def batch_encode(self, video: Tensor) -> Tensor:
        """``(B, 3, T, H, W) → (B, 128, T_lat, H/16, W/16)``.

        Per-frame VAE encode (folded into the batch axis), reshape to a 5D grid,
        then causal-mean-pool along T. No extra normalization — the VAE's
        built-in BatchNorm whitening already runs inside ``core.encode``.
        """
        if video.dim() != 5 or video.shape[1] != 3:
            raise ValueError(f"batch_encode expects (B, 3, T, H, W); got {tuple(video.shape)}")
        b, _, t, h, w = video.shape
        sc = self._spec.spatial_compression
        if h % sc != 0 or w % sc != 0:
            raise ValueError(
                f"FluxVAEVideoEncoder requires H,W divisible by spatial_compression={sc}; got H={h}, W={w}."
            )
        tc = self._spec.temporal_compression
        if (t - 1) % tc != 0:
            raise ValueError(
                f"FluxVAEVideoEncoder expects T ≡ 1 (mod {tc}); got T={t}. "
                "This matches Wan VAE's causal first-frame + 4x temporal grouping."
            )

        flat = rearrange(video, "B C T H W -> (B T) C H W")
        # Gradient enablement is the caller's responsibility — the host backbone's
        # preprocess / prepare_inputs already wrap the encode path in
        # ``@torch.no_grad`` for the frozen-feature use case (matches wan_vae /
        # dinov3). The VAE is in the ``video_backbone.video_encoder`` freeze list.
        z = self._core.encode(flat)  # (B*T, z_dim, H/sc, W/sc)
        grid = rearrange(z, "(B T) D H W -> B D T H W", B=b, T=t)
        return _causal_temporal_pool(grid)

    # decode / to_frames intentionally omitted — defaults from VideoEncoder
    # raise NotImplementedError because spec.is_reversible=False.

    @classmethod
    def from_pretrained(cls, model_path: str, **kw: Any) -> "FluxVAEVideoEncoder":
        from safetensors.torch import load_file

        core_kwargs = _read_flux_vae_config(model_path)
        core = FluxVaeEncoderCore(**core_kwargs)
        weights = _find_vae_weights(model_path)
        sd = load_file(weights)
        converted = convert_diffusers_encoder_sd(sd)
        core.load_state_dict(converted, strict=True)
        core = core.to(dtype=torch.bfloat16).eval()
        logger.info(
            "FluxVAEVideoEncoder loaded %s (z_dim=%d, spatial_compression=%d)",
            weights,
            int(core.z_dim),
            int(core.spatial_compression),
        )
        return cls(core)

    @classmethod
    def from_skeleton(
        cls,
        components_entry: dict,
        *,
        device: str = "cpu",
        encoder_cfg: Any = None,
        ckpt_dir: str | None = None,
    ) -> "FluxVAEVideoEncoder":
        """Deploy-time zero-weight core, sized by ``encoder.model_path/config.json``.

        Mirrors :meth:`DinoV3VideoEncoder.from_skeleton`: no weights are loaded
        here (the architecture's strict checkpoint load fills them in, including
        the BatchNorm running stats). ``components_entry`` / ``ckpt_dir`` are
        accepted for ABC parity but unused — there is no manifest sidecar, so the
        deploy host must be able to read ``encoder.model_path/config.json``.
        """
        model_path = None
        if encoder_cfg is not None:
            if isinstance(encoder_cfg, dict):
                model_path = encoder_cfg.get("model_path")
            else:
                model_path = getattr(encoder_cfg, "model_path", None)
        if not model_path or not os.path.isdir(str(model_path)):
            raise FileNotFoundError(
                "FluxVAEVideoEncoder.from_skeleton requires encoder.model_path to be a "
                f"readable directory holding config.json; got {model_path!r}."
            )
        core_kwargs = _read_flux_vae_config(str(model_path))
        with torch.device(device):
            core = FluxVaeEncoderCore(**core_kwargs)
        core = core.to(dtype=torch.bfloat16).eval()
        logger.info(
            "FluxVAEVideoEncoder.from_skeleton: instantiated from %s "
            "(z_dim=%d, spatial_compression=%d) — weights pending checkpoint load",
            model_path,
            int(core.z_dim),
            int(core.spatial_compression),
        )
        return cls(core)


__all__ = ["FluxVAEVideoEncoder"]
