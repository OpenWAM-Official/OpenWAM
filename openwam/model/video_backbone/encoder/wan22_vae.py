"""Wan VAE wrapped as a :class:`VideoEncoder`.

Reference implementation: serves both as an A/B baseline (functionally
equivalent to the default ``pipe.vae`` path) and as a template that future
non-VAE encoders can copy from. The class deliberately does NOT override
:meth:`VideoEncoder.build_dit_input_proj` / ``build_dit_output_proj`` — the
default hooks produce a Wan-original ``Conv3d(z_dim, dit_dim, (1,2,2),
(1,2,2))`` + ``Linear(dit_dim, z_dim * 4)`` pair, which is what we want.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import reduce, repeat
from PIL import Image
from torch import Tensor

from openwam.model.video_backbone.encoder.base import VideoEncoder, VideoEncoderProperties
from openwam.model.video_backbone.encoder.registry import register_video_encoder

logger = logging.getLogger(__name__)


def _resolve_wan22_vae_class_for_file(file_path: str):
    """Hash-match a VAE weight file against ``MODEL_CONFIGS`` to pick the
    correct concrete class (``WanVideoVAE`` z=16 for Wan2.1 vs.
    ``WanVideoVAE38`` z=48 for Wan2.2) plus its state-dict converter.
    """
    import importlib

    from openwam.model.video_backbone.wan.shared.configs import MODEL_CONFIGS
    from openwam.model.video_backbone.wan.shared.core.loader.file import hash_model_file

    h = hash_model_file(file_path)
    for entry in MODEL_CONFIGS:
        if entry.get("model_hash") == h and entry.get("model_name") == "wan_video_vae":
            mod_path, cls_name = entry["model_class"].rsplit(".", 1)
            cls = getattr(importlib.import_module(mod_path), cls_name)
            converter = None
            conv_path = entry.get("state_dict_converter")
            if conv_path:
                cm, cn = conv_path.rsplit(".", 1)
                converter = getattr(importlib.import_module(cm), cn)
            return cls, converter
    raise FileNotFoundError(f"No MODEL_CONFIGS entry matched VAE file hash {h} at {file_path}.")


def _find_vae_file(model_path: str) -> str:
    """Locate the single ``*VAE*.{pth,safetensors}`` inside ``model_path``.

    Refuses to guess when the directory contains zero or more than one
    candidate — both are surely user-configuration errors.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"encoder model_path is not a directory: {model_path}")
    candidates: list[str] = []
    for pat in ("*VAE*.safetensors", "*VAE*.pth", "*vae*.safetensors", "*vae*.pth"):
        candidates.extend(glob.glob(os.path.join(model_path, pat)))
    seen: set[str] = set()
    uniq = [p for p in candidates if not (p in seen or seen.add(p))]
    if not uniq:
        raise FileNotFoundError(f"No VAE file found in {model_path}")
    if len(uniq) > 1:
        raise RuntimeError(f"Multiple VAE candidates in {model_path}: {uniq}")
    return uniq[0]


def _preprocess_image(image: Image.Image, *, dtype, device, min_value=-1.0, max_value=1.0) -> Tensor:
    arr = torch.tensor(np.array(image, dtype=np.float32), dtype=dtype, device=device)
    arr = arr * ((max_value - min_value) / 255.0) + min_value
    return repeat(arr, "H W C -> B C H W", B=1)


def _vae_output_to_image(t: Tensor, *, min_value=-1.0, max_value=1.0) -> Image.Image:
    img = ((t - min_value) * (255.0 / (max_value - min_value))).clip(0, 255)
    img = img.to(device="cpu", dtype=torch.uint8)
    return Image.fromarray(img.numpy())


@register_video_encoder("wan22_vae")
class WanVideoVAEEncoder(VideoEncoder):
    """:class:`VideoEncoder` wrapping the Wan VAE family.

    Default hooks on :class:`VideoEncoder` already produce Wan-original
    layouts for ``patch_embedding`` and ``head.head``; this class therefore
    intentionally does NOT override them. Future non-VAE encoders that need
    different DiT-side projections override only those two hooks.
    """

    def __init__(self, vae: nn.Module):
        super().__init__()
        self._m = vae
        # Wan VAE family always uses temporal_compression=4 with a causal
        # first-frame token; z_dim and upsampling_factor come from the loaded
        # weights so Wan2.1 (z=16, upsample=8) and Wan2.2 (z=48, upsample=16)
        # share this same wrapper.
        self._spec = VideoEncoderProperties(
            z_dim=int(vae.z_dim),
            spatial_compression=int(vae.upsampling_factor),
            temporal_compression=4,
            causal_temporal=True,
            # pixel_decode defaults to True — Wan VAE has a real pixel decoder.
            # dit_patch_size defaults to (1, 2, 2) — the Wan DiT's native value.
        )

    @property
    def properties(self) -> VideoEncoderProperties:
        return self._spec

    def preprocess_video(self, frames) -> Tensor:
        dtype = next(self._m.parameters()).dtype
        device = next(self._m.parameters()).device
        images = [_preprocess_image(img, dtype=dtype, device=device, min_value=-1.0, max_value=1.0) for img in frames]
        return torch.stack(images, dim=2)

    def batch_encode(self, video: Tensor) -> Tensor:
        return self._m.batch_encode(video, device=video.device)

    def decode(self, latents: Tensor, *, tiled: bool = True) -> Tensor:
        return self._m.decode(latents, device=latents.device, tiled=tiled)

    def to_frames(self, video_tensor: Tensor) -> list:
        t = reduce(video_tensor, "B C T H W -> T H W C", reduction="mean")
        return [_vae_output_to_image(frame, min_value=-1.0, max_value=1.0) for frame in t]

    @classmethod
    def from_pretrained(cls, model_path: str, **kw: Any) -> "WanVideoVAEEncoder":
        from openwam.model.video_backbone.wan.shared.core.loader.model import load_model

        vae_file = _find_vae_file(model_path)
        vae_class, converter = _resolve_wan22_vae_class_for_file(vae_file)
        vae = load_model(
            vae_class,
            path=vae_file,
            torch_dtype=torch.bfloat16,
            device="cpu",
            state_dict_converter=converter,
        )
        logger.info(
            "WanVideoVAEEncoder loaded %s (class=%s, z_dim=%d)",
            vae_file,
            vae_class.__name__,
            vae.z_dim,
        )
        return cls(vae)

    @classmethod
    def from_skeleton(
        cls,
        components_entry: dict,
        *,
        device: str = "cpu",
        encoder_cfg: Any = None,
        ckpt_dir: str | None = None,
    ) -> "WanVideoVAEEncoder":
        """Deploy-time constructor — instantiate the underlying ``WanVideoVAE``
        / ``WanVideoVAE38`` class with empty weights using the saved
        ``components`` entry's ``model_class`` + ``extra_kwargs``. The
        architecture's checkpoint ``load_checkpoint`` strict load fills
        in the weights immediately after.

        ``encoder_cfg`` / ``ckpt_dir`` are accepted (and ignored) for ABC
        signature compatibility — Wan VAE's structural geometry is fully
        captured by ``components_entry``, so no side files / yaml fallback
        is needed.

        Mirrors :func:`wan.loader.build_holder_from_components`'s
        instantiation pattern so the resulting module has bit-identical
        structure (same kwargs, same dtype, same construction-time device)
        — only ``self.video_encoder._m.*`` lives where ``self.vae.*``
        would on the training-side native path.
        """
        import importlib

        cls_path = components_entry["model_class"]
        mod_path, cls_name = cls_path.rsplit(".", 1)
        vae_class = getattr(importlib.import_module(mod_path), cls_name)
        kwargs = components_entry.get("extra_kwargs", {}) or {}
        with torch.device(device):
            vae = vae_class(**kwargs)
        vae.to(dtype=torch.bfloat16)
        logger.info(
            "WanVideoVAEEncoder.from_skeleton: %s instantiated (z_dim=%d, upsample=%d) — "
            "weights pending checkpoint load",
            vae_class.__name__,
            int(vae.z_dim),
            int(vae.upsampling_factor),
        )
        return cls(vae)


__all__ = ["WanVideoVAEEncoder"]
