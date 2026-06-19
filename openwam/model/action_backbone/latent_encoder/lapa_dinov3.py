"""Online LAPA-DINOv3 latent-action target provider."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from openwam.model.action_backbone.latent_encoder.lapa_dinov3_model import LatentActionQuantizationDinov3Feature

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def is_lfs_pointer_file(path: str | os.PathLike[str]) -> bool:
    """Return True when ``path`` is a Git LFS pointer instead of real weights."""
    try:
        with open(path, "rb") as f:
            return f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX
    except OSError:
        return False


def _resolve_path(raw: Any, label: str) -> Path:
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"latent_action.paths.{label} is required")
    return Path(str(raw)).expanduser().resolve()


def _validate_lapa_paths(paths_cfg: Any, expected_dim: int | None = None) -> tuple[Path, Path]:
    lapa_model_dir = _resolve_path(_cfg_get(paths_cfg, "lapa_model_dir"), "lapa_model_dir")
    dinov3_model_dir = _resolve_path(_cfg_get(paths_cfg, "dinov3_model_dir"), "dinov3_model_dir")

    if not lapa_model_dir.is_dir():
        raise FileNotFoundError(f"LAPA-DINOv3 model dir not found: {lapa_model_dir}")
    ckpt = lapa_model_dir / "laq_dinov3.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"LAPA-DINOv3 checkpoint not found: {ckpt}")
    if is_lfs_pointer_file(ckpt):
        raise RuntimeError(
            f"LAPA-DINOv3 checkpoint is still a Git LFS pointer, not real weights: {ckpt}. "
            "Run `git -C models/LAPA-DINOv3 lfs pull` or refresh with Hugging Face snapshot_download."
        )
    if not dinov3_model_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 model dir not found: {dinov3_model_dir}")
    cfg_path = dinov3_model_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"DINOv3 config.json not found: {cfg_path}")
    for weight_path in dinov3_model_dir.rglob("*"):
        if weight_path.is_file() and weight_path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}:
            if is_lfs_pointer_file(weight_path):
                raise RuntimeError(
                    f"DINOv3 weight file is still a Git LFS pointer, not real weights: {weight_path}. "
                    "Run git lfs pull for the DINOv3 model directory or refresh it with Hugging Face snapshot_download."
                )
    if expected_dim is not None:
        try:
            model_cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid DINOv3 config.json: {cfg_path}") from exc
        hidden_size = model_cfg.get("hidden_size", model_cfg.get("embed_dim"))
        if int(hidden_size or 0) != int(expected_dim):
            raise ValueError(
                f"LAPA-DINOv3 checkpoint expects DINOv3 hidden_size={expected_dim}, but "
                f"{dinov3_model_dir} has hidden_size={hidden_size}. Use a 1024-dim DINOv3 backbone, "
                "for example facebook/dinov3-vitl16-pretrain-lvd1689m."
            )
    return lapa_model_dir, dinov3_model_dir


def _video_to_tensor(video: Any) -> torch.Tensor:
    """Convert one sample video to ``(T, 3, H, W)`` float in ``[0, 1]``."""
    if isinstance(video, torch.Tensor):
        x = video.detach()
        if x.ndim != 4:
            raise ValueError(f"video tensor must be 4D, got {tuple(x.shape)}")
        if x.shape[0] == 3 and x.shape[1] == 3:
            raise ValueError(
                "Ambiguous video tensor layout with both C and T equal to 3; "
                "pass channels-last (T,H,W,3) or a list of frames."
            )
        if x.shape[1] == 3:
            # (T, C, H, W)
            pass
        elif x.shape[-1] == 3:
            # (T, H, W, C)
            x = x.permute(0, 3, 1, 2)
        elif x.shape[0] == 3:
            # (C, T, H, W)
            x = x.permute(1, 0, 2, 3)
        else:
            raise ValueError(f"cannot infer channel axis for video tensor shape {tuple(x.shape)}")
        x = x.float()
        if x.numel() and float(x.max()) > 2.0:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    frames = list(video)
    if len(frames) < 2:
        raise ValueError("latent action requires at least 2 video frames")
    tensors: list[torch.Tensor] = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            t = frame.detach()
            if t.ndim != 3:
                raise ValueError(f"video frame tensor must be 3D, got {tuple(t.shape)}")
            if t.shape[0] == 3:
                pass
            elif t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
            else:
                raise ValueError(f"cannot infer channel axis for frame shape {tuple(t.shape)}")
            t = t.float()
            if t.numel() and float(t.max()) > 2.0:
                t = t / 255.0
        elif isinstance(frame, Image.Image):
            arr = np.asarray(frame.convert("RGB"))
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        else:
            arr = np.asarray(frame)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"video frame array must be HWC RGB, got shape {arr.shape}")
            t = torch.from_numpy(arr).permute(2, 0, 1).float()
            if t.numel() and float(t.max()) > 2.0:
                t = t / 255.0
        tensors.append(t.clamp(0.0, 1.0))
    return torch.stack(tensors, dim=0)


def _stack_videos(videos: list[Any]) -> torch.Tensor:
    tensors = [_video_to_tensor(v) for v in videos]
    lengths = {int(t.shape[0]) for t in tensors}
    if len(lengths) != 1:
        raise ValueError(f"LAPA online provider requires uniform T_video in a batch, got {sorted(lengths)}")
    if next(iter(lengths)) < 2:
        raise ValueError("LAPA online provider requires at least 2 video frames")
    return torch.stack(tensors, dim=0)


class LAPADinov3TargetProvider(nn.Module):
    """Frozen online LAPA-DINOv3 target provider.

    Input videos are converted to adjacent frame pairs and resized to the
    LARYBench 224-square protocol. The output is a clean action tensor shaped
    ``[B, (T_video - 1) * tokens_per_pair, token_dim]``.
    """

    def __init__(self, cfg: Any, *, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype

        paths_cfg = _cfg_get(cfg, "paths")
        input_cfg = _cfg_get(cfg, "input")
        output_cfg = _cfg_get(cfg, "output")
        model_cfg = _cfg_get(cfg, "model")

        expected_dim = int(_cfg_get(model_cfg, "dim", 1024))
        self.lapa_model_dir, self.dinov3_model_dir = _validate_lapa_paths(paths_cfg, expected_dim=expected_dim)
        self.image_size = int(_cfg_get(input_cfg, "image_size", 224))
        self.tokens_per_pair = int(_cfg_get(output_cfg, "tokens_per_pair", 16))
        self.token_dim = int(_cfg_get(output_cfg, "token_dim", 1024))
        action_dim = int(_cfg_get(output_cfg, "action_dim", self.token_dim))
        if action_dim != self.token_dim:
            raise ValueError(
                f"latent_action.output.action_dim ({action_dim}) must equal "
                f"latent_action.output.token_dim ({self.token_dim}) for LAPA-DINOv3 targets."
            )
        self.flatten_pairs = bool(_cfg_get(output_cfg, "flatten_pairs", True))
        if not self.flatten_pairs:
            raise ValueError("Only latent_action.output.flatten_pairs=true is supported.")

        kwargs = {
            "dim": int(_cfg_get(model_cfg, "dim", 1024)),
            "quant_dim": int(_cfg_get(model_cfg, "quant_dim", 32)),
            "codebook_size": int(_cfg_get(model_cfg, "codebook_size", 8)),
            "image_size": int(_cfg_get(model_cfg, "image_size", self.image_size)),
            "patch_size": int(_cfg_get(model_cfg, "patch_size", 16)),
            "spatial_depth": int(_cfg_get(model_cfg, "spatial_depth", 4)),
            "temporal_depth": int(_cfg_get(model_cfg, "temporal_depth", 4)),
            "dim_head": int(_cfg_get(model_cfg, "dim_head", 64)),
            "heads": int(_cfg_get(model_cfg, "heads", 16)),
            "code_seq_len": int(_cfg_get(model_cfg, "code_seq_len", self.tokens_per_pair)),
            "dinov3_model_dir": self.dinov3_model_dir,
            "device": self.device,
        }
        self.model = LatentActionQuantizationDinov3Feature(**kwargs).to(self.device)
        state = torch.load(self.lapa_model_dir / "laq_dinov3.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

        self.model.eval()
        self.model.requires_grad_(False)

    def _build_pairs(self, videos: list[Any] | torch.Tensor) -> torch.Tensor:
        if isinstance(videos, torch.Tensor):
            if videos.ndim != 5:
                raise ValueError(f"batch video tensor must be 5D, got {tuple(videos.shape)}")
            if videos.shape[1] == 3 and videos.shape[2] == 3:
                raise ValueError(
                    "Ambiguous batch video tensor layout with both C and T equal to 3; "
                    "pass channels-last (B,T,H,W,3) or a list of frames."
                )
            if videos.shape[2] == 3:
                batch = videos.detach().float()
            elif videos.shape[-1] == 3:
                batch = videos.detach().permute(0, 1, 4, 2, 3).float()
            elif videos.shape[1] == 3:
                batch = videos.detach().permute(0, 2, 1, 3, 4).float()
            else:
                raise ValueError(f"cannot infer channel axis for batch video shape {tuple(videos.shape)}")
            if batch.numel() and float(batch.max()) > 2.0:
                batch = batch / 255.0
            batch = batch.clamp(0.0, 1.0)
        else:
            batch = _stack_videos(list(videos))

        if batch.shape[1] < 2:
            raise ValueError("LAPA online provider requires at least 2 video frames")

        B, T, C, H, W = batch.shape
        flat = batch.reshape(B * T, C, H, W).to(device=self.device, dtype=torch.float32)
        resized = F.interpolate(
            flat,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        resized = resized.reshape(B, T, C, self.image_size, self.image_size)
        src = resized[:, :-1]
        tgt = resized[:, 1:]
        pairs = torch.stack([src, tgt], dim=3)
        return pairs.reshape(B * (T - 1), C, 2, self.image_size, self.image_size)

    @torch.inference_mode()
    def forward(self, videos: list[Any] | torch.Tensor) -> torch.Tensor:
        pairs = self._build_pairs(videos)
        # Frozen label generation must be deterministic. ``return_only_codebook_ids``
        # switches the reference NSVQ path to hard codebook embeddings instead
        # of its stochastic training-time residual noise.
        tokens, _indices = self.model(pairs, return_only_codebook_ids=True)
        if tokens.ndim != 3:
            raise ValueError(f"LAPA-DINOv3 tokens must be 3D, got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.tokens_per_pair or tokens.shape[2] != self.token_dim:
            raise ValueError(
                f"LAPA-DINOv3 token shape {tuple(tokens.shape)} does not match "
                f"tokens_per_pair={self.tokens_per_pair}, token_dim={self.token_dim}"
            )

        B_pairs = tokens.shape[0]
        videos_len = len(videos) if not isinstance(videos, torch.Tensor) else int(videos.shape[0])
        if B_pairs % videos_len != 0:
            raise ValueError(f"Cannot reshape {B_pairs} pair tokens over batch size {videos_len}")
        pairs_per_sample = B_pairs // videos_len
        target = tokens.reshape(videos_len, pairs_per_sample * self.tokens_per_pair, self.token_dim)
        return target.to(device=self.device, dtype=self.dtype)


def build_latent_action_provider(cfg: Any, *, device: torch.device, dtype: torch.dtype) -> LAPADinov3TargetProvider:
    name = str(_cfg_get(cfg, "name", ""))
    if name != "lapa_dinov3":
        raise ValueError(f"Unsupported latent action provider: {name!r}")
    return LAPADinov3TargetProvider(cfg, device=device, dtype=dtype)
