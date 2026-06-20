"""Online LAPA-DINOv3 latent-action target provider."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from openwam.model.action_backbone.latent_encoder.base import LatentActionEncoder
from openwam.model.action_backbone.latent_encoder.lapa_dinov3_src.checkpoint import (
    is_lfs_pointer_file,
    validate_lapa_paths,
)
from openwam.model.action_backbone.latent_encoder.lapa_dinov3_src.model import (
    LatentActionQuantizationDinov3Feature,
)
from openwam.model.action_backbone.latent_encoder.lapa_dinov3_src.video_io import stack_videos

__all__ = ["LAPADinov3TargetProvider", "build_latent_action_provider", "is_lfs_pointer_file"]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class LAPADinov3TargetProvider(LatentActionEncoder):
    """Frozen online LAPA-DINOv3 target provider.

    Input videos are converted to adjacent frame pairs and resized to the
    LARYBench 224-square protocol. The output is a clean action tensor shaped
    ``[B, (T_video - 1) * tokens_per_pair, token_dim]``.
    """

    def __init__(self, cfg: Any, *, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.cfg = cfg

        paths_cfg = _cfg_get(cfg, "paths")
        input_cfg = _cfg_get(cfg, "input")
        output_cfg = _cfg_get(cfg, "output")
        model_cfg = _cfg_get(cfg, "model")

        expected_dim = int(_cfg_get(model_cfg, "dim", 1024))
        self.lapa_model_dir, self.dinov3_model_dir = validate_lapa_paths(paths_cfg, expected_dim=expected_dim)
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
            batch = stack_videos(list(videos))

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
