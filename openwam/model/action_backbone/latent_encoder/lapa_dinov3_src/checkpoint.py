"""LAPA-DINOv3 weight/path validation helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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


def validate_lapa_paths(paths_cfg: Any, expected_dim: int | None = None) -> tuple[Path, Path]:
    """Resolve + validate the LAPA + DINOv3 model dirs; return ``(lapa_dir, dinov3_dir)``."""
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
