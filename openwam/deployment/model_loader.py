"""Model-loading helpers for inference and deployment.

Provides two loading paths:

1. ``load_from_checkpoint_dir`` — load from a self-contained checkpoint
   directory produced by training (config.yaml + .safetensors).  This is
   the recommended path for deployment.

2. ``load_wam_models`` — legacy loader that reads eval-section config and
   assembles models from separate files.  Kept for backward compatibility.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from openwam.model.base import BaseWAMArchitecture
from openwam.model.video_backbone import WanVideoPipeline
from openwam.train.utils.checkpointing import load_trainable_checkpoint
from openwam.train.utils.pipeline_builder import (
    build_training_pipeline,
    build_video_backbone_from_manifest,
)

logger = logging.getLogger(__name__)


def _find_latest_checkpoint(ckpt_dir: str) -> str:
    """Return the path to the highest-step ``checkpoint_step_N.safetensors`` in *ckpt_dir*.

    Malformed filenames that glob-match but don't carry a numeric step are
    skipped (instead of silently getting step=0 and competing for latest).
    If every remaining file has step == 0 we warn — typically that means
    training crashed before the first save_steps interval and the caller is
    about to deploy uninitialized weights.
    """
    pattern = os.path.join(ckpt_dir, "checkpoint_step_*.safetensors")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint_step_*.safetensors found in {ckpt_dir}")

    step_re = re.compile(r"checkpoint_step_(\d+)\.safetensors$")
    numbered: list[tuple[int, str]] = []
    for f in files:
        m = step_re.search(os.path.basename(f))
        if m is not None:
            numbered.append((int(m.group(1)), f))
        else:
            logger.warning("Skipping malformed checkpoint name: %s", f)

    if not numbered:
        raise FileNotFoundError(f"No checkpoint file in {ckpt_dir} matches checkpoint_step_<int>.safetensors")

    numbered.sort(key=lambda p: p[0])
    latest_step, latest_path = numbered[-1]
    if latest_step == 0:
        logger.warning(
            "Latest checkpoint in %s is step 0 (%s) — this usually means training "
            "crashed before completing its first save_steps interval. Verify before deploying.",
            ckpt_dir,
            os.path.basename(latest_path),
        )
    return latest_path


def load_from_checkpoint_dir(
    ckpt_dir: str,
    device: str = "cuda",
    ckpt_name: Optional[str] = None,
) -> Tuple[DictConfig, WanVideoPipeline, BaseWAMArchitecture]:
    """Load full model from a self-contained checkpoint directory.

    The directory must contain:
      - ``config.yaml`` — Hydra config saved during training.
      - One or more ``checkpoint_step_*.safetensors`` files.

    Args:
        ckpt_dir: Path to the checkpoint directory.
        device: Target device (e.g. ``"cuda"`` or ``"cuda:0"``).
        ckpt_name: Specific checkpoint filename.  If *None*, the latest
            (highest step number) checkpoint is used.

    Returns:
        ``(cfg, pipe, architecture)`` — the resolved config, loaded
        pipeline, and architecture with all weights restored.
    """
    # 1. Load config
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in {ckpt_dir}")
    cfg = OmegaConf.load(config_path)

    # 2. Resolve checkpoint file
    if ckpt_name is not None:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    else:
        ckpt_path = _find_latest_checkpoint(ckpt_dir)
    logger.info("Loading checkpoint: %s", ckpt_path)

    # 3. Build video backbone.
    #    Prefer manifest-based path (self-contained: no external backbone source needed);
    #    fall back to training pipeline builder (requires cfg.model.video_backbone.model_path).
    manifest_path = os.path.join(ckpt_dir, "video_backbone_manifest.json")
    if os.path.exists(manifest_path):
        logger.info("Using manifest-based video-backbone builder: %s", manifest_path)
        pipe = build_video_backbone_from_manifest(manifest_path, device="cpu")
    else:
        pipe = build_training_pipeline(cfg)
    pipe.device = device

    # 4. Build architecture (same logic as OpenWAMTrainer.__init__)
    from openwam.model.registry import build_architecture

    m = cfg.model
    video_dim = int(pipe.dit.dim)

    arch_cfg = getattr(m, "architecture", {})
    action_cfg = getattr(m, "action_backbone", {})

    params = {k: v for k, v in arch_cfg.items() if k != "type"}
    if action_cfg:
        params.update({k: v for k, v in action_cfg.items()})
    params["video_dim"] = video_dim

    arch_type = arch_cfg.get("type", "dual_system")
    architecture = build_architecture(arch_type, params)
    logger.info("Architecture: %s (video_dim=%d)", arch_type, video_dim)

    # 5. Resolve action_dit reference for checkpoint loading
    if hasattr(architecture, "action_dit") and architecture.action_dit is not None:
        action_dit = architecture.action_dit
    elif hasattr(architecture, "moe_dit"):
        action_dit = architecture.moe_dit
    else:
        action_dit = architecture

    # 6. Load all weights from checkpoint
    load_trainable_checkpoint(ckpt_path, action_dit, pipe)

    # 7. Move to device and set eval mode.
    #    Dtype is driven by cfg.training.mixed_precision (default: bf16).
    _mp = OmegaConf.select(cfg, "training.mixed_precision", default="bf16")
    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    model_dtype = _DTYPE_MAP.get(str(_mp).strip().lower(), torch.bfloat16)
    logger.info("Loading all models with dtype=%s (mixed_precision=%s)", model_dtype, _mp)

    action_dit.to(dtype=model_dtype, device=device)
    action_dit.eval()
    # VAE is cast to model_dtype too; set mixed_precision: no if fp32 VAE is needed.
    for name in ("dit", "vace", "text_encoder", "vae"):
        mod = getattr(pipe, name, None)
        if mod is not None:
            mod.to(dtype=model_dtype, device=device)
            mod.eval()

    # 8. Attach denormalizer built from saved action_stats.npy + config.
    #    joint_generation will pick this up via getattr(architecture, "action_denormalizer")
    #    and use it in place of the legacy z-score buffer path.
    architecture.action_denormalizer = _build_action_denormalizer(cfg, ckpt_dir)

    logger.info("Model loaded successfully on %s", device)
    return cfg, pipe, architecture


def _build_action_denormalizer(cfg: DictConfig, ckpt_dir: str):
    """Load action_stats.npy from *ckpt_dir* and wrap in an ActionNormalizer.

    Reads ``dataloader.normalize_mode`` and ``dataloader.action_mode`` from the
    saved config to decide which mode to apply and which sub-dict to pull out
    of the nested stats schema.  Returns ``None`` when any prerequisite is
    missing (legacy checkpoint without stats, normalization disabled, etc.);
    the caller then falls back to the buffer-based z-score path.
    """
    logger.info("[normalizer] Resolving deployment denormalizer from checkpoint dir: %s", ckpt_dir)
    stats_path = os.path.join(ckpt_dir, "action_stats.npy")
    if not os.path.exists(stats_path):
        logger.warning(
            "[normalizer] No pre-computed stats file at expected location: %s\n"
            "[normalizer]   → denormalizer DISABLED; falling back to legacy "
            "buffer-based z-score path (action_mean/action_std from checkpoint).\n"
            "[normalizer]   This fallback is only correct if training used z-score normalization.",
            stats_path,
        )
        return None
    logger.info("[normalizer] Found pre-computed stats file: %s (exists ✓)", stats_path)

    dl = OmegaConf.select(cfg, "dataloader", default=None)
    norm_mode = OmegaConf.select(cfg, "dataloader.normalize_mode", default=None)
    action_mode = OmegaConf.select(cfg, "dataloader.action_mode", default="joint")
    if dl is None or norm_mode in (None, "", "none", "null"):
        logger.info(
            "[normalizer] normalize_mode=%r disabled in saved config; denormalizer INACTIVE "
            "(actions will be returned as-is from the model).",
            norm_mode,
        )
        return None
    logger.info(
        "[normalizer] Saved config: normalize_mode=%s, action_mode=%s",
        norm_mode,
        action_mode,
    )

    from openwam.dataloader.robotwin_dataset import _YAML_TO_NORM_MODE, _load_mode_stats
    from openwam.dataloader.transforms.normalize import ActionNormalizer

    if norm_mode not in _YAML_TO_NORM_MODE:
        logger.warning(
            "[normalizer] Unknown normalize_mode %r in checkpoint config; denormalizer DISABLED.",
            norm_mode,
        )
        return None

    mode_stats = _load_mode_stats(stats_path, action_mode)
    if mode_stats is None:
        logger.warning(
            "[normalizer] Stats file %s has no '%s' entry; denormalizer DISABLED.",
            stats_path,
            action_mode,
        )
        return None

    denorm = ActionNormalizer(mode=_YAML_TO_NORM_MODE[norm_mode], stats=mode_stats)
    logger.info(
        "[normalizer] Active: mode=%s action_mode=%s dim=%d stats=%s",
        norm_mode,
        action_mode,
        len(mode_stats["mean"]),
        stats_path,
    )
    return denorm
