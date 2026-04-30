"""Model-loading helpers for inference and deployment.

Provides the checkpoint-first loading path used by OpenWAM deployment.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from openwam.model.architectures.base import BaseWAMArchitecture

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
) -> Tuple[DictConfig, BaseWAMArchitecture]:
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
        ``(cfg, architecture)`` — the resolved config and architecture
        with all weights restored.
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

    # 3. Build architecture (creates video backbone internally from config).
    from openwam.model import build_architecture, resolve_architecture_config

    m = cfg.model
    resolved_arch = resolve_architecture_config(m)
    params = dict(resolved_arch.params)

    # Provide video backbone source: config components (preferred), manifest, or model_path.
    vb_components = OmegaConf.select(cfg, "model.video_backbone.components", default=None)
    if vb_components is not None:
        logger.info("Using config-embedded component specs for video-backbone construction")
        vb_cfg_dict = OmegaConf.to_container(cfg.model.video_backbone, resolve=True)
        params.setdefault("video_backbone", {})["_source"] = vb_cfg_dict
        params["video_backbone"]["_ckpt_dir"] = ckpt_dir
    else:
        manifest_path = os.path.join(ckpt_dir, "video_backbone_manifest.json")
        if os.path.exists(manifest_path):
            logger.info("Using manifest-based video-backbone builder: %s", manifest_path)
            params.setdefault("video_backbone", {})["_source"] = manifest_path
        else:
            model_path = OmegaConf.select(cfg, "model.video_backbone.model_path", default=None)
            if model_path is not None:
                params.setdefault("video_backbone", {})["_source"] = str(model_path)
            else:
                logger.warning(
                    "No components, manifest, or model_path in config; "
                    "architecture __init__ will attempt to build from config."
                )

    architecture = build_architecture(resolved_arch.registry_name, params)
    logger.info(
        "Architecture: %s (framework=%s variant=%s)",
        resolved_arch.registry_name,
        resolved_arch.canonical.framework,
        resolved_arch.canonical.variant,
    )

    # 4. Load all weights from checkpoint
    architecture.load_checkpoint(ckpt_path)

    # 5. Move to device and set eval mode — top-down: architecture → video_backbone → submodules.
    _mp = OmegaConf.select(cfg, "training.mixed_precision", default="bf16")
    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    model_dtype = _DTYPE_MAP.get(str(_mp).strip().lower(), torch.bfloat16)
    logger.info("Loading all models with dtype=%s (mixed_precision=%s)", model_dtype, _mp)

    architecture.set_dtype_device(model_dtype, torch.device(device))

    # 6. Attach denormalizer built from saved action_stats.npy + config.
    architecture.action_denormalizer = _build_action_denormalizer(cfg, ckpt_dir)

    logger.info("Model loaded successfully on %s", device)
    return cfg, architecture


def _build_action_denormalizer(cfg: DictConfig, ckpt_dir: str):
    """Load action_stats.npy from *ckpt_dir* and wrap in an ActionNormalizer.

    Reads ``dataloader.normalize_mode`` and ``dataloader.action_mode`` from the
    saved config to decide which mode to apply and which sub-dict to pull out
    of the nested stats schema.
    """
    logger.info("[normalizer] Resolving deployment denormalizer from checkpoint dir: %s", ckpt_dir)
    stats_path = os.path.join(ckpt_dir, "action_stats.npy")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Missing required action_stats.npy in checkpoint dir: {stats_path}. "
            "Old checkpoints without action_stats.npy are no longer supported."
        )
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

    from openwam.dataloader.transforms.normalize import (
        YAML_TO_NORM_MODE,
        ActionNormalizer,
        load_mode_stats,
    )

    if norm_mode not in YAML_TO_NORM_MODE:
        logger.warning(
            "[normalizer] Unknown normalize_mode %r in checkpoint config; denormalizer DISABLED.",
            norm_mode,
        )
        return None

    mode_stats = load_mode_stats(stats_path, action_mode)
    if mode_stats is None:
        logger.warning(
            "[normalizer] Stats file %s has no '%s' entry; denormalizer DISABLED.",
            stats_path,
            action_mode,
        )
        return None

    denorm = ActionNormalizer(mode=YAML_TO_NORM_MODE[norm_mode], stats=mode_stats)
    logger.info(
        "[normalizer] Active: mode=%s action_mode=%s dim=%d stats=%s",
        norm_mode,
        action_mode,
        len(mode_stats["mean"]),
        stats_path,
    )
    return denorm
