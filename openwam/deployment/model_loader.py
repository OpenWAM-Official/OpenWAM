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
from openwam.train.utils.pipeline_builder import build_training_pipeline

logger = logging.getLogger(__name__)


def _find_latest_checkpoint(ckpt_dir: str) -> str:
    """Return the path to the highest-step .safetensors file in *ckpt_dir*."""
    pattern = os.path.join(ckpt_dir, "checkpoint_step_*.safetensors")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint_step_*.safetensors found in {ckpt_dir}")

    def _step(path):
        m = re.search(r"checkpoint_step_(\d+)", path)
        return int(m.group(1)) if m else 0

    files.sort(key=_step)
    return files[-1]


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

    # 3. Build video pipeline (structure + pretrained weights from model_path)
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

    # 7. Move to device and set eval mode
    action_dit.to(dtype=torch.bfloat16, device=device)
    action_dit.eval()
    for name in ("dit", "vace", "text_encoder", "vae"):
        mod = getattr(pipe, name, None)
        if mod is not None:
            mod.to(device=device)
            mod.eval()

    logger.info("Model loaded successfully on %s", device)
    return cfg, pipe, architecture
