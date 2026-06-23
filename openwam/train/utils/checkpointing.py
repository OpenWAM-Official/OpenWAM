"""Checkpoint save / load / management utilities."""

import glob as _glob
import logging
import os
import re

logger = logging.getLogger(__name__)


def save_config(output_dir: str, cfg):
    """Save Hydra DictConfig as config.yaml in the checkpoint directory.

    Only written once (skipped if the file already exists).
    """
    config_path = os.path.join(output_dir, "config.yaml")
    if os.path.exists(config_path):
        return
    os.makedirs(output_dir, exist_ok=True)
    from omegaconf import OmegaConf

    OmegaConf.save(cfg, config_path)
    logger.info("Saved config to %s", config_path)


def save_normalization_stats(output_dir: str, dataset) -> None:
    """Copy the dataset's resolved action-stats .npy into the checkpoint dir.

    Written once (skipped if ``normalization_stats.npy`` already exists). Silently
    no-ops when the dataset has no stats path. The copied file preserves the nested
    ``{"joint": ..., "eef": ...}`` schema so deployment can pick the sub-dict
    matching the saved config's ``action_mode``.
    """
    import shutil

    dst = os.path.join(output_dir, "normalization_stats.npy")
    if os.path.exists(dst):
        logger.info(
            "[normalizer] normalization_stats.npy already present in checkpoint dir: %s (skip copy)",
            dst,
        )
        return
    src = getattr(dataset, "normalization_stats_path", None)
    if not src:
        logger.info(
            "[normalizer] Dataset has no normalization_stats_path (normalization likely disabled); "
            "nothing copied into checkpoint dir."
        )
        return
    if not os.path.exists(src):
        logger.warning(
            "[normalizer] Dataset reports normalization_stats_path=%s but file does not exist; "
            "nothing copied into checkpoint dir.",
            src,
        )
        return
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(src, dst)
    logger.info("[normalizer] Copied action stats into checkpoint dir:\n  src: %s\n  dst: %s", src, dst)


def manage_checkpoints(output_dir: str, keep_last_k: int):
    """Delete old checkpoints, keeping only the most recent *keep_last_k*.

    Looks for ``checkpoint_step_*`` files in *output_dir* and removes the oldest.
    """
    pattern = os.path.join(output_dir, "checkpoint_step_*")
    files = _glob.glob(pattern)

    def _step_num(path):
        m = re.search(r"checkpoint_step_(\d+)", path)
        return int(m.group(1)) if m else 0

    files.sort(key=_step_num)
    while len(files) > keep_last_k:
        old = files.pop(0)
        if os.path.isfile(old):
            os.remove(old)
            logger.info("Removed old checkpoint: %s", old)
