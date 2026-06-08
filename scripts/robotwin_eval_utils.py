"""Shared helpers for RoboTwin diagnostic evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

EEF_GROUPS = {
    "left_xyz": slice(0, 3),
    "left_rot6d": slice(3, 9),
    "left_grip": [9],
    "right_xyz": slice(10, 13),
    "right_rot6d": slice(13, 19),
    "right_grip": [19],
}


def parse_indices(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def sample_indices(dataset_len: int, num_samples: int, seed: int, explicit: list[int] | None) -> list[int]:
    if explicit is not None:
        return explicit
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    rng = np.random.default_rng(seed)
    replace = num_samples > dataset_len
    return [int(i) for i in rng.choice(dataset_len, size=num_samples, replace=replace)]


def build_dataset_from_checkpoint_cfg(
    ckpt_dir: Path,
    cfg,
    split: str,
    *,
    dataset_dir: str | None = None,
    task_name: str | None = None,
    variant: str | None = None,
):
    from openwam.dataloader.registry import build_dataset

    dl_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataloader, resolve=True))
    OmegaConf.update(dl_cfg, "normalization_stats_path", str(ckpt_dir / "normalization_stats.npy"), merge=False)
    if dataset_dir is not None:
        OmegaConf.update(dl_cfg, "dataset_dir", dataset_dir, merge=False)
    if task_name is not None:
        OmegaConf.update(dl_cfg, "task_name", task_name, merge=False)
        OmegaConf.update(dl_cfg, "train_tasks", None, merge=False)
        OmegaConf.update(dl_cfg, "holdout_tasks", None, merge=False)
    if variant is not None:
        OmegaConf.update(dl_cfg, "variant", variant, merge=False)
    return build_dataset(dl_cfg, split=split)
