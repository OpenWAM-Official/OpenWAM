"""Compute streaming 7-D LIBERO action normalization statistics.

Example:
    python -m openwam.dataloader.utils.stats_computation.libero_stats_computation \
      --config configs/dataloader/libero.yaml \
      --output /path/to/normalization_stats.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from omegaconf import OmegaConf

from openwam.dataloader.libero import LiberoDataset, MultiLiberoDataset
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator


def _iter_buckets(dataset) -> Iterable[LiberoDataset]:
    if isinstance(dataset, MultiLiberoDataset):
        yield from dataset.buckets
    else:
        yield dataset


def _iter_action_arrays(bucket: LiberoDataset):
    seen = set()
    for _, row in bucket._eps_df.iterrows():  # noqa: SLF001
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if key in seen:
            continue
        seen.add(key)
        table = bucket._load_data_table(*key)  # noqa: SLF001
        action = np.stack(table.to_pandas()["action"].values).astype(np.float32)
        if action.ndim != 2 or action.shape[1] != 7:
            raise ValueError(f"LIBERO action must be (T, 7), got {action.shape} in shard {key}")
        yield action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataloader/libero.yaml")
    parser.add_argument("--output", required=True, help="Deploy-compatible .npy output")
    parser.add_argument("--reservoir-cap", type=int, default=1_000_000)
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix != ".npy":
        raise ValueError("--output must end in .npy")

    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "normalize_mode", None, merge=False)
    dataset = LiberoDataset.from_config(cfg, split=str(OmegaConf.select(cfg, "split", default="train")))

    accumulator = Accumulator(dim=7, reservoir_cap=args.reservoir_cap)
    for bucket in _iter_buckets(dataset):
        for action in _iter_action_arrays(bucket):
            accumulator.update_batch(action)
    if accumulator.count == 0:
        raise ValueError("cannot compute LIBERO stats from an empty dataset")

    stats = accumulator.finalize()
    stats["num_timesteps"] = accumulator.count
    stats["pool"] = "action"

    payload = {}
    if output.exists():
        previous = np.load(output, allow_pickle=True).item()
        if isinstance(previous, dict):
            payload.update(previous)
    payload["libero"] = stats
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, payload)
    print(f"wrote {output} action_mode=libero dim=7 rows={accumulator.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
