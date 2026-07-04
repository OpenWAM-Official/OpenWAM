#!/usr/bin/env python3
"""Compute RoboCasa GR1 action normalization stats.

Example:
    python scripts/robocasa_gr1_compute_stats.py \
      --config configs/dataloader/robocasa_gr1.yaml \
      --output /path/to/normalization_stats.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openwam.dataloader.robocasa_gr1 import MultiRoboCasaGR1Dataset, RoboCasaGR1Dataset  # noqa: E402
from openwam.dataloader.robocasa_gr1_stats import compute_array_stats  # noqa: E402


def _iter_buckets(dataset) -> Iterable[RoboCasaGR1Dataset]:
    if isinstance(dataset, MultiRoboCasaGR1Dataset):
        yield from dataset.buckets
    else:
        yield dataset


def _iter_bucket_action_arrays(bucket: RoboCasaGR1Dataset):
    seen = set()
    for _, row in bucket._eps_df.iterrows():  # noqa: SLF001 - stats script uses reader internals intentionally.
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if key in seen:
            continue
        seen.add(key)
        table = bucket._load_data_table(*key)  # noqa: SLF001
        win = table.to_pandas()
        # Stats are computed in the raw selected action space. For unify mode,
        # runtime normalization happens before map_to_unify().
        yield bucket._raw_action(win)  # noqa: SLF001


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataloader/robocasa_gr1.yaml")
    parser.add_argument("--output", required=True, help="Output .npy path")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "normalize_mode", None, merge=False)
    dataset = RoboCasaGR1Dataset.from_config(cfg, split=str(OmegaConf.select(cfg, "split", default="train")))

    arrays = []
    rot6d_slices = ()
    action_mode = "unknown"
    for bucket in _iter_buckets(dataset):
        action_mode = bucket.action_mode
        rot6d_slices = bucket._rot6d_slices_for_stats()  # noqa: SLF001
        arrays.extend(_iter_bucket_action_arrays(bucket))
    stats = compute_array_stats(arrays, rot6d_slices=rot6d_slices)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, {action_mode: stats})
    print(f"wrote {output} action_mode={action_mode} dim={len(stats['mean'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
