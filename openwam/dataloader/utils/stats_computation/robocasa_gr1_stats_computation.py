"""Compute RoboCasa GR1 action normalization stats.

Example:
    python -m openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation \
      --config configs/dataloader/robocasa_gr1.yaml \
      --output /path/to/normalization_stats.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from omegaconf import OmegaConf

from openwam.dataloader.robocasa_gr1 import MultiRoboCasaGR1Dataset, RoboCasaGR1Dataset
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator


def _iter_buckets(dataset) -> Iterable[RoboCasaGR1Dataset]:
    if isinstance(dataset, MultiRoboCasaGR1Dataset):
        yield from dataset.buckets
    else:
        yield dataset


def _iter_bucket_arrays(bucket: RoboCasaGR1Dataset):
    seen = set()
    for _, row in bucket._eps_df.iterrows():  # noqa: SLF001 - stats script uses reader internals intentionally.
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if key in seen:
            continue
        seen.add(key)
        table = bucket._load_data_table(*key)  # noqa: SLF001
        win = table.to_pandas()
        # Action and state share one raw representation and one runtime
        # normalizer. Pool both streams, as the RoboCOIN computation does.
        yield bucket._raw_action(win)  # noqa: SLF001
        yield bucket._raw_state(win)  # noqa: SLF001


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataloader/robocasa_gr1.yaml")
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--reservoir-cap", type=int, default=1_000_000)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "normalize_mode", None, merge=False)
    dataset = RoboCasaGR1Dataset.from_config(cfg, split=str(OmegaConf.select(cfg, "split", default="train")))

    buckets = list(_iter_buckets(dataset))
    if not buckets:
        raise ValueError("RoboCasa GR1 dataset has no buckets")
    action_modes = {bucket.action_mode for bucket in buckets}
    raw_dims = {bucket._raw_action_dim for bucket in buckets}  # noqa: SLF001
    if len(action_modes) != 1 or len(raw_dims) != 1:
        raise ValueError(f"stats require homogeneous modes/dims, got modes={action_modes}, dims={raw_dims}")
    action_mode = next(iter(action_modes))
    raw_dim = next(iter(raw_dims))
    accumulator = Accumulator(dim=raw_dim, reservoir_cap=args.reservoir_cap)
    for bucket in _iter_buckets(dataset):
        for array in _iter_bucket_arrays(bucket):
            accumulator.update_batch(np.asarray(array, dtype=np.float32).reshape(-1, raw_dim))
    if accumulator.count == 0:
        raise ValueError("cannot compute normalization stats from an empty dataset")
    stats = accumulator.finalize()
    stats["num_timesteps"] = accumulator.count
    stats["pool"] = "action+state"
    if action_mode in {"eef", "unify"}:
        pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)

    output = Path(args.output)
    if output.suffix != ".npy":
        raise ValueError("--output must end in .npy so checkpoints receive a deploy-compatible artifact")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if output.exists():
        previous = np.load(output, allow_pickle=True).item()
        if isinstance(previous, dict):
            payload.update(previous)
    payload[action_mode] = stats
    np.save(output, payload)
    print(f"wrote {output} action_mode={action_mode} dim={raw_dim} rows={accumulator.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
