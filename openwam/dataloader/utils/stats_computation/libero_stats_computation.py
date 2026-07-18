"""Compute streaming LIBERO EEF10 action/state normalization statistics.

Emits SEPARATE stats blocks for the action targets (next-frame achieved pose +
recorded gripper command) and the achieved proprio, matching the reader's
asymmetric normalization contract. rot6d dims (3:9) are pinned to identity so
normalization never distorts the rotation representation.

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

from openwam.dataloader.libero import ROT6D_DIMS_EEF10, LiberoDataset, MultiLiberoDataset
from openwam.dataloader.utils.normalization import pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator


def _iter_buckets(dataset) -> Iterable[LiberoDataset]:
    if isinstance(dataset, MultiLiberoDataset):
        yield from dataset.buckets
    else:
        yield dataset


def _iter_bucket_arrays(bucket: LiberoDataset):
    """Yield per-episode ``(action_eef10, state_eef10)`` raw arrays.

    Iterates EPISODES (not shards): the next-frame action target shift must not
    cross episode boundaries, so each episode's rows are sliced out of its
    shard via the reader's file-local row offset.
    """
    for pos, (_, row) in enumerate(bucket._eps_df.iterrows()):  # noqa: SLF001 - stats script uses reader internals.
        table = bucket._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))  # noqa: SLF001
        offset = int(bucket._ep_data_row_offset[pos])  # noqa: SLF001
        win = table.slice(offset, int(row["length"])).to_pandas()
        action = bucket._raw_action_eef10(win)  # noqa: SLF001
        state = bucket._raw_state_eef10(win)  # noqa: SLF001
        # The final row's action target is a clamped copy (no t+1) — exclude it
        # from the statistics exactly as it is excluded from the loss.
        yield action[:-1] if action.shape[0] > 1 else action[:0], state


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

    buckets = list(_iter_buckets(dataset))
    if not buckets:
        raise ValueError("LIBERO dataset has no buckets")
    raw_dims = {bucket._raw_action_dim for bucket in buckets}  # noqa: SLF001
    if len(raw_dims) != 1:
        raise ValueError(f"stats require homogeneous raw dims, got {raw_dims}")
    raw_dim = next(iter(raw_dims))
    action_mode = buckets[0].action_mode

    action_accumulator = Accumulator(dim=raw_dim, reservoir_cap=args.reservoir_cap)
    state_accumulator = Accumulator(dim=raw_dim, reservoir_cap=args.reservoir_cap)
    for bucket in buckets:
        for action, state in _iter_bucket_arrays(bucket):
            if action.shape[0]:
                action_accumulator.update_batch(np.asarray(action, np.float32).reshape(-1, raw_dim))
            state_accumulator.update_batch(np.asarray(state, np.float32).reshape(-1, raw_dim))
    if action_accumulator.count == 0 or state_accumulator.count == 0:
        raise ValueError("cannot compute LIBERO stats from an empty dataset")

    action_stats = action_accumulator.finalize()
    state_stats = state_accumulator.finalize()
    action_stats["num_timesteps"] = action_accumulator.count
    action_stats["pool"] = "action"
    state_stats["num_timesteps"] = state_accumulator.count
    state_stats["pool"] = "state"
    pin_rot6d_identity(action_stats, ROT6D_DIMS_EEF10)
    pin_rot6d_identity(state_stats, ROT6D_DIMS_EEF10)

    payload = {}
    if output.exists():
        previous = np.load(output, allow_pickle=True).item()
        if isinstance(previous, dict):
            payload.update(previous)
    payload[action_mode] = action_stats
    payload[f"{action_mode}_state"] = state_stats
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, payload)
    print(
        f"wrote {output} action_mode={action_mode} state_mode={action_mode}_state "
        f"dim={raw_dim} action_rows={action_accumulator.count} state_rows={state_accumulator.count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
