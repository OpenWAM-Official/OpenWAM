"""Compute RoboCasa GR1 normalization stats with directional hand blocks.

The reader auto-builds this file on first use at its fixed location
(``<dataset_dir>/meta/robocasa_gr1_normalization_stats.npy``; see
``RoboCasaGR1Dataset.from_config``), so this CLI is only needed to force a
rebuild or to write the stats somewhere else.

Arm pose and waist dimensions remain pooled across action/state. Fourier-hand
action is a discrete command, whereas hand state is a continuous joint angle;
those 12 dimensions are therefore stored separately under ``eef`` (action) and
``eef_state`` (proprio).

Example:
    python -m openwam.dataloader.utils.stats_computation.robocasa_gr1_stats_computation \
      --config configs/dataloader/robocasa_gr1.yaml
"""

from __future__ import annotations

import argparse
import os
import pickle
import socket
import uuid
from pathlib import Path
from typing import Iterable

import numpy as np
from omegaconf import OmegaConf

from openwam.dataloader.robocasa_gr1 import (
    NORMALIZATION_STATS_FILENAME,
    STATS_SCHEMA_VERSION,
    MultiRoboCasaGR1Dataset,
    RoboCasaGR1Dataset,
)
from openwam.dataloader.utils.gr1_kinematics import HAND_DIMS_EEF33, ROT6D_DIMS_EEF33
from openwam.dataloader.utils.normalization import pin_rot6d_identity
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
        yield bucket._raw_action(win), bucket._raw_state(win)  # noqa: SLF001


def _compute_global_stats(dataset, reservoir_cap: int):
    """Share pose/waist stats while keeping command-valued hand dims directional.

    GR1 arm and waist actions are physical targets in the same space as state,
    but the 12 Fourier-hand action dims are discrete commands while state holds
    continuous joint angles. Pooling those unlike hand distributions compresses
    proprio and changes the meaning of one affine transform by direction.
    """
    buckets = list(_iter_buckets(dataset))
    if not buckets:
        raise ValueError("RoboCasa GR1 dataset has no buckets")
    action_modes = {bucket.action_mode for bucket in buckets}
    raw_dims = {bucket._raw_action_dim for bucket in buckets}  # noqa: SLF001
    if len(action_modes) != 1 or len(raw_dims) != 1:
        raise ValueError(f"stats require homogeneous modes/dims, got modes={action_modes}, dims={raw_dims}")
    action_mode = next(iter(action_modes))
    raw_dim = next(iter(raw_dims))

    pooled_accumulator = Accumulator(dim=raw_dim, reservoir_cap=reservoir_cap)
    action_accumulator = Accumulator(dim=raw_dim, reservoir_cap=reservoir_cap)
    state_accumulator = Accumulator(dim=raw_dim, reservoir_cap=reservoir_cap)
    action_rows = 0
    state_rows = 0
    for bucket in buckets:
        for action, state in _iter_bucket_arrays(bucket):
            action = np.asarray(action, dtype=np.float32).reshape(-1, raw_dim)
            state = np.asarray(state, dtype=np.float32).reshape(-1, raw_dim)
            pooled_accumulator.update_batch(action)
            pooled_accumulator.update_batch(state)
            action_accumulator.update_batch(action)
            state_accumulator.update_batch(state)
            action_rows += action.shape[0]
            state_rows += state.shape[0]
    if action_rows == 0 or state_rows == 0:
        raise ValueError("cannot compute normalization stats from an empty dataset")

    pooled_stats = pooled_accumulator.finalize()
    action_only = action_accumulator.finalize()
    state_only = state_accumulator.finalize()
    action_stats = {}
    state_stats = {}
    hand_dims = np.asarray(HAND_DIMS_EEF33, dtype=np.int64)
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        action_stats[key] = np.asarray(pooled_stats[key], dtype=np.float32).copy()
        state_stats[key] = np.asarray(pooled_stats[key], dtype=np.float32).copy()
        action_stats[key][hand_dims] = np.asarray(action_only[key], dtype=np.float32)[hand_dims]
        state_stats[key][hand_dims] = np.asarray(state_only[key], dtype=np.float32)[hand_dims]
    for stats, stream in ((action_stats, "action"), (state_stats, "state")):
        stats["num_timesteps"] = pooled_accumulator.count
        stats["pool"] = "action_state_except_hand"
        stats["hand_pool"] = stream
        stats["action_rows"] = action_rows
        stats["state_rows"] = state_rows
    if action_mode == "eef":
        pin_rot6d_identity(action_stats, ROT6D_DIMS_EEF33)
        pin_rot6d_identity(state_stats, ROT6D_DIMS_EEF33)
    return action_mode, raw_dim, action_stats, state_stats, action_rows, state_rows


def build_and_save_robocasa_gr1_stats(dataset, output: str | Path, reservoir_cap: int = 1_000_000):
    """Compute pooled stats for ``dataset`` and atomically write ``output``.

    Shared by the CLI below and the reader's fixed-path auto-build (rank 0 in
    ``RoboCasaGR1Dataset.from_config``): the tmp-file + ``os.replace`` write
    means concurrently polling ranks never observe a torn file. Returns
    ``(action_mode, raw_dim, action_rows, state_rows)`` for the caller's report.
    """
    output = Path(output)
    action_mode, raw_dim, action_stats, state_stats, action_rows, state_rows = _compute_global_stats(
        dataset, reservoir_cap
    )

    payload = {}
    if output.exists():
        try:
            previous = np.load(output, allow_pickle=True).item()
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            # Compatibility checking intentionally routes unreadable/truncated
            # files into this rebuild path. Treat them as having no reusable
            # payload so the atomic write below can replace them.
            previous = None
        if isinstance(previous, dict):
            payload.update(previous)
    payload[action_mode] = action_stats
    payload[f"{action_mode}_state"] = state_stats
    payload["robocasa_gr1_stats_schema"] = STATS_SCHEMA_VERSION
    output.parent.mkdir(parents=True, exist_ok=True)
    # pid alone collides across nodes on a shared filesystem; qualify with
    # hostname + uuid like the LeRobotV3Reader deploy-stats writer.
    tmp_path = output.with_name(f".{output.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp_path.open("wb") as f:
            np.save(f, payload)
        os.replace(tmp_path, output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return action_mode, raw_dim, action_rows, state_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataloader/robocasa_gr1.yaml")
    parser.add_argument(
        "--output",
        default=None,
        help=f"Output .npy path (default: <dataset_dir>/meta/{NORMALIZATION_STATS_FILENAME}, "
        "the fixed location the reader loads)",
    )
    parser.add_argument("--reservoir-cap", type=int, default=1_000_000)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "normalize_mode", None, merge=False)
    dataset_dir = OmegaConf.select(cfg, "dataset_dir")
    if dataset_dir is None:
        raise ValueError(f"{args.config} has no dataset_dir")
    output = Path(args.output) if args.output else Path(str(dataset_dir)) / "meta" / NORMALIZATION_STATS_FILENAME
    if output.suffix != ".npy":
        raise ValueError("--output must end in .npy so checkpoints receive a deploy-compatible artifact")

    dataset = RoboCasaGR1Dataset.from_config(cfg, split=str(OmegaConf.select(cfg, "split", default="train")))
    action_mode, raw_dim, action_rows, state_rows = build_and_save_robocasa_gr1_stats(
        dataset, output, args.reservoir_cap
    )
    print(
        f"wrote {output} mode={action_mode} pool=action_state_except_hand dim={raw_dim} "
        f"action_rows={action_rows} state_rows={state_rows} total_rows={action_rows + state_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
