"""Compute shared LIBERO EEF10 action/state normalization statistics.

Action targets (next-frame achieved pose + recorded gripper command) and
achieved proprio rows are accumulated into one global ``eef`` statistics block.
Both training directions therefore use exactly the same transform. rot6d dims
(3:9) are pinned to identity so normalization never distorts rotation.

The gripper dim (9) IS normalized, so these numbers are tied to the reader's
gripper direction; the payload records it as ``gripper_convention`` and the
reader validates the match at load time.

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

from openwam.dataloader.libero import (
    GRIPPER_CONVENTION,
    ROT6D_DIMS_EEF10,
    LiberoDataset,
    MultiLiberoDataset,
)
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


def _compute_global_stats(dataset, reservoir_cap: int):
    """Pool every valid action and state row into one EEF10 accumulator."""
    buckets = list(_iter_buckets(dataset))
    if not buckets:
        raise ValueError("LIBERO dataset has no buckets")
    raw_dims = {bucket._raw_action_dim for bucket in buckets}  # noqa: SLF001
    if len(raw_dims) != 1:
        raise ValueError(f"stats require homogeneous raw dims, got {raw_dims}")
    raw_dim = next(iter(raw_dims))
    action_mode = buckets[0].action_mode

    accumulator = Accumulator(dim=raw_dim, reservoir_cap=reservoir_cap)
    action_rows = 0
    state_rows = 0
    for bucket in buckets:
        for action, state in _iter_bucket_arrays(bucket):
            action = np.asarray(action, np.float32).reshape(-1, raw_dim)
            state = np.asarray(state, np.float32).reshape(-1, raw_dim)
            if action.shape[0]:
                accumulator.update_batch(action)
                action_rows += action.shape[0]
            accumulator.update_batch(state)
            state_rows += state.shape[0]
    if action_rows == 0 or state_rows == 0:
        raise ValueError("cannot compute LIBERO stats from an empty dataset")

    stats = accumulator.finalize()
    stats["num_timesteps"] = accumulator.count
    stats["pool"] = "action_state"
    stats["action_rows"] = action_rows
    stats["state_rows"] = state_rows
    # Bind the gripper direction these numbers were accumulated under, so the
    # reader can refuse a file computed before/after the open-scale flip
    # (LiberoDataset._check_gripper_convention).
    stats["gripper_convention"] = GRIPPER_CONVENTION
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF10)
    return action_mode, raw_dim, stats, action_rows, state_rows


def build_and_save_libero_stats(dataset, output: str | Path, reservoir_cap: int = 1_000_000):
    """Compute pooled stats for ``dataset`` and atomically write ``output``.

    Shared by the CLI below and the reader's default-path auto-build (rank 0
    during ``__init__``; see ``LiberoDataset._build_default_stats``): the
    tmp-file + ``os.replace`` write means concurrently polling ranks never
    observe a torn file. Returns ``(action_mode, raw_dim, action_rows,
    state_rows)`` for the caller's report.
    """
    import os
    import socket
    import uuid

    output = Path(output)
    action_mode, raw_dim, global_stats, action_rows, state_rows = _compute_global_stats(dataset, reservoir_cap)

    payload = {}
    if output.exists():
        previous = np.load(output, allow_pickle=True).item()
        if isinstance(previous, dict):
            payload.update(previous)
    payload.pop(f"{action_mode}_state", None)
    payload[action_mode] = global_stats
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

    action_mode, raw_dim, action_rows, state_rows = build_and_save_libero_stats(
        dataset, output, args.reservoir_cap
    )
    print(
        f"wrote {output} mode={action_mode} pool=action_state dim={raw_dim} "
        f"action_rows={action_rows} state_rows={state_rows} total_rows={action_rows + state_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
