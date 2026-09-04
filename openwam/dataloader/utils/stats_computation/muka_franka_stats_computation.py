"""Compute MUKA Franka EEF10 action/state normalization statistics.

State rows and real next-state action targets are pooled after the exact reader
conversion (Euler XYZ -> rot6d and closedness -> ``[-1 closed, +1 open]``).
The repeated terminal action row of each episode is excluded.  rot6d dimensions
3..8 are pinned to identity for every normalization mode.

Example::

    python -m openwam.dataloader.utils.stats_computation.muka_franka_stats_computation \
      --config configs/dataloader/pretrain_data/muka_franka.yaml \
      --output /path/to/muka_franka_lerobot_v3/meta/normalization_stats.npy
"""

from __future__ import annotations

import argparse
import os
import socket
import uuid
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from openwam.dataloader.muka_franka import (
    GRIPPER_CONVENTION,
    GRIPPER_TRANSFORM,
    RAW_GRIPPER_CONVENTION,
    ROT6D_DIMS_EEF10,
    ROTATION_CONVENTION,
    STATS_POPULATION,
    MukaFrankaDataset,
)
from openwam.dataloader.utils.normalization import pin_rot6d_identity
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator


def _iter_bucket_arrays(bucket: MukaFrankaDataset):
    """Yield per-episode ``(real_action_eef10, state_eef10)`` arrays."""
    for pos, (_, row) in enumerate(bucket._eps_df.iterrows()):  # noqa: SLF001
        table = bucket._load_data_table(  # noqa: SLF001
            int(row["data/chunk_index"]), int(row["data/file_index"])
        )
        offset = int(bucket._ep_data_row_offset[pos])  # noqa: SLF001
        win = table.slice(offset, int(row["length"])).to_pandas()
        action = bucket._raw_action_eef10(win)[:-1]  # noqa: SLF001 - terminal row is repeated.
        state = bucket._raw_state_eef10(win)  # noqa: SLF001
        yield action, state


def _compute_global_stats(dataset: MukaFrankaDataset, reservoir_cap: int):
    if not isinstance(dataset, MukaFrankaDataset):
        raise TypeError(f"expected MukaFrankaDataset, got {type(dataset).__name__}")
    raw_dim = dataset._raw_action_dim  # noqa: SLF001
    accumulator = Accumulator(dim=raw_dim, reservoir_cap=reservoir_cap)
    action_rows = 0
    state_rows = 0
    for action, state in _iter_bucket_arrays(dataset):
        action = np.asarray(action, np.float32).reshape(-1, raw_dim)
        state = np.asarray(state, np.float32).reshape(-1, raw_dim)
        accumulator.update_batch(action)
        accumulator.update_batch(state)
        action_rows += action.shape[0]
        state_rows += state.shape[0]
    if action_rows == 0 or state_rows == 0:
        raise ValueError("cannot compute MUKA Franka stats from an empty dataset")

    stats = accumulator.finalize()
    stats.update(
        {
            "num_timesteps": accumulator.count,
            "pool": "real_action_state",
            "action_rows": action_rows,
            "state_rows": state_rows,
            "gripper_convention": GRIPPER_CONVENTION,
            "raw_gripper_convention": RAW_GRIPPER_CONVENTION,
            "gripper_transform": GRIPPER_TRANSFORM,
            "rotation_convention": ROTATION_CONVENTION,
            "rotation_transform": "euler_xyz_to_rot6d",
            "action_alignment": "action[t]=state[t+1]; repeated terminal action excluded",
            "stats_population": STATS_POPULATION,
            "split": dataset._split,  # noqa: SLF001
            "num_episodes": len(dataset._eps_df),  # noqa: SLF001
        }
    )
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF10)
    return dataset.action_mode, raw_dim, stats, action_rows, state_rows


def build_and_save_muka_franka_stats(
    dataset: MukaFrankaDataset,
    output: str | Path,
    reservoir_cap: int = 1_000_000,
):
    """Compute stats and atomically merge the EEF block into ``output``."""
    output = Path(output)
    action_mode, raw_dim, global_stats, action_rows, state_rows = _compute_global_stats(dataset, reservoir_cap)
    payload = {}
    if output.exists():
        try:
            previous = np.load(output, allow_pickle=True).item()
            if isinstance(previous, dict):
                payload.update(previous)
        except (ValueError, EOFError):
            pass
    payload[action_mode] = global_stats
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f".{output.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, payload)
        os.replace(tmp_path, output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return action_mode, raw_dim, action_rows, state_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataloader/pretrain_data/muka_franka.yaml")
    parser.add_argument("--output", required=True, help="Deploy-compatible .npy output")
    parser.add_argument("--reservoir-cap", type=int, default=1_000_000)
    args = parser.parse_args()

    output = Path(args.output)
    if output.suffix != ".npy":
        raise ValueError("--output must end in .npy")
    cfg = OmegaConf.load(args.config)
    OmegaConf.update(cfg, "normalize_mode", None, merge=False)
    dataset = MukaFrankaDataset.from_config(cfg, split=str(OmegaConf.select(cfg, "split", default="train")))
    action_mode, raw_dim, action_rows, state_rows = build_and_save_muka_franka_stats(
        dataset, output, args.reservoir_cap
    )
    print(
        f"wrote {output} mode={action_mode} pool=real_action_state dim={raw_dim} "
        f"action_rows={action_rows} state_rows={state_rows} "
        f"total_rows={action_rows + state_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
