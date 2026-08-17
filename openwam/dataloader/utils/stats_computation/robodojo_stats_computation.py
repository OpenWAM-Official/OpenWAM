"""Compute pooled raw-EEF20 normalization statistics for formal RoboDojo data.

The pool contains every achieved state row and, separately, each real
next-state target (episode rows ``1:T``).  Conversion is delegated to the
RoboDojo reader's canonical calibrated EEF20 function so the statistics cannot
drift from training inputs.
"""

from __future__ import annotations

import argparse
import os
import socket
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from openwam.dataloader.robodojo import (
    DEPLOY_ACTION_MODE,
    GRIPPER_CONVENTION,
    ROBODOJO_SOURCE_FRAME,
    calibration_fingerprint,
    read_calibrated_eef20,
    resolve_robodojo_tasks,
    validate_robodojo_episode,
)
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_EEF20,
    STAT_KEYS,
    pin_rot6d_identity,
)
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import (
    Accumulator,
)
from benchmarks.robodojo.contract import (
    EEF20_DIM,
    ENDPOINT_LINK_NAME,
    ROBODOJO_EMBODIMENT,
    discover_episodes,
    resolve_robodojo_calibration,
    validate_embodiment,
)

DEFAULT_RESERVOIR_CAP = 1_000_000


def iter_episode_eef20(
    episode_paths: Iterable[str | Path],
    calibration: Mapping,
):
    """Yield each episode's validated raw calibrated EEF20 state rows."""
    for episode_path in episode_paths:
        path = Path(episode_path)
        validate_robodojo_episode(path)
        with h5py.File(path, "r") as handle:
            yield read_calibrated_eef20(handle, calibration)


def _metadata(
    *,
    tasks: Sequence[str],
    calibration: Mapping,
    state_rows: int,
    action_rows: int,
    reservoir_cap: int,
    reservoir_rows: int,
) -> dict:
    return {
        "pool": "action_state",
        "action_rows": int(action_rows),
        "state_rows": int(state_rows),
        "num_timesteps": int(action_rows + state_rows),
        "source_frame": ROBODOJO_SOURCE_FRAME,
        "target_frame": "per_arm_robot_base",
        "endpoint": ENDPOINT_LINK_NAME,
        "embodiment": ROBODOJO_EMBODIMENT,
        "tasks": list(tasks),
        "calibration_fingerprint": calibration_fingerprint(calibration),
        "gripper_convention": GRIPPER_CONVENTION,
        "reservoir_cap": int(reservoir_cap),
        "reservoir_rows": int(reservoir_rows),
    }


def compute_robodojo_stats(
    dataset_dir: str | Path,
    *,
    calibration: Mapping | None = None,
    calibration_path: str | Path | None = None,
    task_name: str | None = None,
    train_tasks: Sequence[str] | None = None,
    holdout_tasks: Sequence[str] | None = None,
    split: str = "train",
    embodiment: str = ROBODOJO_EMBODIMENT,
    action_mode: str = DEPLOY_ACTION_MODE,
    reservoir_cap: int = DEFAULT_RESERVOIR_CAP,
) -> dict:
    """Compute bounded pooled raw-20D statistics across selected formal tasks."""
    if action_mode != DEPLOY_ACTION_MODE:
        raise ValueError(
            f"RoboDojo stats support only action_mode='eef', got {action_mode!r}"
        )
    validate_embodiment(embodiment)
    if int(reservoir_cap) < 1:
        raise ValueError(f"reservoir_cap must be >= 1, got {reservoir_cap}")

    if calibration_path is not None:
        raise ValueError(
            "RoboDojo uses the built-in dual-X5 base constants; "
            "calibration_path is not accepted"
        )
    calibration = resolve_robodojo_calibration(calibration)
    tasks = resolve_robodojo_tasks(
        dataset_dir,
        split=split,
        task_name=task_name,
        train_tasks=train_tasks,
        holdout_tasks=holdout_tasks,
        embodiment=embodiment,
    )

    accumulator = Accumulator(
        dim=EEF20_DIM,
        reservoir_cap=int(reservoir_cap),
        seed=0,
    )
    state_rows = 0
    action_rows = 0
    for task in tasks:
        episode_paths = discover_episodes(
            dataset_dir,
            task,
            embodiment=embodiment,
        )
        for states in iter_episode_eef20(episode_paths, calibration):
            states = np.asarray(states, dtype=np.float32).reshape(-1, EEF20_DIM)
            if states.shape[0] < 2:
                raise ValueError(
                    f"cannot compute RoboDojo stats from an episode with "
                    f"{states.shape[0]} state rows"
                )
            targets = states[1:]
            accumulator.update_batch(states)
            accumulator.update_batch(targets)
            state_rows += states.shape[0]
            action_rows += targets.shape[0]

    if state_rows == 0 or action_rows == 0 or accumulator.count == 0:
        raise ValueError("cannot compute RoboDojo stats from an empty dataset")

    eef = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in accumulator.finalize().items()
        if key in STAT_KEYS
    }
    pin_rot6d_identity(eef, ROT6D_DIMS_EEF20)
    metadata = _metadata(
        tasks=tasks,
        calibration=calibration,
        state_rows=state_rows,
        action_rows=action_rows,
        reservoir_cap=int(reservoir_cap),
        reservoir_rows=min(accumulator.count, int(reservoir_cap)),
    )
    # Keep operational metadata both beside the nested deploy stats and inside
    # it. Existing deploy loaders consume only the six vectors, while stats
    # tooling in this repository conventionally reads pool/count fields from
    # the active mode block.
    eef.update(metadata)
    return {
        DEPLOY_ACTION_MODE: eef,
        "metadata": metadata,
        "num_timesteps": metadata["num_timesteps"],
    }


def compute_normalization_stats(*args, **kwargs) -> dict:
    """Compatibility alias for callers using the repository-wide naming."""
    return compute_robodojo_stats(*args, **kwargs)


def atomic_save_stats_npy(path: str | Path, payload: dict) -> None:
    """Atomically write a deploy-compatible ``.npy`` statistics payload."""
    output = Path(path)
    if output.suffix != ".npy":
        raise ValueError("RoboDojo stats output must end in .npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.{socket.gethostname()}.{os.getpid()}."
        f"{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            np.save(handle, payload, allow_pickle=True)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_and_save_robodojo_stats(
    dataset_dir: str | Path,
    output: str | Path,
    *,
    calibration: Mapping | None = None,
    calibration_path: str | Path | None = None,
    **kwargs,
) -> Path:
    """Compute RoboDojo stats and atomically replace ``output``."""
    output_path = Path(output)
    if output_path.suffix != ".npy":
        raise ValueError("RoboDojo stats output must end in .npy")
    payload = compute_robodojo_stats(
        dataset_dir=dataset_dir,
        calibration=calibration,
        calibration_path=calibration_path,
        **kwargs,
    )
    atomic_save_stats_npy(output_path, payload)
    return output_path


def _load_config(path: str | Path) -> dict:
    config = OmegaConf.to_container(
        OmegaConf.load(path),
        resolve=True,
    )
    if not isinstance(config, dict):
        raise ValueError(f"RoboDojo config must contain a mapping: {path}")
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/dataloader/robodojo.yaml",
        help="RoboDojo dataloader YAML",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Deploy-compatible .npy output",
    )
    parser.add_argument(
        "--reservoir-cap",
        type=int,
        default=DEFAULT_RESERVOIR_CAP,
    )
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.suffix != ".npy":
        raise ValueError("--output must end in .npy")
    config = _load_config(args.config)
    if config.get("type", "robodojo") != "robodojo":
        raise ValueError(
            f"stats config type must be 'robodojo', got {config.get('type')!r}"
        )
    dataset_dir = config.get("dataset_dir", config.get("dataset_root"))
    if not dataset_dir:
        raise ValueError(f"{args.config} has no dataset_dir")

    # Deliberately do not construct a normalizing reader here: scans consume the
    # shared raw conversion function and never load or auto-build stats.
    build_and_save_robodojo_stats(
        dataset_dir=dataset_dir,
        output=output,
        task_name=config.get("task_name"),
        train_tasks=config.get("train_tasks"),
        holdout_tasks=config.get("holdout_tasks"),
        split=str(config.get("split", "train")),
        embodiment=str(config.get("embodiment", ROBODOJO_EMBODIMENT)),
        action_mode=str(config.get("action_mode", DEPLOY_ACTION_MODE)),
        reservoir_cap=args.reservoir_cap,
    )
    payload = np.load(output, allow_pickle=True).item()
    metadata = payload["metadata"]
    print(
        f"wrote {output} mode=eef pool=action_state dim={EEF20_DIM} "
        f"action_rows={metadata['action_rows']} "
        f"state_rows={metadata['state_rows']} "
        f"total_rows={metadata['num_timesteps']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RESERVOIR_CAP",
    "atomic_save_stats_npy",
    "build_and_save_robodojo_stats",
    "compute_normalization_stats",
    "compute_robodojo_stats",
    "iter_episode_eef20",
    "main",
]
