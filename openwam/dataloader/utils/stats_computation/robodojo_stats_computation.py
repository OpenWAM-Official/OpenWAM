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
    ROBODOJO_CONTRACT_ID,
    ROBODOJO_REAL_CONTRACT_ID,
    ROBODOJO_REAL_SOURCE_FRAME,
    ROBODOJO_REAL_VARIANT,
    ROBODOJO_SIM_SOURCE_FRAME,
    ROBODOJO_SIM_VARIANT,
    calibration_fingerprint,
    read_calibrated_eef20,
    real_frame_contract_fingerprint,
    resolve_robodojo_tasks,
    validate_robodojo_episode,
)
from openwam.dataloader.robodojo_contract import (
    EEF20_DIM,
    ENDPOINT_LINK_NAME,
    REAL_ENDPOINT_NAME,
    ROBODOJO_EMBODIMENT,
    ROBODOJO_TARGET_FRAME,
    discover_episodes,
    resolve_robodojo_calibration,
    validate_embodiment,
)
from openwam.dataloader.utils.normalization import (
    ROT6D_DIMS_EEF20,
    STAT_KEYS,
    pin_rot6d_identity,
)
from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import (
    Accumulator,
)

DEFAULT_RESERVOIR_CAP = 1_000_000


def iter_episode_eef20(
    episode_paths: Iterable[str | Path],
    calibration: Mapping | None,
    *,
    variant: str = ROBODOJO_SIM_VARIANT,
    embodiment: str = ROBODOJO_EMBODIMENT,
):
    """Yield each episode's validated release-specific EEF20 state rows."""
    for episode_path in episode_paths:
        path = Path(episode_path)
        validate_robodojo_episode(path, variant=variant)
        with h5py.File(path, "r") as handle:
            yield read_calibrated_eef20(
                handle,
                calibration,
                variant=variant,
                embodiment=embodiment,
            )


def _metadata(
    *,
    tasks: Sequence[str],
    calibration: Mapping | None,
    variant: str,
    embodiment: str,
    state_rows: int,
    action_rows: int,
    reservoir_cap: int,
    reservoir_rows: int,
) -> dict:
    metadata = {
        "pool": "action_state",
        "action_rows": int(action_rows),
        "state_rows": int(state_rows),
        "num_timesteps": int(action_rows + state_rows),
        "variant": variant,
        "target_frame": ROBODOJO_TARGET_FRAME,
        "embodiment": embodiment,
        "tasks": list(tasks),
        "gripper_convention": GRIPPER_CONVENTION,
        "action_target": "state[1:T]",
        "reservoir_cap": int(reservoir_cap),
        "reservoir_rows": int(reservoir_rows),
    }
    if variant == ROBODOJO_REAL_VARIANT:
        metadata.update(
            {
                "source_frame": ROBODOJO_REAL_SOURCE_FRAME,
                "endpoint": REAL_ENDPOINT_NAME,
                "pose_transform": "identity_before_quaternion_to_rot6d",
                "frame_contract_fingerprint": real_frame_contract_fingerprint(embodiment),
                "contract_id": ROBODOJO_REAL_CONTRACT_ID,
                "gripper_preprocessing": "clip_sensor_noise_to_[0,1]",
            }
        )
    else:
        if calibration is None:
            raise ValueError("RoboDojo sim stats metadata requires calibration")
        metadata.update(
            {
                "source_frame": ROBODOJO_SIM_SOURCE_FRAME,
                "endpoint": ENDPOINT_LINK_NAME,
                "pose_transform": "env_origin_world_to_per_arm_robot_base",
                "calibration_fingerprint": calibration_fingerprint(calibration),
                "contract_id": ROBODOJO_CONTRACT_ID,
                "gripper_preprocessing": "clip_float_noise_to_[0,1]",
            }
        )
    return metadata


def compute_robodojo_stats(
    dataset_dir: str | Path,
    *,
    calibration: Mapping | None = None,
    calibration_path: str | Path | None = None,
    tasks: Sequence[str] | None = None,
    embodiment: str = ROBODOJO_EMBODIMENT,
    variant: str = ROBODOJO_SIM_VARIANT,
    action_mode: str = DEPLOY_ACTION_MODE,
    reservoir_cap: int = DEFAULT_RESERVOIR_CAP,
) -> dict:
    """Compute bounded pooled raw-20D statistics across selected formal tasks."""
    if action_mode != DEPLOY_ACTION_MODE:
        raise ValueError(f"RoboDojo stats support only action_mode='eef', got {action_mode!r}")
    validate_embodiment(embodiment, variant=variant)
    if int(reservoir_cap) < 1:
        raise ValueError(f"reservoir_cap must be >= 1, got {reservoir_cap}")

    if calibration_path is not None:
        raise ValueError("RoboDojo uses a built-in frame contract; calibration_path is not accepted")
    if variant == ROBODOJO_REAL_VARIANT:
        if calibration is not None:
            raise ValueError("RoboDojo_real uses native per-arm base poses; calibration must be None")
        resolved_calibration = None
    else:
        resolved_calibration = resolve_robodojo_calibration(calibration)
    tasks = resolve_robodojo_tasks(
        dataset_dir,
        tasks=tasks,
        embodiment=embodiment,
        variant=variant,
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
            variant=variant,
        )
        for states in iter_episode_eef20(
            episode_paths,
            resolved_calibration,
            variant=variant,
            embodiment=embodiment,
        ):
            states = np.asarray(states, dtype=np.float32).reshape(-1, EEF20_DIM)
            if states.shape[0] < 2:
                raise ValueError(f"cannot compute RoboDojo stats from an episode with {states.shape[0]} state rows")
            targets = states[1:]
            accumulator.update_batch(states)
            accumulator.update_batch(targets)
            state_rows += states.shape[0]
            action_rows += targets.shape[0]

    if state_rows == 0 or action_rows == 0 or accumulator.count == 0:
        raise ValueError("cannot compute RoboDojo stats from an empty dataset")

    eef = {
        key: np.asarray(value, dtype=np.float32) for key, value in accumulator.finalize().items() if key in STAT_KEYS
    }
    pin_rot6d_identity(eef, ROT6D_DIMS_EEF20)
    metadata = _metadata(
        tasks=tasks,
        calibration=resolved_calibration,
        variant=variant,
        embodiment=embodiment,
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
    temporary = output.with_name(f".{output.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
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
        raise ValueError(f"stats config type must be 'robodojo', got {config.get('type')!r}")
    dataset_dir = config.get("dataset_dir", config.get("dataset_root"))
    if not dataset_dir:
        raise ValueError(f"{args.config} has no dataset_dir")

    # Deliberately do not construct a normalizing reader here: scans consume the
    # shared raw conversion function and never load or auto-build stats.
    build_and_save_robodojo_stats(
        dataset_dir=dataset_dir,
        output=output,
        embodiment=str(config.get("embodiment", ROBODOJO_EMBODIMENT)),
        variant=str(config.get("variant", ROBODOJO_SIM_VARIANT)),
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
