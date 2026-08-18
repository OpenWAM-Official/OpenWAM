"""Extract RoboDojo X5 base transforms from an already-live evaluation env.

This module intentionally has no RoboDojo or Isaac imports at module scope.
Callers create their EvalEnv through the normal RoboDojo launcher, then pass it
to :func:`calibrate_and_save`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from openwam.dataloader.robodojo_contract import (
    CALIBRATION_SCHEMA_VERSION,
    ENDPOINT_LINK_NAME,
    ENDPOINT_POSE_FRAME_CONTRACT,
    ROBODOJO_EMBODIMENT,
    save_calibration,
    validate_calibration,
)


def _to_numpy(value: Any, name: str) -> np.ndarray:
    """Move a tensor-like value to CPU lazily, without importing torch."""
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    array = np.asarray(converted)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return array


def _target_x5_robots(eval_env: Any) -> dict[str, Any]:
    try:
        robots = eval_env.robot_manager.robot_list
    except AttributeError as error:
        raise ValueError(
            "eval_env must expose robot_manager.robot_list"
        ) from error

    targets = [robot for robot in robots if getattr(robot, "type", None) == "target"]
    valid_dual_x5 = (
        len(targets) == 2
        and all(getattr(robot, "robot_type", None) == "arm" for robot in targets)
        and all(getattr(robot, "robot_name", None) == "x5" for robot in targets)
    )
    if not valid_dual_x5:
        raise ValueError(
            "live calibration requires the dual-arm arx_x5 configuration "
            "with exactly two target x5 arms"
        )

    by_arm_name = {getattr(robot, "arm_name", None): robot for robot in targets}
    if set(by_arm_name) != {"left_arm", "right_arm"}:
        raise ValueError(
            "dual-arm arx_x5 targets must be named left_arm and right_arm; "
            f"got {sorted(str(name) for name in by_arm_name)}"
        )
    return {"left": by_arm_name["left_arm"], "right": by_arm_name["right_arm"]}


def _live_env_origins(eval_env: Any) -> tuple[Any, str]:
    """Return reset-time origins from the current RoboDojo API or a fallback."""

    direct = getattr(eval_env, "env_origins", None)
    if direct is not None:
        return direct, "env_origins"

    sim = getattr(eval_env, "sim", None)
    sim_scene = getattr(sim, "scene", None)
    simulated = getattr(sim_scene, "env_origins", None)
    if simulated is not None:
        return simulated, "sim.scene.env_origins"

    # Retain compatibility with older wrappers and lightweight external fakes.
    legacy_scene = getattr(eval_env, "scene", None)
    legacy = getattr(legacy_scene, "env_origins", None)
    if legacy is not None:
        return legacy, "scene.env_origins"

    raise RuntimeError(
        "live calibration requires a successful eval_env.reset(...) before "
        "extraction so eval_env.env_origins is available"
    )


def extract_live_calibration(eval_env: Any, *, env_idx: int = 0) -> dict[str, Any]:
    """Measure both target X5 ``base_link`` poses from a live EvalEnv-like object.

    ``RobotManager.get_link_pose(..., is_relative=False)`` returns world
    ``[xyz, qw, qx, qy, qz]``. Only reset-time ``env_origins[env_idx]`` is
    removed from its translation; the orientation remains world-frame.
    """
    if isinstance(env_idx, bool) or not isinstance(env_idx, int) or env_idx < 0:
        raise ValueError(f"env_idx must be a non-negative integer, got {env_idx!r}")
    origins_value, origins_name = _live_env_origins(eval_env)
    origins = _to_numpy(origins_value, origins_name)
    if origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError(
            f"{origins_name} must have shape (num_envs, 3), got {origins.shape}"
        )
    if env_idx >= len(origins):
        raise ValueError(
            f"env_idx {env_idx} is out of range for {len(origins)} environment origins"
        )

    robots = _target_x5_robots(eval_env)
    arms: dict[str, dict[str, list[float]]] = {}
    for side in ("left", "right"):
        robot = robots[side]
        try:
            result = eval_env.robot_manager.get_link_pose(
                robot=robot,
                link_name="base_link",
                env_idx_list=[env_idx],
                is_relative=False,
            )
        except AttributeError as error:
            raise ValueError(
                "eval_env.robot_manager must provide get_link_pose"
            ) from error
        if not isinstance(result, dict) or env_idx not in result or result[env_idx] is None:
            raise ValueError(
                f"get_link_pose did not return a pose for {side} arm env_idx {env_idx}"
            )
        world_pose = _to_numpy(result[env_idx], f"{side} base_link world pose")
        if world_pose.shape != (7,):
            raise ValueError(
                f"{side} base_link world pose must have exact shape (7,), "
                f"got {world_pose.shape}"
            )
        arms[side] = {
            "base_pos_relative_to_env_origin": (
                world_pose[:3] - origins[env_idx]
            ).tolist(),
            "base_quat_wxyz": world_pose[3:7].tolist(),
        }

    return validate_calibration(
        {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "embodiment": ROBODOJO_EMBODIMENT,
            "endpoint": {
                "link_name": ENDPOINT_LINK_NAME,
                "pose_frame_contract": ENDPOINT_POSE_FRAME_CONTRACT,
            },
            "arms": arms,
        }
    )


def calibrate_and_save(
    eval_env: Any,
    output_path: str | Path,
    *,
    env_idx: int = 0,
) -> dict[str, Any]:
    """Extract a live calibration and save the shared versioned JSON."""
    calibration = extract_live_calibration(eval_env, env_idx=env_idx)
    return save_calibration(calibration, output_path)


__all__ = ["calibrate_and_save", "extract_live_calibration"]
