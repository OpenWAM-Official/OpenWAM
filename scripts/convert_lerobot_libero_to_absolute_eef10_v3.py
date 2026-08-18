#!/usr/bin/env python3
"""Convert the official ``lerobot/libero`` v3 dataset to absolute EEF10.

This converter is intentionally separate from
``convert_libero_to_absolute_eef10_v3.py``.  The latter consumes four Fast-WAM
20 Hz buckets with one episode per parquet/video.  The official LeRobot copy is
a single 10 Hz dataset whose parquet and video files contain several episodes.
Those files and the episode offsets must remain packed exactly as they are.

Source rows:

* ``observation.state`` = achieved ``[xyz3, axis_angle3, finger_qpos2]``
* ``action`` = native continuous LIBERO OSC
  ``[delta_xyz3, delta_axis_angle3, close_scale1]``

Output rows:

* ``observation.state`` = achieved ``[xyz3, rot6d6, open_scale1]``
* ``action`` = absolute OSC goal ``[xyz3, rot6d6, open_scale1]``

LIBERO's native gripper direction is ``+1 = close``.  OpenWAM's EEF10
contract is the opposite: ``-1 = closed, +1 = open``.  The native gripper
channel is continuous and is therefore negated without thresholding.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EEF10_DIM = 10
SOURCE_ACTION_DIM = 7
SOURCE_STATE_DIM = 8
LIBERO_GRIPPER_WIDTH_OPEN = 0.08
POSITION_SCALE = 0.05
ROTATION_SCALE = 0.5
OUTPUT_REPRESENTATION = "absolute_eef10"
GRIPPER_CONVENTION = "minus1_closed_plus1_open"
ROT6D_DIMS = tuple(range(3, 9))
DEFAULT_SOURCE_ROOT = Path("/path/to/libero-lerobot")
DEFAULT_OUTPUT_ROOT = Path("/path/to/libero")
DEFAULT_SOURCE_REPO_ID = "lerobot/libero"
DEFAULT_SOURCE_FPS = 10.0
DEFAULT_VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.image2",
)
EEF10_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "rot6d_col0_x",
    "rot6d_col0_y",
    "rot6d_col0_z",
    "rot6d_col1_x",
    "rot6d_col1_y",
    "rot6d_col1_z",
    "gripper_open_scale",
]


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert rotation vectors to rotation matrices with Rodrigues' formula."""
    value = np.asarray(axis_angle, dtype=np.float64)
    if value.shape[-1:] != (3,):
        raise ValueError(f"axis-angle values must end in dimension 3, got {value.shape}")
    angle = np.linalg.norm(value, axis=-1, keepdims=True)
    small = angle[..., 0] < 1e-8
    axis = np.where(angle > 1e-8, value / np.maximum(angle, 1e-8), 0.0)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    cosine = np.cos(angle[..., 0])
    sine = np.sin(angle[..., 0])
    one_minus_cosine = 1.0 - cosine
    matrix = np.empty(value.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = cosine + x * x * one_minus_cosine
    matrix[..., 0, 1] = x * y * one_minus_cosine - z * sine
    matrix[..., 0, 2] = x * z * one_minus_cosine + y * sine
    matrix[..., 1, 0] = y * x * one_minus_cosine + z * sine
    matrix[..., 1, 1] = cosine + y * y * one_minus_cosine
    matrix[..., 1, 2] = y * z * one_minus_cosine - x * sine
    matrix[..., 2, 0] = z * x * one_minus_cosine - y * sine
    matrix[..., 2, 1] = z * y * one_minus_cosine + x * sine
    matrix[..., 2, 2] = cosine + z * z * one_minus_cosine
    matrix[small] = np.eye(3)
    return matrix


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Store the first two columns of a rotation matrix, column by column."""
    value = np.asarray(matrix)
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrices must end in shape (3, 3), got {value.shape}")
    return np.concatenate([value[..., :, 0], value[..., :, 1]], axis=-1).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Project a 6-D rotation representation back onto SO(3)."""
    value = np.asarray(rot6d, np.float64)
    if value.shape[-1:] != (6,):
        raise ValueError(f"rot6d values must end in dimension 6, got {value.shape}")
    first = value[..., 0:3]
    second = value[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1)


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Convert the small relative rotations used by LIBERO back to rotvecs."""
    value = np.asarray(matrix, np.float64)
    cosine = np.clip((np.trace(value, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cosine)
    skew = np.stack(
        [
            value[..., 2, 1] - value[..., 1, 2],
            value[..., 0, 2] - value[..., 2, 0],
            value[..., 1, 0] - value[..., 0, 1],
        ],
        axis=-1,
    )
    scale = np.empty_like(angle)
    small = angle < 1e-7
    scale[small] = 0.5
    scale[~small] = angle[~small] / (2.0 * np.sin(angle[~small]))
    return skew * scale[..., None]


def gripper_qpos_to_open_scale(width: np.ndarray) -> np.ndarray:
    """Map achieved Panda finger separation to ``-1 closed / +1 open``."""
    value = 2.0 * np.asarray(width, np.float64) / LIBERO_GRIPPER_WIDTH_OPEN - 1.0
    return np.clip(value, -1.0, 1.0).astype(np.float32)


def convert_state_action(
    state8: np.ndarray,
    action7: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Convert row-aligned official LeRobot LIBERO state/actions to EEF10.

    ``action7`` is the native normalized OSC command, not a metric pose delta.
    Position and rotation commands are scaled by robosuite's LIBERO OSC_POSE
    output maxima (0.05 m and 0.5 rad) before composing an absolute goal with
    the achieved state on the same row.
    """
    state8 = np.asarray(state8, np.float64)
    action7 = np.asarray(action7, np.float64)
    if state8.ndim != 2 or state8.shape[1] != SOURCE_STATE_DIM:
        raise ValueError(f"source state must be (T, {SOURCE_STATE_DIM}), got {state8.shape}")
    if action7.shape != (state8.shape[0], SOURCE_ACTION_DIM):
        raise ValueError(f"source action must be {(state8.shape[0], SOURCE_ACTION_DIM)}, got {action7.shape}")
    if not np.isfinite(state8).all() or not np.isfinite(action7).all():
        raise ValueError("source state/action contains NaN or infinity")
    if np.any(np.abs(action7[:, :6]) > 1.0 + 1e-5):
        raise ValueError("source OSC position/rotation command falls outside [-1, 1]")
    if np.any(np.abs(action7[:, 6]) > 1.0 + 1e-5):
        raise ValueError("source continuous gripper command falls outside [-1, 1]")

    current_rotation = axis_angle_to_matrix(state8[:, 3:6])
    finger_width = state8[:, 6] - state8[:, 7]
    state_gripper = gripper_qpos_to_open_scale(finger_width)[:, None]
    state10 = np.concatenate(
        [state8[:, 0:3].astype(np.float32), matrix_to_rot6d(current_rotation), state_gripper],
        axis=1,
    ).astype(np.float32)

    delta_rotation = axis_angle_to_matrix(action7[:, 3:6] * ROTATION_SCALE)
    goal_rotation = delta_rotation @ current_rotation
    goal_position = state8[:, 0:3] + action7[:, 0:3] * POSITION_SCALE
    # Native LIBERO: +1 close.  Canonical EEF10: +1 open.  Keep continuity.
    action_gripper = -action7[:, 6:7]
    action10 = np.concatenate(
        [goal_position.astype(np.float32), matrix_to_rot6d(goal_rotation), action_gripper.astype(np.float32)],
        axis=1,
    ).astype(np.float32)

    recovered_position = (action10[:, 0:3].astype(np.float64) - state8[:, 0:3]) / POSITION_SCALE
    rebuilt_goal_rotation = rot6d_to_matrix(action10[:, 3:9])
    recovered_delta_rotation = rebuilt_goal_rotation @ np.swapaxes(current_rotation, -1, -2)
    recovered_rotation = matrix_to_axis_angle(recovered_delta_rotation) / ROTATION_SCALE
    recovered_gripper = -action10[:, 9].astype(np.float64)
    errors = {
        "position": float(np.max(np.abs(recovered_position - action7[:, 0:3]), initial=0.0)),
        "rotation": float(np.max(np.abs(recovered_rotation - action7[:, 3:6]), initial=0.0)),
        "gripper": float(np.max(np.abs(recovered_gripper - action7[:, 6]), initial=0.0)),
    }
    if errors["position"] > 2e-6 or errors["rotation"] > 2e-6 or errors["gripper"] > 1e-7:
        raise ValueError(f"absolute EEF10 round-trip failed: {errors}")
    return state10, action10, errors


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"parquet table is missing {name!r}")
    array = pa.array(values.tolist(), type=pa.list_(pa.float32()))
    return table.set_column(index, name, array)


def _feature_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"statistics require a non-empty 2-D array, got {values.shape}")
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "std": np.std(values, axis=0, dtype=np.float64).tolist(),
        "count": [int(values.shape[0])],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def _pin_rot6d_identity(stats: dict) -> None:
    identity = {"min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0, "mean": 0.0, "std": 1.0}
    for key, value in identity.items():
        for dim in ROT6D_DIMS:
            stats[key][dim] = value


def _copy_independent(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if source.is_symlink() or target.is_symlink():
        raise ValueError(f"independent copy must not be a symlink: {source} -> {target}")
    source_stat = source.stat()
    target_stat = target.stat()
    if source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino:
        raise ValueError(f"independent copy unexpectedly shares an inode: {source} -> {target}")


def _source_revision(source: Path) -> str | None:
    metadata = source / ".cache" / "huggingface" / "download" / "meta" / "info.json.metadata"
    if not metadata.is_file():
        return None
    with metadata.open(encoding="utf-8") as handle:
        revision = handle.readline().strip()
    return revision or None


def _validate_source(
    source: Path,
    info: dict,
    *,
    expected_fps: float = DEFAULT_SOURCE_FPS,
    expected_video_keys: tuple[str, ...] = DEFAULT_VIDEO_KEYS,
) -> tuple[list[Path], list[Path]]:
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    if float(info.get("fps", 0.0)) != float(expected_fps):
        raise ValueError(f"expected a {expected_fps:g} FPS source, got {info.get('fps')!r}")
    features = info.get("features", {})
    if tuple(features.get("observation.state", {}).get("shape", ())) != (SOURCE_STATE_DIM,):
        raise ValueError("source observation.state must be 8-D")
    if tuple(features.get("action", {}).get("shape", ())) != (SOURCE_ACTION_DIM,):
        raise ValueError("source action must be 7-D")
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    if set(video_keys) != set(expected_video_keys):
        raise ValueError(
            f"unexpected video keys: {video_keys}; expected {sorted(expected_video_keys)}"
        )

    data_paths = sorted((source / "data").rglob("*.parquet"))
    video_paths = sorted((source / "videos").rglob("*.mp4"))
    if not data_paths:
        raise FileNotFoundError(f"no source parquet files under {source / 'data'}")
    if not video_paths:
        raise FileNotFoundError(f"no source videos under {source / 'videos'}")
    episodes = sorted((source / "meta" / "episodes").rglob("*.parquet"))
    if not episodes:
        raise FileNotFoundError(f"no episode metadata under {source / 'meta' / 'episodes'}")
    if not (source / "meta" / "tasks.parquet").is_file():
        raise FileNotFoundError(source / "meta" / "tasks.parquet")
    if not (source / "meta" / "stats.json").is_file():
        raise FileNotFoundError(source / "meta" / "stats.json")
    return data_paths, video_paths


def _write_readme(
    path: Path,
    *,
    info: dict,
    source: Path,
    revision: str | None,
    source_repo_id: str = DEFAULT_SOURCE_REPO_ID,
    dataset_title: str = "LIBERO",
) -> None:
    revision_text = revision or "unknown local snapshot"
    text = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# {dataset_title} absolute EEF10 (LeRobot v3.0)

This local dataset was converted from `{source_repo_id}` at revision
`{revision_text}` (local source: `{source}`).  Its original parquet packing,
episode offsets, tasks, {info['fps']:g} FPS timeline, and videos are preserved.

Both primary robot columns use the row-aligned 10-D contract:

* `observation.state`: achieved `[xyz3, rot6d6, gripper_open_scale1]`
* `action`: absolute OSC goal `[xyz3, rot6d6, gripper_open_scale1]`

The gripper convention is `-1 = closed, +1 = open`.  Native LIBERO actions
use the opposite continuous gripper direction.  Absolute action goals are
constructed as `goal_xyz = state_xyz + delta_xyz * 0.05` and
`goal_R = Exp(delta_rotvec * 0.5) @ state_R`.

The sole training/deployment normalization artifact is
`meta/normalization_stats.npy`.

Dataset totals: {info['total_episodes']} episodes, {info['total_frames']} frames,
{info['total_tasks']} tasks, {info['fps']} FPS.  Full provenance and numerical
round-trip checks are recorded in `meta/conversion.json`.
"""
    path.write_text(text, encoding="utf-8")


def convert_dataset(
    source: Path,
    output: Path,
    *,
    workers: int = 8,
    source_repo_id: str = DEFAULT_SOURCE_REPO_ID,
    expected_fps: float = DEFAULT_SOURCE_FPS,
    expected_video_keys: tuple[str, ...] = DEFAULT_VIDEO_KEYS,
    dataset_title: str = "LIBERO",
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output == source or source in output.parents:
        raise ValueError("output must not be the source or live inside the source tree")
    if workers <= 0:
        raise ValueError("workers must be positive")

    info = _read_json(source / "meta" / "info.json")
    data_paths, video_paths = _validate_source(
        source,
        info,
        expected_fps=expected_fps,
        expected_video_keys=expected_video_keys,
    )
    revision = _source_revision(source)
    temp = output.with_name(f".{output.name}.building-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    if temp.exists():
        raise FileExistsError(temp)
    temp.mkdir(parents=True)

    state_arrays: list[np.ndarray] = []
    action_arrays: list[np.ndarray] = []
    total_rows = 0
    max_errors = {"position": 0.0, "rotation": 0.0, "gripper": 0.0}
    state_gripper_projection = {
        "rows_outside_nominal_width": 0,
        "max_width_clip_error_m": 0.0,
        "max_abs_finger_joint_sum": 0.0,
    }
    source_gripper_min = float("inf")
    source_gripper_max = float("-inf")

    try:
        for number, source_data in enumerate(data_paths, 1):
            relative = source_data.relative_to(source)
            target_data = temp / relative
            target_data.parent.mkdir(parents=True, exist_ok=True)
            table = pq.read_table(source_data)
            state8 = np.asarray(table["observation.state"].to_pylist(), np.float32)
            action7 = np.asarray(table["action"].to_pylist(), np.float32)
            state10, action10, errors = convert_state_action(state8, action7)
            for key, value in errors.items():
                max_errors[key] = max(max_errors[key], value)

            finger_width = state8[:, 6].astype(np.float64) - state8[:, 7].astype(np.float64)
            clipped_width = np.clip(finger_width, 0.0, LIBERO_GRIPPER_WIDTH_OPEN)
            state_gripper_projection["rows_outside_nominal_width"] += int(
                np.count_nonzero(finger_width != clipped_width)
            )
            state_gripper_projection["max_width_clip_error_m"] = max(
                state_gripper_projection["max_width_clip_error_m"],
                float(np.max(np.abs(finger_width - clipped_width), initial=0.0)),
            )
            state_gripper_projection["max_abs_finger_joint_sum"] = max(
                state_gripper_projection["max_abs_finger_joint_sum"],
                float(np.max(np.abs(state8[:, 6].astype(np.float64) + state8[:, 7].astype(np.float64)), initial=0.0)),
            )
            source_gripper_min = min(source_gripper_min, float(np.min(action7[:, 6], initial=np.inf)))
            source_gripper_max = max(source_gripper_max, float(np.max(action7[:, 6], initial=-np.inf)))

            converted = _replace_column(table, "observation.state", state10)
            converted = _replace_column(converted, "action", action10)
            pq.write_table(converted, target_data)
            state_arrays.append(state10)
            action_arrays.append(action10)
            total_rows += table.num_rows
            if number % 50 == 0 or number == len(data_paths):
                print(f"converted parquet: {number}/{len(data_paths)} files, {total_rows} rows", flush=True)

        expected_rows = int(info["total_frames"])
        if total_rows != expected_rows:
            raise ValueError(f"source parquet rows={total_rows}, info.total_frames={expected_rows}")

        copy_jobs: list[tuple[Path, Path]] = []
        for source_video in video_paths:
            copy_jobs.append((source_video, temp / source_video.relative_to(source)))
        for relative in (Path(".gitattributes"), Path("meta/tasks.parquet")):
            source_file = source / relative
            if source_file.is_file():
                copy_jobs.append((source_file, temp / relative))
        for source_episode in sorted((source / "meta" / "episodes").rglob("*.parquet")):
            copy_jobs.append((source_episode, temp / source_episode.relative_to(source)))

        print(f"copying {len(copy_jobs)} independent unchanged files with {workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for done, _ in enumerate(executor.map(lambda pair: _copy_independent(*pair), copy_jobs), 1):
                if done % 25 == 0 or done == len(copy_jobs):
                    print(f"copied unchanged files: {done}/{len(copy_jobs)}", flush=True)

        output_features = {key: dict(value) for key, value in info["features"].items()}
        output_features["observation.state"] = {
            "dtype": "float32",
            "shape": [EEF10_DIM],
            "names": {"states": EEF10_NAMES},
            "fps": info["fps"],
        }
        output_features["action"] = {
            "dtype": "float32",
            "shape": [EEF10_DIM],
            "names": {"motors": EEF10_NAMES},
            "fps": info["fps"],
        }
        output_info = dict(info)
        output_info["codebase_version"] = "v3.0"
        output_info["features"] = output_features
        _write_json(temp / "meta" / "info.json", output_info)

        state_all = np.concatenate(state_arrays, axis=0)
        action_all = np.concatenate(action_arrays, axis=0)
        standard_stats = _read_json(source / "meta" / "stats.json")
        standard_stats["observation.state"] = _feature_stats(state_all)
        standard_stats["action"] = _feature_stats(action_all)
        _write_json(temp / "meta" / "stats.json", standard_stats)

        pooled = np.concatenate([action_all, state_all], axis=0)
        eef_stats = _feature_stats(pooled)
        _pin_rot6d_identity(eef_stats)
        eef_stats.update(
            {
                "num_timesteps": int(pooled.shape[0]),
                "pool": "action_state",
                "action_rows": int(action_all.shape[0]),
                "state_rows": int(state_all.shape[0]),
                "gripper_convention": GRIPPER_CONVENTION,
                "representation": OUTPUT_REPRESENTATION,
                "osc_position_scale": POSITION_SCALE,
                "osc_rotation_scale": ROTATION_SCALE,
            }
        )
        # One complete payload is both the training source and the deployment
        # artifact.  Deployment selects its six required vectors from ``eef``
        # and safely ignores the additional metadata and quantiles.
        with (temp / "meta" / "normalization_stats.npy").open("wb") as handle:
            np.save(handle, {"eef": eef_stats}, allow_pickle=True)

        conversion = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source),
            "source_repo_id": source_repo_id,
            "source_revision": revision,
            "output_representation": OUTPUT_REPRESENTATION,
            "state": "achieved [xyz3, rot6d6, gripper_open_scale1]",
            "state_gripper_formula": "clip(2 * (source_qpos0 - source_qpos1) / 0.08 - 1, -1, 1)",
            "state_gripper_projection": {
                **state_gripper_projection,
                "source_joint_pair_is_not_bijectively_recoverable": True,
            },
            "action": "absolute OSC goal [xyz3, rot6d6, gripper_open_scale1]",
            "position_formula": "goal_xyz = state_xyz + source_action_xyz * 0.05",
            "rotation_formula": "goal_R = Exp(source_action_rotvec * 0.5) @ state_R",
            "gripper_formula": "open_scale = -source_native_close_scale",
            "source_gripper_observed_range": [source_gripper_min, source_gripper_max],
            "source_gripper_is_continuous": True,
            "roundtrip_max_abs_error": max_errors,
            "normalization_stats_file": "meta/normalization_stats.npy",
            "preserved": {
                "fps": info["fps"],
                "episodes": info["total_episodes"],
                "frames": info["total_frames"],
                "tasks": info["total_tasks"],
                "parquet_file_count": len(data_paths),
                "video_file_count": len(video_paths),
                "parquet_row_order_and_nonconverted_columns": True,
                "episode_metadata_and_offsets": True,
                "video_bytes_and_timestamps": True,
            },
            "independent_copied_files": True,
            "video_reencoded": False,
        }
        _write_json(temp / "meta" / "conversion.json", conversion)
        _write_readme(
            temp / "README.md",
            info=output_info,
            source=source,
            revision=revision,
            source_repo_id=source_repo_id,
            dataset_title=dataset_title,
        )

        os.replace(temp, output)
        return {
            "output": str(output),
            "episodes": int(info["total_episodes"]),
            "frames": total_rows,
            "tasks": int(info["total_tasks"]),
            "parquet_files": len(data_paths),
            "videos": len(video_paths),
            "source_revision": revision,
            "roundtrip_max_abs_error": max_errors,
        }
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = convert_dataset(args.source, args.output, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
