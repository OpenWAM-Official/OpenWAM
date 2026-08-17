#!/usr/bin/env python3
"""Merge Fast-WAM LIBERO buckets into one independent absolute-EEF10 LeRobot v3 dataset.

The source contract is the Fast-WAM LIBERO conversion:

* ``observation.state``: achieved ``[xyz3, axis_angle3, finger_qpos2]``
* ``action``: normalized OSC ``[delta_xyz3, delta_axis_angle3, open_flag1]``

The output stores the same EEF10 schema in both primary columns:

* ``observation.state``: achieved ``[xyz3, rot6d6, gripper_open_scale1]``
* ``action``: absolute OSC controller goal in that same EEF10 schema

Every output parquet and video is a new file (never a hardlink or symlink), so
the output remains usable after the source buckets are removed.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.libero import (
    EEF10_DIM,
    GRIPPER_CONVENTION,
)
from openwam.dataloader.utils.normalization import pin_rot6d_identity

FILES_PER_CHUNK = 1000
POSITION_SCALE = 0.05
ROTATION_SCALE = 0.5
OUTPUT_REPRESENTATION = "absolute_eef10"
SOURCE_ACTION_DIM = 7
SOURCE_STATE_DIM = 8
LIBERO_GRIPPER_WIDTH_OPEN = 0.08
VIDEO_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
DEFAULT_SOURCE_ROOT = Path("/path/to/libero-fastwam")
DEFAULT_OUTPUT_ROOT = Path("/path/to/libero-fastwam-absolute-eef10-v3")
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
    """Store the first two columns of a rotation matrix."""
    value = np.asarray(matrix)
    return np.concatenate([value[..., :, 0], value[..., :, 1]], axis=-1).astype(np.float32)


def gripper_qpos_to_open_scale(width: np.ndarray) -> np.ndarray:
    """Convert achieved Panda finger separation to -1 closed / +1 open."""
    value = 2.0 * np.asarray(width, np.float64) / LIBERO_GRIPPER_WIDTH_OPEN - 1.0
    return np.clip(value, -1.0, 1.0).astype(np.float32)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _bucket_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if "spatial" in name:
        order = 0
    elif "object" in name:
        order = 1
    elif "goal" in name:
        order = 2
    elif "libero_10" in name:
        order = 3
    else:
        order = 99
    return order, name


def discover_buckets(root: Path) -> list[Path]:
    buckets = sorted(
        (path for path in root.iterdir() if path.is_dir() and (path / "meta" / "info.json").is_file()),
        key=_bucket_sort_key,
    )
    if not buckets:
        raise FileNotFoundError(f"no LeRobot v3 buckets found under {root}")
    return buckets


def _load_episodes(bucket: Path) -> pd.DataFrame:
    paths = sorted((bucket / "meta" / "episodes").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{bucket}: no meta/episodes/*.parquet")
    episodes = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    return episodes.sort_values("episode_index").reset_index(drop=True)


def _load_tasks(bucket: Path) -> dict[int, str]:
    frame = pd.read_parquet(bucket / "meta" / "tasks.parquet")
    return {int(idx): str(task) for idx, task in zip(frame["task_index"], frame.index)}


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"parquet table is missing {name!r}")
    return table.set_column(index, name, values)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rot6d, np.float64)
    first = value[..., 0:3]
    second = value[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1)


def _matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, np.float64)
    cosine = np.clip((np.trace(matrix, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cosine)
    skew = np.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        axis=-1,
    )
    scale = np.empty_like(angle)
    small = angle < 1e-7
    scale[small] = 0.5
    scale[~small] = angle[~small] / (2.0 * np.sin(angle[~small]))
    return skew * scale[..., None]


def convert_state_action(state8: np.ndarray, action7: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Convert one episode and verify the goal can recover the source OSC command."""
    state8 = np.asarray(state8, np.float64)
    action7 = np.asarray(action7, np.float64)
    if state8.ndim != 2 or state8.shape[1] != SOURCE_STATE_DIM:
        raise ValueError(f"source state must be (T, {SOURCE_STATE_DIM}), got {state8.shape}")
    if action7.shape != (state8.shape[0], SOURCE_ACTION_DIM):
        raise ValueError(f"source action must be {(state8.shape[0], SOURCE_ACTION_DIM)}, got {action7.shape}")
    if not np.isfinite(state8).all() or not np.isfinite(action7).all():
        raise ValueError("source state/action contains NaN or infinity")
    if np.any(action7[:, :6] < -1.0 - 1e-5) or np.any(action7[:, :6] > 1.0 + 1e-5):
        raise ValueError("source OSC command falls outside [-1, 1]")
    gripper = action7[:, 6]
    if not np.all(np.isclose(gripper[:, None], np.array([0.0, 1.0]), atol=1e-6).any(axis=1)):
        raise ValueError(f"source gripper is not the Fast-WAM {{0,1}} open flag: {np.unique(gripper)}")

    current_rotation = axis_angle_to_matrix(state8[:, 3:6])
    state_gripper = gripper_qpos_to_open_scale(state8[:, 6] - state8[:, 7])[:, None]
    state10 = np.concatenate(
        [state8[:, 0:3].astype(np.float32), matrix_to_rot6d(current_rotation), state_gripper],
        axis=1,
    ).astype(np.float32)

    delta_rotation = axis_angle_to_matrix(action7[:, 3:6] * ROTATION_SCALE)
    goal_rotation = delta_rotation @ current_rotation
    goal_position = state8[:, 0:3] + action7[:, 0:3] * POSITION_SCALE
    action_gripper = (2.0 * gripper - 1.0)[:, None]
    action10 = np.concatenate(
        [goal_position.astype(np.float32), matrix_to_rot6d(goal_rotation), action_gripper.astype(np.float32)],
        axis=1,
    ).astype(np.float32)

    recovered_position = (action10[:, 0:3].astype(np.float64) - state10[:, 0:3]) / POSITION_SCALE
    rebuilt_goal_rotation = _rot6d_to_matrix(action10[:, 3:9])
    recovered_delta_rotation = rebuilt_goal_rotation @ np.swapaxes(current_rotation, -1, -2)
    recovered_rotation = _matrix_to_axis_angle(recovered_delta_rotation) / ROTATION_SCALE
    recovered_gripper = (action10[:, 9] + 1.0) / 2.0
    errors = {
        "position": float(np.max(np.abs(recovered_position - action7[:, 0:3]))),
        "rotation": float(np.max(np.abs(recovered_rotation - action7[:, 3:6]))),
        "gripper": float(np.max(np.abs(recovered_gripper - gripper))),
    }
    if errors["position"] > 2e-6 or errors["rotation"] > 2e-6 or errors["gripper"] > 1e-7:
        raise ValueError(f"absolute EEF10 round-trip failed: {errors}")
    return state10, action10, errors


def _feature_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values)
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


def _numpy_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return np.asarray(value)
    return value


def _load_legacy_aggregate_stats(legacy_root: Path | None, bucket_names: Iterable[str]) -> dict:
    if legacy_root is None:
        return {}
    episode_stats: list[dict] = []
    for name in bucket_names:
        path = legacy_root / name / "meta" / "episodes_stats.jsonl"
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    episode_stats.append(_numpy_tree(json.loads(line)["stats"]))
    if not episode_stats:
        return {}
    from lerobot.datasets.compute_stats import aggregate_stats

    return aggregate_stats(episode_stats)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _copy_independent(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if source.is_symlink() or target.is_symlink():
        raise ValueError(f"independent copy must not be a symlink: {source} -> {target}")
    source_stat = source.stat()
    target_stat = target.stat()
    if source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino:
        raise ValueError(f"independent copy unexpectedly shares an inode: {source} -> {target}")


def _validate_source_info(bucket: Path, info: dict) -> None:
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{bucket}: expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    if float(info.get("fps", 0)) != 20.0:
        raise ValueError(f"{bucket}: expected Fast-WAM 20 FPS, got {info.get('fps')!r}")
    features = info.get("features", {})
    if tuple(features.get("action", {}).get("shape", ())) != (SOURCE_ACTION_DIM,):
        raise ValueError(f"{bucket}: source action must be 7-D")
    if tuple(features.get("observation.state", {}).get("shape", ())) != (SOURCE_STATE_DIM,):
        raise ValueError(f"{bucket}: source observation.state must be 8-D")
    for key in VIDEO_KEYS:
        if features.get(key, {}).get("dtype") != "video":
            raise ValueError(f"{bucket}: missing video feature {key!r}")


def convert_dataset(
    source_root: Path,
    output: Path,
    *,
    legacy_stats_root: Path | None,
    workers: int,
    episode_limit: int | None,
) -> dict:
    source_root = source_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output == source_root or source_root in output.parents:
        raise ValueError("output must not be the source or live inside the source tree")

    buckets = discover_buckets(source_root)
    if episode_limit is not None and episode_limit <= 0:
        raise ValueError("episode_limit must be positive")
    infos = {bucket.name: _read_json(bucket / "meta" / "info.json") for bucket in buckets}
    for bucket in buckets:
        _validate_source_info(bucket, infos[bucket.name])

    first_info = infos[buckets[0].name]
    base_features = first_info["features"]
    for bucket in buckets[1:]:
        if set(infos[bucket.name]["features"]) != set(base_features):
            raise ValueError(f"{bucket}: source feature keys differ from {buckets[0]}")

    temp = output.with_name(f".{output.name}.building-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    if temp.exists():
        raise FileExistsError(temp)
    (temp / "meta" / "episodes").mkdir(parents=True)

    state_arrays: list[np.ndarray] = []
    action_arrays: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    global_indices: list[np.ndarray] = []
    task_indices: list[np.ndarray] = []
    episode_rows: list[dict] = []
    task_prompts: list[str] = []
    task_suites: list[str] = []
    video_jobs: list[tuple[Path, Path]] = []
    source_manifests: list[dict] = []
    global_episode = 0
    global_frame = 0
    max_errors = {"position": 0.0, "rotation": 0.0, "gripper": 0.0}
    state_gripper_projection = {
        "rows_outside_nominal_width": 0,
        "max_width_clip_error_m": 0.0,
        "max_abs_finger_joint_sum": 0.0,
    }

    try:
        stop = False
        for bucket in buckets:
            info = infos[bucket.name]
            episodes = _load_episodes(bucket)
            tasks = _load_tasks(bucket)
            task_map: dict[int, int] = {}
            for old_task in sorted(tasks):
                task_map[old_task] = len(task_prompts)
                task_prompts.append(tasks[old_task])
                task_suites.append(bucket.name)

            converted_bucket_episodes = 0
            converted_bucket_frames = 0
            for _, episode in episodes.iterrows():
                if episode_limit is not None and global_episode >= episode_limit:
                    stop = True
                    break
                source_chunk = int(episode["data/chunk_index"])
                source_file = int(episode["data/file_index"])
                source_data = bucket / info["data_path"].format(
                    chunk_index=source_chunk,
                    file_index=source_file,
                )
                table = pq.read_table(source_data)
                length = int(episode["length"])
                if table.num_rows != length:
                    raise ValueError(f"{source_data}: rows={table.num_rows}, episode length={length}")
                state8 = np.asarray(table["observation.state"].to_pylist(), np.float32)
                action7 = np.asarray(table["action"].to_pylist(), np.float32)
                finger_width = state8[:, 6].astype(np.float64) - state8[:, 7].astype(np.float64)
                clipped_width = np.clip(finger_width, 0.0, LIBERO_GRIPPER_WIDTH_OPEN)
                state_gripper_projection["rows_outside_nominal_width"] += int(
                    np.count_nonzero(finger_width != clipped_width)
                )
                state_gripper_projection["max_width_clip_error_m"] = max(
                    state_gripper_projection["max_width_clip_error_m"],
                    float(np.max(np.abs(finger_width - clipped_width))),
                )
                state_gripper_projection["max_abs_finger_joint_sum"] = max(
                    state_gripper_projection["max_abs_finger_joint_sum"],
                    float(np.max(np.abs(state8[:, 6].astype(np.float64) + state8[:, 7].astype(np.float64)))),
                )
                state10, action10, errors = convert_state_action(state8, action7)
                for key, value in errors.items():
                    max_errors[key] = max(max_errors[key], value)

                source_task_values = np.unique(np.asarray(table["task_index"].to_numpy()))
                if source_task_values.size != 1:
                    raise ValueError(f"{source_data}: expected one task_index, got {source_task_values.tolist()}")
                source_task = int(source_task_values[0])
                if source_task not in task_map:
                    raise KeyError(f"{source_data}: task_index={source_task} missing from tasks.parquet")
                new_task = task_map[source_task]
                new_chunk = global_episode // FILES_PER_CHUNK
                new_file = global_episode % FILES_PER_CHUNK
                target_data = temp / f"data/chunk-{new_chunk:03d}/file-{new_file:03d}.parquet"
                target_data.parent.mkdir(parents=True, exist_ok=True)

                table = _replace_column(
                    table,
                    "observation.state",
                    pa.array(state10.tolist(), type=pa.list_(pa.float32())),
                )
                table = _replace_column(table, "action", pa.array(action10.tolist(), type=pa.list_(pa.float32())))
                table = _replace_column(
                    table, "episode_index", pa.array(np.full(length, global_episode, np.int64), type=pa.int64())
                )
                table = _replace_column(
                    table, "index", pa.array(np.arange(global_frame, global_frame + length), type=pa.int64())
                )
                table = _replace_column(
                    table, "task_index", pa.array(np.full(length, new_task, np.int64), type=pa.int64())
                )
                pq.write_table(table, target_data)

                row = {
                    "episode_index": global_episode,
                    "tasks": [task_prompts[new_task]],
                    "length": length,
                    "data/chunk_index": new_chunk,
                    "data/file_index": new_file,
                    "dataset_from_index": 0,
                    "dataset_to_index": length,
                    "source_suite": bucket.name,
                    "source_episode_index": int(episode["episode_index"]),
                }
                for video_key in VIDEO_KEYS:
                    source_video = bucket / info["video_path"].format(
                        video_key=video_key,
                        chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
                        file_index=int(episode[f"videos/{video_key}/file_index"]),
                    )
                    if not source_video.is_file():
                        raise FileNotFoundError(source_video)
                    target_video = (
                        temp / "videos" / video_key / f"chunk-{new_chunk:03d}" / f"file-{new_file:03d}.mp4"
                    )
                    video_jobs.append((source_video, target_video))
                    row[f"videos/{video_key}/chunk_index"] = new_chunk
                    row[f"videos/{video_key}/file_index"] = new_file
                    # Each output MP4 contains exactly this one episode, so its
                    # local time span is unambiguous and directly row-count/FPS.
                    row[f"videos/{video_key}/from_timestamp"] = 0.0
                    row[f"videos/{video_key}/to_timestamp"] = length / float(info["fps"])
                episode_rows.append(row)

                state_arrays.append(state10)
                action_arrays.append(action10)
                timestamps.append(np.asarray(table["timestamp"].to_numpy(), np.float32).reshape(-1, 1))
                frame_indices.append(np.asarray(table["frame_index"].to_numpy(), np.int64).reshape(-1, 1))
                episode_indices.append(np.full((length, 1), global_episode, np.int64))
                global_indices.append(np.arange(global_frame, global_frame + length, dtype=np.int64).reshape(-1, 1))
                task_indices.append(np.full((length, 1), new_task, np.int64))
                global_episode += 1
                global_frame += length
                converted_bucket_episodes += 1
                converted_bucket_frames += length
                if global_episode % 100 == 0:
                    print(f"converted parquet: episodes={global_episode} frames={global_frame}", flush=True)

            source_manifests.append(
                {
                    "name": bucket.name,
                    "source": str(bucket),
                    "episodes": converted_bucket_episodes,
                    "frames": converted_bucket_frames,
                }
            )
            if stop:
                break

        if global_episode == 0:
            raise ValueError("no episodes converted")

        print(f"copying {len(video_jobs)} independent video files with {workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for done, _ in enumerate(executor.map(lambda pair: _copy_independent(*pair), video_jobs), 1):
                if done % 200 == 0 or done == len(video_jobs):
                    print(f"copied videos: {done}/{len(video_jobs)}", flush=True)

        # Standard v3 task metadata keeps the task text in the pandas index.
        tasks_frame = pd.DataFrame(
            {
                "task_index": np.arange(len(task_prompts), dtype=np.int64),
                "source_suite": task_suites,
            },
            index=pd.Index(task_prompts, name="task"),
        )
        pq.write_table(pa.Table.from_pandas(tasks_frame), temp / "meta" / "tasks.parquet")
        episodes_frame = pd.DataFrame(episode_rows)
        for chunk, start in enumerate(range(0, len(episodes_frame), FILES_PER_CHUNK)):
            subset = episodes_frame.iloc[start : start + FILES_PER_CHUNK].reset_index(drop=True)
            episode_path = temp / "meta" / "episodes" / f"chunk-{chunk:03d}" / "file-000.parquet"
            episode_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(subset), episode_path)

        output_features = {key: dict(value) for key, value in base_features.items()}
        output_features["observation.state"] = {
            "dtype": "float32",
            "shape": [EEF10_DIM],
            "names": {"states": EEF10_NAMES},
            "fps": 20,
        }
        output_features["action"] = {
            "dtype": "float32",
            "shape": [EEF10_DIM],
            "names": {"motors": EEF10_NAMES},
            "fps": 20,
        }
        output_info = dict(first_info)
        output_info.update(
            {
                "codebase_version": "v3.0",
                "total_episodes": global_episode,
                "total_frames": global_frame,
                "total_tasks": len(task_prompts),
                "chunks_size": FILES_PER_CHUNK,
                "fps": 20,
                "splits": {"train": f"0:{global_episode}"},
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                "features": output_features,
            }
        )
        # These fields occur in older conversion outputs but are not part of
        # LeRobot 0.5's typed v3 DatasetInfo. Full semantics live in
        # meta/conversion.json instead of relying on ignored info.json fields.
        output_info.pop("total_videos", None)
        output_info.pop("total_chunks", None)
        output_info.pop("openwam", None)
        _write_json(temp / "meta" / "info.json", output_info)

        state_all = np.concatenate(state_arrays, axis=0)
        action_all = np.concatenate(action_arrays, axis=0)
        standard_stats = _load_legacy_aggregate_stats(
            legacy_stats_root,
            [entry["name"] for entry in source_manifests if entry["episodes"]],
        )
        # The converted columns and re-indexed scalar columns must never inherit source stats.
        standard_stats.update(
            {
                "observation.state": _numpy_tree(_feature_stats(state_all)),
                "action": _numpy_tree(_feature_stats(action_all)),
                "timestamp": _numpy_tree(_feature_stats(np.concatenate(timestamps))),
                "frame_index": _numpy_tree(_feature_stats(np.concatenate(frame_indices))),
                "episode_index": _numpy_tree(_feature_stats(np.concatenate(episode_indices))),
                "index": _numpy_tree(_feature_stats(np.concatenate(global_indices))),
                "task_index": _numpy_tree(_feature_stats(np.concatenate(task_indices))),
            }
        )
        from lerobot.datasets.io_utils import write_stats

        write_stats(standard_stats, temp)

        pooled = np.concatenate([action_all, state_all], axis=0)
        eef_stats = _feature_stats(pooled)
        # OpenWAM normalization intentionally leaves the geometry-constrained rot6d block untouched.
        pin_rot6d_identity(eef_stats, tuple(range(3, 9)))
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
        # One authoritative artifact serves both training and deployment.  The
        # deploy loader consumes the six standard vectors and ignores the extra
        # representation/provenance fields in this full payload.
        with (temp / "meta" / "normalization_stats.npy").open("wb") as handle:
            np.save(handle, {"eef": eef_stats})

        conversion = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "sources": source_manifests,
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
            "gripper_formula": "open_scale = 2 * source_open_flag - 1",
            "roundtrip_max_abs_error": max_errors,
            "normalization_stats_file": "meta/normalization_stats.npy",
            "independent_files": True,
            "video_reencoded": False,
        }
        _write_json(temp / "meta" / "conversion.json", conversion)

        os.replace(temp, output)
        return {
            "output": str(output),
            "episodes": global_episode,
            "frames": global_frame,
            "tasks": len(task_prompts),
            "videos": len(video_jobs),
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
    parser.add_argument(
        "--legacy-stats-root",
        type=Path,
        default=None,
        help="Optional v2.1 root supplying unchanged image/auxiliary feature statistics",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--episode-limit", type=int, default=None, help="Global smoke-test limit")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    result = convert_dataset(
        args.source,
        args.output,
        legacy_stats_root=args.legacy_stats_root,
        workers=args.workers,
        episode_limit=args.episode_limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
