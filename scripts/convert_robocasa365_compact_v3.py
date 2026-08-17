#!/usr/bin/env python3
"""Build independent RoboCasa365 LeRobot v3 datasets with compact state/action.

The authoritative numeric source is the per-task RoboCasa365 LeRobot v2.1 tree
under ``robocasa365/pretrain``.  The published v3 mirror is used only as the
packing/index template: every packed episode is checked byte-for-byte (after
Arrow decoding) against its v2.1 source before conversion.  Videos are copied
as independent files; links are rejected.

Output robot columns:

* ``observation.state`` (19-D): achieved EEF xyz3 + rot6d6 + gripper1,
  followed by world base xyz3 + rot6d6.
* ``action`` (15-D): current-state-anchored absolute EEF xyz3 + rot6d6 +
  gripper1, followed by base vx/vy/vyaw + torso + control_mode.

The source normalized OSC command on row ``t`` is composed with state ``t``:
``goal_xyz = eef_xyz + 0.05 * delta_xyz`` and
``goal_R = Exp(0.5 * delta_rotvec) @ eef_R``.  No next-frame state and no
control-mode-dependent reference are used.
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
from scipy.spatial.transform import Rotation

SOURCE_STATE_DIM = 16
SOURCE_ACTION_DIM = 12
STATE_DIM = 19
ACTION_DIM = 15
SHARED_EEF_STATS_DIM = 4  # xyz3 + gripper1, pooled across state and action
POSITION_SCALE = 0.05
ROTATION_SCALE = 0.5
GRIPPER_WIDTH_OPEN = 0.1
REPRESENTATION = "robocasa365_compact_absolute_eef_v1"
ACTION_STATS_KEY = "robocasa365"
STATE_STATS_KEY = f"{ACTION_STATS_KEY}_state"
DEFAULT_SOURCE_PRETRAIN = Path("/path/to/robocasa365/pretrain")
DEFAULT_PACKED_V3_ROOT = Path("/path/to/robocasa365_v3")
DEFAULT_OUTPUT_ROOT = Path("/path/to/robocasa365_openwam_v3")
SPLIT_REPO_NAMES = {
    "atomic": "robocasa365-pretrain-atomic",
    "composite": "robocasa365-pretrain-composite",
}
STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_rot6d_col0_x",
    "eef_rot6d_col0_y",
    "eef_rot6d_col0_z",
    "eef_rot6d_col1_x",
    "eef_rot6d_col1_y",
    "eef_rot6d_col1_z",
    "gripper_open_scale",
    "base_x",
    "base_y",
    "base_z",
    "base_rot6d_col0_x",
    "base_rot6d_col0_y",
    "base_rot6d_col0_z",
    "base_rot6d_col1_x",
    "base_rot6d_col1_y",
    "base_rot6d_col1_z",
]
ACTION_NAMES = [
    "eef_target_x",
    "eef_target_y",
    "eef_target_z",
    "eef_target_rot6d_col0_x",
    "eef_target_rot6d_col0_y",
    "eef_target_rot6d_col0_z",
    "eef_target_rot6d_col1_x",
    "eef_target_rot6d_col1_y",
    "eef_target_rot6d_col1_z",
    "gripper_open_command",
    "base_vx_command",
    "base_vy_command",
    "base_vyaw_command",
    "torso_command",
    "control_mode",
]
STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix)
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrices must end in (3, 3), got {value.shape}")
    return np.concatenate([value[..., :, 0], value[..., :, 1]], axis=-1).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    value = np.asarray(rot6d, np.float64)
    first = value[..., 0:3]
    second = value[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    return np.stack([first, second, np.cross(first, second)], axis=-1)


def convert_state_action(state16: np.ndarray, action12: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Convert row-aligned native RoboCasa state/action to compact physical semantics."""
    state16 = np.asarray(state16, np.float64)
    action12 = np.asarray(action12, np.float64)
    if state16.ndim != 2 or state16.shape[1] != SOURCE_STATE_DIM:
        raise ValueError(f"source state must be (T, {SOURCE_STATE_DIM}), got {state16.shape}")
    if action12.shape != (state16.shape[0], SOURCE_ACTION_DIM):
        raise ValueError(f"source action must be {(state16.shape[0], SOURCE_ACTION_DIM)}, got {action12.shape}")
    if not np.isfinite(state16).all() or not np.isfinite(action12).all():
        raise ValueError("source state/action contains NaN or infinity")
    if np.any(np.abs(action12) > 1.0 + 1e-5):
        where = np.argwhere(np.abs(action12) > 1.0 + 1e-5)[0]
        raise ValueError(f"source normalized action outside [-1,1] at row/dim {where.tolist()}")

    base_rotation = Rotation.from_quat(state16[:, 3:7]).as_matrix()
    eef_rotation = Rotation.from_quat(state16[:, 10:14]).as_matrix()
    gripper_state = np.clip(
        2.0 * (state16[:, 14] - state16[:, 15]) / GRIPPER_WIDTH_OPEN - 1.0,
        -1.0,
        1.0,
    )[:, None]
    state19 = np.concatenate(
        [
            state16[:, 7:10],
            matrix_to_rot6d(eef_rotation),
            gripper_state,
            state16[:, 0:3],
            matrix_to_rot6d(base_rotation),
        ],
        axis=1,
    ).astype(np.float32)

    delta_rotation = Rotation.from_rotvec(action12[:, 8:11] * ROTATION_SCALE).as_matrix()
    target_rotation = delta_rotation @ eef_rotation
    target_position = state16[:, 7:10] + action12[:, 5:8] * POSITION_SCALE
    action15 = np.concatenate(
        [
            target_position,
            matrix_to_rot6d(target_rotation),
            -action12[:, 11:12],
            action12[:, 0:5],
        ],
        axis=1,
    ).astype(np.float32)

    recovered_position = (action15[:, 0:3].astype(np.float64) - state16[:, 7:10]) / POSITION_SCALE
    recovered_target_rotation = rot6d_to_matrix(action15[:, 3:9])
    recovered_delta = recovered_target_rotation @ np.swapaxes(eef_rotation, -1, -2)
    recovered_rotation = Rotation.from_matrix(recovered_delta).as_rotvec() / ROTATION_SCALE
    errors = {
        "position": float(np.max(np.abs(recovered_position - action12[:, 5:8]), initial=0.0)),
        "rotation": float(np.max(np.abs(recovered_rotation - action12[:, 8:11]), initial=0.0)),
        "gripper": float(np.max(np.abs(-action15[:, 9] - action12[:, 11]), initial=0.0)),
        "base": float(np.max(np.abs(action15[:, 10:15] - action12[:, 0:5]), initial=0.0)),
    }
    # The converted columns are intentionally float32.  At metre-scale absolute
    # positions, subtracting the current pose before dividing by 0.05 has a
    # worst-case few-e-6 normalized-command quantization error.
    if errors["position"] > 1e-5 or errors["rotation"] > 3e-6 or errors["gripper"] > 1e-7 or errors["base"] > 1e-7:
        raise ValueError(f"compact action round-trip failed: {errors}")
    return state19, action15, errors


class StatsAccumulator:
    """Exact streaming moments/range plus a bounded deterministic quantile sample."""

    def __init__(self, dim: int, *, seed: int, sample_limit: int = 250_000):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)
        self.minimum = np.full(dim, np.inf, np.float64)
        self.maximum = np.full(dim, -np.inf, np.float64)
        self._sample = np.empty((0, dim), np.float32)
        self._priority = np.empty((0,), np.float64)
        self._sample_limit = sample_limit
        self._rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, np.float32)
        if values.ndim != 2 or values.shape[1] != self.dim or values.shape[0] == 0:
            raise ValueError(f"stats expected nonempty (N,{self.dim}), got {values.shape}")
        batch = values.astype(np.float64)
        n = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_m2 = np.square(batch - batch_mean).sum(axis=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            total = self.count + n
            self.mean += delta * n / total
            self.m2 += batch_m2 + np.square(delta) * self.count * n / total
        self.count += n
        self.minimum = np.minimum(self.minimum, batch.min(axis=0))
        self.maximum = np.maximum(self.maximum, batch.max(axis=0))

        take = min(n, 10_000)
        indices = self._rng.choice(n, size=take, replace=False) if take < n else np.arange(n)
        sample = values[indices]
        priority = self._rng.random(take)
        self._sample = np.concatenate([self._sample, sample], axis=0)
        self._priority = np.concatenate([self._priority, priority])
        if self._sample.shape[0] > self._sample_limit:
            keep = np.argpartition(self._priority, -self._sample_limit)[-self._sample_limit :]
            self._sample = self._sample[keep]
            self._priority = self._priority[keep]

    def merge(self, other: "StatsAccumulator") -> None:
        if other.dim != self.dim or other.count == 0:
            if other.dim != self.dim:
                raise ValueError("cannot merge stats with different dimensions")
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            self.minimum = other.minimum.copy()
            self.maximum = other.maximum.copy()
        else:
            total = self.count + other.count
            delta = other.mean - self.mean
            self.m2 += other.m2 + np.square(delta) * self.count * other.count / total
            self.mean += delta * other.count / total
            self.count = total
            self.minimum = np.minimum(self.minimum, other.minimum)
            self.maximum = np.maximum(self.maximum, other.maximum)
        self._sample = np.concatenate([self._sample, other._sample], axis=0)
        self._priority = np.concatenate([self._priority, other._priority])
        if self._sample.shape[0] > self._sample_limit:
            keep = np.argpartition(self._priority, -self._sample_limit)[-self._sample_limit :]
            self._sample = self._sample[keep]
            self._priority = self._priority[keep]

    def finish(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise ValueError("cannot finish empty stats")
        std = np.sqrt(self.m2 / self.count)
        q01, q99 = np.quantile(self._sample.astype(np.float64), [0.01, 0.99], axis=0)
        return {
            "mean": self.mean.astype(np.float32),
            "std": std.astype(np.float32),
            "min": self.minimum.astype(np.float32),
            "max": self.maximum.astype(np.float32),
            "q01": q01.astype(np.float32),
            "q99": q99.astype(np.float32),
        }


def _pin_dims(stats: dict[str, np.ndarray], dims: Iterable[int]) -> None:
    identity = {"mean": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "q01": -1.0, "q99": 1.0}
    for key, value in identity.items():
        stats[key][list(dims)] = value


def _shared_eef_rows(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Return xyz+gripper rows pooled across achieved state and commanded action."""
    state = np.asarray(state, np.float32)
    action = np.asarray(action, np.float32)
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"shared EEF stats expected state (N,{STATE_DIM}), got {state.shape}")
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        raise ValueError(f"shared EEF stats expected action (N,{ACTION_DIM}), got {action.shape}")
    return np.concatenate(
        [
            np.concatenate([state[:, 0:3], state[:, 9:10]], axis=1),
            np.concatenate([action[:, 0:3], action[:, 9:10]], axis=1),
        ],
        axis=0,
    )


def _stats_payload(
    action_acc: StatsAccumulator,
    state_acc: StatsAccumulator,
    shared_eef_acc: StatsAccumulator,
) -> dict:
    action = action_acc.finish()
    state = state_acc.finish()
    shared_eef = shared_eef_acc.finish()

    # EEF xyz and gripper have matching physical semantics on the achieved-state
    # and absolute-command sides.  Fit them on the union and write the same
    # transform into both directional blocks.  Base state is a world pose while
    # base action is a velocity/command, so their independently accumulated
    # statistics intentionally remain separate.
    for key in STAT_KEYS:
        action[key][0:3] = shared_eef[key][0:3]
        action[key][9] = shared_eef[key][3]
        state[key][0:3] = shared_eef[key][0:3]
        state[key][9] = shared_eef[key][3]

    # Rot6D is a geometric representation rather than six independent scalar
    # quantities.  Preserve it exactly under every supported normalization mode.
    _pin_dims(action, range(3, 9))
    _pin_dims(state, (*range(3, 9), *range(13, 19)))
    action.update(
        {
            "representation": REPRESENTATION,
            "gripper_convention": "minus1_closed_plus1_open",
            "osc_position_scale": POSITION_SCALE,
            "osc_rotation_scale": ROTATION_SCALE,
            "normalization_scope": {
                "eef_xyz_gripper": "pooled_state_action",
                "eef_rot6d": "identity",
                "base_vx_vy_vyaw_torso_mode": "action_only",
            },
            "rot6d_identity_dims": list(range(3, 9)),
        }
    )
    state.update(
        {
            "representation": REPRESENTATION,
            "normalization_scope": {
                "eef_xyz_gripper": "pooled_state_action",
                "eef_rot6d": "identity",
                "base_xyz": "state_only",
                "base_rot6d": "identity",
            },
            "rot6d_identity_dims": [*range(3, 9), *range(13, 19)],
        }
    )
    return {
        ACTION_STATS_KEY: action,
        STATE_STATS_KEY: state,
        "num_timesteps": int(action_acc.count),
        "num_shared_eef_values": int(shared_eef_acc.count),
        "quantiles": f"deterministic priority sample <= {action_acc._sample_limit} rows",
    }


def _jsonable_stats(stats: dict[str, np.ndarray]) -> dict[str, list]:
    return {key: np.asarray(stats[key]).tolist() for key in STAT_KEYS}


def _replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"missing required parquet column {name!r}")
    array = pa.array(values.tolist(), type=pa.list_(pa.float32()))
    return table.set_column(index, name, array)


def _copy_independent(source: Path, target: Path) -> None:
    if source.is_symlink():
        source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if target.is_symlink():
        raise ValueError(f"output must not contain symlinks: {target}")
    source_stat, target_stat = source.stat(), target.stat()
    if source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino:
        raise ValueError(f"copy unexpectedly shares inode: {source} -> {target}")


def _load_episodes(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(root / "meta" / "episodes")
    episodes = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    episodes["_data_row_offset"] = (
        episodes.groupby(["data/chunk_index", "data/file_index"], sort=False)["length"].cumsum() - episodes["length"]
    )
    return episodes


def _source_dataset(source_pretrain: Path, source_prefix: str) -> Path:
    prefix = Path(str(source_prefix))
    if prefix.parts and prefix.parts[0] == "pretrain":
        path = source_pretrain.parent / prefix
    else:
        path = source_pretrain / prefix
    return path / "lerobot"


def _source_episode_path(source_dataset: Path, episode_index: int) -> Path:
    info = _read_json(source_dataset / "meta" / "info.json")
    return source_dataset / info["data_path"].format(
        episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
        episode_index=episode_index,
    )


class SourceMetadataCache:
    def __init__(self):
        self._cache: dict[Path, tuple[list[dict], dict[int, str]]] = {}

    def get(self, root: Path) -> tuple[list[dict], dict[int, str]]:
        if root not in self._cache:
            episodes = _read_jsonl(root / "meta" / "episodes.jsonl")
            tasks = {int(row["task_index"]): str(row["task"]) for row in _read_jsonl(root / "meta" / "tasks.jsonl")}
            self._cache[root] = (episodes, tasks)
        return self._cache[root]


def _validate_source_episode(
    *,
    source_pretrain: Path,
    episode_row: pd.Series,
    packed_slice: pa.Table,
    global_tasks: dict[int, str],
    cache: SourceMetadataCache,
) -> None:
    source_root = _source_dataset(source_pretrain, str(episode_row["source_prefix"]))
    source_index = int(episode_row["source_episode_index"])
    source_path = _source_episode_path(source_root, source_index)
    source = pq.read_table(source_path)
    length = int(episode_row["length"])
    if source.num_rows != length or packed_slice.num_rows != length:
        raise ValueError(
            f"episode length mismatch for {source_path}: source={source.num_rows}, packed={packed_slice.num_rows}, meta={length}"
        )
    for column in ("observation.state", "action"):
        left = np.asarray(source[column].to_pylist(), np.float64)
        right = np.asarray(packed_slice[column].to_pylist(), np.float64)
        if not np.array_equal(left, right):
            raise ValueError(f"packed v3 {column} differs from authoritative source: {source_path}")

    source_episodes, source_tasks = cache.get(source_root)
    source_episode = source_episodes[source_index]
    packed_tasks = [str(value) for value in episode_row["tasks"]]
    if [str(value) for value in source_episode.get("tasks", [])] != packed_tasks:
        raise ValueError(f"episode prompt mismatch for {source_path}: {source_episode.get('tasks')} != {packed_tasks}")
    for column in ("annotation.human.task_description", "annotation.human.task_name", "task_index"):
        if column not in source.column_names or column not in packed_slice.column_names:
            continue
        source_ids = np.asarray(source[column]).reshape(-1)
        packed_ids = np.asarray(packed_slice[column]).reshape(-1)
        source_text = [source_tasks[int(index)] for index in source_ids]
        packed_text = [global_tasks[int(index)] for index in packed_ids]
        if source_text != packed_text:
            raise ValueError(f"prompt-id remap mismatch in {column} for {source_path}")


def _validate_controller_contract(source_pretrain: Path, split: str) -> int:
    roots = sorted(source_pretrain.joinpath(split).glob("*/*/lerobot"))
    if not roots:
        raise FileNotFoundError(source_pretrain / split)
    for root in roots:
        info = _read_json(root / "meta" / "info.json")
        if info.get("codebase_version") != "v2.1" or int(info.get("fps", 0)) != 20:
            raise ValueError(
                f"unexpected source format under {root}: version={info.get('codebase_version')}, fps={info.get('fps')}"
            )
        if tuple(info["features"]["observation.state"]["shape"]) != (SOURCE_STATE_DIM,):
            raise ValueError(f"unexpected source state shape under {root}")
        if tuple(info["features"]["action"]["shape"]) != (SOURCE_ACTION_DIM,):
            raise ValueError(f"unexpected source action shape under {root}")
        dataset_meta = _read_json(root / "extras" / "dataset_meta.json")
        controllers: list[dict] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "OSC_POSE" and "output_max" in value:
                    controllers.append(value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(dataset_meta)
        if not controllers:
            raise ValueError(f"no OSC_POSE controller metadata under {root}")
        for controller in controllers:
            output_max = np.asarray(controller["output_max"], np.float64)
            expected = np.array([POSITION_SCALE] * 3 + [ROTATION_SCALE] * 3)
            if (
                controller.get("input_type") != "delta"
                or controller.get("input_ref_frame") != "base"
                or not np.array_equal(output_max, expected)
            ):
                raise ValueError(f"controller contract mismatch under {root}: {controller}")
    return len(roots)


def _write_readme(path: Path, *, split: str, info: dict, source: Path) -> None:
    text = f"""---
license: mit
task_categories:
- robotics
tags:
- LeRobot
- robocasa365
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# RoboCasa365 {split}: compact absolute EEF (LeRobot v3.0)

Independent conversion from `{source}`.  Videos are byte-preserving file copies;
the output contains no hardlinks or symlinks to its sources.

* `observation.state` (19-D): achieved EEF `[xyz3, rot6d6, gripper1]` + world
  base `[xyz3, rot6d6]`.
* `action` (15-D): current-state-anchored absolute EEF target
  `[xyz3, rot6d6, gripper1]` + `[base_vx, base_vy, base_vyaw, torso, control_mode]`.
* Gripper convention: `-1 = closed`, `+1 = open`.
* OSC conversion: `target_xyz = state_xyz + 0.05 * source_delta_xyz` and
  `target_R = Exp(0.5 * source_delta_rotvec) @ state_R` on the same row.

Totals: {info["total_episodes"]} episodes, {info["total_frames"]} frames,
{info["total_tasks"]} prompt strings, {info["fps"]} FPS.
"""
    path.write_text(text, encoding="utf-8")


def convert_split(
    *,
    split: str,
    source_pretrain: Path,
    packed_root: Path,
    output_root: Path,
    workers: int,
    global_action_stats: StatsAccumulator | None = None,
    global_state_stats: StatsAccumulator | None = None,
    global_shared_eef_stats: StatsAccumulator | None = None,
) -> dict[str, Any]:
    repo_name = SPLIT_REPO_NAMES[split]
    packed = (packed_root / repo_name).resolve()
    output = (output_root / repo_name).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    source_pretrain = source_pretrain.resolve()
    if not packed.is_dir():
        raise FileNotFoundError(packed)
    task_roots = _validate_controller_contract(source_pretrain, split)
    info = _read_json(packed / "meta" / "info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"packed template is not LeRobot v3.0: {packed}")
    if tuple(info["features"]["observation.state"]["shape"]) != (SOURCE_STATE_DIM,) or tuple(
        info["features"]["action"]["shape"]
    ) != (SOURCE_ACTION_DIM,):
        raise ValueError(f"packed template does not contain native state16/action12: {packed}")

    temp = output.with_name(f".{output.name}.building-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    temp.mkdir(parents=True)
    action_acc = StatsAccumulator(ACTION_DIM, seed=41 if split == "atomic" else 43)
    state_acc = StatsAccumulator(STATE_DIM, seed=47 if split == "atomic" else 53)
    shared_eef_acc = StatsAccumulator(SHARED_EEF_STATS_DIM, seed=67 if split == "atomic" else 69)
    max_errors = {key: 0.0 for key in ("position", "rotation", "gripper", "base")}
    episodes = _load_episodes(packed)
    tasks_table = pq.read_table(packed / "meta" / "tasks.parquet")
    global_tasks = {int(row["task_index"]): str(row["task"]) for row in tasks_table.to_pylist()}
    source_cache = SourceMetadataCache()
    data_paths = sorted((packed / "data").rglob("*.parquet"))
    total_rows = 0
    verified_episodes = 0

    try:
        for number, source_data in enumerate(data_paths, 1):
            relative = source_data.relative_to(packed)
            target_data = temp / relative
            target_data.parent.mkdir(parents=True, exist_ok=True)
            table = pq.read_table(source_data)
            state16 = np.asarray(table["observation.state"].to_pylist(), np.float64)
            action12 = np.asarray(table["action"].to_pylist(), np.float64)

            name = source_data.stem
            chunk = int(source_data.parent.name.split("-")[-1])
            file_index = int(name.split("-")[-1])
            group = episodes[(episodes["data/chunk_index"] == chunk) & (episodes["data/file_index"] == file_index)]
            if int(group["length"].sum()) != table.num_rows:
                raise ValueError(
                    f"episode lengths do not cover {source_data}: {group['length'].sum()} != {table.num_rows}"
                )
            for _, episode in group.iterrows():
                offset, length = int(episode["_data_row_offset"]), int(episode["length"])
                _validate_source_episode(
                    source_pretrain=source_pretrain,
                    episode_row=episode,
                    packed_slice=table.slice(offset, length),
                    global_tasks=global_tasks,
                    cache=source_cache,
                )
                verified_episodes += 1

            state19, action15, errors = convert_state_action(state16, action12)
            for key, value in errors.items():
                max_errors[key] = max(max_errors[key], value)
            state_acc.update(state19)
            action_acc.update(action15)
            shared_eef_rows = _shared_eef_rows(state19, action15)
            shared_eef_acc.update(shared_eef_rows)
            if global_state_stats is not None:
                global_state_stats.update(state19)
            if global_action_stats is not None:
                global_action_stats.update(action15)
            if global_shared_eef_stats is not None:
                global_shared_eef_stats.update(shared_eef_rows)
            converted = _replace_column(table, "observation.state", state19)
            converted = _replace_column(converted, "action", action15)
            pq.write_table(converted, target_data, compression="zstd")
            total_rows += table.num_rows
            print(
                f"[{split}] parquet {number}/{len(data_paths)}: rows={total_rows:,}, verified_episodes={verified_episodes:,}",
                flush=True,
            )

        if total_rows != int(info["total_frames"]) or verified_episodes != int(info["total_episodes"]):
            raise ValueError(
                f"{split} total mismatch: rows {total_rows}/{info['total_frames']}, episodes {verified_episodes}/{info['total_episodes']}"
            )

        copy_jobs: list[tuple[Path, Path]] = []
        for source_video in sorted((packed / "videos").rglob("*.mp4")):
            copy_jobs.append((source_video, temp / source_video.relative_to(packed)))
        for relative in (Path(".gitattributes"), Path("meta/tasks.parquet")):
            source_file = packed / relative
            if source_file.is_file():
                copy_jobs.append((source_file, temp / relative))
        for source_episode in sorted((packed / "meta" / "episodes").rglob("*.parquet")):
            copy_jobs.append((source_episode, temp / source_episode.relative_to(packed)))
        print(f"[{split}] independently copying {len(copy_jobs)} metadata/video files", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for done, _ in enumerate(executor.map(lambda pair: _copy_independent(*pair), copy_jobs), 1):
                if done % 25 == 0 or done == len(copy_jobs):
                    print(f"[{split}] copied {done}/{len(copy_jobs)} files", flush=True)

        features = {key: dict(value) for key, value in info["features"].items()}
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": {"states": STATE_NAMES},
            "fps": info["fps"],
        }
        features["action"] = {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": {"motors": ACTION_NAMES},
            "fps": info["fps"],
        }
        output_info = dict(info)
        output_info["features"] = features
        _write_json(temp / "meta" / "info.json", output_info)

        stats_payload = _stats_payload(action_acc, state_acc, shared_eef_acc)
        with (temp / "meta" / "normalization_stats.npy").open("wb") as handle:
            np.save(handle, stats_payload, allow_pickle=True)
        _write_json(
            temp / "meta" / "stats.json",
            {
                "observation.state": _jsonable_stats(stats_payload[STATE_STATS_KEY]),
                "action": _jsonable_stats(stats_payload[ACTION_STATS_KEY]),
            },
        )
        _write_json(
            temp / "meta" / "modality.json",
            {
                "state": {
                    "end_effector": {"original_key": "observation.state", "start": 0, "end": 10},
                    "base_pose": {"original_key": "observation.state", "start": 10, "end": 19},
                },
                "action": {
                    "end_effector": {"original_key": "action", "start": 0, "end": 10},
                    "base_velocity": {"original_key": "action", "start": 10, "end": 13},
                    "torso": {"original_key": "action", "start": 13, "end": 14},
                    "control_mode": {"original_key": "action", "start": 14, "end": 15},
                },
                "unified_mapping": {
                    "state": {"0:10": "0:10", "10:19": "68:77"},
                    "action": {"0:10": "0:10", "10:15": "68:73"},
                },
            },
        )
        conversion = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_pretrain": str(source_pretrain),
            "packed_v3_template": str(packed),
            "split": split,
            "source_task_roots": task_roots,
            "source_state_dim": SOURCE_STATE_DIM,
            "source_action_dim": SOURCE_ACTION_DIM,
            "output_state_dim": STATE_DIM,
            "output_action_dim": ACTION_DIM,
            "representation": REPRESENTATION,
            "same_row_state_action": True,
            "uses_next_state": False,
            "mode_dependent_eef_reference": False,
            "osc_position_scale": POSITION_SCALE,
            "osc_rotation_scale": ROTATION_SCALE,
            "verified_source_episodes": verified_episodes,
            "roundtrip_max_abs_error": max_errors,
            "video_reencoded": False,
            "independent_copied_files": True,
            "hardlinks": False,
            "symlinks": False,
            "prompt_rows": len(global_tasks),
        }
        _write_json(temp / "meta" / "conversion.json", conversion)
        _write_readme(temp / "README.md", split=split, info=output_info, source=source_pretrain / split)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, output)
        return {
            "split": split,
            "output": str(output),
            "task_roots": task_roots,
            "episodes": verified_episodes,
            "frames": total_rows,
            "prompts": len(global_tasks),
            "data_files": len(data_paths),
            "video_files": len(list((output / "videos").rglob("*.mp4"))),
            "roundtrip_max_abs_error": max_errors,
        }
    except BaseException:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def convert_all(
    *,
    source_pretrain: Path = DEFAULT_SOURCE_PRETRAIN,
    packed_v3_root: Path = DEFAULT_PACKED_V3_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    splits: Iterable[str] = ("atomic", "composite"),
    workers: int = 8,
) -> dict[str, Any]:
    source_pretrain = source_pretrain.resolve()
    packed_v3_root = packed_v3_root.resolve()
    output_root = output_root.resolve()
    selected = tuple(splits)
    unknown = set(selected).difference(SPLIT_REPO_NAMES)
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    global_action = StatsAccumulator(ACTION_DIM, seed=59)
    global_state = StatsAccumulator(STATE_DIM, seed=61)
    global_shared_eef = StatsAccumulator(SHARED_EEF_STATS_DIM, seed=71)
    results = []
    try:
        for split in selected:
            results.append(
                convert_split(
                    split=split,
                    source_pretrain=source_pretrain,
                    packed_root=packed_v3_root,
                    output_root=output_root,
                    workers=workers,
                    global_action_stats=global_action,
                    global_state_stats=global_state,
                    global_shared_eef_stats=global_shared_eef,
                )
            )
        global_payload = _stats_payload(global_action, global_state, global_shared_eef)
        stats_path = output_root / "robocasa365_multitask_compact_stats.npy"
        with stats_path.open("wb") as handle:
            np.save(handle, global_payload, allow_pickle=True)
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "representation": REPRESENTATION,
            "source_pretrain": str(source_pretrain),
            "packed_v3_template": str(packed_v3_root),
            "independent": True,
            "hardlinks": False,
            "symlinks": False,
            "normalization_stats": str(stats_path),
            "splits": results,
        }
        _write_json(output_root / "conversion_manifest.json", manifest)
        return manifest
    except BaseException:
        # Completed split repos remain valid and inspectable, but a failed run is
        # clearly marked and never receives the top-level success manifest.
        _write_json(
            output_root / "conversion_failed.json",
            {"failed_at": datetime.now(timezone.utc).isoformat(), "completed_splits": results},
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pretrain", type=Path, default=DEFAULT_SOURCE_PRETRAIN)
    parser.add_argument("--packed-v3-root", type=Path, default=DEFAULT_PACKED_V3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", choices=sorted(SPLIT_REPO_NAMES), default=["atomic", "composite"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = convert_all(
        source_pretrain=args.source_pretrain,
        packed_v3_root=args.packed_v3_root,
        output_root=args.output_root,
        splits=args.splits,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
