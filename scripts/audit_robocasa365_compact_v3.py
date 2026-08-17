#!/usr/bin/env python3
"""Full integrity audit for the compact independent RoboCasa365 conversion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.convert_robocasa365_compact_v3 import (  # noqa: E402
    ACTION_DIM,
    ACTION_STATS_KEY,
    SPLIT_REPO_NAMES,
    STATE_DIM,
    STATE_STATS_KEY,
    convert_state_action,
)

DEFAULT_OUTPUT_ROOT = Path("/path/to/robocasa365_openwam_v3")
DEFAULT_PACKED_ROOT = Path("/path/to/robocasa365_v3")
VIDEO_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _regular_files(root: Path) -> list[Path]:
    links = [path for path in root.rglob("*") if path.is_symlink()]
    if links:
        raise ValueError(f"output contains symlinks: {links[:5]}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    linked = [path for path in files if path.stat().st_nlink != 1]
    if linked:
        raise ValueError(f"output contains multiply-linked files: {linked[:5]}")
    return files


def _independent_equal_copy(source: Path, target: Path) -> int:
    source_stat, target_stat = source.stat(), target.stat()
    if source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino:
        raise ValueError(f"hardlink detected: {source} -> {target}")
    if source_stat.st_size != target_stat.st_size:
        raise ValueError(f"copied-file size mismatch: {source} -> {target}")
    compared = 0
    with source.open("rb") as left, target.open("rb") as right:
        while True:
            left_chunk = left.read(8 * 1024 * 1024)
            right_chunk = right.read(8 * 1024 * 1024)
            if left_chunk != right_chunk:
                raise ValueError(f"copied-file bytes differ: {source} -> {target}")
            compared += len(left_chunk)
            if not left_chunk:
                break
    return compared


def _episodes(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(root / "meta" / "episodes")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _vector_column(table, name: str, dim: int, dtype) -> np.ndarray:
    """Materialize an Arrow list column without constructing Python lists."""
    array = table[name].combine_chunks()
    offsets = np.asarray(array.offsets)
    if not np.all(np.diff(offsets) == dim):
        raise ValueError(f"{name} contains a row whose width is not {dim}")
    return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=dtype).reshape(-1, dim)


def _video_contract(root: Path, episodes: pd.DataFrame, info: dict) -> dict:
    checked = 0
    total_frames = 0
    for key in VIDEO_KEYS:
        chunk_column = f"videos/{key}/chunk_index"
        file_column = f"videos/{key}/file_index"
        if chunk_column not in episodes or file_column not in episodes:
            raise KeyError(f"episode metadata has no video mapping for {key}")
        grouped = episodes.groupby([chunk_column, file_column], sort=True)["length"].sum()
        for (chunk, file_index), expected_frames in grouped.items():
            relative = info["video_path"].format(
                video_key=key,
                chunk_index=int(chunk),
                file_index=int(file_index),
            )
            path = root / relative
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if stream.frames != int(expected_frames):
                    raise ValueError(f"video frame count mismatch {path}: {stream.frames} != {expected_frames}")
                if float(stream.average_rate) != float(info["fps"]):
                    raise ValueError(f"video FPS mismatch {path}: {stream.average_rate}")
                if (stream.height, stream.width) != (256, 256):
                    raise ValueError(f"video resolution mismatch {path}: {stream.height}x{stream.width}")
            checked += 1
            total_frames += int(expected_frames)
    return {"files": checked, "frames_across_three_views": total_frames}


def _audit_split(output_root: Path, packed_root: Path, split: str, workers: int) -> dict:
    repo_name = SPLIT_REPO_NAMES[split]
    output = output_root / repo_name
    packed = packed_root / repo_name
    info = _json(output / "meta" / "info.json")
    conversion = _json(output / "meta" / "conversion.json")
    if info["codebase_version"] != "v3.0":
        raise ValueError(f"{output} is not LeRobot v3.0")
    if tuple(info["features"]["observation.state"]["shape"]) != (STATE_DIM,):
        raise ValueError(f"{output} does not declare state{STATE_DIM}")
    if tuple(info["features"]["action"]["shape"]) != (ACTION_DIM,):
        raise ValueError(f"{output} does not declare action{ACTION_DIM}")
    if not conversion["same_row_state_action"] or conversion["uses_next_state"]:
        raise ValueError(f"{output} conversion does not declare the same-row contract")

    episodes = _episodes(output)
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"episode count mismatch under {output}")
    if int(episodes["length"].sum()) != int(info["total_frames"]):
        raise ValueError(f"episode frame total mismatch under {output}")
    if episodes["episode_index"].nunique() != len(episodes):
        raise ValueError(f"duplicate episode_index under {output}")
    if any(not list(tasks) for tasks in episodes["tasks"]):
        raise ValueError(f"empty episode prompt list under {output}")

    data_source = sorted((packed / "data").rglob("*.parquet"))
    data_target = sorted((output / "data").rglob("*.parquet"))
    if [path.relative_to(packed) for path in data_source] != [path.relative_to(output) for path in data_target]:
        raise ValueError(f"data shard set mismatch under {output}")
    rows = 0
    max_abs_recompute_error = 0.0
    state_min = np.full(STATE_DIM, np.inf, np.float64)
    state_max = np.full(STATE_DIM, -np.inf, np.float64)
    action_min = np.full(ACTION_DIM, np.inf, np.float64)
    action_max = np.full(ACTION_DIM, -np.inf, np.float64)
    for index, (source_path, target_path) in enumerate(zip(data_source, data_target), 1):
        source = pq.read_table(source_path)
        target = pq.read_table(target_path)
        if source.num_rows != target.num_rows:
            raise ValueError(f"data row mismatch: {source_path} -> {target_path}")
        state16 = _vector_column(source, "observation.state", 16, np.float64)
        action12 = _vector_column(source, "action", 12, np.float64)
        expected_state, expected_action, _ = convert_state_action(state16, action12)
        actual_state = _vector_column(target, "observation.state", STATE_DIM, np.float32)
        actual_action = _vector_column(target, "action", ACTION_DIM, np.float32)
        if not np.array_equal(actual_state, expected_state) or not np.array_equal(actual_action, expected_action):
            raise ValueError(f"converted numeric values differ from recomputation: {target_path}")
        if not np.isfinite(actual_state).all() or not np.isfinite(actual_action).all():
            raise ValueError(f"non-finite compact values: {target_path}")
        for binary_dim in (9, 14):
            if np.any(np.abs(np.abs(actual_action[:, binary_dim]) - 1.0) > 1e-6):
                raise ValueError(f"action dim {binary_dim} is not binary in {target_path}")
        other_columns = [name for name in source.column_names if name not in ("observation.state", "action")]
        if not source.select(other_columns).equals(target.select(other_columns)):
            raise ValueError(f"non-robot columns (including prompt IDs) changed: {target_path}")
        max_abs_recompute_error = max(
            max_abs_recompute_error,
            float(np.max(np.abs(actual_action - expected_action), initial=0.0)),
        )
        state_min = np.minimum(state_min, actual_state.min(axis=0))
        state_max = np.maximum(state_max, actual_state.max(axis=0))
        action_min = np.minimum(action_min, actual_action.min(axis=0))
        action_max = np.maximum(action_max, actual_action.max(axis=0))
        rows += target.num_rows
        print(f"[audit:{split}] data {index}/{len(data_target)} rows={rows:,}", flush=True)
    if rows != int(info["total_frames"]):
        raise ValueError(f"data frame total mismatch under {output}: {rows}")

    stats = np.load(output / "meta" / "normalization_stats.npy", allow_pickle=True).item()
    for key, dim in ((ACTION_STATS_KEY, ACTION_DIM), (STATE_STATS_KEY, STATE_DIM)):
        if key not in stats:
            raise KeyError(f"missing {key} in split statistics")
        for name in ("mean", "std", "min", "max", "q01", "q99"):
            if np.asarray(stats[key][name]).shape != (dim,):
                raise ValueError(f"bad statistics shape {key}.{name}")

    copied_relatives = []
    for source_path in sorted((packed / "videos").rglob("*.mp4")):
        copied_relatives.append(source_path.relative_to(packed))
    for relative in (Path(".gitattributes"), Path("meta/tasks.parquet")):
        if (packed / relative).is_file():
            copied_relatives.append(relative)
    copied_relatives.extend(
        source_path.relative_to(packed) for source_path in sorted((packed / "meta" / "episodes").rglob("*.parquet"))
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        byte_counts = list(
            executor.map(
                lambda relative: _independent_equal_copy(packed / relative, output / relative),
                copied_relatives,
            )
        )
    print(
        f"[audit:{split}] independent byte comparison {len(copied_relatives)} files ({sum(byte_counts):,} bytes)",
        flush=True,
    )

    video = _video_contract(output, episodes, info)
    files = _regular_files(output)
    return {
        "repo": str(output),
        "episodes": len(episodes),
        "frames": rows,
        "prompts": int(info["total_tasks"]),
        "data_files": len(data_target),
        "video": video,
        "all_regular_files": len(files),
        "all_files_have_link_count_one": True,
        "copied_files_byte_identical": len(copied_relatives),
        "copied_bytes_compared": sum(byte_counts),
        "hardlinks_to_packed_source": 0,
        "symlinks": 0,
        "numeric_recompute_max_abs_error": max_abs_recompute_error,
        "state_min": state_min.tolist(),
        "state_max": state_max.tolist(),
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "action_binary_dims_verified": [9, 14],
        "non_robot_columns_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--packed-root", type=Path, default=DEFAULT_PACKED_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    packed_root = args.packed_root.resolve()
    manifest = _json(output_root / "conversion_manifest.json")
    if not manifest.get("independent"):
        raise ValueError("conversion manifest does not declare an independent output")
    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "packed_root": str(packed_root),
        "independent": True,
        "splits": [_audit_split(output_root, packed_root, split, max(1, args.workers)) for split in SPLIT_REPO_NAMES],
    }
    global_stats = np.load(output_root / "robocasa365_multitask_compact_stats.npy", allow_pickle=True).item()
    if int(global_stats["num_timesteps"]) != sum(item["frames"] for item in report["splits"]):
        raise ValueError("global normalization statistics count does not match converted rows")
    report["global_stats_rows"] = int(global_stats["num_timesteps"])
    report["total_output_bytes"] = sum(path.stat().st_size for path in _regular_files(output_root))
    report_path = output_root / "conversion_audit.json"
    temp_path = report_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_path, report_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
