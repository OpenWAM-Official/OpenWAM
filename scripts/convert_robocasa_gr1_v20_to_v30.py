#!/usr/bin/env python3
"""Re-index NVIDIA RoboCasa GR1 LeRobot v2.0 buckets as LeRobot v3.

OpenWAM's GR1 integration intentionally trains only bimanual EEF20. The public
NVIDIA files contain joint44 and must first be enriched by a trusted simulator
or FK pipeline with EEF pose/gripper columns. This converter validates those
columns, preserves episode payloads (hard-linking by default), and writes the
LeRobot v3 metadata/path contract. It never invents EEF values from joints.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_V20 = "v2.0"
_V30 = "v3.0"
_FILES_PER_CHUNK = 1000
_EEF_FEATURES = {
    "eef_sim_pose_action": (12,),
    "gripper_open_scale_action": (2,),
    "eef_sim_pose_state": (12,),
    "gripper_open_scale_state": (2,),
}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_buckets(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").is_file():
        return [root]
    return sorted(path.parent.parent for path in root.glob("*/meta/info.json"))


def _link_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, target)
        except OSError as exc:
            raise OSError(
                f"cannot hard-link {source} to {target}; use --link-mode symlink "
                "or --link-mode copy when input/output are on different filesystems"
            ) from exc
    elif mode == "symlink":
        target.symlink_to(os.path.relpath(source, target.parent))
    elif mode == "copy":
        shutil.copy2(source, target)
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(mode)


def _format_v20_path(template: str, episode_index: int, *, video_key: str | None = None) -> str:
    values = {
        "episode_chunk": episode_index // _FILES_PER_CHUNK,
        "episode_index": episode_index,
        "video_key": video_key,
    }
    return template.format(**values)


def _validate_episode(
    data_path: Path,
    episode: dict,
    tasks: dict[int, str],
) -> int:
    table = pq.read_table(data_path)
    length = int(episode["length"])
    if table.num_rows != length:
        raise ValueError(f"{data_path}: parquet rows={table.num_rows}, episode length={length}")
    required = {"episode_index", "task_index", *_EEF_FEATURES}
    missing = required.difference(table.column_names)
    if missing:
        raise KeyError(f"{data_path}: missing required columns {sorted(missing)}")

    frame = table.select(["episode_index", "task_index"]).to_pandas()
    episode_indices = np.unique(frame["episode_index"].to_numpy())
    task_indices = np.unique(frame["task_index"].to_numpy())
    if episode_indices.tolist() != [int(episode["episode_index"])]:
        raise ValueError(f"{data_path}: inconsistent episode_index values {episode_indices.tolist()}")
    if len(task_indices) != 1:
        raise ValueError(f"{data_path}: expected one task_index, got {task_indices.tolist()}")
    task_index = int(task_indices[0])
    prompt = str(tasks.get(task_index, "")).strip()
    if not prompt:
        raise ValueError(f"{data_path}: task_index={task_index} has no non-empty tasks.jsonl prompt")
    episode_tasks = [str(value).strip() for value in episode.get("tasks", []) if str(value).strip()]
    if episode_tasks and prompt not in episode_tasks:
        raise ValueError(
            f"{data_path}: tasks.jsonl prompt {prompt!r} does not match episode tasks {episode_tasks!r}"
        )

    for key, expected in _EEF_FEATURES.items():
        values = np.stack(table[key].to_pylist())
        if values.shape != (length, *expected):
            raise ValueError(f"{data_path}: {key} shape={values.shape}, expected {(length, *expected)}")
    return task_index


def convert_bucket(
    source: Path,
    output: Path,
    *,
    link_mode: str = "hardlink",
    overwrite: bool = False,
    episode_limit: int | None = None,
) -> dict:
    info = _read_json(source / "meta" / "info.json")
    version = info.get("codebase_version")
    if version != _V20:
        raise ValueError(f"{source}: expected codebase_version={_V20!r}, got {version!r}")
    features = info.get("features", {})
    missing_eef = sorted(set(_EEF_FEATURES).difference(features))
    if missing_eef:
        raise KeyError(
            f"{source}: missing required EEF features {missing_eef}. The native NVIDIA joint44 "
            "download is not directly trainable by this EEF-only integration; enrich it with "
            "trusted pose/gripper values before conversion."
        )
    for key, expected in _EEF_FEATURES.items():
        shape = tuple(features[key].get("shape", ()))
        if shape != expected:
            raise ValueError(f"{source}: feature {key!r} must have shape {list(expected)}, got {shape}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)

    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    if episode_limit is not None:
        if episode_limit <= 0:
            raise ValueError("episode_limit must be positive")
        episodes = episodes[:episode_limit]
    if not episodes:
        raise ValueError(f"{source}: no episodes selected")
    expected_indices = list(range(len(episodes)))
    actual_indices = [int(row["episode_index"]) for row in episodes]
    if actual_indices != expected_indices:
        raise ValueError(
            f"{source}: selected episodes must be contiguous from zero; got first/last "
            f"{actual_indices[:1]}/{actual_indices[-1:]}"
        )

    task_rows = _read_jsonl(source / "meta" / "tasks.jsonl")
    tasks = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    video_keys = sorted(key for key, spec in info["features"].items() if spec.get("dtype") == "video")
    episode_rows = []
    used_tasks: set[int] = set()
    total_frames = 0

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        chunk_index = episode_index // _FILES_PER_CHUNK
        file_index = episode_index % _FILES_PER_CHUNK
        source_data = source / _format_v20_path(info["data_path"], episode_index)
        if not source_data.is_file():
            raise FileNotFoundError(source_data)
        task_index = _validate_episode(source_data, episode, tasks)
        used_tasks.add(task_index)

        target_data = output / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        _link_file(source_data, target_data, link_mode)

        length = int(episode["length"])
        row = dict(episode)
        row.update(
            {
                "data/chunk_index": chunk_index,
                "data/file_index": file_index,
                "dataset_from_index": 0,
                "dataset_to_index": length,
            }
        )
        for video_key in video_keys:
            source_video = source / _format_v20_path(
                info["video_path"],
                episode_index,
                video_key=video_key,
            )
            if not source_video.is_file():
                raise FileNotFoundError(source_video)
            target_video = (
                output
                / "videos"
                / video_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            _link_file(source_video, target_video, link_mode)
            row[f"videos/{video_key}/chunk_index"] = chunk_index
            row[f"videos/{video_key}/file_index"] = file_index
        episode_rows.append(row)
        total_frames += length

    output_info = dict(info)
    output_info.update(
        {
            "codebase_version": _V30,
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": len(used_tasks),
            "total_videos": len(episodes) * len(video_keys),
            "total_chunks": (len(episodes) + _FILES_PER_CHUNK - 1) // _FILES_PER_CHUNK,
            "chunks_size": _FILES_PER_CHUNK,
            "fps": int(info["fps"]),
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        }
    )
    output_info["features"] = {key: dict(value) for key, value in info["features"].items()}
    for feature in output_info["features"].values():
        if feature.get("dtype") != "video":
            feature["fps"] = int(info["fps"])

    (output / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    with (output / "meta" / "info.json").open("w", encoding="utf-8") as handle:
        json.dump(output_info, handle, indent=2)
    selected_tasks = [(task_index, tasks[task_index]) for task_index in sorted(used_tasks)]
    task_frame = pd.DataFrame(
        {"task_index": [item[0] for item in selected_tasks]},
        index=pd.Index([item[1] for item in selected_tasks], name="task"),
    )
    pq.write_table(pa.Table.from_pandas(task_frame), output / "meta" / "tasks.parquet")
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(episode_rows)),
        output / "meta" / "episodes" / "chunk-000.parquet",
    )
    for name in ("modality.json", "stats.json"):
        source_meta = source / "meta" / name
        if source_meta.is_file():
            shutil.copy2(source_meta, output / "meta" / name)

    return {
        "source": str(source),
        "output": str(output),
        "episodes": len(episodes),
        "frames": total_frames,
        "tasks": len(used_tasks),
        "video_keys": video_keys,
        "raw_eef_dim": 20,
        "representation": "bimanual_eef20",
    }


def convert_root(
    source_root: Path,
    output_root: Path,
    *,
    link_mode: str,
    overwrite: bool,
    episode_limit: int | None,
) -> Iterable[dict]:
    buckets = discover_buckets(source_root)
    if not buckets:
        raise FileNotFoundError(f"no LeRobot v2.0 buckets found under {source_root}")
    single = len(buckets) == 1 and buckets[0] == source_root
    for bucket in buckets:
        output = output_root if single else output_root / bucket.name
        yield convert_bucket(
            bucket,
            output,
            link_mode=link_mode,
            overwrite=overwrite,
            episode_limit=episode_limit,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One v2.0 bucket or its multi-bucket root")
    parser.add_argument("--output", type=Path, required=True, help="New v3 bucket/root; input is never modified")
    parser.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--episode-limit", type=int, default=None, help="Debug-only leading episode limit per bucket")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must differ from --input; conversion is deliberately non-destructive")
    reports = list(
        convert_root(
            args.input.resolve(),
            args.output.resolve(),
            link_mode=args.link_mode,
            overwrite=args.overwrite,
            episode_limit=args.episode_limit,
        )
    )
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
