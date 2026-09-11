#!/usr/bin/env python3
"""Convert the Wuji/Grasp Anything v2 bucket for OpenWAM training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.utils.rot6d import convert_wuji_58

CAMERAS = (
    "observation.images.head_view",
    "observation.images.left_wrist_view",
    "observation.images.right_wrist_view",
)
STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")
ROT6D_DIMS = (3, 4, 5, 6, 7, 8, 32, 33, 34, 35, 36, 37)


def _converted_feature_names(old_names: list[str]) -> list[str]:
    if len(old_names) != 58:
        raise ValueError(f"expected 58 feature names, got {len(old_names)}")

    def eef_names(prefix: str, position_names: list[str]) -> list[str]:
        return position_names + [
            f"{prefix}.rot6d_r0c0",
            f"{prefix}.rot6d_r1c0",
            f"{prefix}.rot6d_r2c0",
            f"{prefix}.rot6d_r0c1",
            f"{prefix}.rot6d_r1c1",
            f"{prefix}.rot6d_r2c1",
        ]

    return (
        eef_names("left_eef", old_names[0:3])
        + old_names[18:38]
        + eef_names("right_eef", old_names[9:12])
        + old_names[38:58]
    )


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _replace_vector_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"source parquet is missing {name!r}")
    return table.set_column(index, name, pa.array(values.tolist(), type=pa.list_(pa.float32())))


def _stats(rows: np.ndarray, *, pin_rot6d: bool) -> dict:
    rows = np.asarray(rows, dtype=np.float32)
    result = {
        "mean": rows.mean(axis=0, dtype=np.float64).astype(np.float32),
        "std": rows.std(axis=0, dtype=np.float64).astype(np.float32),
        "min": rows.min(axis=0).astype(np.float32),
        "max": rows.max(axis=0).astype(np.float32),
        "q01": np.quantile(rows, 0.01, axis=0).astype(np.float32),
        "q99": np.quantile(rows, 0.99, axis=0).astype(np.float32),
    }
    if pin_rot6d:
        for key in STAT_KEYS:
            result[key][list(ROT6D_DIMS)] = {
                "mean": 0.0,
                "std": 1.0,
                "min": -1.0,
                "max": 1.0,
                "q01": -1.0,
                "q99": 1.0,
            }[key]
    return result


def _copy_meta(
    source: Path,
    destination: Path,
    episodes: list[dict],
    info: dict,
    descriptive_stats: dict,
    normalization_stats: dict,
) -> None:
    meta = destination / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "meta" / "tasks.jsonl", meta / "tasks.jsonl")
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(ep, ensure_ascii=False) + "\n" for ep in episodes)
    )
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    modality = {
        "action": {
            "left_eef": {"start": 0, "end": 9},
            "left_hand_joint": {"start": 9, "end": 29},
            "right_eef": {"start": 29, "end": 38},
            "right_hand_joint": {"start": 38, "end": 58},
        },
        "state": {
            "left_eef": {"start": 0, "end": 9},
            "left_hand_joint": {"start": 9, "end": 29},
            "right_eef": {"start": 29, "end": 38},
            "right_hand_joint": {"start": 38, "end": 58},
        },
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
        "video": {
            "head_view": {"original_key": CAMERAS[0]},
            "left_wrist_view": {"original_key": CAMERAS[1]},
            "right_wrist_view": {"original_key": CAMERAS[2]},
        },
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=2) + "\n")
    serial_stats = {
        stream: {key: value.tolist() for key, value in block.items()}
        for stream, block in descriptive_stats.items()
    }
    (meta / "stats.json").write_text(json.dumps(serial_stats, indent=2) + "\n")
    np.save(meta / "normalization_stats.npy", normalization_stats, allow_pickle=True)
    manifest = {
        "adapter": "grasp_anything_openwam",
        "version": 1,
        "source": str(source.resolve()),
        "target_layout": "[L EEF9, L hand20, R EEF9, R hand20]",
        "rot6d_convention": "column",
        "canonical_map": ["0-8", "10-29", "34-42", "44-63"],
        "episodes": len(episodes),
        "frames": int(sum(ep["length"] for ep in episodes)),
    }
    (meta / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def convert(source: Path, destination: Path, *, copy_videos: bool, overwrite: bool) -> None:
    if not source.is_dir() or not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"source is not a valid bucket: {source}")
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"destination is non-empty; pass --overwrite: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    source_info = json.loads((source / "meta" / "info.json").read_text())
    source_eps = _jsonl(source / "meta" / "episodes.jsonl")
    episode_indices = [int(ep["episode_index"]) for ep in source_eps]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError("source episodes.jsonl contains duplicate episode_index values")
    declared_episodes = source_info.get("total_episodes")
    if declared_episodes is not None and int(declared_episodes) != len(source_eps):
        raise ValueError(
            f"source declares total_episodes={declared_episodes}, but episodes.jsonl has {len(source_eps)} records"
        )
    action_rows, state_rows = [], []
    out_eps = []
    for ep in sorted(source_eps, key=lambda item: int(item["episode_index"])):
        ep_idx = int(ep["episode_index"])
        src_path = source / source_info["data_path"].format(
            episode_chunk=0,
            episode_index=ep_idx,
            chunk_index=0,
            file_index=ep_idx,
        )
        if not src_path.is_file():
            raise FileNotFoundError(f"source episode parquet missing: {src_path}")
        dst_path = destination / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(src_path)
        if table.num_rows != int(ep["length"]):
            raise ValueError(
                f"episode {ep_idx} metadata length={ep['length']} but parquet has {table.num_rows} rows"
            )
        action = convert_wuji_58(np.asarray(table["action"].to_pylist(), dtype=np.float32))
        state = convert_wuji_58(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
        table = _replace_vector_column(table, "action", action)
        table = _replace_vector_column(table, "observation.state", state)
        pq.write_table(table, dst_path, compression="zstd")
        action_rows.append(action)
        state_rows.append(state)
        out_ep = dict(ep)
        out_ep["action_space"] = "eef_absolute_hand_absolute_rot6d_column"
        out_ep["source_file"] = ep.get("source_file", "")
        out_eps.append(out_ep)

    video_root = destination / "videos" / "chunk-000"
    for camera in CAMERAS:
        src_dir = source / "videos" / "chunk-000" / camera
        dst_dir = video_root / camera
        dst_dir.mkdir(parents=True, exist_ok=True)
        for ep in out_eps:
            idx = int(ep["episode_index"])
            src = src_dir / f"episode_{idx:06d}.mp4"
            dst = dst_dir / src.name
            if not src.is_file():
                raise FileNotFoundError(f"source episode video missing: {src}")
            if copy_videos:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())

    features = dict(source_info["features"])
    for name in ("action", "observation.state"):
        features[name] = dict(features[name])
        features[name]["shape"] = [58]
        old_names = list(features[name].get("names") or [])
        features[name]["names"] = _converted_feature_names(old_names)
    info = dict(source_info)
    total_frames = int(sum(ep["length"] for ep in out_eps))
    info.update(
        {
            "codebase_version": "openwam-grasp-anything-v2-adapter-1",
            "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
            "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4",
            "robot_type": "WUJI_ASTRIBOT_EEF_ABSOLUTE_HAND_ABSOLUTE_ROT6D_COLUMN",
            "features": features,
            "total_episodes": len(out_eps),
            "total_frames": total_frames,
            "total_videos": len(out_eps) * len(CAMERAS),
            "total_chunks": 1,
            "splits": {"train": f"0:{len(out_eps)}"},
        }
    )
    descriptive_stats = {
        "action": _stats(np.concatenate(action_rows, axis=0), pin_rot6d=False),
        "observation.state": _stats(np.concatenate(state_rows, axis=0), pin_rot6d=False),
    }
    source_stats = json.loads((source / "meta" / "stats.json").read_text())
    if "timestamp" in source_stats:
        descriptive_stats["timestamp"] = {
            key: np.asarray(value, dtype=np.float32) for key, value in source_stats["timestamp"].items()
        }
    normalization_stats = {
        "eef": _stats(np.concatenate(action_rows + state_rows, axis=0), pin_rot6d=True)
    }
    _copy_meta(source, destination, out_eps, info, descriptive_stats, normalization_stats)
    print(f"converted {len(out_eps)} episodes / {total_frames} frames -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--copy-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args.source, args.destination, copy_videos=args.copy_videos, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
