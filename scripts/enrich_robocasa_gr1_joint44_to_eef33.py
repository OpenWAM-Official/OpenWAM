#!/usr/bin/env python3
"""Enrich NVIDIA RoboCasa GR1 LeRobot v2.0 buckets with simulator FK EEF33.

The input is never modified.  Each selected episode is decoded from its native
44-D ``action`` / ``observation.state`` columns and projected through the exact
RoboCasa GR1 MuJoCo model into:

* ``eef33_action``
* ``eef33_state``

Run this script in the RoboCasa-GR1 environment (MuJoCo + robosuite + the
``robocasa-gr1-tabletop-tasks`` checkout), then pass the output to
``convert_robocasa_gr1_v20_to_v30.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openwam.dataloader.utils.gr1_kinematics import (  # noqa: E402
    EEF33_DIM,
    JOINT44_DIM,
    ROT6D_DIMS_EEF33,
    GR1Kinematics,
)

_FILES_PER_CHUNK = 1000


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _format_path(template: str, episode_index: int, *, video_key: str | None = None) -> str:
    return template.format(
        episode_chunk=episode_index // _FILES_PER_CHUNK,
        episode_index=episode_index,
        video_key=video_key,
    )


def _discover_buckets(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").is_file():
        return [root]
    return sorted(path.parent.parent for path in root.glob("*/meta/info.json"))


def _env_id_from_bucket(bucket: Path) -> str:
    name = bucket.name
    if not name.startswith("gr1_unified."):
        raise ValueError(
            f"{bucket}: cannot derive env id; expected bucket name starting with 'gr1_unified.' "
            "(override with --env-id for a single bucket)"
        )
    task = re.sub(r"_\d+$", "", name.removeprefix("gr1_unified."))
    if not task.endswith("_GR1ArmsAndWaistFourierHands"):
        raise ValueError(f"{bucket}: unsupported GR1 embodiment in bucket name {name!r}")
    return f"gr1_unified/{task}_Env"


def _link(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError as exc:
            raise OSError(f"cannot hard-link {source} -> {target}; use --link-mode copy") from exc
    if mode == "symlink":
        target.symlink_to(os.path.relpath(source, target.parent))
    elif mode == "copy":
        shutil.copy2(source, target)
    else:  # pragma: no cover
        raise ValueError(mode)


def _validate_rot6d(values: np.ndarray, *, source: Path) -> None:
    for start in (3, 18):
        first = values[:, start : start + 3]
        second = values[:, start + 3 : start + 6]
        norm_error = max(
            float(np.max(np.abs(np.linalg.norm(first, axis=1) - 1.0))),
            float(np.max(np.abs(np.linalg.norm(second, axis=1) - 1.0))),
        )
        dot_error = float(np.max(np.abs(np.sum(first * second, axis=1))))
        if norm_error > 1e-4 or dot_error > 1e-4:
            raise ValueError(
                f"{source}: invalid rot6d from FK (norm_error={norm_error:.3e}, dot_error={dot_error:.3e})"
            )
    if not np.isfinite(values[:, ROT6D_DIMS_EEF33]).all():
        raise ValueError(f"{source}: non-finite FK rotation")


def _make_env(env_id: str, repo_path: Path | None):
    if repo_path is not None and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    import gymnasium as gym  # noqa: PLC0415
    import robocasa  # noqa: F401, PLC0415
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401, PLC0415

    return gym.make(env_id, enable_render=False)


def enrich_bucket(
    source: Path,
    output: Path,
    *,
    env_id: str | None = None,
    repo_path: Path | None = None,
    link_mode: str = "hardlink",
    overwrite: bool = False,
    episode_limit: int | None = None,
) -> dict:
    info = _read_json(source / "meta" / "info.json")
    if info.get("codebase_version") != "v2.0":
        raise ValueError(f"{source}: expected codebase_version='v2.0'")
    features = info.get("features", {})
    for column in ("action", "observation.state"):
        shape = tuple(features.get(column, {}).get("shape", ()))
        if shape != (JOINT44_DIM,):
            raise ValueError(f"{source}: {column!r} must be native joint44, got shape={shape}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)

    episodes = _read_jsonl(source / "meta" / "episodes.jsonl")
    if episode_limit is not None:
        if episode_limit <= 0:
            raise ValueError("--episode-limit must be positive")
        episodes = episodes[:episode_limit]
    if not episodes:
        raise ValueError(f"{source}: no episodes selected")

    selected_env = env_id or _env_id_from_bucket(source)
    env = _make_env(selected_env, repo_path)
    try:
        env.reset(seed=0)
        kinematics = GR1Kinematics.from_env(env)
        output.mkdir(parents=True)
        (output / "meta").mkdir()

        output_info = dict(info)
        output_info["total_episodes"] = len(episodes)
        output_info["total_frames"] = int(sum(int(row["length"]) for row in episodes))
        output_info["splits"] = {"train": f"0:{len(episodes)}"}
        output_info["features"] = {key: dict(value) for key, value in features.items()}
        eef_feature = {"dtype": "float32", "shape": [EEF33_DIM]}
        output_info["features"]["eef33_action"] = dict(eef_feature)
        output_info["features"]["eef33_state"] = dict(eef_feature)
        (output / "meta" / "info.json").write_text(json.dumps(output_info, indent=2), encoding="utf-8")
        _write_jsonl(output / "meta" / "episodes.jsonl", episodes)
        for name in ("tasks.jsonl", "modality.json", "stats.json"):
            src = source / "meta" / name
            if src.is_file():
                shutil.copy2(src, output / "meta" / name)

        video_keys = sorted(key for key, spec in features.items() if spec.get("dtype") == "video")
        total_rows = 0
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            data_rel = _format_path(info["data_path"], episode_index)
            src_data = source / data_rel
            table = pq.read_table(src_data)
            frame = table.to_pandas()
            actions = np.stack(frame["action"].values).astype(np.float32)
            states = np.stack(frame["observation.state"].values).astype(np.float32)
            if actions.shape != (len(frame), JOINT44_DIM) or states.shape != (len(frame), JOINT44_DIM):
                raise ValueError(
                    f"{src_data}: expected action/state ({len(frame)}, {JOINT44_DIM}), "
                    f"got {actions.shape}/{states.shape}"
                )
            eef_action = np.stack([kinematics.joint44_to_eef33(row) for row in actions])
            eef_state = np.stack([kinematics.joint44_to_eef33(row) for row in states])
            _validate_rot6d(eef_action, source=src_data)
            _validate_rot6d(eef_state, source=src_data)
            frame["eef33_action"] = list(eef_action)
            frame["eef33_state"] = list(eef_state)
            target_data = output / data_rel
            target_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), target_data)
            total_rows += len(frame)

            for video_key in video_keys:
                video_rel = _format_path(info["video_path"], episode_index, video_key=video_key)
                _link(source / video_rel, output / video_rel, link_mode)
        return {
            "source": str(source),
            "output": str(output),
            "env_id": selected_env,
            "episodes": len(episodes),
            "frames": total_rows,
            "representation": "eef33",
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robocasa-path", type=Path, default=os.environ.get("ROBOCASA_GR1_PATH"))
    parser.add_argument("--env-id", help="Single-bucket override; root conversion derives each id from its name")
    parser.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--episode-limit", type=int)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("--output must differ from --input")
    buckets = _discover_buckets(source)
    if not buckets:
        raise FileNotFoundError(f"no v2.0 buckets found under {source}")
    if args.env_id and len(buckets) != 1:
        raise ValueError("--env-id is valid only for a single bucket")
    reports = []
    for bucket in buckets:
        target = output if len(buckets) == 1 and bucket == source else output / bucket.name
        reports.append(
            enrich_bucket(
                bucket,
                target,
                env_id=args.env_id,
                repo_path=args.robocasa_path,
                link_mode=args.link_mode,
                overwrite=args.overwrite,
                episode_limit=args.episode_limit,
            )
        )
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
