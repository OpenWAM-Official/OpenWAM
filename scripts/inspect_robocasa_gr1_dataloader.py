#!/usr/bin/env python3
"""Inspect one RoboCasa GR1 dataloader sample and save visual artifacts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openwam.dataloader.robocasa_gr1 import MultiRoboCasaGR1Dataset, RoboCasaGR1Dataset  # noqa: E402
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20  # noqa: E402
from openwam.dataloader.utils.unify_action import unmap_from_unify  # noqa: E402


def _contact_sheet(frames: list[Image.Image], labels: list[str], columns: int = 2) -> Image.Image:
    frames = [frame.convert("RGB") for frame in frames]
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (frame, label) in enumerate(zip(frames, labels)):
        x = (index % columns) * width
        y = (index // columns) * (height + 24)
        sheet.paste(frame, (x, y))
        draw.text((x + 4, y + height + 4), label, fill="black")
    return sheet


def _rot6d_report(raw: np.ndarray) -> dict:
    errors = []
    for start in (3, 13):
        first = raw[..., start : start + 3]
        second = raw[..., start + 3 : start + 6]
        errors.append(
            {
                "first_norm_max_error": float(np.max(np.abs(np.linalg.norm(first, axis=-1) - 1.0))),
                "second_norm_max_error": float(np.max(np.abs(np.linalg.norm(second, axis=-1) - 1.0))),
                "orthogonality_max_error": float(np.max(np.abs(np.sum(first * second, axis=-1)))),
            }
        )
    return {"left": errors[0], "right": errors[1]}


def _locate_sample(dataset, sample_index: int):
    if isinstance(dataset, MultiRoboCasaGR1Dataset):
        bucket_index = int(np.searchsorted(dataset._cum_lens, sample_index, side="right") - 1)  # noqa: SLF001
        bucket = dataset.buckets[bucket_index]
        local_index = sample_index - int(dataset._cum_lens[bucket_index])  # noqa: SLF001
    else:
        bucket = dataset
        local_index = sample_index
    episode_local = int(np.searchsorted(bucket._cum_n_starts, local_index, side="right") - 1)  # noqa: SLF001
    return bucket, episode_local


def _load_pair(config_path: Path, sample_index: int, seed: int, dataset_dir: Path | None = None):
    configured = OmegaConf.load(config_path)
    if dataset_dir is not None:
        OmegaConf.update(configured, "dataset_dir", str(dataset_dir), merge=False)
    plain_cfg = OmegaConf.create(OmegaConf.to_container(configured, resolve=True))
    OmegaConf.update(plain_cfg, "color_jitter", None, merge=False)

    random.seed(seed)
    plain_dataset = RoboCasaGR1Dataset.from_config(plain_cfg, split="train")
    plain = plain_dataset[sample_index]
    random.seed(seed)
    jitter_dataset = RoboCasaGR1Dataset.from_config(configured, split="train")
    jittered = jitter_dataset[sample_index]
    return plain_dataset, plain, jitter_dataset, jittered


def inspect_sample(
    config_path: Path,
    sample_index: int,
    output_dir: Path,
    seed: int,
    dataset_dir: Path | None = None,
) -> dict:
    plain_dataset, plain, jitter_dataset, jittered = _load_pair(
        config_path,
        sample_index,
        seed,
        dataset_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plain_frames = plain["video"]
    jitter_frames = jittered["video"]
    if len(plain_frames) != len(jitter_frames) or not plain_frames:
        raise ValueError("plain and jittered samples must contain the same non-zero frame count")
    plain_frames[0].save(output_dir / "plain_composed_frame.png")
    jitter_frames[0].save(output_dir / "color_jitter_composed_frame.png")
    ImageChops.difference(plain_frames[0], jitter_frames[0]).save(output_dir / "color_jitter_difference.png")
    preview_count = min(4, len(plain_frames))
    _contact_sheet(
        [*plain_frames[:preview_count], *jitter_frames[:preview_count]],
        [
            *[f"plain t={index}" for index in range(preview_count)],
            *[f"color_jitter t={index}" for index in range(preview_count)],
        ],
    ).save(output_dir / "temporal_contact_sheet.png")

    plain_pixels = np.asarray(plain_frames[0], dtype=np.float32)
    jitter_pixels = np.asarray(jitter_frames[0], dtype=np.float32)
    top_height = round(plain_pixels.shape[0] * 2 / 3)
    report = {
        "config": str(config_path),
        "sample_index": sample_index,
        "action_mode": plain_dataset.buckets[0].action_mode
        if isinstance(plain_dataset, MultiRoboCasaGR1Dataset)
        else plain_dataset.action_mode,
        "action_shape": list(plain["action"].shape),
        "action_mask_valid_dims": int(plain["action_mask"][0].sum().item()),
        "proprio_shape": list(plain["proprio"].shape),
        "prompt": plain["prompt"],
        "video_frames": len(plain_frames),
        "composed_size_wh": list(plain_frames[0].size),
        "top_nonblack_fraction": float(np.mean(np.any(plain_pixels[:top_height] != 0, axis=-1))),
        "bottom_nonblack_fraction": float(np.mean(np.any(plain_pixels[top_height:] != 0, axis=-1))),
        "color_jitter_mean_abs_pixel_delta": float(np.mean(np.abs(jitter_pixels - plain_pixels))),
        "artifacts": {
            "plain": str(output_dir / "plain_composed_frame.png"),
            "color_jitter": str(output_dir / "color_jitter_composed_frame.png"),
            "difference": str(output_dir / "color_jitter_difference.png"),
            "contact_sheet": str(output_dir / "temporal_contact_sheet.png"),
        },
    }
    if not report["prompt"].strip():
        raise ValueError("resolved prompt is empty")
    if report["color_jitter_mean_abs_pixel_delta"] <= 0:
        raise ValueError("configured color_jitter produced no pixel change")

    bucket, episode_local = _locate_sample(jitter_dataset, sample_index)
    row = bucket._eps_df.iloc[episode_local]  # noqa: SLF001 - intentional inspection.
    table = bucket._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))  # noqa: SLF001
    offset = int(row["_data_row_offset"])
    window = table.slice(offset, int(row["length"])).to_pandas()
    raw_action = bucket._raw_action(window)  # noqa: SLF001
    raw_state = bucket._raw_state(window)  # noqa: SLF001
    task_index = int(window["task_index"].iloc[0])
    indexed_prompt = str(bucket._task_idx_to_text.get(task_index, "")).strip()  # noqa: SLF001
    report.update(
        {
            "raw_action_shape": list(raw_action.shape),
            "raw_state_shape": list(raw_state.shape),
            "raw_action_all_finite": bool(np.isfinite(raw_action).all()),
            "raw_state_all_finite": bool(np.isfinite(raw_state).all()),
            "raw_action_range": [float(raw_action.min()), float(raw_action.max())],
            "raw_state_range": [float(raw_state.min()), float(raw_state.max())],
            "task_index": task_index,
            "task_index_prompt": indexed_prompt,
            "prompt_matches_task_index": plain["prompt"] == indexed_prompt,
        }
    )
    if not report["raw_action_all_finite"] or not report["raw_state_all_finite"]:
        raise ValueError("decoded state/action contains NaN or infinity")
    if not report["prompt_matches_task_index"]:
        raise ValueError("sample prompt does not match task_index -> tasks.parquet")

    modality_path = Path(bucket._dataset_dir) / "meta" / "modality.json"  # noqa: SLF001
    if modality_path.is_file():
        with modality_path.open(encoding="utf-8") as handle:
            report["modality"] = json.load(handle)

    mode = bucket.action_mode
    if mode in {"eef", "unify"}:
        raw = raw_action
        if raw.shape[-1] != 20:
            raise ValueError(f"EEF/unify reader must emit raw EEF20, got {raw.shape}")
        report["eef20_layout"] = "[L xyz3, rot6d6, grip1, R xyz3, rot6d6, grip1]"
        report["rot6d"] = _rot6d_report(raw)
        normalized = bucket._normalize_array(raw)  # noqa: SLF001
        rotation_indices = list(ROT6D_DIMS_EEF20)
        report["rot6d_normalization_max_delta"] = float(
            np.max(np.abs(normalized[..., rotation_indices] - raw[..., rotation_indices]))
        )
        if report["rot6d_normalization_max_delta"] > 1e-6:
            raise ValueError("normalization changed rot6d; regenerate stats with identity-pinned rotation dims")
        if mode == "unify":
            restored = unmap_from_unify(
                jittered["action"].numpy(),
                bucket._unify_dst_index,  # noqa: SLF001
            )
            expected = normalized[: restored.shape[0]]
            report["unify_roundtrip_max_error"] = float(np.max(np.abs(restored - expected)))
            if report["unify_roundtrip_max_error"] > 1e-6:
                raise ValueError("unify_action mapping does not round-trip to normalized EEF20")
    else:
        report["representation"] = (
            "native joint/body vector; not xyz+rot6d+gripper and not semantically mappable to EEF80"
        )

    with (output_dir / "inspection_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataloader/robocasa_gr1.yaml"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = inspect_sample(args.config, args.sample_index, args.output_dir, args.seed, args.dataset_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
