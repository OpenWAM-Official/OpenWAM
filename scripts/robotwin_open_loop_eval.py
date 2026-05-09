"""Open-loop RoboTwin evaluation through the current deploy inference path.

Loads a checkpoint with ``openwam.deploy.model_loader.load_from_checkpoint_dir``,
samples windows using the current RoboTwin dataloader, feeds the first-frame
observation and raw proprio state into ``JointInferenceEngine.generate()``, and
compares the generated action chunk against the dataset ground-truth action
chunk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))


EEF_GROUPS = {
    "left_xyz": slice(0, 3),
    "left_rot6d": slice(3, 9),
    "left_grip": [9],
    "right_xyz": slice(10, 13),
    "right_rot6d": slice(13, 19),
    "right_grip": [19],
}


def _parse_indices(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    err = pred - gt
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))),
        "first_mae": float(np.mean(np.abs(err[0]))),
        "first_l2": float(np.linalg.norm(err[0])),
    }
    for name, idx in EEF_GROUPS.items():
        out[f"mae_{name}"] = float(np.mean(np.abs(err[:, idx])))
        out[f"first_mae_{name}"] = float(np.mean(np.abs(err[0, idx])))
    return out


def _summarize(rows: list[dict[str, Any]], key_prefix: str) -> dict[str, float]:
    keys = [k for k in rows[0][key_prefix] if isinstance(rows[0][key_prefix][k], (int, float))]
    summary = {}
    for k in keys:
        vals = np.asarray([r[key_prefix][k] for r in rows], dtype=np.float64)
        summary[f"{key_prefix}.{k}.mean"] = float(vals.mean())
        summary[f"{key_prefix}.{k}.median"] = float(np.median(vals))
        summary[f"{key_prefix}.{k}.max"] = float(vals.max())
    return summary


def _sample_indices(dataset_len: int, num_samples: int, seed: int, explicit: list[int] | None) -> list[int]:
    if explicit is not None:
        return explicit
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    rng = np.random.default_rng(seed)
    replace = num_samples > dataset_len
    return [int(i) for i in rng.choice(dataset_len, size=num_samples, replace=replace)]


def _build_dataset_from_checkpoint_cfg(
    ckpt_dir: Path,
    cfg,
    split: str,
    *,
    dataset_dir: str | None = None,
    task_name: str | None = None,
    variant: str | None = None,
):
    from openwam.dataloader.registry import build_dataset

    dl_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataloader, resolve=True))
    OmegaConf.update(dl_cfg, "action_stats_path", str(ckpt_dir / "action_stats.npy"), merge=False)
    if dataset_dir is not None:
        OmegaConf.update(dl_cfg, "dataset_dir", dataset_dir, merge=False)
    if task_name is not None:
        OmegaConf.update(dl_cfg, "task_name", task_name, merge=False)
        OmegaConf.update(dl_cfg, "train_tasks", None, merge=False)
        OmegaConf.update(dl_cfg, "holdout_tasks", None, merge=False)
    if variant is not None:
        OmegaConf.update(dl_cfg, "variant", variant, merge=False)
    return build_dataset(dl_cfg, split=split)


def _build_engine(
    ckpt_dir: Path, device: str, *, denoise_steps: int | None, schedule_type: str | None, disable_compile: bool
):
    from openwam.deploy.joint_engine import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from scripts.deploy import _load_deploy_config, _merge_with_training_cfg

    training_cfg, architecture = load_from_checkpoint_dir(str(ckpt_dir), device=device)
    deploy_cfg = _load_deploy_config()
    if denoise_steps is not None:
        OmegaConf.update(deploy_cfg, "inference.denoise_steps", int(denoise_steps), merge=False)
    if schedule_type is not None:
        OmegaConf.update(deploy_cfg, "inference.schedule_type", str(schedule_type), merge=False)
    if disable_compile:
        OmegaConf.update(deploy_cfg, "optimization.compile.enabled", False, merge=True)
        OmegaConf.update(deploy_cfg, "optimization.compile.video_dit", False, merge=True)
        OmegaConf.update(deploy_cfg, "optimization.compile.vae", False, merge=True)
    cfg = _merge_with_training_cfg(training_cfg, deploy_cfg)
    engine = JointInferenceEngine(cfg=cfg, architecture=architecture)
    return cfg, architecture, engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-loop RoboTwin eval through current deploy engine.")
    parser.add_argument(
        "--ckpt-dir",
        default="/path/to/openwam_checkpoints/robotwin_dual_system_joint_self_attention",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-dir", default=None, help="Override checkpoint dataloader.dataset_dir.")
    parser.add_argument(
        "--task-name", default=None, help="Restrict sampling to one RoboTwin task, e.g. stack_blocks_three."
    )
    parser.add_argument("--variant", default=None, help="Override RoboTwin variant, e.g. clean_50 or randomized_500.")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--indices", default=None, help="Comma-separated dataset indices. Overrides --num-samples.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--inference-seed", type=int, default=42)
    parser.add_argument("--denoise-steps", type=int, default=None)
    parser.add_argument("--schedule-type", default=None)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_cfg = OmegaConf.load(str(ckpt_dir / "config.yaml"))

    print(f"[open-loop] checkpoint={ckpt_dir}")
    print(
        "[open-loop] checkpoint cfg: "
        f"action_mode={OmegaConf.select(ckpt_cfg, 'dataloader.action_mode')} "
        f"normalize_mode={OmegaConf.select(ckpt_cfg, 'dataloader.normalize_mode')} "
        f"num_frames={OmegaConf.select(ckpt_cfg, 'dataloader.num_frames')} "
        f"canvas={OmegaConf.select(ckpt_cfg, 'dataloader.height')}x{OmegaConf.select(ckpt_cfg, 'dataloader.width')}"
    )

    print("[open-loop] building RoboTwin dataset from checkpoint config...")
    dataset = _build_dataset_from_checkpoint_cfg(
        ckpt_dir,
        ckpt_cfg,
        split=args.split,
        dataset_dir=args.dataset_dir,
        task_name=args.task_name,
        variant=args.variant,
    )
    indices = _sample_indices(len(dataset), args.num_samples, args.sample_seed, _parse_indices(args.indices))
    print(f"[open-loop] dataset_len={len(dataset)} selected_indices={indices}")

    print("[open-loop] loading deploy engine...")
    cfg, architecture, engine = _build_engine(
        ckpt_dir,
        args.device,
        denoise_steps=args.denoise_steps,
        schedule_type=args.schedule_type,
        disable_compile=args.disable_compile,
    )
    print(
        "[open-loop] engine ready: "
        f"denoise_steps={OmegaConf.select(cfg, 'inference.denoise_steps')} "
        f"schedule={OmegaConf.select(cfg, 'inference.schedule_type')} "
        f"compile_enabled={OmegaConf.select(cfg, 'optimization.compile.enabled', default=None)}"
    )

    rows: list[dict[str, Any]] = []
    for sample_i, dataset_idx in enumerate(indices):
        sample = dataset[dataset_idx]
        gt_norm = _to_numpy(sample["action"]).astype(np.float32)
        proprio_norm = _to_numpy(sample["proprio"]).astype(np.float32)

        gt_raw = dataset.denormalize_action(gt_norm).astype(np.float32)
        proprio_raw = dataset.denormalize_action(proprio_norm).astype(np.float32).reshape(-1)

        deploy_norm_state = architecture.normalize_deploy_proprio(torch.from_numpy(proprio_raw))
        deploy_norm_state_np = _to_numpy(deploy_norm_state).reshape(-1)
        state_alignment_err = float(np.max(np.abs(deploy_norm_state_np - proprio_norm.reshape(-1))))

        conditions = {
            "prompt": sample["prompt"],
            "first_frame_image": sample["first_frame_image"],
            "proprio_state": proprio_raw,
            "num_frames": int(OmegaConf.select(cfg, "inference.num_frames", default=gt_raw.shape[0] + 1)),
            "video_num_frames": int(
                OmegaConf.select(
                    cfg,
                    "inference.video_num_frames",
                    default=len(sample["video"]),
                )
            ),
            "height": int(OmegaConf.select(cfg, "inference.height", default=384)),
            "width": int(OmegaConf.select(cfg, "inference.width", default=320)),
            "seed": int(args.inference_seed),
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            result = engine.generate(conditions)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_s = time.time() - t0

        pred_raw = np.asarray(result["actions"], dtype=np.float32)
        horizon = min(len(pred_raw), len(gt_raw))
        pred_raw = pred_raw[:horizon]
        gt_raw_cmp = gt_raw[:horizon]
        gt_norm_cmp = gt_norm[:horizon]
        pred_norm = architecture.action_normalizer.normalize(pred_raw).astype(np.float32)

        row = {
            "sample_i": sample_i,
            "dataset_idx": int(dataset_idx),
            "task_name": sample.get("task_name", ""),
            "episode_path": sample.get("episode_path", ""),
            "start_frame": int(sample.get("start_frame", -1)),
            "horizon": int(horizon),
            "latency_s": float(latency_s),
            "state_alignment_max_abs_err": state_alignment_err,
            "raw": _metrics(pred_raw, gt_raw_cmp),
            "normalized": _metrics(pred_norm, gt_norm_cmp),
            "pred_first_raw": pred_raw[0].tolist(),
            "gt_first_raw": gt_raw_cmp[0].tolist(),
            "pred_first_norm": pred_norm[0].tolist(),
            "gt_first_norm": gt_norm_cmp[0].tolist(),
        }
        rows.append(row)
        print(
            "[open-loop] "
            f"{sample_i + 1}/{len(indices)} idx={dataset_idx} task={row['task_name']!r} "
            f"start={row['start_frame']} horizon={horizon} latency={latency_s:.2f}s "
            f"raw_mae={row['raw']['mae']:.6f} raw_rmse={row['raw']['rmse']:.6f} "
            f"first_l2={row['raw']['first_l2']:.6f} norm_mae={row['normalized']['mae']:.6f} "
            f"state_align_err={state_alignment_err:.3g}"
        )

    summary = {
        "num_samples": len(rows),
        "indices": indices,
        "raw_and_normalized_summary": {
            **_summarize(rows, "raw"),
            **_summarize(rows, "normalized"),
        },
        "latency_s_mean": float(np.mean([r["latency_s"] for r in rows])),
        "state_alignment_max_abs_err_max": float(max(r["state_alignment_max_abs_err"] for r in rows)),
    }

    payload = {
        "checkpoint": str(ckpt_dir),
        "device": args.device,
        "inference_seed": args.inference_seed,
        "denoise_steps": int(OmegaConf.select(cfg, "inference.denoise_steps")),
        "schedule_type": str(OmegaConf.select(cfg, "inference.schedule_type")),
        "compile_enabled": bool(OmegaConf.select(cfg, "optimization.compile.enabled", default=False)),
        "summary": summary,
        "samples": rows,
    }

    print("[open-loop] summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_path = args.output_json
    if out_path is None:
        out_path = str(ckpt_dir / f"robotwin_open_loop_eval_{int(time.time())}.json")
    Path(out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[open-loop] wrote {out_path}")


if __name__ == "__main__":
    main()
