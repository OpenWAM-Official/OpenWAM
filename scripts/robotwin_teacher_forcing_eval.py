"""Teacher-forcing RoboTwin loss evaluation for a deployed checkpoint.

This script measures the same denoising objective used during training by
calling ``architecture.prepare_inputs()`` and ``architecture.compute_loss()``
on windows sampled from the current RoboTwin dataloader. It is intentionally
separate from open-loop sampling: low teacher-forcing loss with high open-loop
MAE points at a sampling/deploy-generation gap, while high teacher-forcing loss
points at checkpoint/data/preprocessing mismatch.
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

from scripts.robotwin_eval_utils import (  # noqa: E402
    build_dataset_from_checkpoint_cfg,
    parse_indices,
    sample_indices,
)


def _float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def _summarize(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint teacher-forcing loss on RoboTwin samples.")
    parser.add_argument(
        "--ckpt-dir",
        default="/path/to/openwam_checkpoints/robotwin_dual_system_joint_self_attention",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-dir", default=None, help="Override checkpoint dataloader.dataset_dir.")
    parser.add_argument("--task-name", default=None, help="Restrict sampling to one RoboTwin task.")
    parser.add_argument("--variant", default=None, help="Override RoboTwin variant, e.g. clean_50.")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--indices", default=None, help="Comma-separated dataset indices. Overrides --num-samples.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--loss-repeats", type=int, default=3, help="Different noise/timestep seeds per sample.")
    parser.add_argument("--loss-seed", type=int, default=1234)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--lambda-video", type=float, default=0.0)
    parser.add_argument("--lambda-action", type=float, default=1.0)
    parser.add_argument("--train-mode", action="store_true", help="Use architecture.train() instead of eval().")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_cfg = OmegaConf.load(str(ckpt_dir / "config.yaml"))

    print(f"[teacher] checkpoint={ckpt_dir}")
    print(
        "[teacher] checkpoint cfg: "
        f"action_mode={OmegaConf.select(ckpt_cfg, 'dataloader.action_mode')} "
        f"normalize_mode={OmegaConf.select(ckpt_cfg, 'dataloader.normalize_mode')} "
        f"num_frames={OmegaConf.select(ckpt_cfg, 'dataloader.num_frames')} "
        f"canvas={OmegaConf.select(ckpt_cfg, 'dataloader.height')}x{OmegaConf.select(ckpt_cfg, 'dataloader.width')}"
    )

    print("[teacher] building RoboTwin dataset from checkpoint config...")
    dataset = build_dataset_from_checkpoint_cfg(
        ckpt_dir,
        ckpt_cfg,
        split=args.split,
        dataset_dir=args.dataset_dir,
        task_name=args.task_name,
        variant=args.variant,
    )
    indices = sample_indices(len(dataset), args.num_samples, args.sample_seed, parse_indices(args.indices))
    print(f"[teacher] dataset_len={len(dataset)} selected_indices={indices}")

    print("[teacher] loading architecture...")
    from openwam.deploy.model_loader import load_from_checkpoint_dir

    _, architecture = load_from_checkpoint_dir(str(ckpt_dir), device=args.device)
    if args.train_mode:
        architecture.train()
    else:
        architecture.eval()
    architecture.set_training_runtime(
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    )
    architecture.init_training_schedulers(args.num_train_timesteps)
    print(f"[teacher] architecture mode={'train' if architecture.training else 'eval'}")

    rows: list[dict[str, Any]] = []
    for sample_i, dataset_idx in enumerate(indices):
        sample = dataset[dataset_idx]
        per_repeat = []
        t0 = time.time()
        for repeat_i in range(args.loss_repeats):
            seed = int(args.loss_seed + sample_i * args.loss_repeats + repeat_i)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize()
            with torch.no_grad():
                inputs = architecture.prepare_inputs([sample])
                out = architecture.compute_loss(
                    **inputs,
                    lambda_video=args.lambda_video,
                    lambda_action=args.lambda_action,
                    current_step=0,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            per_repeat.append(
                {
                    "seed": seed,
                    "loss": _float(out["loss"]),
                    "loss_action": _float(out["loss_action"]),
                    "loss_video": _float(out["loss_video"]),
                }
            )
        latency_s = time.time() - t0
        action_vals = [x["loss_action"] for x in per_repeat]
        row = {
            "sample_i": sample_i,
            "dataset_idx": int(dataset_idx),
            "task_name": sample.get("task_name", ""),
            "episode_path": sample.get("episode_path", ""),
            "start_frame": int(sample.get("start_frame", -1)),
            "latency_s": float(latency_s),
            "repeats": per_repeat,
            "loss_action_mean": float(np.mean(action_vals)),
            "loss_action_min": float(np.min(action_vals)),
            "loss_action_max": float(np.max(action_vals)),
        }
        rows.append(row)
        print(
            "[teacher] "
            f"{sample_i + 1}/{len(indices)} idx={dataset_idx} task={row['task_name']!r} "
            f"start={row['start_frame']} repeats={args.loss_repeats} latency={latency_s:.2f}s "
            f"loss_action_mean={row['loss_action_mean']:.8f} "
            f"range=[{row['loss_action_min']:.8f}, {row['loss_action_max']:.8f}]"
        )

    all_action = [x["loss_action"] for row in rows for x in row["repeats"]]
    all_total = [x["loss"] for row in rows for x in row["repeats"]]
    all_video = [x["loss_video"] for row in rows for x in row["repeats"]]
    summary = {
        "num_samples": len(rows),
        "loss_repeats": args.loss_repeats,
        "indices": indices,
        "loss_action": _summarize(all_action),
        "loss": _summarize(all_total),
        "loss_video": _summarize(all_video),
        "latency_s_mean_per_sample": float(np.mean([r["latency_s"] for r in rows])),
    }
    payload = {
        "checkpoint": str(ckpt_dir),
        "device": args.device,
        "lambda_video": args.lambda_video,
        "lambda_action": args.lambda_action,
        "train_mode": bool(args.train_mode),
        "summary": summary,
        "samples": rows,
    }

    print("[teacher] summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_path = args.output_json
    if out_path is None:
        out_path = str(ckpt_dir / f"robotwin_teacher_forcing_eval_{int(time.time())}.json")
    Path(out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[teacher] wrote {out_path}")


if __name__ == "__main__":
    main()
