"""Probe RoboTwin action flow quality at fixed noise levels.

This uses the current training/deploy code path to load a checkpoint and
RoboTwin dataloader sample, then runs one forward pass at chosen action sigmas.
For each sigma it reports:

    x_sigma = (1 - sigma) * action + sigma * noise
    target_velocity = noise - action
    x0_hat = x_sigma - sigma * pred_velocity

If teacher-forcing loss is low but x0_hat is poor at high sigma, the open-loop
gap is in the action flow/sampling regime rather than in state normalization or
RoboTwin adapter wiring.
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

from scripts.robotwin_open_loop_eval import (  # noqa: E402
    EEF_GROUPS,
    _build_dataset_from_checkpoint_cfg,
    _parse_indices,
    _sample_indices,
)


def _parse_sigmas(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _to_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().float())
    return float(x)


def _nearest_training_weight(architecture, sigma: float) -> tuple[int, float, float]:
    scheduler = architecture.action_scheduler
    sigmas = scheduler.sigmas.to(dtype=torch.float32)
    idx = int(torch.argmin(torch.abs(sigmas - float(sigma))).item())
    weight = scheduler.training_weight(torch.tensor([idx], device=sigmas.device))[0]
    return idx, float(sigmas[idx]), _to_float(weight)


def _metrics(err: torch.Tensor) -> dict[str, float]:
    err_f = err.detach().float()
    out = {
        "mae": _to_float(err_f.abs().mean()),
        "rmse": _to_float(torch.sqrt((err_f**2).mean())),
        "max_abs": _to_float(err_f.abs().max()),
        "first_mae": _to_float(err_f[:, 0].abs().mean()),
        "first_l2": _to_float(torch.linalg.vector_norm(err_f[:, 0], dim=-1).mean()),
    }
    for name, idx in EEF_GROUPS.items():
        if isinstance(idx, slice):
            group_err = err_f[:, :, idx]
        else:
            group_err = err_f[:, :, idx]
        out[f"mae_{name}"] = _to_float(group_err.abs().mean())
        out[f"first_mae_{name}"] = _to_float(group_err[:, 0].abs().mean())
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_sigma: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['sigma']:.6g}"
        by_sigma.setdefault(key, []).append(row)

    summary = {}
    for key, vals in by_sigma.items():
        item = {}
        for metric in (
            "velocity_mse",
            "velocity_mae",
            "x0_norm_mae",
            "x0_raw_mae",
            "x0_norm_first_l2",
            "x0_raw_first_l2",
            "train_weight",
        ):
            arr = np.asarray([v[metric] for v in vals], dtype=np.float64)
            item[f"{metric}.mean"] = float(arr.mean())
            item[f"{metric}.median"] = float(np.median(arr))
            item[f"{metric}.max"] = float(arr.max())
        summary[key] = item
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe fixed-sigma action flow quality on RoboTwin samples.")
    parser.add_argument(
        "--ckpt-dir",
        default="/path/to/openwam_checkpoints/robotwin_dual_system_joint_self_attention",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--noise-seed", type=int, default=123)
    parser.add_argument("--sigmas", default="1.0,0.95,0.8,0.5,0.2,0.05")
    parser.add_argument(
        "--video-sigma-mode",
        choices=("clean", "matched"),
        default="clean",
        help="clean = condition on clean GT video latents; matched = add video noise at the same sigma.",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_cfg = OmegaConf.load(str(ckpt_dir / "config.yaml"))
    sigmas = _parse_sigmas(args.sigmas)

    print(f"[field] checkpoint={ckpt_dir}")
    print(
        "[field] checkpoint cfg: "
        f"action_mode={OmegaConf.select(ckpt_cfg, 'dataloader.action_mode')} "
        f"normalize_mode={OmegaConf.select(ckpt_cfg, 'dataloader.normalize_mode')} "
        f"num_frames={OmegaConf.select(ckpt_cfg, 'dataloader.num_frames')} "
        f"video_stride={OmegaConf.select(ckpt_cfg, 'dataloader.video_stride')}"
    )

    dataset = _build_dataset_from_checkpoint_cfg(
        ckpt_dir,
        ckpt_cfg,
        split=args.split,
        dataset_dir=args.dataset_dir,
        task_name=args.task_name,
        variant=args.variant,
    )
    indices = _sample_indices(len(dataset), args.num_samples, args.sample_seed, _parse_indices(args.indices))
    print(f"[field] dataset_len={len(dataset)} selected_indices={indices}")

    from openwam.deploy.model_loader import load_from_checkpoint_dir

    _, architecture = load_from_checkpoint_dir(str(ckpt_dir), device=args.device)
    architecture.eval()
    architecture.set_training_runtime(
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    )
    architecture.init_training_schedulers(1000)

    rows: list[dict[str, Any]] = []
    for sample_i, dataset_idx in enumerate(indices):
        sample = dataset[dataset_idx]
        with torch.no_grad():
            inputs = architecture.prepare_inputs([sample])

        actions = inputs["actions"].to(device=architecture.device, dtype=architecture.dtype)
        proprio_state = inputs.get("proprio_state")
        base_latents = inputs["input_latents"].to(device=architecture.device, dtype=architecture.dtype)

        for repeat_i in range(args.noise_repeats):
            gen = torch.Generator(device=architecture.device).manual_seed(args.noise_seed + sample_i * 1000 + repeat_i)
            action_noise = torch.randn(actions.shape, dtype=actions.dtype, device=actions.device, generator=gen)
            video_noise = torch.randn(
                base_latents.shape, dtype=base_latents.dtype, device=base_latents.device, generator=gen
            )

            for sigma in sigmas:
                sigma_f = float(sigma)
                sigma_t = torch.tensor([sigma_f * 1000.0], dtype=architecture.dtype, device=architecture.device)
                noisy_actions = (1.0 - sigma_f) * actions + sigma_f * action_noise

                if args.video_sigma_mode == "matched":
                    video_latents = (1.0 - sigma_f) * base_latents + sigma_f * video_noise
                    ref_latents = inputs.get("first_frame_latents")
                    if ref_latents is not None:
                        ref_latents = ref_latents.to(device=architecture.device, dtype=architecture.dtype)
                        video_latents = video_latents.clone()
                        video_latents[:, :, : ref_latents.shape[2]] = ref_latents
                    video_timestep = sigma_t
                else:
                    video_latents = base_latents
                    video_timestep = torch.zeros_like(sigma_t)

                forward_inputs = dict(inputs)
                forward_inputs["latents"] = video_latents
                forward_inputs.pop("actions", None)
                forward_inputs.pop("action_is_pad", None)
                forward_inputs.pop("video_is_pad", None)
                forward_inputs.pop("proprio_state", None)
                forward_inputs.pop("use_gradient_checkpointing", None)
                forward_inputs.pop("use_gradient_checkpointing_offload", None)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    _, pred_velocity = architecture.forward(
                        noisy_actions,
                        sigma_t,
                        proprio_state=proprio_state,
                        use_gradient_checkpointing=False,
                        use_gradient_checkpointing_offload=False,
                        **forward_inputs,
                        timestep=video_timestep,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency_s = time.time() - t0

                target_velocity = action_noise - actions
                velocity_err = pred_velocity - target_velocity
                x0_hat_norm = noisy_actions - sigma_f * pred_velocity
                x0_err_norm = x0_hat_norm - actions

                x0_hat_raw = architecture.action_normalizer.unnormalize(
                    x0_hat_norm.squeeze(0).detach().float().cpu().numpy()
                )
                gt_raw = dataset.denormalize_action(actions.squeeze(0).detach().float().cpu().numpy())
                x0_err_raw = torch.from_numpy((x0_hat_raw - gt_raw).astype(np.float32)).unsqueeze(0)

                train_idx, nearest_sigma, train_weight = _nearest_training_weight(architecture, sigma_f)
                norm_metrics = _metrics(x0_err_norm)
                raw_metrics = _metrics(x0_err_raw)
                row = {
                    "sample_i": sample_i,
                    "dataset_idx": int(dataset_idx),
                    "task_name": sample.get("task_name", ""),
                    "start_frame": int(sample.get("start_frame", -1)),
                    "repeat_i": repeat_i,
                    "sigma": sigma_f,
                    "nearest_train_timestep_id": train_idx,
                    "nearest_train_sigma": nearest_sigma,
                    "train_weight": train_weight,
                    "latency_s": float(latency_s),
                    "velocity_mse": _to_float((velocity_err.detach().float() ** 2).mean()),
                    "velocity_mae": _to_float(velocity_err.detach().float().abs().mean()),
                    "x0_norm_mae": norm_metrics["mae"],
                    "x0_norm_first_l2": norm_metrics["first_l2"],
                    "x0_raw_mae": raw_metrics["mae"],
                    "x0_raw_first_l2": raw_metrics["first_l2"],
                    "x0_norm": norm_metrics,
                    "x0_raw": raw_metrics,
                }
                rows.append(row)
                print(
                    "[field] "
                    f"idx={dataset_idx} repeat={repeat_i} sigma={sigma_f:.3f} "
                    f"w={train_weight:.4f} vel_mse={row['velocity_mse']:.6f} "
                    f"x0_norm_mae={row['x0_norm_mae']:.6f} "
                    f"x0_raw_mae={row['x0_raw_mae']:.6f} latency={latency_s:.2f}s"
                )

    payload = {
        "checkpoint": str(ckpt_dir),
        "device": args.device,
        "indices": indices,
        "sigmas": sigmas,
        "video_sigma_mode": args.video_sigma_mode,
        "noise_repeats": args.noise_repeats,
        "summary_by_sigma": _summarize(rows),
        "samples": rows,
    }

    print("[field] summary_by_sigma")
    print(json.dumps(payload["summary_by_sigma"], indent=2, ensure_ascii=False))

    out_path = args.output_json
    if out_path is None:
        out_path = str(ckpt_dir / f"robotwin_action_field_probe_{int(time.time())}.json")
    Path(out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[field] wrote {out_path}")


if __name__ == "__main__":
    main()
