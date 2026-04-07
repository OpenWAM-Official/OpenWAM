"""
Hydra entry point for OpenWAM evaluation.

Usage:
    # Offline evaluation (default)
    python scripts/eval.py eval=robotwin_offline \
        eval.ckpt_path=/path/to/checkpoint.safetensors \
        eval.task_name=adjust_bottle

    # Online evaluation
    python scripts/eval.py eval=robotwin_online \
        eval.ckpt_path=/path/to/checkpoint.safetensors

    # Print resolved config without running
    python scripts/eval.py --cfg job
"""

import json
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"


def _save_results(results: dict, output_dir: str) -> str:
    """Save evaluation results to disk and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    return result_path


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("OpenWAM Evaluation — Hydra Config")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(THIRD_PARTY))

    eval_cfg = cfg.eval

    # Load models
    device = getattr(eval_cfg, "device", "cuda")
    from open_wam.evaluation.registry import build_evaluator
    from open_wam.inference import JointInferenceEngine, load_wam_models

    pipe, action_dit = load_wam_models(cfg, device=device)

    # Create inference engine
    engine = JointInferenceEngine(cfg=cfg, pipeline=pipe, action_dit=action_dit)

    eval_type = eval_cfg.type
    evaluator = build_evaluator(eval_type, cfg, engine)

    # Build eval target (dataset or env) depending on evaluator type
    eval_target = None
    if eval_type == "offline":
        from open_wam.data.robotwin import MultiTaskRoboTwinActionDataset, RoboTwinActionDataset

        d = cfg.data
        if d.type == "robotwin_multitask":
            eval_target = MultiTaskRoboTwinActionDataset(
                dataset_dir=d.dataset_dir,
                robot=d.robot,
                variant=d.variant,
                num_frames=int(d.num_frames),
                height=int(d.height),
                width=int(d.width),
                split="val",
                val_ratio=float(d.val_ratio),
                multiview=bool(d.multiview),
                backbone=cfg.model.backbone.name,
            )
        else:
            eval_target = RoboTwinActionDataset(
                data_root=d.hdf5_data_root,
                num_frames=int(d.num_frames),
                height=int(d.height),
                width=int(d.width),
                split="val",
                val_ratio=float(d.val_ratio),
                multiview=bool(d.multiview),
                backbone=cfg.model.backbone.name,
            )
    elif eval_type == "online":
        from open_wam.evaluation.envs.robotwin import RoboTwinEnvAdapter

        task_name = getattr(eval_cfg, "task_name", "adjust_bottle")
        eval_target = RoboTwinEnvAdapter(task_name=task_name, robot=cfg.data.robot)

    # Run evaluation
    if eval_target is not None:
        results = evaluator.evaluate(eval_target)
    else:
        results = evaluator.evaluate()

    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results ({eval_type}):")
    skip_keys = {"per_task", "per_category", "per_length_success", "completion_distribution"}
    for k, v in results.items():
        if k not in skip_keys:
            print(f"  {k}: {v}")
    if "per_length_success" in results:
        print("  Per-length success rates:")
        for k, v in results["per_length_success"].items():
            print(f"    {k}: {v:.3f}")
    if "per_category" in results:
        print("  Per-category success rates:")
        for cat, cat_data in results["per_category"].items():
            print(f"    {cat}: {cat_data['success_rate']:.3f}")
    print("=" * 60)

    output_dir = getattr(eval_cfg, "output_dir", f"eval_results/{eval_type}")
    result_path = _save_results(results, output_dir)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
