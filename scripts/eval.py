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

import os
import sys
import json
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
    from open_wam.inference import JointInferenceEngine, load_wam_models

    pipe, action_dit = load_wam_models(cfg, device=device)

    # Create inference engine
    engine = JointInferenceEngine(cfg=cfg, pipeline=pipe, action_dit=action_dit)

    eval_type = eval_cfg.type

    if eval_type == "offline":
        from open_wam.evaluation import RoboTwinOfflineEvaluator
        from open_wam.data.robotwin import RoboTwinActionDataset, MultiTaskRoboTwinActionDataset

        # Build dataset
        d = cfg.data
        if d.type == "robotwin_multitask":
            dataset = MultiTaskRoboTwinActionDataset(
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
            dataset = RoboTwinActionDataset(
                data_root=d.hdf5_data_root,
                num_frames=int(d.num_frames),
                height=int(d.height),
                width=int(d.width),
                split="val",
                val_ratio=float(d.val_ratio),
                multiview=bool(d.multiview),
                backbone=cfg.model.backbone.name,
            )

        evaluator = RoboTwinOfflineEvaluator(cfg=cfg, engine=engine)
        results = evaluator.evaluate(dataset)

        print("\n" + "=" * 60)
        print("Evaluation Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print("=" * 60)

        output_dir = getattr(eval_cfg, "output_dir", "eval_results")
        result_path = _save_results(results, output_dir)
        print(f"Results saved to {result_path}")

    elif eval_type == "online":
        from open_wam.evaluation import RoboTwinOnlineEvaluator
        from open_wam.evaluation.envs.robotwin import RoboTwinEnvAdapter

        task_name = getattr(eval_cfg, "task_name", "adjust_bottle")
        robot = cfg.data.robot
        env = RoboTwinEnvAdapter(task_name=task_name, robot=robot)

        evaluator = RoboTwinOnlineEvaluator(cfg=cfg, engine=engine)
        results = evaluator.evaluate(env)

        print("\n" + "=" * 60)
        print("Online Evaluation Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print("=" * 60)

        output_dir = getattr(eval_cfg, "output_dir", "eval_results")
        result_path = _save_results(results, output_dir)
        print(f"Results saved to {result_path}")

    elif eval_type == "simpler_env":
        from open_wam.evaluation import SimplerEnvEvaluator

        evaluator = SimplerEnvEvaluator(cfg=cfg, engine=engine)
        results = evaluator.evaluate()

        print("\n" + "=" * 60)
        print("SimplerEnv Evaluation Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print("=" * 60)

        output_dir = getattr(eval_cfg, "output_dir", "eval_results/simpler_env")
        result_path = _save_results(results, output_dir)
        print(f"Results saved to {result_path}")

    elif eval_type == "libero":
        from open_wam.evaluation import LIBEROEvaluator

        evaluator = LIBEROEvaluator(cfg=cfg, engine=engine)
        results = evaluator.evaluate()

        print("\n" + "=" * 60)
        print("LIBERO Evaluation Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
        print("=" * 60)

        output_dir = getattr(eval_cfg, "output_dir", "eval_results/libero")
        result_path = _save_results(results, output_dir)
        print(f"Results saved to {result_path}")

    elif eval_type == "robocasa":
        from open_wam.evaluation import RoboCasaEvaluator

        evaluator = RoboCasaEvaluator(cfg=cfg, engine=engine)
        results = evaluator.evaluate()

        print("\n" + "=" * 60)
        print("RoboCasa Evaluation Results:")
        for k, v in results.items():
            if k != "per_task":
                print(f"  {k}: {v}")
        print("=" * 60)

        output_dir = getattr(eval_cfg, "output_dir", "eval_results/robocasa")
        result_path = _save_results(results, output_dir)
        print(f"Results saved to {result_path}")

    elif eval_type == "calvin":
        from open_wam.evaluation import CalvinEvaluator

        evaluator = CalvinEvaluator(cfg=cfg, engine=engine)
        results = evaluator.evaluate()

        print("\n" + "=" * 60)
        print("Calvin Evaluation Results:")
        for k, v in results.items():
            if k not in ("per_length_success", "completion_distribution"):
                print(f"  {k}: {v}")
        if "per_length_success" in results:
            print("  Per-length success rates:")
            for k, v in results["per_length_success"].items():
                print(f"    {k}: {v:.3f}")
        print("=" * 60)

        output_dir = getattr(eval_cfg, "output_dir", "eval_results/calvin")
        result_path = _save_results(results, output_dir)
        print(f"Results saved to {result_path}")

    elif eval_type == "behavior":
        from open_wam.evaluation import BehaviorEvaluator

        evaluator = BehaviorEvaluator(cfg=cfg, engine=engine)
        results = evaluator.evaluate()

        print("\n" + "=" * 60)
        print("BEHAVIOR-1K Evaluation Results:")
        for k, v in results.items():
            if k not in ("per_task", "per_category"):
                print(f"  {k}: {v}")
        if "per_category" in results:
            print("  Per-category success rates:")
            for cat, cat_data in results["per_category"].items():
                print(f"    {cat}: {cat_data['success_rate']:.3f}")
        print("=" * 60)

        output_dir = getattr(eval_cfg, "output_dir", "eval_results/behavior")
        result_path = _save_results(results, output_dir)
        print(f"Results saved to {result_path}")

    else:
        raise ValueError(
            f"Unknown eval type: {eval_type}. "
            "Use 'offline', 'online', 'simpler_env', 'libero', "
            "'robocasa', 'calvin', or 'behavior'."
        )


if __name__ == "__main__":
    main()
