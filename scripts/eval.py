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
WAM_DIR = PROJECT_ROOT / "examples" / "wanvideo" / "wam"


def _load_models(cfg, device="cuda"):
    """Load pipeline and ActionDiT from config."""
    import torch

    sys.path.insert(0, str(WAM_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.models.action_dit import ActionDiT

    eval_cfg = cfg.eval
    m = cfg.model
    b = cfg.model.backbone

    # Model paths
    model_paths = getattr(eval_cfg, "model_paths", None)
    tokenizer_path = getattr(eval_cfg, "tokenizer_path", None)

    if model_paths is None:
        model_paths = [
            "models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
            "models/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth",
        ]
    if isinstance(model_paths, str):
        model_paths = json.loads(model_paths)
    if tokenizer_path is None:
        tokenizer_path = "models/Wan-AI/Wan2.1-VACE-1.3B/google/umt5-xxl"

    model_configs = [ModelConfig(p) for p in model_paths]
    tokenizer_config = ModelConfig(tokenizer_path)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    bridge_layers = tuple(int(x) for x in m.bridge_layers)
    action_dit = ActionDiT(
        action_dim=int(m.action_dim),
        dim=int(m.dim),
        ffn_dim=int(m.ffn_dim),
        num_heads=int(m.num_heads),
        num_layers=int(m.num_layers),
        video_dim=int(b.video_dim),
        bridge_layers=bridge_layers,
        bridge_type=m.bridge_type,
    ).to(dtype=torch.bfloat16, device=device)

    # Load checkpoint
    ckpt_path = eval_cfg.ckpt_path
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    action_keys = {k: v for k, v in state_dict.items() if k.startswith("action_dit.")}
    if action_keys:
        cleaned = {k.removeprefix("action_dit."): v for k, v in action_keys.items()}
        action_dit.load_state_dict(cleaned, strict=False)
        print(f"Loaded ActionDiT from {ckpt_path} ({len(action_keys)} keys)")

    # Load VACE weights if present
    candidate_keys = {k: v for k, v in state_dict.items() if not k.startswith("action_dit.")}
    if candidate_keys and hasattr(pipe, "vace"):
        vace_expected = set(pipe.vace.state_dict().keys())
        vace_keys = {k: v for k, v in candidate_keys.items() if k in vace_expected}
        if vace_keys:
            pipe.vace.load_state_dict(vace_keys, strict=False)
            print(f"Loaded VACE weights: {len(vace_keys)}/{len(vace_expected)} keys")

    action_dit.eval()
    return pipe, action_dit


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("OpenWAM Evaluation — Hydra Config")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    sys.path.insert(0, str(WAM_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

    eval_cfg = cfg.eval

    # Load models
    device = getattr(eval_cfg, "device", "cuda")
    pipe, action_dit = _load_models(cfg, device=device)

    # Create inference engine
    from open_wam.inference import JointInferenceEngine
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

        # Save results
        output_dir = getattr(eval_cfg, "output_dir", "eval_results")
        os.makedirs(output_dir, exist_ok=True)
        import json
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_dir}/results.json")

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

    else:
        raise ValueError(f"Unknown eval type: {eval_type}. Use 'offline' or 'online'.")


if __name__ == "__main__":
    main()
