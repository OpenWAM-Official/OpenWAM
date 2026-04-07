"""
Hydra entry point for OpenWAM inference.

Usage:
    # Sync schedule inference
    python scripts/infer.py inference=sync \
        inference.prompt="robot picks up the bottle" \
        inference.seed=42

    # Action-only mode (with pre-encoded video latents)
    python scripts/infer.py inference=action_only \
        inference.seed=42

    # Print resolved config without running
    python scripts/infer.py --cfg job
"""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("OpenWAM Inference — Hydra Config")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(THIRD_PARTY))

    import numpy as np

    inf_cfg = cfg.inference

    # Load models via package-native loader
    device = getattr(inf_cfg, "device", "cuda")
    from open_wam.inference import JointInferenceEngine, load_wam_models

    pipe, action_dit = load_wam_models(cfg, device=device)

    # Create inference engine
    engine = JointInferenceEngine(cfg=cfg, pipeline=pipe, action_dit=action_dit)

    # Build conditions from config
    conditions = {
        "prompt": getattr(inf_cfg, "prompt", ""),
        "negative_prompt": getattr(inf_cfg, "negative_prompt", ""),
        "seed": getattr(inf_cfg, "seed", 42),
        "num_frames": getattr(inf_cfg, "num_frames", cfg.data.num_frames),
        "height": getattr(inf_cfg, "height", cfg.data.height),
        "width": getattr(inf_cfg, "width", cfg.data.width),
    }

    # Load reference image if provided
    ref_image_path = getattr(inf_cfg, "reference_image_path", None)
    if ref_image_path is not None:
        from PIL import Image

        ref_img = Image.open(ref_image_path).convert("RGB")
        conditions["vace_reference_image"] = [ref_img]

    # Load context video if provided
    context_video_path = getattr(inf_cfg, "context_video_path", None)
    if context_video_path is not None:
        import imageio
        from PIL import Image

        reader = imageio.get_reader(context_video_path)
        frames = [Image.fromarray(f) for f in reader]
        reader.close()
        conditions["vace_video"] = frames

    # Generate
    print("\nGenerating...")
    result = engine.generate(conditions)

    video_frames = result["video"]
    actions = result["actions"]

    print(f"Generated {len(video_frames)} video frames")
    print(f"Generated actions shape: {actions.shape}")

    # Save outputs
    output_dir = getattr(inf_cfg, "output_dir", "inference_output")
    os.makedirs(output_dir, exist_ok=True)

    # Save video
    if video_frames:
        import imageio

        video_path = os.path.join(output_dir, "generated_video.mp4")
        imageio.mimsave(video_path, [np.array(f) for f in video_frames], fps=15)
        print(f"Video saved to {video_path}")

    # Save actions
    actions_path = os.path.join(output_dir, "actions.npz")
    np.savez(actions_path, actions=actions)
    print(f"Actions saved to {actions_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
