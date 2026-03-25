"""
Hydra entry point for OpenWAM inference.

Usage:
    python scripts/infer.py inference=action_only \
        inference.seed=42
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAM_DIR = PROJECT_ROOT / "examples" / "wanvideo" / "wam"


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("OpenWAM Inference — Hydra Config")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    sys.path.insert(0, str(WAM_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

    # TODO: Phase 2 will implement the inference adapter.
    # For now, delegate to joint_inference.py.
    raise NotImplementedError(
        "Hydra inference entry point will be implemented in Phase 2. "
        "Use the legacy module: from joint_inference import generate_video_and_actions"
    )


if __name__ == "__main__":
    main()
