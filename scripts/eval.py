"""
Hydra entry point for OpenWAM evaluation.

Usage:
    python scripts/eval.py eval=robotwin_offline \
        evaluator.env.task_name=adjust_bottle
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
    print("OpenWAM Evaluation — Hydra Config")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    sys.path.insert(0, str(WAM_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))

    # TODO: Phase 2 will implement the evaluation adapter.
    # For now, delegate to the legacy eval_robotwin.py.
    raise NotImplementedError(
        "Hydra eval entry point will be implemented in Phase 2. "
        "Use the legacy script: python examples/wanvideo/wam/eval_robotwin.py"
    )


if __name__ == "__main__":
    main()
