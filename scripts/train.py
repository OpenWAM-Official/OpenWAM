"""Hydra entry point for OpenWAM training.

Usage:
    # Train with default config
    python scripts/train.py

    # Override from CLI
    python scripts/train.py training=video_only model/backbone=ti2v_5b \
        data.dataset_dir=/data/robotwin training.learning_rate=5e-5

    # Print resolved config without running
    python scripts/train.py --cfg job

    # Multi-run sweep
    python scripts/train.py -m training.learning_rate=1e-4,5e-5,1e-5
"""

import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"


def _train(cfg: DictConfig) -> None:
    """Package-native training path."""
    import accelerate

    from open_wam.training.native_trainer import NativeTrainer
    from open_wam.training.runtime import (
        build_training_dataset,
        cfg_to_flat_namespace,
        parse_task_overrides,
    )
    from open_wam.training.config_tracking import (
        build_run_metadata,
        get_git_commit,
        make_run_id,
    )

    t = cfg.training

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=int(t.gradient_accumulation_steps),
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=bool(t.find_unused_parameters)
            )
        ],
    )

    # Build dataset
    args = cfg_to_flat_namespace(cfg)
    train_tasks, holdout_tasks = parse_task_overrides(args)
    dataset = build_training_dataset(args, train_tasks=train_tasks, holdout_tasks=holdout_tasks)

    # Build trainer
    trainer = NativeTrainer(cfg, accelerator=accelerator, dataset=dataset)

    # Save config artifacts
    run_id = make_run_id()
    hydra_output_dir = None
    if HydraConfig.initialized():
        hydra_output_dir = HydraConfig.get().runtime.output_dir
    run_metadata = build_run_metadata(
        run_id=run_id,
        output_dir=t.output_path,
        hydra_output_dir=hydra_output_dir,
        git_commit=get_git_commit(PROJECT_ROOT),
    )

    # Run training
    trainer.train()


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("OpenWAM Training — Hydra Config")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(THIRD_PARTY))

    _train(cfg)


if __name__ == "__main__":
    main()
