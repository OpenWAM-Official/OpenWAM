"""Hydra entry point for the supported OpenWAM training workflow.

Usage:
    # Default: native trainer (no legacy dependencies)
    python scripts/train.py training.trainer=native

    # Legacy trainer (backward-compatible, deprecated)
    python scripts/train.py training.trainer=legacy

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
WAM_DIR = PROJECT_ROOT / "examples" / "wanvideo" / "wam"
THIRD_PARTY = PROJECT_ROOT / "third_party"


def _train_native(cfg: DictConfig) -> None:
    """Package-native training path — no legacy dependencies."""
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

    # Build dataset (still uses cfg_to_flat_namespace for dataset routing)
    args = cfg_to_flat_namespace(cfg)
    train_tasks, holdout_tasks = parse_task_overrides(args)
    dataset = build_training_dataset(args, train_tasks=train_tasks, holdout_tasks=holdout_tasks)

    # Build native trainer
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


def _train_legacy(cfg: DictConfig) -> None:
    """Legacy training path — uses VideoActionTrainingModule + third_party/diffsynth."""
    from open_wam.training.config_tracking import (
        build_run_metadata,
        get_git_commit,
        make_run_id,
    )
    from open_wam.training.legacy import ModelLogger, launch_training_task
    from open_wam.training.runtime import (
        build_training_dataset,
        build_training_module,
        build_validation_datasets,
        cfg_to_flat_namespace,
        parse_task_overrides,
    )

    args = cfg_to_flat_namespace(cfg)
    resolved_config_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    resolved_config_dict = OmegaConf.to_container(cfg, resolve=True)
    run_id = make_run_id()
    hydra_output_dir = None
    if HydraConfig.initialized():
        hydra_output_dir = HydraConfig.get().runtime.output_dir
    run_metadata = build_run_metadata(
        run_id=run_id,
        output_dir=args.output_path,
        hydra_output_dir=hydra_output_dir,
        git_commit=get_git_commit(PROJECT_ROOT),
    )

    import accelerate
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )

    train_tasks, holdout_tasks = parse_task_overrides(args)
    dataset = build_training_dataset(args, train_tasks=train_tasks, holdout_tasks=holdout_tasks)
    model = build_training_module(args, accelerator=accelerator, dataset=dataset)

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config=vars(args),
        rolling_save_steps=args.rolling_save_steps,
        keep_last_k_ckpts=args.keep_last_k_ckpts,
    )

    # --- Callback system ---
    from open_wam.training.callbacks import (
        CallbackRunner, ValidationLossCallback, VideoLogCallback, SetupCallback,
    )

    runner = CallbackRunner()
    runner.add(
        SetupCallback(
            output_dir=args.output_path,
            config_dict=vars(args),
            resolved_config_yaml=resolved_config_yaml,
            resolved_config_dict=resolved_config_dict,
            flat_args_dict=vars(args),
            run_metadata=run_metadata,
        )
    )

    if args.val_steps is not None:
        _video_log_steps = args.video_log_steps if args.video_log_steps is not None else args.val_steps
        val_datasets, video_datasets = build_validation_datasets(
            args,
            train_tasks=train_tasks,
            holdout_tasks=holdout_tasks,
        )

        runner.add(ValidationLossCallback(
            model=model,
            datasets=val_datasets,
            wandb_run=model_logger.wandb_run,
            every_n_steps=args.val_steps,
            max_samples=args.max_val_samples,
        ))
        runner.add(VideoLogCallback(
            model=model,
            datasets=video_datasets,
            wandb_run=model_logger.wandb_run,
            every_n_steps=_video_log_steps,
        ))

    val_callback, callback_interval = runner.as_legacy_val_callback()

    launch_training_task(
        accelerator, dataset, model, model_logger, args=args,
        val_callback=val_callback, val_steps=callback_interval,
    )


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("OpenWAM Training — Hydra Config")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Ensure WAM example scripts and third-party packages are importable
    sys.path.insert(0, str(WAM_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(THIRD_PARTY))

    trainer_type = getattr(cfg.training, "trainer", "legacy")

    if trainer_type == "native":
        _train_native(cfg)
    else:
        _train_legacy(cfg)


if __name__ == "__main__":
    main()
