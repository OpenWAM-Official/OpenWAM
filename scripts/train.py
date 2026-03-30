"""
Hydra entry point for OpenWAM training.

Translates the hierarchical Hydra config into the flat argparse Namespace
expected by the existing VideoActionTrainingModule pipeline, then delegates
to the legacy training loop.

Usage:
    # Default config (joint training, vace backbone, robotwin_multitask)
    python scripts/train.py

    # Override from CLI
    python scripts/train.py training=video_only model/backbone=ti2v_5b \
        data.dataset_dir=/data/robotwin training.learning_rate=5e-5

    # Print resolved config without running
    python scripts/train.py --cfg job

    # Multi-run sweep
    python scripts/train.py -m training.learning_rate=1e-4,5e-5,1e-5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAM_DIR = PROJECT_ROOT / "examples" / "wanvideo" / "wam"
THIRD_PARTY = PROJECT_ROOT / "third_party"


def _cfg_to_flat_namespace(cfg: DictConfig) -> argparse.Namespace:
    """Convert hierarchical Hydra config to flat argparse-like Namespace.

    Maps config groups to the exact arg names expected by the legacy
    train_video_action.py code.

    Hydra group structure (after defaults resolution):
        cfg.model.*     — ActionDiT params (from model/action_dit_*.yaml)
        cfg.model.backbone.*  — backbone params (from model/backbone/*.yaml)
        cfg.data.*      — dataset params (from data/*.yaml)
        cfg.training.*  — trainer params (from training/*.yaml)
        cfg.inference.* — inference params (from inference/*.yaml)
        cfg.eval.*      — evaluator params (from eval/*.yaml)
        cfg.project.*   — project-level settings
    """
    t = cfg.training
    d = cfg.data
    m = cfg.model
    b = cfg.model.backbone

    bridge_layers_str = ",".join(str(x) for x in m.bridge_layers)

    args = argparse.Namespace(
        # Model loading
        model_paths=t.model_paths,
        model_id_with_origin_paths=t.model_id_with_origin_paths,
        tokenizer_path=t.tokenizer_path,
        audio_processor_path=None,
        extra_inputs="vace_video,vace_reference_image,action_trajectory",
        fp8_models=t.fp8_models,
        offload_models=t.offload_models,
        task="sft",

        # Training
        trainable_models=",".join(t.trainable_models) if t.trainable_models else None,
        learning_rate=float(t.learning_rate),
        weight_decay=float(t.weight_decay),
        num_epochs=int(t.num_epochs),
        max_steps=t.max_steps,
        batch_size=int(t.batch_size),
        gradient_accumulation_steps=int(t.gradient_accumulation_steps),
        find_unused_parameters=bool(t.find_unused_parameters),

        # Gradient
        use_gradient_checkpointing=bool(t.use_gradient_checkpointing),
        use_gradient_checkpointing_offload=bool(t.use_gradient_checkpointing_offload),

        # LoRA
        lora_base_model=t.lora_base_model,
        lora_target_modules=t.lora_target_modules,
        lora_rank=int(t.lora_rank),
        lora_checkpoint=t.lora_checkpoint,
        preset_lora_path=t.preset_lora_path,
        preset_lora_model=t.preset_lora_model,

        # Output
        output_path=t.output_path,
        remove_prefix_in_ckpt=t.remove_prefix_in_ckpt,
        save_steps=t.save_steps,
        rolling_save_steps=t.rolling_save_steps,
        keep_last_k_ckpts=int(t.keep_last_k_ckpts),
        dataset_num_workers=int(t.dataset_num_workers),

        # W&B
        wandb_project=cfg.project.wandb.project,
        wandb_run_name=cfg.project.wandb.run_name,

        # Timestep boundary
        max_timestep_boundary=float(t.max_timestep_boundary),
        min_timestep_boundary=float(t.min_timestep_boundary),
        initialize_model_on_cpu=bool(t.initialize_model_on_cpu),

        # ActionDiT
        action_dim=int(m.action_dim),
        action_dit_dim=int(m.dim),
        action_dit_ffn_dim=int(m.ffn_dim),
        action_dit_num_heads=int(m.num_heads),
        action_dit_num_layers=int(m.num_layers),
        action_dit_bridge_layers=bridge_layers_str,
        video_dim=int(b.video_dim),
        lambda_video=float(t.lambda_video),
        lambda_action=float(t.lambda_action),
        bridge_type=m.bridge_type,
        action_lr=float(t.action_lr) if t.action_lr is not None else None,
        action_stats_path=d.action_stats_path,

        # Backbone
        backbone=b.name,

        # Data (common)
        height=int(d.height),
        width=int(d.width),
        num_frames=int(d.num_frames),
        target_camera=d.target_camera,
        window_stride=int(d.window_stride),
        multiview=bool(d.multiview),
        robot=d.robot,
        variant=d.variant,
        val_ratio=float(d.val_ratio),
        dataset_repeat=int(d.repeat),
        dataset_base_path="",
        dataset_metadata_path=None,
        data_file_keys="image,video",

        # Validation
        val_steps=t.val_steps,
        video_log_steps=t.video_log_steps,
        max_val_samples=int(d.get("max_val_samples", 500)),
    )

    # Dataset-type specific fields
    dtype = d.type
    args.dataset_type = dtype

    if dtype == "robotwin_multitask":
        args.dataset_dir = d.dataset_dir
        args.hdf5_data_root = None
        args.task_name = None
        args.train_tasks = ",".join(d.train_tasks) if d.train_tasks else None
        args.holdout_tasks = ",".join(d.holdout_tasks) if d.holdout_tasks else None
        args.val_variant = d.val_variant
    else:
        args.dataset_dir = None
        args.hdf5_data_root = d.hdf5_data_root
        args.task_name = d.get("task_name", None)
        args.train_tasks = None
        args.holdout_tasks = None
        args.val_variant = None

    return args


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

    args = _cfg_to_flat_namespace(cfg)

    import accelerate
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )

    # --- Dataset setup (mirrors train_video_action.py __main__) ---
    _train_tasks = None
    if getattr(args, "train_tasks", None):
        _train_tasks = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
    _holdout_tasks = None
    if getattr(args, "holdout_tasks", None):
        _holdout_tasks = [t.strip() for t in args.holdout_tasks.split(",") if t.strip()]

    if args.dataset_type == "robotwin_multitask":
        from video_action_dataset import (
            MultiTaskRoboTwinDataset, ROBOTWIN_TRAIN_TASKS, ROBOTWIN_ALL_TASKS,
        )
        if not args.dataset_dir:
            raise ValueError("data.dataset_dir is required for robotwin_multitask")
        if _train_tasks is not None:
            train_tasks = _train_tasks
        elif _holdout_tasks is not None:
            train_tasks = sorted(t for t in ROBOTWIN_ALL_TASKS if t not in _holdout_tasks)
        else:
            train_tasks = ROBOTWIN_TRAIN_TASKS
        dataset = MultiTaskRoboTwinDataset(
            dataset_dir=args.dataset_dir,
            robot=args.robot,
            variant=args.variant,
            tasks=train_tasks,
            action_stats_path=args.action_stats_path,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            split="train",
            val_ratio=args.val_ratio,
            repeat=args.dataset_repeat,
            target_camera=args.target_camera,
            window_stride=args.window_stride,
            multiview=args.multiview,
            backbone=args.backbone,
        )
    else:
        from video_action_dataset import RoboTwinDataset
        dataset = RoboTwinDataset(
            data_root=args.hdf5_data_root,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            split="train",
            val_ratio=args.val_ratio,
            repeat=args.dataset_repeat,
            task_name=args.task_name,
            action_stats_path=args.action_stats_path,
            target_camera=args.target_camera,
            window_stride=args.window_stride,
            multiview=args.multiview,
            robot=args.robot,
            variant=args.variant,
            backbone=args.backbone,
        )

    # --- Model setup ---
    import torch
    import numpy as np
    import math
    from train_video_action import VideoActionTrainingModule
    from diffsynth.diffusion import ModelLogger
    from diffsynth.diffusion.runner import launch_training_task

    model = VideoActionTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        action_dim=args.action_dim,
        action_dit_dim=args.action_dit_dim,
        action_dit_ffn_dim=args.action_dit_ffn_dim,
        action_dit_num_heads=args.action_dit_num_heads,
        action_dit_num_layers=args.action_dit_num_layers,
        action_dit_bridge_layers=args.action_dit_bridge_layers,
        video_dim=args.video_dim,
        lambda_video=args.lambda_video,
        lambda_action=args.lambda_action,
        bridge_type=args.bridge_type,
        action_lr=args.action_lr,
        action_stats_path=args.action_stats_path,
    )

    if args.lambda_action > 0:
        stats = dataset.action_stats
        if stats is not None:
            model.action_dit.action_mean.copy_(torch.from_numpy(stats["mean"].astype(np.float32)))
            model.action_dit.action_std.copy_(torch.from_numpy(stats["std"].astype(np.float32)))

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config=vars(args),
        rolling_save_steps=args.rolling_save_steps,
        keep_last_k_ckpts=args.keep_last_k_ckpts,
    )

    # --- Callback system (DynamiCrafter-inspired) ---
    from open_wam.training.callbacks import (
        CallbackRunner, ValidationLossCallback, VideoLogCallback, SetupCallback,
    )

    runner = CallbackRunner()
    runner.add(SetupCallback(output_dir=args.output_path, config_dict=vars(args)))

    if args.val_steps is not None:
        val_datasets = {}
        video_datasets = {}
        _video_log_steps = args.video_log_steps if args.video_log_steps is not None else args.val_steps

        if args.dataset_type == "robotwin_multitask":
            from video_action_dataset import MultiTaskRoboTwinDataset, ROBOTWIN_HOLDOUT_TASKS
            val_variant = args.val_variant or args.variant
            holdout_tasks = _holdout_tasks if _holdout_tasks else ROBOTWIN_HOLDOUT_TASKS
            _mt_val_common = dict(
                action_stats_path=args.action_stats_path,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                split="val",
                repeat=1,
                target_camera=args.target_camera,
                window_stride=args.window_stride,
                multiview=args.multiview,
                backbone=args.backbone,
            )
            val_datasets["val_id"] = MultiTaskRoboTwinDataset(
                dataset_dir=args.dataset_dir,
                robot=args.robot,
                variant=args.variant,
                tasks=train_tasks,
                val_ratio=args.val_ratio,
                num_val_samples=0,
                **_mt_val_common,
            )
            val_datasets["val_ood"] = MultiTaskRoboTwinDataset(
                dataset_dir=args.dataset_dir,
                robot=args.robot,
                variant=val_variant,
                tasks=holdout_tasks,
                val_ratio=1.0,
                num_val_samples=5,
                **_mt_val_common,
            )
            video_datasets = dict(val_datasets)
        else:
            from video_action_dataset import RoboTwinDataset
            val_ds = RoboTwinDataset(
                data_root=args.hdf5_data_root,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                split="val",
                val_ratio=args.val_ratio,
                repeat=1,
                task_name=args.task_name,
                action_stats_path=args.action_stats_path,
                num_val_samples=4,
                target_camera=args.target_camera,
                window_stride=args.window_stride,
                multiview=args.multiview,
                robot=args.robot,
                variant=args.variant,
                backbone=args.backbone,
            )
            val_datasets["val"] = val_ds
            video_datasets["val"] = val_ds

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


if __name__ == "__main__":
    main()
