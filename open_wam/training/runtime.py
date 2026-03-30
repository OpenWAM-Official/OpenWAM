"""Package-native builders for the supported OpenWAM training path."""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from open_wam.data.robotwin import (
    MultiTaskRoboTwinActionDataset,
    RoboTwinActionDataset,
    ROBOTWIN_ALL_TASKS,
    ROBOTWIN_HOLDOUT_TASKS,
    ROBOTWIN_TRAIN_TASKS,
)
from open_wam.training.legacy import VideoActionTrainingModule


def cfg_to_flat_namespace(cfg: DictConfig) -> argparse.Namespace:
    """Convert Hydra config groups to the legacy training module arguments."""
    t = cfg.training
    d = cfg.data
    m = cfg.model
    b = cfg.model.backbone

    bridge_layers_str = ",".join(str(x) for x in m.bridge_layers)

    args = argparse.Namespace(
        model_paths=t.model_paths,
        model_id_with_origin_paths=t.model_id_with_origin_paths,
        tokenizer_path=t.tokenizer_path,
        audio_processor_path=None,
        extra_inputs="vace_video,vace_reference_image,action_trajectory",
        fp8_models=t.fp8_models,
        offload_models=t.offload_models,
        task="sft",
        trainable_models=",".join(t.trainable_models) if t.trainable_models else None,
        learning_rate=float(t.learning_rate),
        weight_decay=float(t.weight_decay),
        num_epochs=int(t.num_epochs),
        max_steps=t.max_steps,
        batch_size=int(t.batch_size),
        gradient_accumulation_steps=int(t.gradient_accumulation_steps),
        find_unused_parameters=bool(t.find_unused_parameters),
        use_gradient_checkpointing=bool(t.use_gradient_checkpointing),
        use_gradient_checkpointing_offload=bool(t.use_gradient_checkpointing_offload),
        lora_base_model=t.lora_base_model,
        lora_target_modules=t.lora_target_modules,
        lora_rank=int(t.lora_rank),
        lora_checkpoint=t.lora_checkpoint,
        preset_lora_path=t.preset_lora_path,
        preset_lora_model=t.preset_lora_model,
        output_path=t.output_path,
        remove_prefix_in_ckpt=t.remove_prefix_in_ckpt,
        save_steps=t.save_steps,
        rolling_save_steps=t.rolling_save_steps,
        keep_last_k_ckpts=int(t.keep_last_k_ckpts),
        dataset_num_workers=int(t.dataset_num_workers),
        wandb_project=cfg.project.wandb.project,
        wandb_run_name=cfg.project.wandb.run_name,
        max_timestep_boundary=float(t.max_timestep_boundary),
        min_timestep_boundary=float(t.min_timestep_boundary),
        initialize_model_on_cpu=bool(t.initialize_model_on_cpu),
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
        backbone=b.name,
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
        val_steps=t.val_steps,
        video_log_steps=t.video_log_steps,
        max_val_samples=int(d.get("max_val_samples", 500)),
    )

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


def parse_task_overrides(args: argparse.Namespace) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """Parse task override comma lists from the flat training namespace."""
    train_tasks = None
    if getattr(args, "train_tasks", None):
        train_tasks = [t.strip() for t in args.train_tasks.split(",") if t.strip()]

    holdout_tasks = None
    if getattr(args, "holdout_tasks", None):
        holdout_tasks = [t.strip() for t in args.holdout_tasks.split(",") if t.strip()]

    return train_tasks, holdout_tasks


def resolve_train_tasks(
    args: argparse.Namespace,
    train_tasks: Optional[list[str]] = None,
    holdout_tasks: Optional[list[str]] = None,
) -> Optional[list[str]]:
    """Resolve the multitask training split used by the supported train path."""
    if args.dataset_type != "robotwin_multitask":
        return None
    if train_tasks is not None:
        return train_tasks
    if holdout_tasks is not None:
        return sorted(t for t in ROBOTWIN_ALL_TASKS if t not in holdout_tasks)
    return ROBOTWIN_TRAIN_TASKS


def build_training_dataset(
    args: argparse.Namespace,
    train_tasks: Optional[list[str]] = None,
    holdout_tasks: Optional[list[str]] = None,
):
    """Build the training dataset from the supported package-native adapters."""
    if args.dataset_type == "robotwin_multitask":
        if not args.dataset_dir:
            raise ValueError("data.dataset_dir is required for robotwin_multitask")
        resolved_tasks = resolve_train_tasks(args, train_tasks=train_tasks, holdout_tasks=holdout_tasks)
        return MultiTaskRoboTwinActionDataset(
            dataset_dir=args.dataset_dir,
            robot=args.robot,
            variant=args.variant,
            tasks=resolved_tasks,
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

    return RoboTwinActionDataset(
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


def build_validation_datasets(
    args: argparse.Namespace,
    train_tasks: Optional[list[str]] = None,
    holdout_tasks: Optional[list[str]] = None,
) -> tuple[dict, dict]:
    """Build validation and video logging datasets for training callbacks."""
    if args.dataset_type == "robotwin_multitask":
        resolved_train_tasks = resolve_train_tasks(
            args, train_tasks=train_tasks, holdout_tasks=holdout_tasks
        )
        val_variant = args.val_variant or args.variant
        resolved_holdout = holdout_tasks if holdout_tasks else ROBOTWIN_HOLDOUT_TASKS
        common_kwargs = dict(
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
        val_datasets = {
            "val_id": MultiTaskRoboTwinActionDataset(
                dataset_dir=args.dataset_dir,
                robot=args.robot,
                variant=args.variant,
                tasks=resolved_train_tasks,
                val_ratio=args.val_ratio,
                num_val_samples=0,
                **common_kwargs,
            ),
            "val_ood": MultiTaskRoboTwinActionDataset(
                dataset_dir=args.dataset_dir,
                robot=args.robot,
                variant=val_variant,
                tasks=resolved_holdout,
                val_ratio=1.0,
                num_val_samples=5,
                **common_kwargs,
            ),
        }
        return val_datasets, dict(val_datasets)

    val_ds = RoboTwinActionDataset(
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
    return {"val": val_ds}, {"val": val_ds}


def load_action_stats_into_model(model, dataset) -> None:
    """Populate ActionDiT normalization buffers from dataset action stats."""
    stats = getattr(dataset, "action_stats", None)
    if callable(stats):
        stats = stats()
    elif hasattr(dataset, "_legacy") and hasattr(dataset._legacy, "action_stats"):
        stats = dataset._legacy.action_stats

    if stats is None:
        return

    model.action_dit.action_mean.copy_(torch.from_numpy(stats["mean"].astype(np.float32)))
    model.action_dit.action_std.copy_(torch.from_numpy(stats["std"].astype(np.float32)))


def build_training_module(
    args: argparse.Namespace,
    accelerator=None,
    dataset=None,
):
    """Construct the legacy training module behind the package-native API."""
    device = "cpu"
    if not getattr(args, "initialize_model_on_cpu", True) and accelerator is not None:
        device = accelerator.device

    model = VideoActionTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=int(args.lora_rank),
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=bool(args.use_gradient_checkpointing),
        use_gradient_checkpointing_offload=bool(args.use_gradient_checkpointing_offload),
        extra_inputs=getattr(args, "extra_inputs", "vace_video,vace_reference_image,action_trajectory"),
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=getattr(args, "task", "sft"),
        device=device,
        max_timestep_boundary=float(args.max_timestep_boundary),
        min_timestep_boundary=float(args.min_timestep_boundary),
        action_dim=int(args.action_dim),
        action_dit_dim=int(args.action_dit_dim),
        action_dit_ffn_dim=int(args.action_dit_ffn_dim),
        action_dit_num_heads=int(args.action_dit_num_heads),
        action_dit_num_layers=int(args.action_dit_num_layers),
        action_dit_bridge_layers=args.action_dit_bridge_layers,
        video_dim=int(args.video_dim),
        lambda_video=float(args.lambda_video),
        lambda_action=float(args.lambda_action),
        bridge_type=args.bridge_type,
        action_lr=float(args.action_lr) if args.action_lr is not None else None,
        action_stats_path=getattr(args, "action_stats_path", None),
    )

    if dataset is not None and float(args.lambda_action) > 0:
        load_action_stats_into_model(model, dataset)

    return model


__all__ = [
    "build_training_dataset",
    "build_training_module",
    "build_validation_datasets",
    "cfg_to_flat_namespace",
    "load_action_stats_into_model",
    "parse_task_overrides",
    "resolve_train_tasks",
]
