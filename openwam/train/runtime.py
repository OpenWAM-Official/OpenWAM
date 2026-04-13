"""Package-native builders for the supported OpenWAM training path."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from omegaconf import DictConfig


def cfg_to_flat_namespace(cfg: DictConfig) -> argparse.Namespace:
    """Convert Hydra config groups to the legacy training module arguments.

    .. deprecated::
        Used by the legacy training path. Prefer :class:`NativeTrainer`
        which consumes DictConfig directly.
    """
    t = cfg.training
    d = cfg.data
    m = cfg.model
    b = cfg.model.backbone

    bridge_layers_str = ",".join(str(x) for x in m.bridge_layers)

    args = argparse.Namespace(
        model_paths=t.model_paths,
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
        video_lr=float(t.video_lr) if t.get("video_lr", None) is not None else None,
        lora_lr=float(t.lora_lr) if t.get("lora_lr", None) is not None else None,
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
        max_val_samples=int(d.get("max_val_samples", 500)),
    )

    dtype = d.type
    args.dataset_type = dtype
    args.action_mode = d.get("action_mode", "eef")

    if dtype in ("robotwin", "robotwin_multitask"):
        args.dataset_dir = d.dataset_dir
        task_name = d.get("task_name", None)
        args.task_name = task_name
        if task_name:
            args.hdf5_data_root = os.path.join(d.dataset_dir, task_name, f"{d.robot}_{d.variant}", "data")
        else:
            args.hdf5_data_root = None
    else:
        raise ValueError(f"Unknown dataset type '{dtype}'")

    return args


def build_training_dataset(
    args: argparse.Namespace,
    data_config=None,
):
    """Build the training dataset via the dataset registry.

    Args:
        args: Flat training namespace (from ``cfg_to_flat_namespace``).
        data_config: Original Hydra ``cfg.data`` DictConfig.  Required for
            robotwin datasets — passed to ``MultiTaskRoboTwinDataset.from_config``.
    """
    if args.dataset_type in ("robotwin", "robotwin_multitask"):
        if data_config is None:
            raise ValueError(
                "data_config (cfg.data) is required for robotwin datasets. Pass it to build_training_dataset()."
            )
        from openwam.dataloader.registry import build_dataset

        return build_dataset(data_config, split="train")

    raise ValueError(f"Unknown dataset_type '{args.dataset_type}'")


def build_validation_datasets(
    args: argparse.Namespace,
    data_config=None,
) -> tuple[dict, dict]:
    """Build validation and video logging datasets for training callbacks."""
    if args.dataset_type in ("robotwin", "robotwin_multitask"):
        if data_config is None:
            raise ValueError(
                "data_config (cfg.data) is required for robotwin datasets. Pass it to build_validation_datasets()."
            )
        from openwam.dataloader.registry import build_dataset

        val_ds = build_dataset(data_config, split="val")
        return {"val": val_ds}, {"val": val_ds}

    raise ValueError(f"Unknown dataset_type '{args.dataset_type}'")


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


__all__ = [
    "build_training_dataset",
    "build_validation_datasets",
    "cfg_to_flat_namespace",
    "load_action_stats_into_model",
]
