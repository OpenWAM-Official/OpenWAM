"""OpenWAM trainer that consumes Hydra DictConfig directly.

Composes package-native components:
  - Pipeline building: openwam.train.utils.pipeline_builder
  - Loss: openwam.train.loss.flow_match_loss.FlowMatchVideoActionLoss
  - Optimizer groups: openwam.train.utils.optimizer_groups
  - Checkpointing: openwam.train.utils.checkpointing
  - Architecture: openwam.model.registry (DualSystem / MoE / SharedBackbone)

Usage:
    trainer = OpenWAMTrainer(cfg, accelerator, dataset)
    trainer.train()
"""

import logging
import math
import os

import numpy as np
import torch
from omegaconf import DictConfig

from openwam.train.base import BaseTrainer
from openwam.train.loss.flow_match_loss import FlowMatchVideoActionLoss
from openwam.train.utils.checkpointing import (
    load_trainable_checkpoint,
    manage_checkpoints,
    save_trainable_checkpoint,
)
from openwam.train.utils.optimizer_groups import build_trainable_parameters
from openwam.train.utils.pipeline_builder import build_training_pipeline

logger = logging.getLogger(__name__)

VAE_TEMPORAL_FACTOR = 4  # Wan VAE encodes every 4 video frames into 1 latent time step


def _downsample_video_mask_to_latent(video_is_pad: torch.Tensor) -> torch.Tensor:
    """Downsample frame-level padding mask to VAE latent temporal dimension.

    Following FastWAM: separate frame 0 (conditioning, excluded from loss),
    then group the tail frames by VAE_TEMPORAL_FACTOR. A latent step is
    padded only if ALL frames in the group are padded.

    The returned mask covers tail latent steps only (frame 0 excluded),
    matching the loss which trims pred/target via ``[:, :, 1:]``.

    Args:
        video_is_pad: (T_video,) bool, True=padded.

    Returns:
        (T_latent_tail,) bool mask where
        T_latent_tail = ceil((T_video - 1) / VAE_TEMPORAL_FACTOR).
    """
    T = video_is_pad.shape[0]
    if T <= 1:
        return torch.zeros(0, dtype=torch.bool)

    # Separate frame 0 (conditioning), group tail by VAE_TEMPORAL_FACTOR
    tail_is_pad = video_is_pad[1:]  # (T_video - 1,)

    T_tail = tail_is_pad.shape[0]
    pad_len = (VAE_TEMPORAL_FACTOR - T_tail % VAE_TEMPORAL_FACTOR) % VAE_TEMPORAL_FACTOR
    if pad_len > 0:
        tail_is_pad = torch.cat([tail_is_pad, torch.ones(pad_len, dtype=torch.bool)])

    return tail_is_pad.view(-1, VAE_TEMPORAL_FACTOR).all(dim=1)


class TrainableModuleWrapper(torch.nn.Module):
    """Thin nn.Module wrapper around trainable components for DeepSpeed.

    DeepSpeed requires a single nn.Module to wrap with its engine.
    This collects the ActionDiT and trainable pipeline sub-modules (DiT, VACE)
    so DeepSpeed can manage their optimizer states and gradient sync.

    Not used for forward pass — OpenWAMTrainer.compute_loss() drives execution.
    """

    def __init__(self, action_dit, pipe_trainable_modules: dict):
        super().__init__()
        self.action_dit = action_dit
        self.pipe_modules = torch.nn.ModuleDict(pipe_trainable_modules)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass is handled by OpenWAMTrainer.compute_loss()")


class OpenWAMTrainer(BaseTrainer):
    """Joint video-action trainer for OpenWAM.

    Directly consumes Hydra DictConfig without argparse conversion.
    Builds all components from package-native modules, with no dependency
    on third_party/diffsynth training infrastructure.

    Args:
        cfg: Hydra DictConfig with model, training, data, project sections.
        accelerator: HuggingFace Accelerator instance.
        dataset: Training dataset (used for action stats loading).
    """

    def __init__(self, cfg: DictConfig, accelerator=None, dataset=None):
        super().__init__(cfg, model=None, dataset=dataset, accelerator=accelerator)

        t = cfg.training
        m = cfg.model

        # Build video pipeline (delegated to utils/pipeline_builder.py)
        self.pipe = build_training_pipeline(cfg)

        # Derive video_dim from the loaded model instead of config
        video_dim = int(self.pipe.dit.dim)

        # Build architecture from 3 config sources:
        #   1. cfg.model.architecture  — type, action_dim, bridge_type, bridge_layers
        #   2. cfg.model.action_backbone — dim, ffn_dim, num_heads, ... (DualSystem only)
        #   3. video_dim — derived from loaded model
        from openwam.model.registry import build_architecture

        arch_cfg = getattr(m, "architecture", {})
        action_cfg = getattr(m, "action_backbone", {})

        # Merge: architecture params + action_backbone params + video_dim
        params = {k: v for k, v in arch_cfg.items() if k != "type"}
        if action_cfg:
            params.update({k: v for k, v in action_cfg.items()})
        params["video_dim"] = video_dim

        arch_type = arch_cfg.get("type", "dual_system")
        self.architecture = build_architecture(arch_type, params)
        logger.info("Architecture: %s (video_dim=%d)", arch_type, video_dim)

        # Build ActionDiT for DualSystem, or get it from the architecture
        if hasattr(self.architecture, "action_dit") and self.architecture.action_dit is not None:
            self.action_dit = self.architecture.action_dit
        elif hasattr(self.architecture, "moe_dit"):
            self.action_dit = self.architecture.moe_dit
        else:
            # SharedBackbone: the architecture IS the action model
            self.action_dit = self.architecture

        # Device placement: skip .to(device) when initialize_model_on_cpu + DeepSpeed,
        # because DeepSpeed's prepare() will handle the move.
        _init_on_cpu = bool(t.get("initialize_model_on_cpu", False))
        _use_deepspeed = (
            accelerator is not None
            and hasattr(accelerator, "distributed_type")
            and str(accelerator.distributed_type).endswith("DEEPSPEED")
        )
        if not (_init_on_cpu and _use_deepspeed):
            self.action_dit.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        # --- Freeze: apply after all models are built ---
        # Read from training_strategy config (e.g. joint.yaml / video_only.yaml)
        strategy = cfg.training_strategy
        freeze_list = list(getattr(strategy, "freeze", []))
        for name in freeze_list:
            # Check pipeline sub-modules (text_encoder, vae, dit, vace, ...)
            module = getattr(self.pipe, name, None)
            # Check trainer-level modules (action_dit)
            if module is None:
                module = getattr(self, name, None)
            if module is not None:
                module.requires_grad_(False)
                logger.info("Frozen: %s", name)

        # Build trainable module wrapper for DeepSpeed
        pipe_trainable = {}
        if self.pipe.dit is not None and any(p.requires_grad for p in self.pipe.dit.parameters()):
            pipe_trainable["dit"] = self.pipe.dit
        if getattr(self.pipe, "vace", None) is not None and any(p.requires_grad for p in self.pipe.vace.parameters()):
            pipe_trainable["vace"] = self.pipe.vace
        self.trainable_wrapper = TrainableModuleWrapper(self.action_dit, pipe_trainable)

        # Schedulers (video + action, independent timesteps)
        from openwam.deployment.flow_match_scheduler import FlowMatchScheduler

        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler = FlowMatchScheduler("Wan")
        self.action_scheduler.set_timesteps(1000, training=True)

        # Loss function (lambda weights from training_strategy config)
        self.lambda_video = float(strategy.lambda_video)
        self.lambda_action = float(strategy.lambda_action)
        bridge_type = getattr(m.architecture, "bridge_type", "cross_attn_detach")

        self.loss_fn = FlowMatchVideoActionLoss(
            lambda_video=self.lambda_video,
            lambda_action=self.lambda_action,
            detach_bridge=(bridge_type == "cross_attn_detach"),
        )

        # Decoupled training support
        decoupled_cfg = getattr(t, "decoupled", None)
        self.decoupled_sampler = None
        if decoupled_cfg is not None and getattr(decoupled_cfg, "enabled", False):
            from openwam.train.loss.decoupled_loss import DecoupledFlowMatchLoss

            self.decoupled_sampler = DecoupledFlowMatchLoss(
                video_beta_a=float(getattr(decoupled_cfg, "video_beta_a", 0.5)),
                video_beta_b=float(getattr(decoupled_cfg, "video_beta_b", 1.0)),
                warmup_steps=int(getattr(decoupled_cfg, "warmup_steps", 0)),
            )

        # Load action stats
        if dataset is not None and self.lambda_action > 0:
            self._load_action_stats(dataset)

        # Training config
        self.use_gradient_checkpointing = bool(t.use_gradient_checkpointing)
        self.use_gradient_checkpointing_offload = bool(t.use_gradient_checkpointing_offload)
        self.max_timestep_boundary = float(t.max_timestep_boundary)
        self.min_timestep_boundary = float(t.min_timestep_boundary)

        # Extra inputs
        extra_inputs = getattr(t, "extra_inputs", "vace_video,vace_reference_image,action_trajectory")
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []

        # Store reference for BaseTrainer interface
        self.model = self

        # Pipeline-level conditioning transform (adds VACE fields if missing)
        from openwam.dataloader.transforms.pipeline import VACEConditioningTransform

        self._pipeline_transform = VACEConditioningTransform()

        # Step counter
        self._current_step = 0
        self._last_loss_components = {}

        # Print param counts
        action_params = sum(p.numel() for p in self.action_dit.parameters())
        logger.info("OpenWAMTrainer: ActionDiT %.1fM params", action_params / 1e6)

    def _load_action_stats(self, dataset):
        """Load action normalization stats from dataset into architecture buffers."""
        stats = getattr(dataset, "action_stats", None)
        if callable(stats):
            stats = stats()

        if stats is None:
            return

        mean = torch.from_numpy(stats["mean"].astype(np.float32))
        std = torch.from_numpy(np.maximum(stats["std"].astype(np.float32), 1e-3))
        self.architecture.action_mean.copy_(mean)
        self.architecture.action_std.copy_(std)
        logger.info("Loaded action stats into architecture buffers from dataset")

    def get_trainable_parameters(self):
        """Return optimizer parameter groups."""
        t = self.cfg.training
        return build_trainable_parameters(
            self,
            action_lr=float(t.action_lr) if getattr(t, "action_lr", None) else None,
            video_lr=float(t.video_lr) if getattr(t, "video_lr", None) else None,
            lora_lr=float(t.lora_lr) if getattr(t, "lora_lr", None) else None,
        )

    def compute_loss(self, batch) -> dict:
        """Compute joint video-action loss.

        Args:
            batch: A single sample dict or list of sample dicts from the dataset.

        Returns:
            dict with keys: ``total``, ``video``, ``action``.
        """
        if isinstance(batch, list):
            return self._forward_batch(batch)
        return self._forward_single(batch)

    def _forward_single(self, data) -> dict:
        """Single-sample forward pass."""
        # Add pipeline-specific conditioning (VACE fields) if missing
        data = self._pipeline_transform.apply(data)

        # Extract action data
        action_data = data.get("action_trajectory", None)
        if self.lambda_action > 0 and action_data is None:
            raise ValueError("lambda_action > 0 but no action_trajectory in data.")
        if action_data is not None:
            if isinstance(action_data, np.ndarray):
                action_data = torch.from_numpy(action_data)
            action_data = action_data.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            if action_data.dim() == 2:
                action_data = action_data.unsqueeze(0)

        # Prepare pipeline inputs
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Extra inputs (vace_video, vace_reference_image)
        for key in self.extra_inputs:
            if key == "vace_reference_image":
                val = data.get(key)
                inputs_shared[key] = val[0] if val is not None else None
            elif key == "action_trajectory":
                pass
            else:
                inputs_shared[key] = data.get(key)

        inputs_posi = {"prompt": data["prompt"]}

        # Run pipeline units (data preprocessing)
        inputs = (inputs_shared, inputs_posi, {})
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, _ = inputs

        # Padding masks: mask (True=valid) → is_pad (True=padded)
        #
        # action_mask: (num_frames,) full resolution — matches action_trajectory.
        # video_mask:  (num_video_frames,) after video_stride subsampling.
        #   For video loss: VAE temporally downsamples ~4x. Following FastWAM,
        #   group frames by 4, mark padded only if ALL in group are padded.
        #   First-frame exclusion is handled inside _compute_video_loss.
        action_mask = data.get("action_mask", None)
        video_mask = data.get("video_mask", None)
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.from_numpy(action_mask)
            inputs_shared["action_is_pad"] = (~action_mask).unsqueeze(0).to(device=self.pipe.device)
        if video_mask is not None:
            if isinstance(video_mask, np.ndarray):
                video_mask = torch.from_numpy(video_mask)
            inputs_shared["video_is_pad"] = (
                _downsample_video_mask_to_latent(
                    ~video_mask,
                )
                .unsqueeze(0)
                .to(device=self.pipe.device)
            )

        # Compute loss
        result = self.loss_fn(
            pipe=self.pipe,
            architecture=self.architecture,
            action_scheduler=self.action_scheduler,
            action_data=action_data,
            current_step=self._current_step,
            decoupled_sampler=self.decoupled_sampler,
            **inputs_shared,
            **inputs_posi,
        )

        return {
            "total": result["loss"],
            "video": result.get("loss_video", torch.tensor(0.0)),
            "action": result.get("loss_action", torch.tensor(0.0)),
        }

    def _forward_batch(self, data_list) -> dict:
        """Batched forward pass — encode all samples in one VAE/text-encoder pass."""
        import torch.nn.functional as F
        from einops import rearrange

        # Add pipeline-specific conditioning (VACE fields) if missing
        data_list = [self._pipeline_transform.apply(s) for s in data_list]

        B = len(data_list)
        pipe = self.pipe

        height, width, num_frames = pipe.check_resize_height_width(
            data_list[0]["video"][0].size[1],
            data_list[0]["video"][0].size[0],
            len(data_list[0]["video"]),
        )

        # Collect data
        all_input_videos = []
        all_vace_videos = []
        all_ref_images = []
        all_prompts = []
        all_actions = []
        all_action_masks = []
        all_video_masks = []

        for sample in data_list:
            all_input_videos.append(pipe.preprocess_video(sample["video"]))

            vv = sample.get("vace_video")
            if vv is not None:
                all_vace_videos.append(pipe.preprocess_video(vv))
            else:
                all_vace_videos.append(
                    torch.zeros(1, 3, num_frames, height, width, dtype=pipe.torch_dtype, device=pipe.device)
                )

            ref = sample.get("vace_reference_image")
            if ref is not None:
                if not isinstance(ref, list):
                    ref = [ref]
                all_ref_images.append(pipe.preprocess_video(ref))
            else:
                all_ref_images.append(None)

            all_prompts.append(sample["prompt"])

            action = sample.get("action_trajectory")
            if self.lambda_action > 0 and action is None:
                raise ValueError("lambda_action > 0 but no action_trajectory in data.")
            if action is not None:
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action)
                action = action.to(dtype=pipe.torch_dtype, device=pipe.device).unsqueeze(0)
            all_actions.append(action)

            # Collect masks (True=valid) for padding-aware loss
            amask = sample.get("action_mask", None)
            vmask = sample.get("video_mask", None)
            if amask is not None:
                if isinstance(amask, np.ndarray):
                    amask = torch.from_numpy(amask)
                all_action_masks.append(amask)
            else:
                all_action_masks.append(None)
            if vmask is not None:
                if isinstance(vmask, np.ndarray):
                    vmask = torch.from_numpy(vmask)
                all_video_masks.append(vmask)
            else:
                all_video_masks.append(None)

        ref_flags = [r is not None for r in all_ref_images]
        if any(ref_flags) and not all(ref_flags):
            raise ValueError("Mixed reference images in batch: all samples must be consistent.")
        has_ref = ref_flags[0] if ref_flags else False

        # Batch text encoding
        pipe.load_models_to_device(["text_encoder"])
        ids, mask = pipe.tokenizer(all_prompts, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(pipe.device), mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            context[i, v:] = 0

        # Batch VAE encoding
        pipe.load_models_to_device(["vae"])
        stacked_inputs = torch.cat(all_input_videos, dim=0)
        input_latents = pipe.vae.batch_encode(stacked_inputs, pipe.device).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )

        if has_ref:
            stacked_refs = torch.cat(all_ref_images, dim=0)
            ref_latents = pipe.vae.batch_encode(stacked_refs, pipe.device).to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            input_latents = torch.cat([ref_latents, input_latents], dim=2)

        # VACE context assembly
        vace_context = None
        if pipe.vace is not None:
            stacked_vace = torch.cat(all_vace_videos, dim=0)
            reactive_latents = pipe.vae.batch_encode(stacked_vace, pipe.device).to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            single_zero = torch.zeros(1, 3, num_frames, height, width, dtype=pipe.torch_dtype, device=pipe.device)
            inactive_latent = pipe.vae.batch_encode(single_zero, pipe.device).to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            inactive_latents = inactive_latent.expand(B, -1, -1, -1, -1)
            vace_video_latents = torch.cat([inactive_latents, reactive_latents], dim=1)

            vace_mask = torch.ones(B, 1, num_frames, height, width, dtype=pipe.torch_dtype, device=pipe.device)
            vace_mask_latents = rearrange(vace_mask[:, 0], "B T (H P) (W Q) -> B (P Q) T H W", P=8, Q=8)
            T_lat = (vace_mask_latents.shape[2] + 3) // 4
            vace_mask_latents = F.interpolate(
                vace_mask_latents,
                size=(T_lat, vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
                mode="nearest-exact",
            )

            if has_ref:
                ref_f = ref_latents.shape[2]
                vace_ref_latents = torch.cat([ref_latents, torch.zeros_like(ref_latents)], dim=1)
                vace_video_latents = torch.cat([vace_ref_latents, vace_video_latents], dim=2)
                vace_mask_latents = torch.cat(
                    [
                        torch.zeros(
                            B,
                            vace_mask_latents.shape[1],
                            ref_f,
                            vace_mask_latents.shape[3],
                            vace_mask_latents.shape[4],
                            dtype=pipe.torch_dtype,
                            device=pipe.device,
                        ),
                        vace_mask_latents,
                    ],
                    dim=2,
                )

            vace_context = torch.cat([vace_video_latents, vace_mask_latents], dim=1)

        # TI2V handling
        is_ti2v = getattr(pipe.dit, "fuse_vae_embedding_in_latents", False)
        first_frame_latents = None
        num_clean_prefix = 0
        if is_ti2v and has_ref:
            first_frame_latents = ref_latents[:, :, 0:1].clone()
            num_clean_prefix += ref_latents.shape[2]

        # Assemble inputs
        batched_shared = {
            "latents": None,
            "input_latents": input_latents,
            "vace_context": vace_context,
            "vace_scale": 1.0,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": 1,
            "cfg_merge": False,
            "tiled": False,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "fuse_vae_embedding_in_latents": is_ti2v and has_ref,
            "num_clean_prefix_frames": num_clean_prefix,
            "first_frame_latents": first_frame_latents,
        }

        action_data = torch.cat(all_actions, dim=0) if all_actions[0] is not None else None

        # Padding masks: action at full resolution, video downsampled to latent.
        if all_action_masks[0] is not None:
            batched_shared["action_is_pad"] = torch.stack(
                [~m for m in all_action_masks],
                dim=0,
            ).to(device=pipe.device)  # (B, num_frames) full resolution
        if all_video_masks[0] is not None:
            latent_masks = [_downsample_video_mask_to_latent(~m) for m in all_video_masks]
            batched_shared["video_is_pad"] = torch.stack(latent_masks, dim=0).to(device=pipe.device)

        result = self.loss_fn(
            pipe=self.pipe,
            architecture=self.architecture,
            action_scheduler=self.action_scheduler,
            action_data=action_data,
            current_step=self._current_step,
            decoupled_sampler=self.decoupled_sampler,
            **batched_shared,
            context=context,
        )

        return {
            "total": result["loss"],
            "video": result.get("loss_video", torch.tensor(0.0)),
            "action": result.get("loss_action", torch.tensor(0.0)),
        }

    def _init_wandb(self):
        """Initialize wandb run from project config. Returns the run or None."""
        wandb_cfg = self.cfg.project.get("wandb", None)
        if wandb_cfg is None:
            return None
        project = getattr(wandb_cfg, "project", None)
        if not project:
            return None
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed, skipping wandb logging")
            return None

        run_name = getattr(wandb_cfg, "run_name", None)
        entity = getattr(wandb_cfg, "entity", None)
        from omegaconf import OmegaConf

        run = wandb.init(
            project=project,
            name=run_name,
            entity=entity,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            resume="allow",
        )
        logger.info("wandb initialized: %s/%s", project, run.name)
        return run

    def train(self, num_epochs: int = None, max_steps: int = None):
        """Run the training loop.

        Uses HuggingFace Accelerate for distributed training.

        Args:
            num_epochs: Override for ``training.num_epochs``.
            max_steps: Override for ``training.max_steps``.
        """
        t = self.cfg.training
        num_epochs = num_epochs or int(t.num_epochs)
        max_steps = max_steps or getattr(t, "max_steps", None)
        batch_size = int(t.batch_size)
        lr = float(t.learning_rate)
        grad_accum = int(t.gradient_accumulation_steps)

        # Debug mode: override to a short sanity-check run
        debug = bool(getattr(t, "debug", False))
        if debug:
            max_steps = 20
            save_steps_override = 5
            logger.info("DEBUG mode: max_steps=20, save@5, constant LR")

        # Build optimizer
        params = self.get_trainable_parameters()
        betas = tuple(getattr(t, "adam_betas", [0.9, 0.95]))
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=float(t.weight_decay), betas=betas)

        # Build dataloader
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(t.dataset_num_workers),
            collate_fn=lambda x: x,  # Return list of dicts
            pin_memory=True,
        )

        # Gradient clipping
        max_grad_norm = float(t.max_grad_norm) if getattr(t, "max_grad_norm", None) else None

        # LR scheduler (cosine with linear warmup; disabled in debug mode)
        scheduler = None
        lr_scheduler_type = getattr(t, "lr_scheduler", None)
        if debug:
            lr_scheduler_type = None  # constant LR in debug mode
        if lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup_ratio = float(getattr(t, "warmup_ratio", 0.05))
            lr_min_ratio = float(getattr(t, "lr_min_ratio", 0.01))
            steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
            total_opt_steps = steps_per_epoch * num_epochs
            if max_steps:
                total_opt_steps = min(total_opt_steps, max_steps)
            warmup_steps = int(total_opt_steps * warmup_ratio)
            cosine_steps = max(total_opt_steps - warmup_steps, 1)
            warmup_sched = LinearLR(
                optimizer,
                start_factor=1.0 / max(warmup_steps, 1),
                total_iters=warmup_steps,
            )
            cosine_sched = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=lr * lr_min_ratio,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_steps],
            )
            logger.info(
                "LR scheduler: cosine | total_opt_steps=%d warmup=%d eta_min=%.2e",
                total_opt_steps,
                warmup_steps,
                lr * lr_min_ratio,
            )

        # Checkpoint intervals
        if debug:
            save_steps = save_steps_override
        else:
            save_steps = getattr(t, "save_steps", None)
            if save_steps is not None:
                save_steps = int(save_steps)
        keep_last_k = int(getattr(t, "keep_last_k_ckpts", 3))
        base_output_path = getattr(t, "output_path", "./models")
        from datetime import datetime

        run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if debug:
            run_dir_name += "_debug"
        output_path = os.path.join(base_output_path, run_dir_name)
        os.makedirs(output_path, exist_ok=True)
        logger.info("Checkpoints will be saved to %s", output_path)

        # Detect DeepSpeed
        use_deepspeed = (
            self.accelerator is not None
            and hasattr(self.accelerator, "distributed_type")
            and str(self.accelerator.distributed_type).endswith("DEEPSPEED")
        )

        # Prepare with accelerator
        if use_deepspeed:
            # DeepSpeed needs the model wrapper to manage params/optimizer/gradients
            prepare_args = [self.trainable_wrapper, optimizer, dataloader]
            if scheduler is not None:
                prepare_args.append(scheduler)
                self.trainable_wrapper, optimizer, dataloader, scheduler = self.accelerator.prepare(*prepare_args)
            else:
                self.trainable_wrapper, optimizer, dataloader = self.accelerator.prepare(*prepare_args)

            # Update references — DeepSpeed wraps the module
            unwrapped = self.accelerator.unwrap_model(self.trainable_wrapper)
            self.action_dit = unwrapped.action_dit

            # Sync pipe's trainable sub-modules with DeepSpeed-managed versions
            if "dit" in unwrapped.pipe_modules:
                self.pipe.dit = unwrapped.pipe_modules["dit"]
            if "vace" in unwrapped.pipe_modules:
                self.pipe.vace = unwrapped.pipe_modules["vace"]

            # Fix pipe.device — may still be "cpu" after initialize_model_on_cpu
            self.pipe.device = self.accelerator.device

            # Move frozen modules (T5, VAE) to device
            for name in ("text_encoder", "vae"):
                mod = getattr(self.pipe, name, None)
                if mod is not None:
                    mod.to(device=self.accelerator.device)

            logger.info("DeepSpeed: model wrapped, device=%s", self.accelerator.device)
        elif self.accelerator is not None:
            # Plain DDP / single GPU
            optimizer, dataloader = self.accelerator.prepare(optimizer, dataloader)

        # Collect all trainable params for grad clipping
        all_params = [p for group in optimizer.param_groups for p in group["params"]]

        # Initialize wandb (skip in debug mode; rank 0 only for multi-GPU)
        _is_main = self.accelerator is None or self.accelerator.is_main_process
        wandb_run = None if (debug or not _is_main) else self._init_wandb()

        from tqdm import tqdm

        # Estimate total steps for progress bar
        total_steps = len(dataloader) * num_epochs
        if max_steps:
            total_steps = min(total_steps, max_steps)

        import time as _time

        opt_step = 0
        global_step = 0
        _step_t0 = _time.monotonic()
        pbar = tqdm(total=total_steps, desc="Training", unit="step")
        for epoch in range(num_epochs):
            for batch in dataloader:
                losses = self.compute_loss(batch)

                loss = losses["total"]
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                # Gradient clipping & optimizer step
                grad_norm = torch.tensor(0.0, device=loss.device)
                if (global_step + 1) % grad_accum == 0:
                    if max_grad_norm is not None:
                        if use_deepspeed:
                            grad_norm_val = self.accelerator.clip_grad_norm_(all_params, max_grad_norm)
                        else:
                            grad_norm_val = torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                        grad_norm = torch.tensor(float(grad_norm_val), device=loss.device)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    opt_step += 1

                self._current_step = global_step
                global_step += 1

                # --- Gather losses across all ranks ---
                _device = loss.device
                if self.accelerator is not None and self.accelerator.num_processes > 1:
                    # Build tensor of metrics to gather in one call
                    local_metrics = torch.tensor(
                        [
                            loss.detach().float().item(),
                            losses["video"].item()
                            if isinstance(losses["video"], torch.Tensor)
                            else float(losses["video"]),
                            losses["action"].item()
                            if isinstance(losses["action"], torch.Tensor)
                            else float(losses["action"]),
                            grad_norm.item(),
                        ],
                        device=_device,
                        dtype=torch.float32,
                    ).reshape(1, -1)
                    gathered = self.accelerator.gather(local_metrics)  # (num_processes, 4)
                    global_metrics = gathered.mean(dim=0)
                    loss_total = global_metrics[0].item()
                    loss_video = global_metrics[1].item()
                    loss_action = global_metrics[2].item()
                    global_grad_norm = global_metrics[3].item()
                else:
                    loss_total = loss.detach().item()
                    loss_video = (
                        losses["video"].item() if isinstance(losses["video"], torch.Tensor) else float(losses["video"])
                    )
                    loss_action = (
                        losses["action"].item()
                        if isinstance(losses["action"], torch.Tensor)
                        else float(losses["action"])
                    )
                    global_grad_norm = grad_norm.item()

                # --- Progress bar ---
                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(
                    loss=f"{loss_total:.4f}",
                    video=f"{loss_video:.4f}",
                    action=f"{loss_action:.4f}",
                    lr=f"{current_lr:.2e}",
                    epoch=epoch,
                )
                pbar.update(1)

                # --- wandb (rank 0 only, with global-averaged metrics) ---
                if wandb_run is not None:
                    _now = _time.monotonic()
                    steps_per_sec = 1.0 / max(_now - _step_t0, 1e-9)
                    _step_t0 = _now
                    _num_procs = self.accelerator.num_processes if self.accelerator is not None else 1
                    log_dict = {
                        "train/loss": loss_total,
                        "train/loss_video": loss_video,
                        "train/loss_action": loss_action,
                        "train/grad_norm": global_grad_norm,
                        "train/lr": current_lr,
                        "performance/steps_per_sec": steps_per_sec,
                        "performance/samples_per_sec": steps_per_sec * batch_size * _num_procs,
                    }
                    wandb_run.log(log_dict, step=global_step)

                # Periodic checkpoint saving (rank 0 only for multi-GPU)
                _is_main = self.accelerator is None or self.accelerator.is_main_process
                if save_steps and global_step % save_steps == 0 and _is_main:
                    ckpt_path = os.path.join(output_path, f"checkpoint_step_{global_step}.safetensors")
                    self.save_checkpoint(ckpt_path)
                    manage_checkpoints(output_path, keep_last_k)

                if max_steps and global_step >= max_steps:
                    pbar.close()
                    if wandb_run is not None:
                        wandb_run.finish()
                    return

        pbar.close()

        # Save final checkpoint (rank 0 only)
        _is_main = self.accelerator is None or self.accelerator.is_main_process
        if save_steps and _is_main:
            ckpt_path = os.path.join(output_path, f"checkpoint_step_{global_step}.safetensors")
            self.save_checkpoint(ckpt_path)
            manage_checkpoints(output_path, keep_last_k)

        if wandb_run is not None:
            wandb_run.finish()

    def save_checkpoint(self, path: str):
        """Export trainable state dict to safetensors."""
        save_trainable_checkpoint(path, self.action_dit, self.pipe, self.lambda_action)

    def load_checkpoint(self, path: str):
        """Load a checkpoint into the model."""
        load_trainable_checkpoint(path, self.action_dit, self.pipe)

    # --- Compatibility with build_trainable_parameters ---
    # These properties allow optimizer_groups.py to work on OpenWAMTrainer

    @property
    def lambda_action_compat(self):
        return self.lambda_action

    def parameters(self):
        """Yield all parameters (for compatibility)."""
        yield from self.action_dit.parameters()
        yield from self.pipe.parameters()

    def named_parameters(self, prefix="", recurse=True):
        """Yield named parameters (for compatibility)."""
        for name, param in self.action_dit.named_parameters(prefix="action_dit"):
            yield name, param
        for name, param in self.pipe.named_parameters():
            yield name, param

    def state_dict(self):
        """Return full state dict."""
        state = {}
        for name, param in self.action_dit.named_parameters():
            state[f"action_dit.{name}"] = param.data
        for name, buf in self.action_dit.named_buffers():
            state[f"action_dit.{name}"] = buf
        for name, param in self.pipe.named_parameters():
            state[name] = param.data
        return state
