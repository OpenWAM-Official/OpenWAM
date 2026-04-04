"""Package-native trainer that consumes Hydra DictConfig directly.

Replaces the legacy training path that required:
  cfg_to_flat_namespace → argparse → VideoActionTrainingModule → third_party/diffsynth

This trainer composes existing package-native components:
  - Model loading: open_wam.inference.model_loader.load_wam_models
  - Loss: open_wam.training.flow_match_loss.FlowMatchVideoActionLoss
  - Optimizer groups: open_wam.training.optimizer_groups.build_trainable_parameters
  - Architecture: open_wam.models.architectures (DualSystemArchitecture)

Usage:
    trainer = NativeTrainer(cfg, accelerator, dataset)
    trainer.train()
"""

import logging
import math
import os
import warnings
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from open_wam.training.base import BaseTrainer
from open_wam.training.flow_match_loss import FlowMatchVideoActionLoss
from open_wam.training.optimizer_groups import build_trainable_parameters

logger = logging.getLogger(__name__)


class NativeTrainer(BaseTrainer):
    """Package-native joint video-action trainer.

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
        b = cfg.model.backbone

        # Build models
        self.pipe, self.action_dit = self._build_models(cfg)

        # Wrap in architecture interface
        from open_wam.models.architectures.dual_system import DualSystemArchitecture
        self.architecture = DualSystemArchitecture(cfg=None)
        self.architecture.action_dit = self.action_dit

        # Action scheduler (independent from video)
        from third_party.diffsynth.diffusion import FlowMatchScheduler
        self.action_scheduler = FlowMatchScheduler("Wan")
        self.action_scheduler.set_timesteps(1000, training=True)

        # Loss function
        self.lambda_video = float(t.lambda_video)
        self.lambda_action = float(t.lambda_action)
        bridge_type = m.bridge_type

        self.loss_fn = FlowMatchVideoActionLoss(
            lambda_video=self.lambda_video,
            lambda_action=self.lambda_action,
            detach_bridge=(bridge_type == "cross_attn_detach"),
        )

        # Decoupled training support
        decoupled_cfg = getattr(t, "decoupled", None)
        self.decoupled_sampler = None
        if decoupled_cfg is not None and getattr(decoupled_cfg, "enabled", False):
            from open_wam.training.decoupled_loss import DecoupledFlowMatchLoss
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

        # Step counter
        self._current_step = 0
        self._last_loss_components = {}

        # Print param counts
        action_params = sum(p.numel() for p in self.action_dit.parameters())
        logger.info("NativeTrainer: ActionDiT %.1fM params", action_params / 1e6)

    def _build_models(self, cfg: DictConfig):
        """Build pipeline and ActionDiT from Hydra config."""
        import json

        from third_party.diffsynth.models.action_dit import ActionDiT
        from third_party.diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

        t = cfg.training
        m = cfg.model
        b = cfg.model.backbone

        device = "cpu" if bool(t.initialize_model_on_cpu) else "cuda"

        # Parse model paths
        model_paths = t.model_paths
        if isinstance(model_paths, str):
            model_paths = json.loads(model_paths)
        model_id_with_origin = t.model_id_with_origin_paths

        model_configs = []
        if model_paths:
            for p in model_paths:
                model_configs.append(ModelConfig(p))
        if model_id_with_origin:
            for entry in model_id_with_origin:
                parts = entry.split(",")
                if len(parts) == 2:
                    model_configs.append(ModelConfig(parts[0], origin_file_pattern=parts[1]))
                else:
                    model_configs.append(ModelConfig(entry))

        tokenizer_path = t.tokenizer_path
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )

        # Load pipeline
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        # Training mode setup for pipeline
        trainable_models = t.trainable_models
        if trainable_models:
            trainable_str = ",".join(trainable_models) if isinstance(trainable_models, (list, tuple)) else trainable_models
        else:
            trainable_str = None

        # Apply LoRA if configured
        lora_base_model = getattr(t, "lora_base_model", None)
        if lora_base_model:
            pipe = self._setup_training_mode(
                pipe, trainable_str, lora_base_model,
                t.lora_target_modules, int(t.lora_rank),
                t.lora_checkpoint, t.preset_lora_path, t.preset_lora_model,
            )

        # Gradient checkpointing
        if bool(t.use_gradient_checkpointing):
            for module in pipe.modules():
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable()

        # Build ActionDiT
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
        ).to(dtype=torch.bfloat16)

        return pipe, action_dit

    def _setup_training_mode(self, pipe, trainable_models, lora_base_model,
                              lora_target_modules, lora_rank, lora_checkpoint,
                              preset_lora_path, preset_lora_model):
        """Set up LoRA and training mode on the pipeline."""
        # Delegate to DiffusionTrainingModule's static methods if needed.
        # For NativeTrainer, we use a simpler approach: direct PEFT injection.
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            logger.warning("PEFT not available, skipping LoRA setup")
            return pipe

        if lora_base_model and hasattr(pipe, lora_base_model):
            base_model = getattr(pipe, lora_base_model)
            target_modules = lora_target_modules.split(",") if lora_target_modules else None
            lora_config = LoraConfig(
                r=lora_rank,
                target_modules=target_modules,
            )
            setattr(pipe, lora_base_model, get_peft_model(base_model, lora_config))

        return pipe

    def _load_action_stats(self, dataset):
        """Load action normalization stats from dataset into ActionDiT buffers."""
        stats = getattr(dataset, "action_stats", None)
        if callable(stats):
            stats = stats()

        if stats is None:
            return

        self.action_dit.action_mean.copy_(
            torch.from_numpy(stats["mean"].astype(np.float32))
        )
        self.action_dit.action_std.copy_(
            torch.from_numpy(np.maximum(stats["std"].astype(np.float32), 1e-3))
        )
        logger.info("Loaded action stats into ActionDiT buffers from dataset")

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
        # Extract action data
        action_data = data.get("action_trajectory", None)
        if self.lambda_action > 0 and action_data is None:
            raise ValueError(
                "lambda_action > 0 but no action_trajectory in data."
            )
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
        self._last_loss_components = {k: v for k, v in result.items() if k != "loss"}

        return {
            "total": result["loss"],
            "video": result.get("loss_video", torch.tensor(0.0)),
            "action": result.get("loss_action", torch.tensor(0.0)),
            **{k: v for k, v in result.items() if k != "loss"},
        }

    def _forward_batch(self, data_list) -> dict:
        """Batched forward pass — encode all samples in one VAE/text-encoder pass."""
        from einops import rearrange
        import torch.nn.functional as F

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

        for sample in data_list:
            all_input_videos.append(pipe.preprocess_video(sample["video"]))

            vv = sample.get("vace_video")
            if vv is not None:
                all_vace_videos.append(pipe.preprocess_video(vv))
            else:
                all_vace_videos.append(
                    torch.zeros(1, 3, num_frames, height, width,
                                dtype=pipe.torch_dtype, device=pipe.device)
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

        ref_flags = [r is not None for r in all_ref_images]
        if any(ref_flags) and not all(ref_flags):
            raise ValueError(
                "Mixed reference images in batch: all samples must be consistent."
            )
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
            single_zero = torch.zeros(
                1, 3, num_frames, height, width, dtype=pipe.torch_dtype, device=pipe.device
            )
            inactive_latent = pipe.vae.batch_encode(single_zero, pipe.device).to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            inactive_latents = inactive_latent.expand(B, -1, -1, -1, -1)
            vace_video_latents = torch.cat([inactive_latents, reactive_latents], dim=1)

            vace_mask = torch.ones(
                B, 1, num_frames, height, width, dtype=pipe.torch_dtype, device=pipe.device
            )
            vace_mask_latents = rearrange(
                vace_mask[:, 0], "B T (H P) (W Q) -> B (P Q) T H W", P=8, Q=8
            )
            T_lat = (vace_mask_latents.shape[2] + 3) // 4
            vace_mask_latents = F.interpolate(
                vace_mask_latents,
                size=(T_lat, vace_mask_latents.shape[3], vace_mask_latents.shape[4]),
                mode="nearest-exact",
            )

            if has_ref:
                ref_f = ref_latents.shape[2]
                vace_ref_latents = torch.cat(
                    [ref_latents, torch.zeros_like(ref_latents)], dim=1
                )
                vace_video_latents = torch.cat(
                    [vace_ref_latents, vace_video_latents], dim=2
                )
                vace_mask_latents = torch.cat([
                    torch.zeros(
                        B, vace_mask_latents.shape[1], ref_f,
                        vace_mask_latents.shape[3], vace_mask_latents.shape[4],
                        dtype=pipe.torch_dtype, device=pipe.device,
                    ),
                    vace_mask_latents,
                ], dim=2)

            vace_context = torch.cat([vace_video_latents, vace_mask_latents], dim=1)

        # TI2V handling
        is_ti2v = getattr(pipe.dit, 'fuse_vae_embedding_in_latents', False)
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
        self._last_loss_components = {k: v for k, v in result.items() if k != "loss"}

        return {
            "total": result["loss"],
            "video": result.get("loss_video", torch.tensor(0.0)),
            "action": result.get("loss_action", torch.tensor(0.0)),
            **{k: v for k, v in result.items() if k != "loss"},
        }

    def train(self, num_epochs: int = None, max_steps: int = None):
        """Run the native training loop.

        This replaces ``launch_training_task`` from third_party/diffsynth.
        Uses HuggingFace Accelerate for distributed training.
        """
        t = self.cfg.training
        num_epochs = num_epochs or int(t.num_epochs)
        max_steps = max_steps or getattr(t, "max_steps", None)
        batch_size = int(t.batch_size)
        lr = float(t.learning_rate)
        grad_accum = int(t.gradient_accumulation_steps)

        # Build optimizer
        params = self.get_trainable_parameters()
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=float(t.weight_decay))

        # Build dataloader
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(t.dataset_num_workers),
            collate_fn=lambda x: x,  # Return list of dicts
            pin_memory=True,
        )

        # Prepare with accelerator
        if self.accelerator is not None:
            optimizer, dataloader = self.accelerator.prepare(optimizer, dataloader)

        global_step = 0
        for epoch in range(num_epochs):
            for batch in dataloader:
                losses = self.compute_loss(batch)

                loss = losses["total"]
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                if (global_step + 1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                self._current_step = global_step
                global_step += 1

                if global_step % 100 == 0:
                    logger.info(
                        "Step %d | loss=%.4f video=%.4f action=%.4f",
                        global_step,
                        losses["total"].item(),
                        losses["video"].item() if isinstance(losses["video"], torch.Tensor) else losses["video"],
                        losses["action"].item() if isinstance(losses["action"], torch.Tensor) else losses["action"],
                    )

                if max_steps and global_step >= max_steps:
                    return

    def save_checkpoint(self, path: str):
        """Export trainable state dict to safetensors."""
        state_dict = {}

        # ActionDiT parameters and buffers
        if self.lambda_action > 0:
            for name, param in self.action_dit.named_parameters():
                if param.requires_grad:
                    state_dict[f"action_dit.{name}"] = param.data
            for buf_name in ("action_mean", "action_std"):
                buf = getattr(self.action_dit, buf_name, None)
                if buf is not None:
                    state_dict[f"action_dit.{buf_name}"] = buf

        # Video pipeline trainable parameters
        for name, param in self.pipe.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".safetensors"):
            from safetensors.torch import save_file
            save_file(state_dict, path)
        else:
            torch.save(state_dict, path)

        logger.info("Saved checkpoint to %s (%d keys)", path, len(state_dict))

    def load_checkpoint(self, path: str):
        """Load a checkpoint into the model."""
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu")

        action_keys = {k: v for k, v in state_dict.items() if k.startswith("action_dit.")}
        if action_keys:
            cleaned = {k.removeprefix("action_dit."): v for k, v in action_keys.items()}
            self.action_dit.load_state_dict(cleaned, strict=False)

        pipe_keys = {k: v for k, v in state_dict.items() if not k.startswith("action_dit.")}
        if pipe_keys:
            self.pipe.load_state_dict(pipe_keys, strict=False)

        logger.info("Loaded checkpoint from %s", path)

    # --- Compatibility with build_trainable_parameters ---
    # These properties allow optimizer_groups.py to work on NativeTrainer

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
