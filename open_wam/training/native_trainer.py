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

import numpy as np
import torch
from omegaconf import DictConfig

from open_wam.training.base import BaseTrainer
from open_wam.training.flow_match_loss import FlowMatchVideoActionLoss
from open_wam.training.optimizer_groups import build_trainable_parameters

logger = logging.getLogger(__name__)


class TrainableModuleWrapper(torch.nn.Module):
    """Thin nn.Module wrapper around trainable components for DeepSpeed.

    DeepSpeed requires a single nn.Module to wrap with its engine.
    This collects the ActionDiT and trainable pipeline sub-modules (DiT, VACE)
    so DeepSpeed can manage their optimizer states and gradient sync.

    Not used for forward pass — NativeTrainer.compute_loss() drives execution.
    """

    def __init__(self, action_dit, pipe_trainable_modules: dict):
        super().__init__()
        self.action_dit = action_dit
        self.pipe_modules = torch.nn.ModuleDict(pipe_trainable_modules)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass is handled by NativeTrainer.compute_loss()")


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

        # Build video pipeline
        self.pipe = self._build_pipeline(cfg)

        # Build architecture from config (supports dual_system, moe_expert, shared_backbone)
        arch_cfg = getattr(m, "architecture", None)
        if arch_cfg is not None and hasattr(arch_cfg, "type"):
            from open_wam.models.architectures.registry import build_architecture

            arch_type = arch_cfg.type
            # Convert OmegaConf to plain dict for architecture constructor
            arch_params = {k: v for k, v in arch_cfg.items() if k != "type"}
            self.architecture = build_architecture(arch_type, arch_params)
            logger.info("Architecture: %s (from config)", arch_type)
        else:
            # Fallback: build DualSystem from flat model config (backward compat)
            from open_wam.models.architectures.dual_system import DualSystemArchitecture

            # Merge video_dim from backbone config into model config for DualSystem
            b = cfg.model.backbone
            dual_cfg = {k: v for k, v in m.items()}
            dual_cfg.setdefault("video_dim", int(b.video_dim))
            self.architecture = DualSystemArchitecture(cfg=dual_cfg)
            logger.info("Architecture: dual_system (default)")

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

        # Build trainable module wrapper for DeepSpeed
        pipe_trainable = {}
        if self.pipe.dit is not None and any(p.requires_grad for p in self.pipe.dit.parameters()):
            pipe_trainable["dit"] = self.pipe.dit
        if getattr(self.pipe, "vace", None) is not None and any(p.requires_grad for p in self.pipe.vace.parameters()):
            pipe_trainable["vace"] = self.pipe.vace
        self.trainable_wrapper = TrainableModuleWrapper(self.action_dit, pipe_trainable)

        # Schedulers (video + action, independent timesteps)
        from open_wam.inference.flow_match_scheduler import FlowMatchScheduler

        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler = FlowMatchScheduler("Wan")
        self.action_scheduler.set_timesteps(1000, training=True)

        # Loss function
        self.lambda_video = float(t.lambda_video)
        self.lambda_action = float(t.lambda_action)
        bridge_type = getattr(m, "bridge_type", "cross_attn_detach")

        action_mode = cfg.data.get("action_mode", "joint")

        self.loss_fn = FlowMatchVideoActionLoss(
            lambda_video=self.lambda_video,
            lambda_action=self.lambda_action,
            detach_bridge=(bridge_type == "cross_attn_detach"),
            action_mode=action_mode,
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

        # Pipeline-level conditioning transform (adds VACE fields if missing)
        from open_wam.data.transforms.pipeline import VACEConditioningTransform

        self._pipeline_transform = VACEConditioningTransform()

        # Step counter
        self._current_step = 0
        self._last_loss_components = {}

        # Print param counts
        action_params = sum(p.numel() for p in self.action_dit.parameters())
        logger.info("NativeTrainer: ActionDiT %.1fM params", action_params / 1e6)

    def _build_pipeline(self, cfg: DictConfig):
        """Build WanVideoPipeline from Hydra config."""
        import json

        from open_wam.inference.model_config import ModelConfig
        from open_wam.inference.video_pipeline import WanVideoPipeline

        t = cfg.training

        device = "cpu" if bool(t.initialize_model_on_cpu) else "cuda"

        # Parse model paths
        # Accepts:
        #   - A directory path (str): auto-groups sharded safetensors by name prefix,
        #     each .pth file becomes its own ModelConfig.
        #   - A list where each element is either a str (single file) or a list[str]
        #     (multiple shards that form one model, e.g. DiT split across 3 files).
        #   - A JSON string (for CLI overrides).
        model_paths = t.model_paths
        if isinstance(model_paths, str):
            try:
                model_paths = json.loads(model_paths)
            except json.JSONDecodeError:
                model_paths = [model_paths]

        # Expand a single directory into grouped model entries
        if model_paths and len(model_paths) == 1 and isinstance(model_paths[0], str) and os.path.isdir(model_paths[0]):
            import glob as _glob
            from collections import defaultdict

            model_dir = model_paths[0]
            safetensors = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
            pth_files = sorted(_glob.glob(os.path.join(model_dir, "*.pth")))
            if not safetensors and not pth_files:
                raise FileNotFoundError(f"No *.safetensors or *.pth files found in {model_dir}")

            # Group sharded safetensors by prefix (e.g. "diffusion_pytorch_model-0000X-of-00003")
            # Files matching *-NNNNN-of-NNNNN.safetensors are shards of the same model.
            import re

            shard_groups = defaultdict(list)
            standalone = []
            for f in safetensors:
                basename = os.path.basename(f)
                m = re.match(r"^(.+)-\d{5}-of-\d{5}\.safetensors$", basename)
                if m:
                    shard_groups[m.group(1)].append(f)
                else:
                    standalone.append(f)

            model_paths = []
            for prefix in sorted(shard_groups):
                shards = sorted(shard_groups[prefix])
                model_paths.append(shards)  # list[str] → one ModelConfig with multiple files
                logger.info("Grouped %d shards as one model: %s-*", len(shards), prefix)
            for f in standalone:
                model_paths.append(f)
            for f in pth_files:
                model_paths.append(f)
            logger.info("Auto-discovered %d model entries from %s", len(model_paths), model_dir)

        model_configs = []
        if model_paths:
            for p in model_paths:
                # p is either str (single file) or list[str] (sharded model)
                model_configs.append(ModelConfig(p))

        tokenizer_path = t.tokenizer_path
        if tokenizer_path is None:
            # Auto-detect: look for google/umt5-xxl under model_paths directory
            _raw = t.model_paths
            _model_dir = _raw if isinstance(_raw, str) and os.path.isdir(_raw) else None
            _auto_tok = os.path.join(_model_dir, "google", "umt5-xxl") if _model_dir else None
            if _auto_tok and os.path.isdir(_auto_tok):
                tokenizer_config = ModelConfig(_auto_tok)
                logger.info("Auto-detected tokenizer at %s", _auto_tok)
            else:
                tokenizer_config = ModelConfig(
                    model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"
                )
        else:
            tokenizer_config = ModelConfig(tokenizer_path)

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
            trainable_str = (
                ",".join(trainable_models) if isinstance(trainable_models, (list, tuple)) else trainable_models
            )
        else:
            trainable_str = None

        # Apply LoRA if configured
        lora_base_model = getattr(t, "lora_base_model", None)
        if lora_base_model:
            pipe = self._setup_training_mode(
                pipe,
                trainable_str,
                lora_base_model,
                t.lora_target_modules,
                int(t.lora_rank),
                t.lora_checkpoint,
                t.preset_lora_path,
                t.preset_lora_model,
            )

        # Freeze inference-only components (text encoder, VAE, image encoder)
        # Only DiT, VACE, and ActionDiT should be trainable
        for name in ("text_encoder", "vae", "image_encoder"):
            module = getattr(pipe, name, None)
            if module is not None:
                module.requires_grad_(False)
                logger.info("Frozen: pipe.%s", name)

        # Gradient checkpointing
        if bool(t.use_gradient_checkpointing):
            for module in pipe.modules():
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable()

        return pipe

    def _setup_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model,
        lora_target_modules,
        lora_rank,
        lora_checkpoint,
        preset_lora_path,
        preset_lora_model,
    ):
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

    def _manage_checkpoints(self, output_dir: str, keep_last_k: int):
        """Delete old checkpoints, keeping only the most recent *keep_last_k*."""
        import glob as _glob
        import re

        pattern = os.path.join(output_dir, "checkpoint_step_*")
        files = _glob.glob(pattern)

        # Sort numerically by step number
        def _step_num(path):
            m = re.search(r"checkpoint_step_(\d+)", path)
            return int(m.group(1)) if m else 0

        files.sort(key=_step_num)
        while len(files) > keep_last_k:
            old = files.pop(0)
            if os.path.isfile(old):
                os.remove(old)
                logger.info("Removed old checkpoint: %s", old)

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
        """Run the native training loop.

        This replaces ``launch_training_task`` from third_party/diffsynth.
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
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=float(t.weight_decay), betas=(0.9, 0.95))

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

        opt_step = 0
        global_step = 0
        pbar = tqdm(total=total_steps, desc="Training", unit="step")
        for epoch in range(num_epochs):
            for batch in dataloader:
                losses = self.compute_loss(batch)

                loss = losses["total"]
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                if (global_step + 1) % grad_accum == 0:
                    if max_grad_norm is not None:
                        if use_deepspeed:
                            self.accelerator.clip_grad_norm_(all_params, max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    opt_step += 1

                self._current_step = global_step
                global_step += 1

                # --- Progress bar ---
                current_lr = optimizer.param_groups[0]["lr"]
                loss_total = losses["total"].item()
                loss_video = losses["video"].item() if isinstance(losses["video"], torch.Tensor) else losses["video"]
                loss_action = (
                    losses["action"].item() if isinstance(losses["action"], torch.Tensor) else losses["action"]
                )
                pbar.set_postfix(
                    loss=f"{loss_total:.4f}",
                    video=f"{loss_video:.4f}",
                    action=f"{loss_action:.4f}",
                    lr=f"{current_lr:.2e}",
                    epoch=epoch,
                )
                pbar.update(1)

                # --- wandb ---
                if wandb_run is not None:
                    log_dict = {
                        "train/loss": loss_total,
                        "train/loss_video": loss_video,
                        "train/loss_action": loss_action,
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                    }
                    for key in ("loss_video_unweighted", "loss_action_unweighted", "loss_scale_ratio"):
                        if key in losses and isinstance(losses[key], torch.Tensor):
                            log_dict[f"train/{key}"] = losses[key].item()
                    wandb_run.log(log_dict, step=global_step)

                # Periodic checkpoint saving (rank 0 only for multi-GPU)
                _is_main = self.accelerator is None or self.accelerator.is_main_process
                if save_steps and global_step % save_steps == 0 and _is_main:
                    ckpt_path = os.path.join(output_path, f"checkpoint_step_{global_step}.safetensors")
                    self.save_checkpoint(ckpt_path)
                    self._manage_checkpoints(output_path, keep_last_k)

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
            self._manage_checkpoints(output_path, keep_last_k)

        if wandb_run is not None:
            wandb_run.finish()

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
