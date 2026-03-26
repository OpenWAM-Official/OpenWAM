"""
Video-Action Joint Training Module and Loss Functions.

Extends the DiffSynth-Studio training framework to support synchronized
video + action generation using independent diffusion timesteps.

Training Strategies (via CLI flags)
------------------------------------
  Video-only:     --lambda_action 0  (train video model only, no action loss)
  Joint:          --lambda_video 1.0 --lambda_action 1.0  (train both together)

Bridge types (--bridge_type):
  cross_attn          — Unidirectional cross-attention (video -> action).
  cross_attn_detach   — Same architecture, gradients detached at bridge.
  joint_self_attn     — MMDiT-style bidirectional joint self-attention.

Key Design Decisions
--------------------
- Independent timesteps for video and action (UWM-style)
- Bridge features extracted from video DiT and passed to ActionDiT via
  cross-attention or joint self-attention (configurable via --bridge_type)
- Gradients flow from action loss back through bridge features to the video DiT,
  unless --bridge_type cross_attn_detach is set (REPA-inspired)
- Flow matching loss for both modalities with video loss warmup
- Configurable loss weighting and freezing for multi-strategy support

Training Optimization (Batched Path)
-------------------------------------
When the dataloader provides a list of samples (B > 1), ``forward()`` dispatches
to ``_forward_batch()`` which applies two key optimizations:

1. **Per-sample timestep sampling** — Each sample in the batch draws its own
   (video_t, action_t) independently, so one gradient step covers B points in the
   timestep distribution instead of 1.  This is standard practice in modern
   diffusion training (Stable Diffusion, DDPM, diffusers).  The scheduler's
   ``add_noise`` / ``training_target`` helpers only accept scalar timesteps, so
   the noise formula ``x_t = (1 - σ) * x_0 + σ * ε`` is inlined with
   per-sample σ broadcast via ``view(B, 1, 1, 1, 1)``.

2. **Batched VAE / text encoding** — Instead of running pipeline units
   per-sample (B sequential VAE forward passes), the method stacks all input
   tensors along the batch dim and calls ``vae.batch_encode()`` once.  The text
   encoder similarly processes all prompts in a single forward pass.  The VACE
   context (inactive + reactive latents, mask, reference frames) is assembled
   manually, replicating the logic of three pipeline units:

     - ``WanVideoUnit_NoiseInitializer``  (noise generation — skipped, see below)
     - ``WanVideoUnit_InputVideoEmbedder``  (VAE encode + reference prepend)
     - ``WanVideoUnit_VACE``  (inactive/reactive split, mask rearrange, ref handling)

   **The noise tensor passed to the loss function is a placeholder** —
   ``FlowMatchVideoActionSFTLoss`` generates its own per-sample noise and
   overwrites ``inputs["latents"]`` unconditionally.  This is also true for the
   single-sample path (pipeline units generate noise that is equally discarded).

   **Limitation**: ``_forward_batch`` assumes mask = all-ones (standard VACE SFT)
   and does not set ``first_frame_latents`` (I2V-style conditioning).  It also
   requires all samples in a batch to have consistent spatial/temporal dimensions
   and consistent reference image presence.
"""

import torch
import torch.nn.functional as F
import math
import os
import argparse
import accelerate
import warnings
import numpy as np
from einops import rearrange

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.models.action_dit import ActionDiT, ActionDiTState

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def FlowMatchVideoActionSFTLoss(
    pipe,
    action_dit: ActionDiT,
    action_scheduler,
    action_data: torch.Tensor,
    lambda_video: float = 1.0,
    lambda_action: float = 1.0,
    current_step: int = 0,
    detach_bridge: bool = False,
    decoupled_sampler=None,
    **inputs,
):
    """
    Joint video-action supervised finetuning loss with per-sample timesteps.

    Implements the training objective::

        L = lambda_video(step) * L_video + lambda_action * L_action

    where each sample in the batch draws its own independent video timestep t_v[i]
    and action timestep t_a[i] (UWM-style).

    Timestep & noise handling
    -------------------------
    The scheduler's ``add_noise()`` / ``training_target()`` only accept scalar
    timesteps, so per-sample noise is computed inline:

        σ = sigmas[timestep_ids]            # (B,)
        x_t = (1 - σ) * x_0 + σ * ε        # σ broadcast via view(B,1,1,1,1)
        target = ε - x_0                    # flow matching velocity target

    This function **generates its own noise** and **overwrites inputs["latents"]**,
    so the caller does not need to provide meaningful noise — only ``input_latents``
    (the clean encoded video) matters.

    Per-sample loss weighting
    -------------------------
    Each sample's MSE is weighted by ``scheduler.linear_timesteps_weights[t_id]``
    (a Gaussian-like schedule centered at t=500).  The final loss is the mean of
    per-sample weighted MSE values.  When B=1, a fused ``mse_loss(reduction='mean')``
    fast path avoids materializing the full unreduced (1,C,T,H,W) intermediate.

    Bridge features
    ---------------
    Passed WITH gradients (no detach) by default, so the action loss provides a
    training signal to the video DiT.  Set ``detach_bridge=True`` to block gradients.

    Args:
        pipe: WanVideoPipeline
        action_dit: ActionDiT model
        action_scheduler: FlowMatchScheduler for action stream
        action_data: (B, T_action, action_dim) ground truth actions
        lambda_video: Target weight for video loss
        lambda_action: Weight for action loss
        current_step: Current training step
        detach_bridge: If True, detach bridge features to prevent action loss
            gradients from flowing back to the video DiT
        **inputs: Must contain ``input_latents`` (B, C, T, H, W) clean video latents.
            ``latents`` key is ignored and overwritten.
    """
    # ==================== Common Setup ====================
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    B = inputs["input_latents"].shape[0]

    # Sample per-sample video timesteps
    if decoupled_sampler is not None:
        # Decoupled training: Beta-distributed video timesteps
        _video_t, _action_t = decoupled_sampler.sample_timesteps(
            B, current_step=current_step, device="cpu"
        )
        # Map continuous timesteps to scheduler indices
        num_ts = len(pipe.scheduler.timesteps)
        video_timestep_ids = (_video_t / decoupled_sampler.num_train_timesteps * num_ts).long().clamp(
            min_timestep_boundary, max_timestep_boundary - 1
        )
    else:
        video_timestep_ids = torch.randint(min_timestep_boundary, max_timestep_boundary, (B,))
    video_timesteps = pipe.scheduler.timesteps[video_timestep_ids].to(
        dtype=pipe.torch_dtype, device=pipe.device
    )
    video_sigmas = pipe.scheduler.sigmas[video_timestep_ids].to(
        dtype=pipe.torch_dtype, device=pipe.device
    )

    # Add noise with per-sample sigmas: x_t = (1 - σ) * x_0 + σ * ε
    video_noise = torch.randn_like(inputs["input_latents"])
    sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
    inputs["latents"] = (1 - sigma_bc) * inputs["input_latents"] + sigma_bc * video_noise
    video_training_target = video_noise - inputs["input_latents"]

    if inputs.get("first_frame_latents") is not None:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    use_interleaved = (action_dit.bridge_type == "joint_self_attn" and lambda_action > 0)

    # ==================== Prepare action data ====================
    action_dit_state = None
    action_timesteps = None
    action_timestep_ids = None
    action_training_target = None
    noisy_actions = None

    if lambda_action > 0:
        # Sample per-sample independent action timesteps
        if decoupled_sampler is not None:
            # Decoupled training: use action timesteps from sampler
            num_ts_a = len(action_scheduler.timesteps)
            action_timestep_ids = (_action_t / decoupled_sampler.num_train_timesteps * num_ts_a).long().clamp(0, num_ts_a - 1)
        else:
            action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))
        action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )

        # Ensure action_data is on correct device
        action_data = action_data.to(dtype=pipe.torch_dtype, device=pipe.device)
        if action_data.dim() == 2:
            action_data = action_data.unsqueeze(0)  # (1, T, action_dim)

        # Subsample actions to match video frame count
        T_action = action_data.shape[1]
        T_video_frames = inputs.get("num_frames", 49)
        if T_action > T_video_frames:
            indices = torch.linspace(0, T_action - 1, T_video_frames).long()
            action_data = action_data[:, indices]

        # Add noise with per-sample sigmas
        action_noise = torch.randn_like(action_data)
        action_sigma_bc = action_sigmas.view(B, 1, 1)
        noisy_actions = (1 - action_sigma_bc) * action_data + action_sigma_bc * action_noise
        action_training_target = action_noise - action_data

        if use_interleaved:
            action_dit_state = action_dit.prepare_action_state(
                noisy_actions, action_timesteps,
                use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
                use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
            )

    # ==================== Video Forward ====================
    if use_interleaved:
        video_noise_pred = pipe.model_fn(
            **models, **inputs, timestep=video_timesteps,
            action_dit_state=action_dit_state,
        )
    else:
        bridge_features = []
        video_noise_pred = pipe.model_fn(
            **models, **inputs, timestep=video_timesteps,
            bridge_feature_store=bridge_features,
            bridge_feature_layers=action_dit.bridge_layers_set,
            bridge_feature_detach=detach_bridge,
        )

    # ==================== Video Loss ====================
    _has_first_frame = inputs.get("first_frame_latents") is not None
    if _has_first_frame:
        video_noise_pred_loss = video_noise_pred[:, :, 1:]
        video_training_target = video_training_target[:, :, 1:]
    else:
        video_noise_pred_loss = video_noise_pred

    # MSE weighted by per-sample training weights
    video_tw = pipe.scheduler.linear_timesteps_weights[video_timestep_ids].to(
        dtype=torch.float32, device=pipe.device
    )
    if B == 1:
        loss_video = torch.nn.functional.mse_loss(
            video_noise_pred_loss.float(), video_training_target.float()
        ) * video_tw[0]
    else:
        video_mse_per_sample = torch.nn.functional.mse_loss(
            video_noise_pred_loss.float(), video_training_target.float(), reduction='none'
        ).mean(dim=list(range(1, video_noise_pred_loss.ndim)))  # (B,)
        loss_video = (video_mse_per_sample * video_tw).mean()

    # ==================== Action Loss ====================
    if lambda_action == 0:
        return {
            "loss": lambda_video * loss_video,
            "loss_video": loss_video.detach(),
            "video_weight": lambda_video,
        }

    # Get action noise prediction
    if use_interleaved:
        action_noise_pred = action_dit_state.action_noise_pred
    else:
        assert len(bridge_features) == len(action_dit.bridge_layers), (
            f"Expected {len(action_dit.bridge_layers)} bridge features, got {len(bridge_features)}. "
            f"Check that bridge_layers indices are valid for the video DiT."
        )
        action_noise_pred = action_dit(
            action_tokens=noisy_actions,
            video_features=bridge_features,
            timestep=action_timesteps,
            use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
            use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
        )

    # Action MSE weighted by per-sample training weights
    action_tw = action_scheduler.linear_timesteps_weights[action_timestep_ids].to(
        dtype=torch.float32, device=pipe.device
    )
    if B == 1:
        loss_action = torch.nn.functional.mse_loss(
            action_noise_pred.float(), action_training_target.float()
        ) * action_tw[0]
    else:
        action_mse_per_sample = torch.nn.functional.mse_loss(
            action_noise_pred.float(), action_training_target.float(), reduction='none'
        ).mean(dim=list(range(1, action_noise_pred.ndim)))  # (B,)
        loss_action = (action_mse_per_sample * action_tw).mean()

    # ==================== Combined Loss ====================
    video_weight = lambda_video

    if video_weight == 0:
        loss = lambda_action * loss_action
    else:
        loss = video_weight * loss_video + lambda_action * loss_action

    # Unweighted losses for scale monitoring
    loss_video_uw = (loss_video / (video_tw.mean() + 1e-8)).detach()
    loss_action_uw = (loss_action / (action_tw.mean() + 1e-8)).detach()

    return {
        "loss": loss,
        "loss_video": loss_video.detach(),
        "loss_action": loss_action.detach(),
        "video_weight": video_weight,
        "loss_video_unweighted": loss_video_uw,
        "loss_action_unweighted": loss_action_uw,
        "loss_scale_ratio": (loss_video.detach() / (loss_action.detach() + 1e-8)),
    }


class VideoActionTrainingModule(DiffusionTrainingModule):
    """
    Training module for joint video + action generation.

    Supports multiple training strategies:
    - video-only: lambda_action=0, trains only VACE/DiT
    - joint: both video and action trained together
    """

    def __init__(
        self,
        # Video model config (same as WanTrainingModule)
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        # Action model config (new)
        action_dim=7,
        action_dit_dim=768,
        action_dit_ffn_dim=3072,
        action_dit_num_heads=12,
        action_dit_num_layers=8,
        action_dit_bridge_layers="5,10,15,20,25,29",
        video_dim=1536,
        lambda_video=1.0,
        lambda_action=1.0,
        bridge_type="cross_attn",
        action_stats_path=None,
        action_lr=None,
    ):
        super().__init__()

        # Warning
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is detected as disabled. "
                "The training framework will forcibly enable gradient checkpointing."
            )
            use_gradient_checkpointing = True

        # Load video models (same as WanTrainingModule)
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths,
            fp8_models=fp8_models, offload_models=offload_models, device=device
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode for video model
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # Create ActionDiT
        bridge_layers = tuple(int(x) for x in action_dit_bridge_layers.split(","))
        self.action_dit = ActionDiT(
            action_dim=action_dim,
            dim=action_dit_dim,
            ffn_dim=action_dit_ffn_dim,
            num_heads=action_dit_num_heads,
            num_layers=action_dit_num_layers,
            video_dim=video_dim,
            bridge_layers=bridge_layers,
            bridge_type=bridge_type,
        ).to(dtype=torch.bfloat16)

        # Load action normalization stats into ActionDiT buffers
        if action_stats_path is not None and os.path.exists(action_stats_path):
            stats = np.load(action_stats_path, allow_pickle=True).item()
            self.action_dit.action_mean.copy_(torch.from_numpy(stats["mean"].astype(np.float32)))
            std = np.maximum(stats["std"].astype(np.float32), 1e-3)
            self.action_dit.action_std.copy_(torch.from_numpy(std))
            print(f"Loaded action stats into ActionDiT buffers from {action_stats_path}")

        # Print ActionDiT param count
        total_params = sum(p.numel() for p in self.action_dit.parameters())
        print(f"ActionDiT created: {total_params / 1e6:.1f}M params, dtype=bfloat16")

        # Action scheduler (independent from video)
        self.action_scheduler = FlowMatchScheduler("Wan")
        self.action_scheduler.set_timesteps(1000, training=True)

        # Store configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.lambda_video = lambda_video
        self.lambda_action = lambda_action
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.bridge_type = bridge_type
        self.action_lr = action_lr
        self._model_logger = None
        self._last_loss_components = {}

    def set_step_counter(self, logger):
        """Store a reference to ModelLogger for step synchronization."""
        self._model_logger = logger

    @property
    def current_step(self):
        """Read current step from the logger if available, else 0."""
        if self._model_logger is not None:
            return self._model_logger.num_steps
        return 0

    def validate_during_training(self, val_sample, step, wandb_run=None, prefix="val"):
        """Run validation on a single sample during training.

        Generates video + actions and computes metrics vs ground truth.
        Logs results to wandb if available.
        """
        from joint_inference import generate_video_and_actions, make_schedule

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                gt_actions = val_sample["action_trajectory"]
                if isinstance(gt_actions, torch.Tensor):
                    gt_actions = gt_actions.numpy()

                num_frames = len(val_sample["video"])
                height = val_sample["video"][0].size[1]
                width = val_sample["video"][0].size[0]

                task_name = val_sample.get("task_name", "unknown")
                episode_path = val_sample.get("episode_path", "")
                start_frame = val_sample.get("start_frame", -1)
                end_frame = val_sample.get("end_frame", -1)
                episode_length = val_sample.get("episode_length", -1)
                prompt_text = val_sample["prompt"]

                schedule = make_schedule("sync", num_steps=20, shift=5.0)
                try:
                    video, generated_denorm = generate_video_and_actions(
                        pipe=self.pipe,
                        action_dit=self.action_dit,
                        schedule=schedule,
                        prompt=prompt_text,
                        vace_video=val_sample.get("vace_video"),
                        vace_reference_image=val_sample.get("vace_reference_image"),
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        seed=42,
                        cfg_scale=1.0,
                        tiled=True,
                    )
                except Exception as e:
                    print(f"  [{prefix} step {step}] WARNING: video generation failed: {e}")
                    print(f"    Skipping video metrics and logging for this step.")
                    return

                # Denormalize ground truth for metric comparison
                action_mean = self.action_dit.action_mean.float().cpu().numpy()
                action_std = self.action_dit.action_std.float().cpu().numpy()
                gt_denorm = gt_actions * action_std + action_mean

                # Compute action metrics
                action_mse = np.mean((generated_denorm - gt_denorm) ** 2)
                action_mae = np.mean(np.abs(generated_denorm - gt_denorm))

                # Compute video metrics
                from eval_robotwin import compute_video_metrics
                vid_metrics = compute_video_metrics(video, val_sample["video"])

                print(f"  [{prefix} step {step}] task={task_name} "
                      f"action_MSE: {action_mse:.6f}, action_MAE: {action_mae:.6f}, "
                      f"video_MSE: {vid_metrics['video_mse']:.2f}, "
                      f"PSNR: {vid_metrics['video_psnr']:.2f}, "
                      f"SSIM: {vid_metrics['video_ssim']:.4f}, "
                      f"LPIPS: {vid_metrics['video_lpips']:.4f}")
                print(f"    episode: {episode_path} frames [{start_frame}:{end_frame}]/{episode_length}")
                print(f"    prompt: {prompt_text}")

                # Log to wandb
                if wandb_run is not None:
                    import wandb
                    log_dict = {
                        f"{prefix}/action_mse": action_mse,
                        f"{prefix}/action_mae": action_mae,
                        f"{prefix}/video_mse": vid_metrics["video_mse"],
                        f"{prefix}/video_psnr": vid_metrics["video_psnr"],
                        f"{prefix}/video_ssim": vid_metrics["video_ssim"],
                        f"{prefix}/video_lpips": vid_metrics["video_lpips"],
                        f"{prefix}/step": step,
                    }

                    try:
                        import imageio.v2 as imageio
                        import tempfile

                        # Generated video
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                            imageio.mimwrite(f.name, video, fps=15, quality=6)
                            log_dict[f"{prefix}/video_generated"] = wandb.Video(
                                f.name, fps=15, format="mp4")

                        # Ground truth target video
                        gt_frames = val_sample["video"]
                        gt_np = [np.array(frame) for frame in gt_frames]
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                            imageio.mimwrite(f.name, gt_np, fps=15, quality=6)
                            log_dict[f"{prefix}/video_gt_target"] = wandb.Video(
                                f.name, fps=15, format="mp4")
                    except Exception:
                        pass

                    wandb_run.log(log_dict, step=step)

        finally:
            if was_training:
                self.train()
            self.pipe.scheduler.set_timesteps(1000, training=True)
            self.pipe.load_models_to_device(list(self.pipe.in_iteration_models))
            torch.cuda.empty_cache()

    def compute_val_losses(self, val_dataset, step, wandb_run=None, prefix="val",
                           max_samples=0):
        """Compute average flow matching loss over validation samples."""
        import random as _random
        from collections import defaultdict

        was_training = self.training
        self.eval()

        n = len(val_dataset)
        if max_samples > 0 and n > max_samples:
            indices = sorted(_random.sample(range(n), max_samples))
        else:
            indices = list(range(n))

        all_components = defaultdict(list)

        try:
            with torch.no_grad():
                for idx in indices:
                    sample = val_dataset[idx]
                    loss = self.forward(sample)
                    all_components["loss"].append(loss.item())
                    for k, v in self._last_loss_components.items():
                        if isinstance(v, torch.Tensor):
                            all_components[k].append(v.item())
                        elif isinstance(v, (int, float)):
                            all_components[k].append(v)
                    self._last_loss_components = {}

            # Average
            avg = {k: sum(v) / len(v) for k, v in all_components.items() if v}

            print(f"  [{prefix} step {step}] {len(indices)}/{n} samples: "
                  f"loss={avg.get('loss', 0):.6f}, "
                  f"loss_video={avg.get('loss_video', 0):.6f}, "
                  f"loss_action={avg.get('loss_action', 0):.6f}")

            if wandb_run is not None:
                log_dict = {f"{prefix}/{k}": v for k, v in avg.items()}
                log_dict[f"{prefix}/num_samples"] = len(indices)
                wandb_run.log(log_dict, step=step)

            return avg
        finally:
            if was_training:
                self.train()

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        """Extended to handle action_trajectory data."""
        for extra_input in extra_inputs:
            if extra_input == "vace_reference_image":
                val = data[extra_input]
                inputs_shared[extra_input] = val[0] if val is not None else None
            elif extra_input == "action_trajectory":
                pass
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
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
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if isinstance(data, list):
            return self._forward_batch(data)
        return self._forward_single(data, inputs)

    def _forward_single(self, data, inputs=None):
        """Original single-sample forward path."""
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        # Extract action data before running pipeline units
        action_data = data.get("action_trajectory", None)
        if self.lambda_action > 0 and action_data is None:
            raise ValueError(
                "lambda_action > 0 but no action_trajectory in data. "
                "Either set --lambda_action 0 or provide action data."
            )
        if action_data is not None:
            if isinstance(action_data, np.ndarray):
                action_data = torch.from_numpy(action_data)
            action_data = action_data.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            if action_data.dim() == 2:
                action_data = action_data.unsqueeze(0)

        # Run pipeline units (data preprocessing)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        inputs_shared, inputs_posi, inputs_nega = inputs

        # Compute joint loss
        result = FlowMatchVideoActionSFTLoss(
            pipe=self.pipe,
            action_dit=self.action_dit,
            action_scheduler=self.action_scheduler,
            action_data=action_data,
            lambda_video=self.lambda_video,
            lambda_action=self.lambda_action,
            current_step=self.current_step,
            detach_bridge=(self.bridge_type == "cross_attn_detach"),
            decoupled_sampler=getattr(self, "decoupled_sampler", None),
            **inputs_shared,
            **inputs_posi,
        )
        self._last_loss_components = {k: v for k, v in result.items() if k != "loss"}
        return result["loss"]

    def _forward_batch(self, data_list):
        """Batched forward: encode all samples in one VAE / text-encoder pass."""
        B = len(data_list)
        pipe = self.pipe

        # 1. Validate shapes
        height, width, num_frames = pipe.check_resize_height_width(
            data_list[0]["video"][0].size[1],
            data_list[0]["video"][0].size[0],
            len(data_list[0]["video"]),
        )

        # 2. CPU preprocessing: PIL → tensors, collect prompts and actions
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
                raise ValueError(
                    "lambda_action > 0 but no action_trajectory in data. "
                    "Either set --lambda_action 0 or provide action data."
                )
            if action is not None:
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action)
                action = action.to(dtype=pipe.torch_dtype, device=pipe.device).unsqueeze(0)
            all_actions.append(action)

        ref_flags = [r is not None for r in all_ref_images]
        if any(ref_flags) and not all(ref_flags):
            raise ValueError(
                "Mixed reference images in batch: some samples have vace_reference_image "
                "and others don't. All samples in a batch must be consistent."
            )
        has_ref = ref_flags[0]

        # 3. Batch text encoding
        pipe.load_models_to_device(["text_encoder"])
        ids, mask = pipe.tokenizer(all_prompts, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(pipe.device), mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = pipe.text_encoder(ids, mask)  # (B, seq_len, D)
        for i, v in enumerate(seq_lens):
            context[i, v:] = 0

        # 4. Batch VAE encoding
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

        # 5. VACE context assembly (only when model has a VACE module)
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
                vace_mask_latents = torch.cat(
                    [
                        torch.zeros(
                            B, vace_mask_latents.shape[1], ref_f,
                            vace_mask_latents.shape[3], vace_mask_latents.shape[4],
                            dtype=pipe.torch_dtype, device=pipe.device,
                        ),
                        vace_mask_latents,
                    ],
                    dim=2,
                )

            vace_context = torch.cat([vace_video_latents, vace_mask_latents], dim=1)

        # 5b. Activate TI2V-5B separated timestep conditioning
        is_ti2v = getattr(pipe.dit, 'fuse_vae_embedding_in_latents', False)
        first_frame_latents = None
        num_clean_prefix = 0
        if is_ti2v and has_ref:
            if has_ref:
                first_frame_latents = ref_latents[:, :, 0:1].clone()
                num_clean_prefix += ref_latents.shape[2]

        # 6. Assemble batched inputs for the loss function
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
        batched_posi = {"context": context}

        # 8. Action data
        if all_actions[0] is not None:
            action_data = torch.cat(all_actions, dim=0)
        else:
            action_data = None

        # 9. Compute joint loss
        result = FlowMatchVideoActionSFTLoss(
            pipe=self.pipe,
            action_dit=self.action_dit,
            action_scheduler=self.action_scheduler,
            action_data=action_data,
            lambda_video=self.lambda_video,
            lambda_action=self.lambda_action,
            current_step=self.current_step,
            detach_bridge=(self.bridge_type == "cross_attn_detach"),
            decoupled_sampler=getattr(self, "decoupled_sampler", None),
            **batched_shared,
            **batched_posi,
        )
        self._last_loss_components = {k: v for k, v in result.items() if k != "loss"}
        return result["loss"]

    def trainable_modules(self):
        """Return all trainable parameters (video + action).

        When action_lr is set, returns parameter groups with separate LRs.
        When lambda_action == 0 (video-only), ActionDiT params are excluded
        from the optimizer and checkpoint.
        """
        if self.action_lr is not None:
            groups = []
            if self.lambda_action > 0:
                groups.append({
                    "params": list(self.action_dit.parameters()),
                    "lr": self.action_lr,
                })
            else:
                self.action_dit.requires_grad_(False)
            video_params = [p for p in self.pipe.parameters() if p.requires_grad]
            if video_params:
                groups.append({"params": video_params})
            return groups

        trainable = []
        if self.lambda_action > 0:
            trainable.extend(self.action_dit.parameters())
        else:
            self.action_dit.requires_grad_(False)
        for p in self.pipe.parameters():
            if p.requires_grad:
                trainable.append(p)
        return trainable

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        """Export trainable state dict with proper key handling."""
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items()
                      if name in trainable_param_names}
        if self.lambda_action > 0:
            for buf_name in ("action_mean", "action_std"):
                full_name = f"action_dit.{buf_name}"
                if full_name not in state_dict:
                    buf = getattr(self.action_dit, buf_name, None)
                    if buf is not None:
                        state_dict[full_name] = buf
        result = {}
        for name, param in state_dict.items():
            if name.startswith("action_dit."):
                result[name] = param
            elif remove_prefix and name.startswith(remove_prefix):
                result[name[len(remove_prefix):]] = param
            else:
                result[name] = param
        return result

def video_action_parser():
    """Argument parser for video-action joint training."""
    parser = argparse.ArgumentParser(description="Video-Action joint training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)

    # Standard video model args
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--audio_processor_path", type=str, default=None)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true")

    # Action-specific args
    parser.add_argument("--action_dim", type=int, default=7,
                        help="Dimension of action vector (e.g., 7 for 6-DoF + gripper)")
    parser.add_argument("--action_dit_dim", type=int, default=768,
                        help="Hidden dimension of ActionDiT")
    parser.add_argument("--action_dit_ffn_dim", type=int, default=3072,
                        help="FFN dimension of ActionDiT")
    parser.add_argument("--action_dit_num_heads", type=int, default=12,
                        help="Number of attention heads in ActionDiT")
    parser.add_argument("--action_dit_num_layers", type=int, default=8,
                        help="Number of transformer layers in ActionDiT")
    parser.add_argument("--action_dit_bridge_layers", type=str, default="3,7,11,15,19,23,26,29",
                        help="Comma-separated video DiT layer indices for per-block bridge attention. "
                             "Must have exactly action_dit_num_layers entries (1:1 mapping).")
    parser.add_argument("--video_dim", type=int, default=1536,
                        help="Hidden dimension of the video DiT (1536 for VACE-1.3B, 3072 for TI2V-5B)")
    parser.add_argument("--lambda_video", type=float, default=1.0,
                        help="Weight for video generation loss")
    parser.add_argument("--lambda_action", type=float, default=1.0,
                        help="Weight for action prediction loss")
    parser.add_argument("--bridge_type", type=str, default="cross_attn",
                        choices=["cross_attn", "cross_attn_detach", "joint_self_attn"],
                        help="Bridge attention type between video and action streams.")
    parser.add_argument("--action_lr", type=float, default=None,
                        help="Separate learning rate for ActionDiT. "
                             "Defaults to --learning_rate if not set.")
    parser.add_argument("--action_stats_path", type=str, default=None,
                        help="Path to action normalization stats file (.npy). "
                             "Defaults to <dataset_base_path>/action_stats.npy if not set.")

    # Backbone selection
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["vace", "ti2v"],
                        help="Video backbone: 'vace' (Wan2.1-VACE-1.3B/14B) or 'ti2v' (Wan2.2-TI2V-5B).")

    # HDF5 dataset options
    parser.add_argument("--dataset_type", type=str, default="robotwin",
                        choices=["robotwin", "robotwin_multitask"],
                        help="Dataset type: 'robotwin' (single-task RoboTwin 2.0), "
                             "'robotwin_multitask' (multi-task RoboTwin 2.0)")
    parser.add_argument("--hdf5_data_root", type=str, default=None,
                        help="Root directory with episode HDF5 files (RoboTwin single-task)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of episodes for validation split")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Task name for prompt generation (e.g., 'push box')")
    parser.add_argument("--target_camera", type=str, default="head_camera",
                        help="Target camera name for robot view (RoboTwin only)")
    parser.add_argument("--window_stride", type=int, default=1,
                        help="Stride for exhaustive temporal window enumeration.")
    parser.add_argument("--multiview", default=False, action="store_true",
                        help="Use 2x2 multi-view grid (head + third_view + left + right).")
    parser.add_argument("--robot", type=str, default="aloha-agilex",
                        help="Robot name (e.g., 'aloha-agilex')")
    parser.add_argument("--variant", type=str, default="clean_50",
                        help="Data variant for training (e.g., 'clean_50')")

    # Multi-task RoboTwin options
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Top-level RoboTwin dataset directory containing per-task folders. "
                             "Required for --dataset_type robotwin_multitask")
    parser.add_argument("--val_variant", type=str, default=None,
                        help="Data variant for validation (e.g., 'randomized_500').")
    parser.add_argument("--train_tasks", type=str, default=None,
                        help="Comma-separated training task names. "
                             "Overrides the default ROBOTWIN_TRAIN_TASKS.")
    parser.add_argument("--holdout_tasks", type=str, default=None,
                        help="Comma-separated holdout task names for zero-shot eval. "
                             "Defaults to ROBOTWIN_HOLDOUT_TASKS (8 tasks)")

    # Validation during training
    parser.add_argument("--val_steps", type=int, default=None,
                        help="Run validation every N steps")
    parser.add_argument("--video_log_steps", type=int, default=None,
                        help="Log generated videos to wandb every N steps.")
    parser.add_argument("--max_val_samples", type=int, default=500,
                        help="Max samples for val loss computation.")

    return parser


if __name__ == "__main__":
    parser = video_action_parser()
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters
            )
        ],
    )

    # Parse train/holdout tasks (robotwin_multitask only)
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
            raise ValueError("--dataset_dir is required for --dataset_type robotwin_multitask")
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
        action_stats_path = args.action_stats_path
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
        action_stats_path = args.action_stats_path

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
        # Action-specific
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
        action_stats_path=action_stats_path,
    )

    # Load action stats directly from dataset into model's ActionDiT buffers
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

    # Set up validation callback
    val_callback = None
    if args.val_steps is not None:
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
            val_id_dataset = MultiTaskRoboTwinDataset(
                dataset_dir=args.dataset_dir,
                robot=args.robot,
                variant=args.variant,
                tasks=train_tasks,
                val_ratio=args.val_ratio,
                num_val_samples=0,
                **_mt_val_common,
            )
            val_ood_dataset = MultiTaskRoboTwinDataset(
                dataset_dir=args.dataset_dir,
                robot=args.robot,
                variant=val_variant,
                tasks=holdout_tasks,
                val_ratio=1.0,
                num_val_samples=5,
                **_mt_val_common,
            )
            _max_val = args.max_val_samples
            _val_steps = args.val_steps
            _video_log_steps = args.video_log_steps if args.video_log_steps is not None else args.val_steps
            def val_callback(step):
                if step % _val_steps == 0:
                    model.compute_val_losses(
                        val_id_dataset, step, model_logger.wandb_run,
                        prefix="val_id", max_samples=_max_val)
                    model.compute_val_losses(
                        val_ood_dataset, step, model_logger.wandb_run,
                        prefix="val_ood")
                if step % _video_log_steps == 0:
                    model.validate_during_training(
                        val_id_dataset[0], step, model_logger.wandb_run, prefix="val_id")
                    model.validate_during_training(
                        val_ood_dataset[0], step, model_logger.wandb_run, prefix="val_ood")
        else:
            from video_action_dataset import RoboTwinDataset
            val_dataset = RoboTwinDataset(
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
            _max_val = args.max_val_samples
            _val_steps = args.val_steps
            _video_log_steps = args.video_log_steps if args.video_log_steps is not None else args.val_steps
            def val_callback(step):
                if step % _val_steps == 0:
                    model.compute_val_losses(
                        val_dataset, step, model_logger.wandb_run,
                        prefix="val", max_samples=_max_val)
                if step % _video_log_steps == 0:
                    model.validate_during_training(
                        val_dataset[0], step, model_logger.wandb_run)

    # Callback is invoked every gcd(val_steps, video_log_steps)
    callback_interval = args.val_steps
    if args.video_log_steps is not None:
        callback_interval = math.gcd(args.val_steps, args.video_log_steps)
    launch_training_task(
        accelerator, dataset, model, model_logger, args=args,
        val_callback=val_callback, val_steps=callback_interval,
    )
