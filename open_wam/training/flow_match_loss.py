"""Standalone flow matching loss for joint video-action training.

Replaces the legacy ``FlowMatchVideoActionSFTLoss`` function from
``examples/wanvideo/wam/train_video_action.py`` with a clean,
self-contained implementation that has no sys.path manipulation or
legacy module imports.

The loss implements the UWM-style training objective:

    L = lambda_video * L_video + lambda_action * L_action

where each sample draws independent per-sample timesteps for video
and action streams. Flow matching velocity target: v = epsilon - x_0.

Usage:
    loss_fn = FlowMatchVideoActionLoss(lambda_video=1.0, lambda_action=1.0)
    result = loss_fn(
        pipe=pipe,
        action_dit=action_dit,
        action_scheduler=action_scheduler,
        action_data=action_tensor,
        **pipeline_inputs,
    )
"""

from typing import Optional, Union

import torch
import torch.nn.functional as F

from open_wam.models.architectures.base import BaseWAMArchitecture


class FlowMatchVideoActionLoss:
    """Joint video-action flow matching loss.

    Implements per-sample timestep sampling, noise injection, and weighted
    MSE loss computation for both video and action modalities.

    Args:
        lambda_video: Weight for video loss term.
        lambda_action: Weight for action loss term.
        detach_bridge: If True, detach bridge features to prevent action
            gradients from flowing to the video DiT.
    """

    def __init__(
        self,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        detach_bridge: bool = False,
    ):
        self.lambda_video = lambda_video
        self.lambda_action = lambda_action
        self.detach_bridge = detach_bridge

    def __call__(
        self,
        pipe,
        action_dit=None,
        action_scheduler=None,
        action_data: Optional[torch.Tensor] = None,
        current_step: int = 0,
        decoupled_sampler=None,
        architecture: Optional[BaseWAMArchitecture] = None,
        **inputs,
    ) -> dict:
        """Compute joint video-action flow matching loss.

        Args:
            pipe: WanVideoPipeline with scheduler and model_fn.
            action_dit: Raw ActionDiT model (legacy, prefer ``architecture``).
            action_scheduler: FlowMatchScheduler for action stream.
            action_data: (B, T_action, action_dim) ground truth actions.
            current_step: Current training step (for decoupled warmup).
            decoupled_sampler: Optional DecoupledFlowMatchLoss for
                Beta-distributed video timesteps.
            architecture: WAM architecture (preferred over raw action_dit).
            **inputs: Pipeline inputs, must include ``input_latents``
                (B, C, T, H, W) clean video latents.

        Returns:
            dict with keys: loss, loss_video, loss_action, video_weight,
            loss_video_unweighted, loss_action_unweighted, loss_scale_ratio.
        """
        # Resolve architecture from action_dit for backward compat
        if architecture is None and action_dit is not None:
            from open_wam.models.architectures.dual_system import DualSystemArchitecture
            architecture = DualSystemArchitecture(cfg=None)
            architecture.action_dit = action_dit
        max_timestep_boundary = int(
            inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps)
        )
        min_timestep_boundary = int(
            inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps)
        )
        B = inputs["input_latents"].shape[0]

        # --- Sample video timesteps ---
        video_timestep_ids = self._sample_video_timesteps(
            B, min_timestep_boundary, max_timestep_boundary,
            decoupled_sampler, current_step, pipe,
        )
        video_timesteps = pipe.scheduler.timesteps[video_timestep_ids].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        video_sigmas = pipe.scheduler.sigmas[video_timestep_ids].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )

        # --- Add video noise: x_t = (1 - sigma) * x_0 + sigma * epsilon ---
        video_noise = torch.randn_like(inputs["input_latents"])
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        inputs["latents"] = (1 - sigma_bc) * inputs["input_latents"] + sigma_bc * video_noise
        video_target = video_noise - inputs["input_latents"]

        if inputs.get("first_frame_latents") is not None:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

        # --- Prepare action data ---
        action_dit_state, noisy_actions, action_target, action_timesteps, action_timestep_ids = (
            self._prepare_actions(
                B, architecture, action_scheduler, action_data,
                decoupled_sampler, current_step, pipe, inputs,
            )
        )

        # --- Video forward pass ---
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        use_interleaved = (
            architecture.is_interleaved and self.lambda_action > 0
        )

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
                bridge_feature_layers=set(architecture.bridge_layers),
                bridge_feature_detach=self.detach_bridge,
            )

        # --- Video loss ---
        loss_video = self._compute_video_loss(
            video_noise_pred, video_target, video_timestep_ids,
            pipe, inputs, B,
        )

        if self.lambda_action == 0:
            return {
                "loss": self.lambda_video * loss_video,
                "loss_video": loss_video.detach(),
                "video_weight": self.lambda_video,
            }

        # --- Action loss ---
        if use_interleaved:
            action_noise_pred = action_dit_state.action_noise_pred
        else:
            # Use architecture interface: prepare → feed bridge features → extract
            action_state = architecture.prepare_action_tokens(
                noisy_actions, action_timesteps,
                use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
                use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
            )
            sorted_layers = sorted(architecture.bridge_layers)
            for layer_idx, layer_id in enumerate(sorted_layers):
                if layer_idx < len(bridge_features):
                    _, action_state = architecture.on_dit_block(
                        layer_id, bridge_features[layer_idx], action_state
                    )
            action_noise_pred = architecture.extract_action_prediction(action_state)

        loss_action = self._compute_action_loss(
            action_noise_pred, action_target, action_timestep_ids,
            action_scheduler, pipe, B,
        )

        # --- Combined loss ---
        if self.lambda_video == 0:
            loss = self.lambda_action * loss_action
        else:
            loss = self.lambda_video * loss_video + self.lambda_action * loss_action

        # Unweighted for monitoring
        video_tw = pipe.scheduler.linear_timesteps_weights[video_timestep_ids].to(
            dtype=torch.float32, device=pipe.device
        )
        action_tw = action_scheduler.linear_timesteps_weights[action_timestep_ids].to(
            dtype=torch.float32, device=pipe.device
        )
        loss_video_uw = (loss_video / (video_tw.mean() + 1e-8)).detach()
        loss_action_uw = (loss_action / (action_tw.mean() + 1e-8)).detach()

        return {
            "loss": loss,
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
            "video_weight": self.lambda_video,
            "loss_video_unweighted": loss_video_uw,
            "loss_action_unweighted": loss_action_uw,
            "loss_scale_ratio": (loss_video.detach() / (loss_action.detach() + 1e-8)),
        }

    def _sample_video_timesteps(
        self, B, min_b, max_b, decoupled_sampler, current_step, pipe,
    ):
        """Sample per-sample video timestep indices."""
        if decoupled_sampler is not None:
            video_t, self._decoupled_action_t = decoupled_sampler.sample_timesteps(
                B, current_step=current_step, device="cpu"
            )
            num_ts = len(pipe.scheduler.timesteps)
            return (video_t / decoupled_sampler.num_train_timesteps * num_ts).long().clamp(min_b, max_b - 1)
        else:
            self._decoupled_action_t = None
            return torch.randint(min_b, max_b, (B,))

    def _prepare_actions(
        self, B, architecture, action_scheduler, action_data,
        decoupled_sampler, current_step, pipe, inputs,
    ):
        """Prepare noisy actions, targets, and optional interleaved state."""
        if self.lambda_action == 0:
            return None, None, None, None, None

        # Sample action timesteps
        if decoupled_sampler is not None and self._decoupled_action_t is not None:
            num_ts_a = len(action_scheduler.timesteps)
            action_timestep_ids = (
                self._decoupled_action_t / decoupled_sampler.num_train_timesteps * num_ts_a
            ).long().clamp(0, num_ts_a - 1)
        else:
            action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))

        action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )

        action_data = action_data.to(dtype=pipe.torch_dtype, device=pipe.device)
        if action_data.dim() == 2:
            action_data = action_data.unsqueeze(0)

        # Subsample actions to match video frames
        T_action = action_data.shape[1]
        T_video_frames = inputs.get("num_frames", 49)
        if T_action > T_video_frames:
            indices = torch.linspace(0, T_action - 1, T_video_frames).long()
            action_data = action_data[:, indices]

        # Add noise
        action_noise = torch.randn_like(action_data)
        sigma_bc = action_sigmas.view(B, 1, 1)
        noisy_actions = (1 - sigma_bc) * action_data + sigma_bc * action_noise
        action_target = action_noise - action_data

        # Interleaved state for joint_self_attn — the pipe.model_fn needs
        # the internal dit_state from the architecture's ActionState.
        action_dit_state = None
        if architecture.is_interleaved:
            action_state = architecture.prepare_action_tokens(
                noisy_actions, action_timesteps,
                use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
                use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
            )
            action_dit_state = action_state.extra.get("dit_state")

        return action_dit_state, noisy_actions, action_target, action_timesteps, action_timestep_ids

    def _compute_video_loss(
        self, noise_pred, target, timestep_ids, pipe, inputs, B,
    ):
        """Compute per-sample weighted video MSE loss."""
        if inputs.get("first_frame_latents") is not None:
            noise_pred = noise_pred[:, :, 1:]
            target = target[:, :, 1:]

        tw = pipe.scheduler.linear_timesteps_weights[timestep_ids].to(
            dtype=torch.float32, device=pipe.device
        )
        if B == 1:
            return F.mse_loss(noise_pred.float(), target.float()) * tw[0]

        per_sample = F.mse_loss(
            noise_pred.float(), target.float(), reduction="none"
        ).mean(dim=list(range(1, noise_pred.ndim)))
        return (per_sample * tw).mean()

    def _compute_action_loss(
        self, noise_pred, target, timestep_ids, scheduler, pipe, B,
    ):
        """Compute per-sample weighted action MSE loss."""
        tw = scheduler.linear_timesteps_weights[timestep_ids].to(
            dtype=torch.float32, device=pipe.device
        )
        if B == 1:
            return F.mse_loss(noise_pred.float(), target.float()) * tw[0]

        per_sample = F.mse_loss(
            noise_pred.float(), target.float(), reduction="none"
        ).mean(dim=list(range(1, noise_pred.ndim)))
        return (per_sample * tw).mean()
