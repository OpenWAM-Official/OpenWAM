"""Standalone flow matching loss for joint video-action training.

Standalone flow matching loss for the UWM-style training objective:

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

from typing import Optional

import torch
import torch.nn.functional as F

from openwam.model.action_model.action_repr.base import BaseActionRepresentation
from openwam.model.base import BaseWAMArchitecture


class FlowMatchVideoActionLoss:
    """Joint video-action flow matching loss.

    Implements per-sample timestep sampling, noise injection, and weighted
    MSE loss computation for both video and action modalities.

    Supports padding masks for variable-length batches:
      - video_is_pad: (B, T) bool mask, True for padded video frames.
      - action_is_pad: (B, T_action) bool mask, True for padded action steps.

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
        action_repr: Optional[BaseActionRepresentation] = None,
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
                Optional: ``video_is_pad`` (B, T) and ``action_is_pad``
                (B, T_action) boolean padding masks.

        Returns:
            dict with keys: loss, loss_video, loss_action.
        """
        # Resolve architecture from action_dit for backward compat
        if architecture is None and action_dit is not None:
            from openwam.model.dual_system import DualSystemArchitecture

            architecture = DualSystemArchitecture(cfg=None)
            architecture.action_dit = action_dit
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
        B = inputs["input_latents"].shape[0]

        # --- Sample video timesteps ---
        video_timestep_ids = self._sample_video_timesteps(
            B,
            min_timestep_boundary,
            max_timestep_boundary,
            decoupled_sampler,
            current_step,
            pipe,
        )
        video_timesteps = pipe.scheduler.timesteps[video_timestep_ids].to(dtype=pipe.torch_dtype, device=pipe.device)
        video_sigmas = pipe.scheduler.sigmas[video_timestep_ids].to(dtype=pipe.torch_dtype, device=pipe.device)

        # --- Add video noise: x_t = (1 - sigma) * x_0 + sigma * epsilon ---
        video_noise = torch.randn_like(inputs["input_latents"])
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        inputs["latents"] = (1 - sigma_bc) * inputs["input_latents"] + sigma_bc * video_noise
        video_target = video_noise - inputs["input_latents"]

        if inputs.get("first_frame_latents") is not None:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

        # --- Prepare action data ---
        ap = self._prepare_actions(
            B,
            architecture,
            action_scheduler,
            action_data,
            decoupled_sampler,
            current_step,
            pipe,
            inputs,
            action_repr=action_repr,
        )
        action_state = ap["action_state"]
        noisy_actions = ap["noisy_actions"]
        action_target = ap["action_target"]
        action_timesteps = ap["action_timesteps"]
        action_timestep_ids = ap["action_timestep_ids"]

        # --- Video forward pass ---
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        use_interleaved = architecture.is_interleaved and self.lambda_action > 0

        if use_interleaved:
            # Build model_fn kwargs based on architecture state
            interleaved_kwargs = {}
            if action_state is not None:
                extra = action_state.extra
                if "dit_state" in extra:
                    interleaved_kwargs["action_dit_state"] = extra["dit_state"]
                if "moe_state" in extra:
                    interleaved_kwargs["moe_expert_state"] = extra["moe_state"]
                if "shared_backbone_state" in extra:
                    interleaved_kwargs["shared_backbone_state"] = extra["shared_backbone_state"]
            video_noise_pred = pipe.model_fn(
                **models,
                **inputs,
                timestep=video_timesteps,
                **interleaved_kwargs,
            )
        else:
            bridge_features = []
            video_noise_pred = pipe.model_fn(
                **models,
                **inputs,
                timestep=video_timesteps,
                bridge_feature_store=bridge_features,
                bridge_feature_layers=set(architecture.bridge_layers),
                bridge_feature_detach=self.detach_bridge,
            )

        # --- Video loss ---
        video_is_pad = inputs.get("video_is_pad", None)  # (B, T) or None
        loss_video = self._compute_video_loss(
            video_noise_pred,
            video_target,
            video_timestep_ids,
            pipe,
            inputs,
            B,
            video_is_pad=video_is_pad,
        )

        if self.lambda_action == 0:
            return {
                "loss": self.lambda_video * loss_video,
                "loss_video": self.lambda_video * loss_video.detach(),
                "loss_action": torch.tensor(0.0, device=loss_video.device),
            }

        # --- Action loss ---
        if use_interleaved:
            extra = action_state.extra
            if "dit_state" in extra:
                action_noise_pred = extra["dit_state"].action_noise_pred
            elif "moe_state" in extra:
                action_noise_pred = extra["moe_state"].action_noise_pred
            elif "shared_backbone_state" in extra:
                action_noise_pred = extra["shared_backbone_state"].action_noise_pred
            else:
                raise RuntimeError("Interleaved architecture did not produce action predictions")
        else:
            # Use architecture interface: prepare → feed bridge features → extract
            action_state = architecture.prepare_action_tokens(
                noisy_actions,
                action_timesteps,
                use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
                use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
                proprio_state=inputs.get("proprio_state", None),
            )
            sorted_layers = sorted(architecture.bridge_layers)
            for layer_idx, layer_id in enumerate(sorted_layers):
                if layer_idx < len(bridge_features):
                    _, action_state = architecture.on_dit_block(layer_id, bridge_features[layer_idx], action_state)
            action_noise_pred = architecture.extract_action_prediction(action_state)

        action_is_pad = inputs.get("action_is_pad", None)  # (B, T_action) or None
        loss_action = self._compute_action_loss(
            action_noise_pred,
            action_target,
            action_timestep_ids,
            action_scheduler,
            pipe,
            B,
            action_is_pad=action_is_pad,
        )

        # --- Combined loss ---
        if self.lambda_video == 0:
            loss = self.lambda_action * loss_action
        else:
            loss = self.lambda_video * loss_video + self.lambda_action * loss_action

        return {
            "loss": loss,
            "loss_video": self.lambda_video * loss_video.detach(),
            "loss_action": self.lambda_action * loss_action.detach(),
        }

    def _sample_video_timesteps(
        self,
        B,
        min_b,
        max_b,
        decoupled_sampler,
        current_step,
        pipe,
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
        self,
        B,
        architecture,
        action_scheduler,
        action_data,
        decoupled_sampler,
        current_step,
        pipe,
        inputs,
        action_repr=None,
    ):
        """Prepare noisy actions, targets, and optional interleaved state.

        Returns:
            dict with keys: action_state, noisy_actions, action_target,
            action_timesteps, action_timestep_ids, action_data_clean,
            action_sigmas.
        """
        _empty = {
            "action_state": None,
            "noisy_actions": None,
            "action_target": None,
            "action_timesteps": None,
            "action_timestep_ids": None,
            "action_data_clean": None,
            "action_sigmas": None,
        }
        if self.lambda_action == 0:
            return _empty

        # Sample action timesteps
        if decoupled_sampler is not None and self._decoupled_action_t is not None:
            num_ts_a = len(action_scheduler.timesteps)
            action_timestep_ids = (
                (self._decoupled_action_t / decoupled_sampler.num_train_timesteps * num_ts_a)
                .long()
                .clamp(0, num_ts_a - 1)
            )
        else:
            action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))

        action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(dtype=pipe.torch_dtype, device=pipe.device)

        action_data = action_data.to(dtype=pipe.torch_dtype, device=pipe.device)
        if action_data.dim() == 2:
            action_data = action_data.unsqueeze(0)

        # Actions stay at full temporal resolution (e.g. 33 steps),
        # independent of video frame count (e.g. 9 frames after stride).

        # Encode actions into diffusion latent space (identity for continuous)
        if action_repr is not None:
            action_data = action_repr.encode(action_data)

        # Add noise
        action_noise = torch.randn_like(action_data)
        sigma_bc = action_sigmas.view(B, 1, 1)
        noisy_actions = (1 - sigma_bc) * action_data + sigma_bc * action_noise
        action_target = action_noise - action_data

        # Interleaved architectures (MoE, SharedBackbone, joint_self_attn)
        # need their state prepared before the video forward pass.
        action_state = None
        if architecture.is_interleaved:
            action_state = architecture.prepare_action_tokens(
                noisy_actions,
                action_timesteps,
                use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
                use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
                proprio_state=inputs.get("proprio_state", None),
            )

        return {
            "action_state": action_state,
            "noisy_actions": noisy_actions,
            "action_target": action_target,
            "action_timesteps": action_timesteps,
            "action_timestep_ids": action_timestep_ids,
            "action_data_clean": action_data,
            "action_sigmas": action_sigmas,
        }

    def _compute_video_loss(
        self,
        noise_pred,
        target,
        timestep_ids,
        pipe,
        inputs,
        B,
        video_is_pad=None,
    ):
        """Compute per-sample weighted video MSE loss.

        Reduction follows FastWAM: mean over (C, H, W), then masked mean
        over T, then weighted mean over B with timestep weights.

        Args:
            noise_pred: (B, C, T, H, W) predicted noise.
            target: (B, C, T, H, W) target noise.
            timestep_ids: (B,) sampled timestep indices.
            pipe: Pipeline (for scheduler weights and device).
            inputs: Dict with optional ``first_frame_latents``.
            B: Batch size.
            video_is_pad: (B, T_latent) bool, True for padded frames.
                T_latent is the temporal dim of the latent (after VAE
                temporal downsampling). If None, no masking.
        """
        if inputs.get("first_frame_latents") is not None:
            # Skip clean prefix: ref frame(s) + video frame 0 (conditioning, t=0).
            # num_clean_prefix_frames counts ref frame latent steps;
            # +1 for the video's own frame 0 whose latent is also excluded from loss.
            # video_is_pad already excludes frame 0 (tail-only), so no mask trim needed.
            n_skip = inputs.get("num_clean_prefix_frames", 0) + 1
            noise_pred = noise_pred[:, :, n_skip:]
            target = target[:, :, n_skip:]

        tw = pipe.scheduler.linear_timesteps_weights[timestep_ids].to(dtype=torch.float32, device=pipe.device)

        # Per-element MSE → mean over (C, H, W), keep (B, T)
        # noise_pred shape: (B, C, T, H, W)
        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        per_frame = per_element.mean(dim=(1, 3, 4))  # (B, T)

        if video_is_pad is not None:
            # video_is_pad: (B, T_latent) — already downsampled to latent
            # temporal dim by the trainer.
            video_is_pad = video_is_pad.to(device=per_frame.device, dtype=torch.bool)
            valid_mask = ~video_is_pad  # True for valid frames
            per_frame = per_frame * valid_mask.float()
            valid_count = valid_mask.float().sum(dim=1).clamp(min=1)
            per_sample = per_frame.sum(dim=1) / valid_count  # (B,)
        else:
            per_sample = per_frame.mean(dim=1)  # (B,)

        return (per_sample * tw).mean()

    def _compute_action_loss(
        self,
        noise_pred,
        target,
        timestep_ids,
        scheduler,
        pipe,
        B,
        action_is_pad=None,
    ):
        """Compute per-sample weighted action MSE loss.

        Reduction: mean over action_dim, then masked mean over T,
        then weighted mean over B with timestep weights.

        Args:
            noise_pred: (B, T, action_dim) predicted noise.
            target: (B, T, action_dim) target noise.
            timestep_ids: (B,) sampled timestep indices.
            scheduler: Action flow matching scheduler.
            pipe: Pipeline (for device).
            B: Batch size.
            action_is_pad: (B, T) bool, True for padded timesteps.
                If None, no masking.
        """
        tw = scheduler.linear_timesteps_weights[timestep_ids].to(dtype=torch.float32, device=pipe.device)
        pred_f = noise_pred.float()
        target_f = target.float()

        # Per-element MSE → mean over action_dim, keep (B, T)
        per_element = F.mse_loss(pred_f, target_f, reduction="none")
        per_step = per_element.mean(dim=2)  # (B, T)

        if action_is_pad is not None:
            # action_is_pad: (B, T) — already aligned to the subsampled
            # action temporal dim by the trainer.
            action_is_pad = action_is_pad.to(device=per_step.device, dtype=torch.bool)
            valid_mask = ~action_is_pad
            per_step = per_step * valid_mask.float()
            valid_count = valid_mask.float().sum(dim=1).clamp(min=1)
            per_sample = per_step.sum(dim=1) / valid_count  # (B,)
        else:
            per_sample = per_step.mean(dim=1)  # (B,)

        return (per_sample * tw).mean()
