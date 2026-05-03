"""Abstract base class for WAM (World-Action Model) architectures.

Two supported architecture families:

1. **Shared Backbone** (`framework=shared_backbone`)
   Action tokens are concatenated to the video DiT sequence and ride
   through the shared blocks. Variants: `vanilla` (no extra capacity) /
   `moe` (expert FFN at selected layers).

2. **Dual-System** (`framework=dual_system`)
   A separate ActionDiT consumes features from the video DiT. Variants:
   `joint_cross_attn` (bridge cross-attention after a full video forward)
   / `joint_self_attn` (MMDiT-style mixed attention at every layer, driven
   by :class:`MoTJointDriver`).

Each architecture composes a ``video_backbone`` and an ``action_backbone``
and owns its own ``forward()``.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openwam.model.action_backbone.backbone import ActionBackbone
    from openwam.model.video_backbone.adapter import VideoBackbone


@dataclass
class ActionState:
    """Mutable state container used by the joint self-attention path.

    Only ``DualSystemSelfAttnArchitecture`` needs this — its action stream is
    threaded through ``MoTJointDriver``, which mutates the payload across
    layers. SharedBackbone and DualSystem cross-attn don't go through this
    container.

    Fields:
        action_latents: (B, T_action, action_dim) noisy actions (input to forward).
        timestep: action diffusion timestep (raw shape preserved for the action
            backbone's internal use).
        payload: backbone-specific per-forward state (typically
            ``ActionDiTState``).
    """

    action_latents: Optional[Tensor] = None
    timestep: Optional[Tensor] = None
    payload: Optional[Any] = None


class BaseWAMArchitecture(ABC, nn.Module):
    """Base class for WAM architecture variants.

    Composes a ``video_backbone`` and an ``action_backbone``. Subclasses
    instantiate the appropriate ActionBackbone subclass in ``__init__`` —
    they do NOT implement any action processing logic themselves. The
    unified ``forward()`` defined here drives both backbones through the
    standard block-loop adapter interface.

    Args:
        cfg: Architecture-specific configuration (OmegaConf DictConfig or dict).
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.video_backbone: Optional["VideoBackbone"] = None
        self.action_backbone: Optional["ActionBackbone"] = None
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16

        # Forward-time training runtime flags. Trainer calls
        # ``set_training_runtime`` once during construction so ``prepare_inputs``
        # can read these without the trainer having to thread them through.
        self._use_gradient_checkpointing = False
        self._use_gradient_checkpointing_offload = False
        self._max_timestep_boundary = 1.0
        self._min_timestep_boundary = 0.0

        if cfg is not None:
            self._init_video_backbone(cfg)

    def _init_video_backbone(self, cfg):
        """Build video backbone from config.

        Supports two source types in ``cfg.video_backbone``:
        - ``_source``: direct path (directory or manifest .json) or dict with components → deploy-time path
        - ``name``: registry key → training-time path

        Both paths flow through the public :func:`build_video_backbone`.
        """
        from openwam.model.video_backbone import build_video_backbone

        vb_cfg = cfg.get("video_backbone", {}) if isinstance(cfg, dict) else getattr(cfg, "video_backbone", None)
        if vb_cfg is None:
            return

        source = vb_cfg.get("_source") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_source", None)
        vb_name = vb_cfg.get("name") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "name", None)

        if source is not None:
            ckpt_dir = vb_cfg.get("_ckpt_dir") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "_ckpt_dir", None)
            self.video_backbone = build_video_backbone(vb_name, cfg, source=source, device="cpu", ckpt_dir=ckpt_dir)
            return

        if vb_name is not None:
            self.video_backbone = build_video_backbone(vb_name, cfg)

    def _resolve_video_dim(self, cfg) -> int:
        """Resolve video_dim from config or video_backbone; raise if neither provides it."""
        dim = int(cfg.get("video_dim", 0)) if isinstance(cfg, dict) else int(getattr(cfg, "video_dim", 0))
        if dim == 0 and self.video_backbone is not None:
            dim = self.video_backbone.dim
        if not dim:
            raise ValueError("video_dim must be specified in config or inferred from video_backbone")
        return dim

    @property
    def backbones(self) -> dict[str, nn.Module]:
        """All backbone modules owned by this architecture.

        Subclasses with additional backbones (e.g. TriSystem with a VLM
        backbone) should override this to include them. The returned dict
        is used by ``init_training_schedulers``, ``set_dtype_device``,
        ``move_frozen_to_device``, and ``get_component_specs`` to iterate
        over all backbones generically.
        """
        result = {}
        if self.video_backbone is not None:
            result["video_backbone"] = self.video_backbone
        if self.action_backbone is not None:
            result["action_backbone"] = self.action_backbone
        return result

    # --- Action-side properties (delegate to action_backbone) ---

    @property
    def action_scheduler(self):
        """Flow-matching scheduler for the action stream (owned by action_backbone)."""
        if self.action_backbone is None:
            raise RuntimeError("action_backbone is not initialized")
        return self.action_backbone.scheduler

    @property
    def action_dim(self) -> int:
        return self.action_backbone.action_dim if self.action_backbone is not None else 0

    @property
    def bridge_layers(self) -> tuple:
        return self.action_backbone.bridge_layers if self.action_backbone is not None else ()

    @property
    def trainable_action_module(self) -> Optional[nn.Module]:
        """The nn.Module whose parameters are trained as the action model."""
        return self.action_backbone

    @property
    def uses_proprioception(self) -> bool:
        return self.action_backbone is not None and self.action_backbone.uses_proprioception

    @property
    def action_mean(self) -> Tensor:
        if self.action_backbone is not None:
            return self.action_backbone.action_mean
        return torch.zeros(self.action_dim)

    @property
    def action_std(self) -> Tensor:
        if self.action_backbone is not None:
            return self.action_backbone.action_std
        return torch.ones(self.action_dim)

    # --- Device / dtype (top-level authority) ---

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def set_dtype_device(self, dtype: torch.dtype, device: torch.device) -> None:
        """Dispatch to each backbone — they own their own dtype/device handling."""
        self._dtype = dtype
        self._device = device
        for bb in self.backbones.values():
            bb.set_dtype_device(dtype, device)

    # --- Checkpoint save / load ---

    def save_checkpoint(self, path: str) -> None:
        """Save full architecture state (video backbone + action module) to safetensors."""
        from safetensors.torch import save_file

        save_file(self.state_dict(), path)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        """Load full architecture state from a safetensors checkpoint.

        ``strict`` defaults to ``True`` so a renamed state-dict (e.g. v1.0 → v1.1
        where ``moe_expert_dit.*`` became ``shared_moe.*`` and ``action_dit.*``
        moved into ``dualsystem_dit.*``) raises explicitly rather than dropping
        weights silently. Pass ``strict=False`` only for deliberate partial loads.
        """
        from safetensors.torch import load_file

        sd = load_file(path)
        self.load_state_dict(sd, strict=strict)

    # --- Training: module management ---

    def init_training_schedulers(self, num_timesteps: int = 1000) -> None:
        """Initialize all backbone schedulers for training."""
        for bb in self.backbones.values():
            if hasattr(bb, "scheduler"):
                bb.scheduler.set_timesteps(num_timesteps, training=True)

    def freeze_modules(self, names: list[str]) -> list[str]:
        """Freeze named sub-modules by dotted path. Returns actually frozen names.

        Uses nn.Module.get_submodule() so dotted paths like
        ``video_backbone._pipe.text_encoder`` work naturally.
        """
        frozen = []
        for name in names:
            try:
                module = self.get_submodule(name)
            except (AttributeError, KeyError):
                module = None
            if module is not None:
                module.requires_grad_(False)
                frozen.append(name)
        return frozen

    def get_trainable_modules(self, freeze_list: list[str] = ()) -> dict[str, nn.Module]:
        """Return top-level trainable sub-modules.

        Walks ``self.named_children()`` and returns modules that have at
        least one parameter with ``requires_grad=True``, excluding those
        in *freeze_list*. Used by ``optimizer_groups.build_trainable_parameters``
        to source the param groups for the optimizer.
        """
        result = {}
        freeze_set = set(freeze_list)
        for name, mod in self.named_children():
            if name in freeze_set:
                continue
            if any(p.requires_grad for p in mod.parameters()):
                result[name] = mod
        return result

    def move_frozen_to_device(self, device: torch.device, names: tuple[str, ...] = ("text_encoder", "vae")) -> None:
        """Move named frozen modules to device.

        Searches via ``get_submodule`` on self first, then on each backbone.
        """
        for name in names:
            mod = None
            try:
                mod = self.get_submodule(name)
            except (AttributeError, KeyError):
                pass
            if mod is None:
                for bb in self.backbones.values():
                    found = bb.get_submodule(name) if hasattr(bb, "get_submodule") else None
                    if found is not None:
                        mod = found
                        break
            if mod is not None:
                mod.to(device=device)

    def get_component_specs(self, model_path: str) -> Optional[dict]:
        """Get component specs from all backbones for self-contained checkpoint config."""
        for bb in self.backbones.values():
            if hasattr(bb, "get_component_specs"):
                specs = bb.get_component_specs(model_path)
                if specs is not None:
                    return specs
        return None

    # --- Training: preprocessing ---

    @torch.no_grad()
    def preprocess(self, **kwargs) -> dict:
        """Encode raw frames/text into latents + context for training.

        Delegates to ``video_backbone.preprocess_input()``. External code
        (trainer) should call this instead of touching video_backbone directly.
        """
        return self.video_backbone.preprocess_input(**kwargs)

    def set_training_runtime(
        self,
        *,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,
    ) -> None:
        """Set forward-time training flags consumed by ``prepare_inputs``.

        Trainers call this once during construction. Keeping these on the
        architecture keeps ``prepare_inputs(batch)`` self-contained — the
        trainer no longer needs to thread these flags through every loss call.
        """
        self._use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self._use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self._max_timestep_boundary = float(max_timestep_boundary)
        self._min_timestep_boundary = float(min_timestep_boundary)

    @torch.no_grad()
    def prepare_inputs(self, batch: list[dict]) -> dict:
        """Aggregate a list of dataset samples into a batched inputs dict.

        Absorbs the per-sample field collection that previously lived in
        ``OpenWAMTrainer._forward_batch``. The returned dict is designed to be
        unpacked directly into ``compute_loss`` via ``**inputs``.

        Args:
            batch: List of dataset samples (each a dict). A single dict is
                accepted as well and treated as a one-sample batch.

        Returns:
            Dict with all preprocessed video latents, text embeddings, action
            tensors, masks, and forward-time flags ready for ``compute_loss``.
        """
        from openwam.dataloader.transforms.pipeline import FirstFrameConditioningTransform
        from openwam.utils import downsample_video_mask_to_latent

        if isinstance(batch, dict):
            batch = [batch]

        if not hasattr(self, "_pipeline_transform_instance"):
            self._pipeline_transform_instance = FirstFrameConditioningTransform()
        samples = [self._pipeline_transform_instance.apply(s) for s in batch]

        _dtype = self.dtype
        _device = self.device

        all_frames: list = []
        all_prompts: list = []
        all_vace_videos: list = []
        all_ref_images: list = []
        all_actions: list = []
        all_action_masks: list = []
        all_video_masks: list = []

        for sample in samples:
            all_frames.append(sample["video"])
            all_prompts.append(sample["prompt"])
            all_vace_videos.append(sample.get("vace_video"))
            all_ref_images.append(sample.get("first_frame_image"))

            action = sample.get("action")
            if action is not None:
                if isinstance(action, np.ndarray):
                    action = torch.from_numpy(action)
                action = action.to(dtype=_dtype, device=_device).unsqueeze(0)
            all_actions.append(action)

            amask = sample.get("action_mask", None)
            vmask = sample.get("video_mask", None)
            if isinstance(amask, np.ndarray):
                amask = torch.from_numpy(amask)
            if isinstance(vmask, np.ndarray):
                vmask = torch.from_numpy(vmask)
            all_action_masks.append(amask)
            all_video_masks.append(vmask)

        ref_flags = [r is not None for r in all_ref_images]
        if any(ref_flags) and not all(ref_flags):
            raise ValueError("Mixed reference images in batch: all samples must be consistent.")

        preprocessed = self.preprocess(
            frames=all_frames,
            text=all_prompts,
            vace_videos=all_vace_videos,
            ref_images=all_ref_images if ref_flags[0] else None,
        )

        action_data = torch.cat(all_actions, dim=0) if all_actions[0] is not None else None

        inputs = {
            **preprocessed,
            "latents": None,
            "cfg_scale": 1,
            "cfg_merge": False,
            "tiled": False,
            "use_gradient_checkpointing": self._use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self._use_gradient_checkpointing_offload,
            "max_timestep_boundary": self._max_timestep_boundary,
            "min_timestep_boundary": self._min_timestep_boundary,
            "actions": action_data,
        }

        if action_data is not None and self.uses_proprioception:
            inputs["proprio_state"] = action_data[:, 0, :].contiguous()

        if all_action_masks[0] is not None:
            inputs["action_is_pad"] = torch.stack([~m for m in all_action_masks], dim=0).to(device=_device)
        if all_video_masks[0] is not None:
            latent_masks = [downsample_video_mask_to_latent(~m) for m in all_video_masks]
            inputs["video_is_pad"] = torch.stack(latent_masks, dim=0).to(device=_device)

        return inputs

    # --- Training: loss computation ---

    def compute_loss(
        self,
        *,
        actions: Optional[torch.Tensor] = None,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        current_step: int = 0,
        decoupled_sampler=None,
        action_timestep_per_token: bool = False,
        **inputs,
    ) -> dict:
        """Compute joint video-action flow matching loss.

        This is the single entry point for training loss computation.
        Handles timestep sampling, noise injection, forward pass, and
        loss calculation internally.

        Callers should produce ``inputs`` via ``self.prepare_inputs(batch)``
        (preferred) or assemble it manually with the same keys: the output of
        ``self.preprocess()`` plus any of ``actions / proprio_state /
        action_is_pad / video_is_pad / use_gradient_checkpointing[_offload] /
        max_timestep_boundary / min_timestep_boundary``.

        Args:
            actions: (B, T_action, action_dim) ground truth actions. May also
                be passed via ``inputs["actions"]``.
            lambda_video: Weight for video loss term.
            lambda_action: Weight for action loss term.
            current_step: Current training step.
            decoupled_sampler: Optional DecoupledFlowMatchLoss.
            action_timestep_per_token: Per-token action timestep sampling.
            **inputs: Preprocessed video/text tensors plus forward-time flags.

        Returns:
            dict with keys: loss, loss_video, loss_action.
        """
        vb = self.video_backbone
        action_scheduler = self.action_backbone.scheduler
        _dtype = self.dtype
        _device = self.device

        if actions is None:
            actions = inputs.pop("actions", None)
        else:
            inputs.pop("actions", None)

        max_tb = int(inputs.pop("max_timestep_boundary", 1) * len(vb.scheduler.timesteps))
        min_tb = int(inputs.pop("min_timestep_boundary", 0) * len(vb.scheduler.timesteps))
        B = inputs["input_latents"].shape[0]

        # --- Sample video timesteps ---
        if decoupled_sampler is not None:
            video_t, decoupled_action_t = decoupled_sampler.sample_timesteps(B, current_step=current_step, device="cpu")
            num_ts = len(vb.scheduler.timesteps)
            video_timestep_ids = (
                (video_t / decoupled_sampler.num_train_timesteps * num_ts).long().clamp(min_tb, max_tb - 1)
            )
        else:
            decoupled_action_t = None
            video_timestep_ids = torch.randint(min_tb, max_tb, (B,))

        video_timesteps = vb.scheduler.timesteps[video_timestep_ids].to(dtype=_dtype, device=_device)
        video_sigmas = vb.scheduler.sigmas[video_timestep_ids].to(dtype=_dtype, device=_device)

        # --- Add video noise ---
        video_noise = torch.randn_like(inputs["input_latents"])
        sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
        inputs["latents"] = (1 - sigma_bc) * inputs["input_latents"] + sigma_bc * video_noise
        video_target = video_noise - inputs["input_latents"]

        if inputs.get("first_frame_latents") is not None:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

        # --- Prepare action noise ---
        noisy_actions, action_target, action_timesteps, action_timestep_ids, action_sigmas = (
            None,
            None,
            None,
            None,
            None,
        )
        if lambda_action > 0 and actions is not None:
            if decoupled_action_t is not None:
                num_ts_a = len(action_scheduler.timesteps)
                action_timestep_ids = (
                    (decoupled_action_t / decoupled_sampler.num_train_timesteps * num_ts_a)
                    .long()
                    .clamp(0, num_ts_a - 1)
                )
            elif action_timestep_per_token:
                T_action = actions.shape[-2]
                action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B, T_action))
            else:
                action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))

            action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(dtype=_dtype, device=_device)
            action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(dtype=_dtype, device=_device)

            actions = actions.to(dtype=_dtype, device=_device)
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)

            action_noise = torch.randn_like(actions)
            if action_sigmas.dim() == 1:
                a_sigma_bc = action_sigmas.view(B, 1, 1)
            else:
                a_sigma_bc = action_sigmas.unsqueeze(-1)
            noisy_actions = action_scheduler.add_noise(actions, action_noise, a_sigma_bc)
            action_target = action_scheduler.training_target(actions, action_noise)

        # --- Joint forward pass ---
        forward_inputs = dict(inputs)
        proprio_state = forward_inputs.pop("proprio_state", None)
        use_grad_ckpt = forward_inputs.pop("use_gradient_checkpointing", False)
        use_grad_ckpt_offload = forward_inputs.pop("use_gradient_checkpointing_offload", False)
        # Padding masks are kept in `inputs` for loss-side masking but dropped
        # from `forward_inputs` so they don't leak into vb.prepare(). Per FastWAM
        # MoT design, attention itself does not consume sample-level padding.
        forward_inputs.pop("action_is_pad", None)
        forward_inputs.pop("video_is_pad", None)

        video_noise_pred, action_noise_pred = self.forward(
            noisy_actions if lambda_action > 0 else None,
            action_timesteps if lambda_action > 0 else None,
            proprio_state=proprio_state,
            use_gradient_checkpointing=use_grad_ckpt,
            use_gradient_checkpointing_offload=use_grad_ckpt_offload,
            **forward_inputs,
            timestep=video_timesteps,
        )

        # --- Video loss ---
        loss_video = self._compute_video_loss(
            video_noise_pred,
            video_target,
            video_timestep_ids,
            inputs,
            _device,
        )

        if lambda_action == 0 or action_noise_pred is None:
            return {
                "loss": lambda_video * loss_video,
                "loss_video": lambda_video * loss_video.detach(),
                "loss_action": torch.tensor(0.0, device=loss_video.device),
            }

        # --- Action loss ---
        loss_action = self._compute_action_loss(
            action_noise_pred,
            action_target,
            action_timestep_ids,
            action_scheduler,
            inputs,
            _device,
        )

        if lambda_video == 0:
            loss = lambda_action * loss_action
        else:
            loss = lambda_video * loss_video + lambda_action * loss_action

        return {
            "loss": loss,
            "loss_video": lambda_video * loss_video.detach(),
            "loss_action": lambda_action * loss_action.detach(),
        }

    def _compute_video_loss(self, noise_pred, target, timestep_ids, inputs, device):
        """Per-sample weighted video MSE loss."""
        import torch.nn.functional as F

        if inputs.get("first_frame_latents") is not None:
            n_skip = inputs.get("num_clean_prefix_frames", 0) + 1
            noise_pred = noise_pred[:, :, n_skip:]
            target = target[:, :, n_skip:]

        vb = self.video_backbone
        tw = vb.scheduler.linear_timesteps_weights[timestep_ids].to(dtype=torch.float32, device=device)

        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        per_frame = per_element.mean(dim=(1, 3, 4))

        video_is_pad = inputs.get("video_is_pad")
        if video_is_pad is not None:
            video_is_pad = video_is_pad.to(device=per_frame.device, dtype=torch.bool)
            valid_mask = ~video_is_pad
            per_frame = per_frame * valid_mask.float()
            valid_count = valid_mask.float().sum(dim=1).clamp(min=1)
            per_sample = per_frame.sum(dim=1) / valid_count
        else:
            per_sample = per_frame.mean(dim=1)

        return (per_sample * tw).mean()

    def _compute_action_loss(self, noise_pred, target, timestep_ids, scheduler, inputs, device):
        """Per-sample weighted action MSE loss."""
        import torch.nn.functional as F

        tw = scheduler.training_weight(timestep_ids).to(dtype=torch.float32, device=device)
        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        per_step = per_element.mean(dim=2)

        per_token_weight = tw.dim() == 2
        action_is_pad = inputs.get("action_is_pad")

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=per_step.device, dtype=torch.bool)
            valid_mask = ~action_is_pad
            per_step = per_step * valid_mask.float()
            valid_count = valid_mask.float().sum(dim=1).clamp(min=1)
            if per_token_weight:
                per_step = per_step * tw
                return (per_step.sum(dim=1) / valid_count).mean()
            per_sample = per_step.sum(dim=1) / valid_count
            return (per_sample * tw).mean()

        if per_token_weight:
            return (per_step * tw).mean()
        per_sample = per_step.mean(dim=1)
        return (per_sample * tw).mean()

    # --- Inference: generation ---

    @torch.no_grad()
    def generate(
        self,
        schedule,
        prompt: str,
        *,
        vace_video=None,
        first_frame_image=None,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        seed: int = 42,
        tiled: bool = True,
        input_video_latents: Optional[Tensor] = None,
        num_inference_steps: int = 50,
        shift: float = 5.0,
        tile_size: tuple = None,
        tile_stride: tuple = None,
        dit_cache=None,
        decode_video: bool = True,
        profile: bool = False,
        vace_cache: Optional[dict] = None,
        prompt_embed_cache: Optional[dict] = None,
    ) -> dict:
        """Execute joint video-action denoising driven by a schedule.

        This is the single entry point for inference. External code
        (engine) should call this instead of touching video_backbone directly.

        Returns:
            dict with ``video`` (list of PIL images or None) and
            ``actions`` ((T, action_dim) numpy array).
        """
        import time

        from tqdm import tqdm

        vb = self.video_backbone
        device = self.device
        dtype = self.dtype

        t0 = time.time()

        prep_kwargs = {}
        if tile_size is not None:
            prep_kwargs["tile_size"] = tile_size
        if tile_stride is not None:
            prep_kwargs["tile_stride"] = tile_stride

        inputs_shared = vb.prepare_inputs_for_inference(
            prompt,
            vace_video=vace_video,
            first_frame_image=first_frame_image,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            tiled=tiled,
            num_inference_steps=num_inference_steps,
            shift=shift,
            vace_cache=vace_cache,
            prompt_embed_cache=prompt_embed_cache,
            **prep_kwargs,
        )

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[WAM_PROFILE] pipeline_prep: %.3fs", time.time() - t0)

        if input_video_latents is not None:
            inputs_shared["latents"] = input_video_latents

        action_latents = torch.randn(
            1,
            num_frames - 1,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        num_train_ts = float(self.action_scheduler.num_train_timesteps)

        t_loop = time.time()

        for i in tqdm(range(len(schedule) - 1), desc="Joint denoising"):
            t_v, t_a = schedule[i]
            t_v_next, t_a_next = schedule[i + 1]

            sigma_v = t_v / num_train_ts
            sigma_a = t_a / num_train_ts
            sigma_v_next = t_v_next / num_train_ts
            sigma_a_next = t_a_next / num_train_ts

            video_stepping = sigma_v != sigma_v_next
            action_stepping = sigma_a != sigma_a_next

            if not video_stepping and not action_stepping:
                continue

            v_timestep = torch.tensor([t_v], dtype=dtype, device=device)
            a_timestep = torch.tensor([t_a], dtype=dtype, device=device) if action_stepping else None

            if dit_cache is not None and video_stepping and not dit_cache.should_recompute(sigma_v):
                # Reuse cached video noise prediction; still call forward() with
                # noisy_actions=None to skip the action stream cleanly. This is
                # only valid when video_stepping=True (the only path that
                # populates the cache).
                noise_pred = dit_cache.get_cached()
                action_noise_pred = None
                if action_stepping:
                    # Re-run action with a fresh forward pass; without cached
                    # bridges we just rerun video too. Acceptable at this scale.
                    torch.compiler.cudagraph_mark_step_begin()
                    noise_pred, action_noise_pred = self.forward(
                        action_latents,
                        a_timestep,
                        **inputs_shared,
                        timestep=v_timestep,
                    )
            else:
                torch.compiler.cudagraph_mark_step_begin()
                noise_pred, action_noise_pred = self.forward(
                    action_latents if action_stepping else None,
                    a_timestep,
                    **inputs_shared,
                    timestep=v_timestep,
                )
                if dit_cache is not None and video_stepping:
                    dit_cache.update(noise_pred, sigma_v)

            if video_stepping:
                new_latents = inputs_shared["latents"] + noise_pred * (sigma_v_next - sigma_v)
                if "first_frame_latents" in inputs_shared:
                    new_latents = new_latents.clone()
                    new_latents[:, :, 0:1] = inputs_shared["first_frame_latents"]
                inputs_shared["latents"] = new_latents

            if action_stepping and action_noise_pred is not None:
                action_latents = self.action_scheduler.flow_step(
                    action_noise_pred, sigma_a, sigma_a_next, action_latents
                )

        if profile:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[WAM_PROFILE] denoising_loop: %.3fs", time.time() - t_loop)

        # VAE decode
        if decode_video:
            if first_frame_image is not None:
                ref_count = len(first_frame_image) if isinstance(first_frame_image, list) else 1
                inputs_shared["latents"] = inputs_shared["latents"][:, :, ref_count:]
            video_frames = vb.decode_video(inputs_shared["latents"], tiled=tiled)
        else:
            video_frames = None

        actions = action_latents.squeeze(0).float().cpu().numpy()
        denorm = getattr(self, "action_denormalizer", None)
        if denorm is not None:
            actions = denorm.unnormalize(actions)

        return {"video": video_frames, "actions": actions}

    # --- Deploy helpers (combine action module + video backbone) ---

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Apply torch.compile to action module and all backbones.

        Args:
            compile_cfg: Config object with optional bool flags.
                ``enabled`` compiles the action module; backbone-specific
                flags (``video_dit``, ``vae``, etc.) are forwarded to each
                backbone's ``apply_compile`` method.
        """
        if getattr(compile_cfg, "enabled", False):
            action_module = self.trainable_action_module
            if action_module is not None and action_module is not self:
                for name, child in self.named_children():
                    if child is action_module:
                        setattr(self, name, torch.compile(action_module, dynamic=True))
                        logger.info("torch.compile enabled for action module (%s)", name)
                        break

        for bb_name, bb in self.backbones.items():
            if hasattr(bb, "apply_compile"):
                bb.apply_compile(compile_cfg)

    @abstractmethod
    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        proprio_state: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Run the joint video + action forward.

        Each concrete architecture implements its own forward end-to-end —
        runs the video DiT block loop, captures or interleaves with the
        action stream as appropriate, and returns
        ``(video_noise_pred, action_noise_pred)``. When ``noisy_actions`` is
        None (CFG nega pass / video-only generation) the action term is
        None.
        """
        ...
