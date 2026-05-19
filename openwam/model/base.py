"""Abstract base class for WAM (World-Action Model) architectures.

Supported architecture families:

1. **Shared Backbone** (`framework=shared_backbone`)
   Action tokens are concatenated to the video DiT sequence and ride
   through the shared blocks. Variants: `vanilla` (no extra capacity) /
   `moe` (expert FFN at selected layers).

2. **Dual-System** (`framework=dual_system`)
   A separate ActionDiT consumes features from the video DiT. Variants:
   `joint_cross_attn` (bridge cross-attention after a full video forward)
   / `joint_self_attn` (MMDiT-style mixed attention at every layer, driven
   by :class:`MoTJointDriver`).

3. **Tri-System** (`framework=tri_system`)
   Motus-style mixture of transformers: Wan video DiT + action expert +
   frozen VLM / understanding expert, with mixed attention implemented via
   the video backbone adapter.

Each architecture composes the backbones it owns and implements its own
``forward()``.
"""

import functools
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from openwam.model.compile_options import compile_mode


def _wrap_single_forward(module: nn.Module) -> None:
    """Wrap a single module's ``forward`` in ``torch.no_grad``. Idempotent."""
    if getattr(module, "_openwam_no_grad_wrapped", False):
        return
    original_forward = module.forward

    @functools.wraps(original_forward)
    def wrapped(*args, **kwargs):
        with torch.no_grad():
            return original_forward(*args, **kwargs)

    module.forward = wrapped
    module._openwam_no_grad_wrapped = True


def _wrap_forward_in_no_grad(module: nn.Module) -> None:
    """Wrap ``forward`` of ``module`` AND every submodule in its subtree in ``torch.no_grad``.

    Recursion matters because callers commonly bypass the root forward and call a
    nested submodule directly. The canonical case in OpenWAM is
    ``Qwen3VLBackbone.extract_features`` which calls ``self.vlm_model.model(...)``
    (the inner ``Qwen3VLModel``, skipping the LM head) — wrapping only
    ``vlm_model.forward`` would leave that path grad-tracking. Recursively wrapping
    every descendant makes the semantic complete: any entry point into the frozen
    subtree is in ``no_grad``.

    Idempotent — a marker attribute on each module prevents double-wrapping if
    ``freeze_modules`` runs more than once. ``nn.Module.modules()`` deduplicates
    via its internal memo, so cyclic registrations (e.g. test fakes with
    ``self.model = self``) are visited once.

    Safe to apply to any module: if the caller already wraps the call in a
    ``no_grad`` context (e.g. ``prepare_inputs``), the inner ``no_grad`` is a
    no-op; if the caller is inside a grad-tracking forward (the tri_system VLM
    case this actually saves memory in), it short-circuits activation saving.

    **Subtree-level semantic, not per-parameter**: a trainable child under a frozen
    parent will NOT receive gradients, because every descendant ``forward`` is
    wrapped in ``no_grad``. For partial-freeze setups (e.g. LoRA on a frozen base,
    or training only the LM head of an otherwise frozen VLM), do NOT pass the
    parent's dotted path to ``freeze_modules``; pass the specific leaves you want
    frozen instead. The current freeze list in ``configs/training_strategy/*.yaml``
    only names complete subtrees, so this limitation does not bite today.
    """
    for sub in module.modules():
        _wrap_single_forward(sub)


logger = logging.getLogger(__name__)

# Prefix for VLM backbone parameters in the architecture state_dict.
# VLM weights are saved as a separate checkpoint directory (not in safetensors)
# to avoid tied-weight deduplication complexity.
VLM_STATE_DICT_PREFIX = "vlm_backbone."


def _exclude_vlm_from_state_dict(state_dict: dict[str, "Tensor"]) -> dict[str, "Tensor"]:
    """Filter out VLM backbone parameters from a state dict.

    Note: this exclusion is prefix-based (``vlm_backbone.*``).  Future
    trainable modules on the VLM (e.g. LoRA adapters) must be registered at
    the architecture top level (as siblings of ``vlm_backbone``), NOT as
    children under ``vlm_backbone``, otherwise they will be silently excluded
    from the checkpoint.
    """
    return {k: v for k, v in state_dict.items() if not k.startswith(VLM_STATE_DICT_PREFIX)}


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

    Composes a ``video_backbone`` and an ``action_backbone`` plus optional
    extra backbones. Subclasses instantiate the appropriate ActionBackbone
    subclass in ``__init__`` and own the complete ``forward()`` control flow.

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

        # Optional action normalizer for deployment. ``generate`` uses it to
        # return real-scale actions; deploy-side proprio preprocessing uses it
        # to normalize raw robot state into the model's training space.
        self.action_normalizer = None

        if cfg is not None:
            self._init_video_backbone(cfg)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _init_video_backbone(self, cfg):
        """Build video backbone from config.

        Supports two source types in ``cfg.video_backbone``:
        - ``_source``: direct model directory or dict with components → deploy-time path
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
        elif vb_name is not None:
            self.video_backbone = build_video_backbone(vb_name, cfg)

        # Optional from-scratch DiT: keep the Wan video backbone structure but
        # discard the loaded DiT weights and re-randomize them in place. VAE
        # and the text encoder stay pretrained and are frozen by the training
        # strategy yaml. Reproducibility comes from ``cfg.project.seed`` which
        # ``OpenWAMTrainer`` applies before architecture construction. Applies
        # uniformly to every architecture that builds its video backbone via
        # this method (dual_system / shared_backbone / tri_system).
        if self.video_backbone is not None and self._cfg_get(vb_cfg, "from_scratch", False):
            pipe = getattr(self.video_backbone, "_pipe", None)
            if pipe is None:
                logger.warning(
                    "video_backbone.from_scratch=true but backbone has no '_pipe'; skipping. (Non-Wan backbone?)"
                )
            else:
                from openwam.model.video_backbone.wan_adapter import reinit_dit_from_scratch

                reinit_dit_from_scratch(pipe)
                logger.info(
                    "video_backbone.from_scratch=true: DiT re-initialized; VAE / text_encoder keep pretrained weights"
                )

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
    def video_scheduler(self):
        """Flow-matching scheduler for the video stream (owned by video_backbone)."""
        if self.video_backbone is None:
            raise RuntimeError("video_backbone is not initialized")
        return self.video_backbone.scheduler

    @property
    def action_dim(self) -> int:
        return self.action_backbone.action_dim if self.action_backbone is not None else 0

    @property
    def bridge_layers(self) -> tuple:
        return getattr(self.action_backbone, "bridge_layers", ()) if self.action_backbone is not None else ()

    @property
    def expert_layers(self) -> tuple:
        return getattr(self.action_backbone, "expert_layers", ()) if self.action_backbone is not None else ()

    @property
    def trainable_action_module(self) -> Optional[nn.Module]:
        """The nn.Module whose parameters are trained as the action model."""
        return self.action_backbone

    @property
    def uses_proprioception(self) -> bool:
        return bool(getattr(self, "_use_proprioception_context", False)) or (
            self.action_backbone is not None and self.action_backbone.uses_proprioception
        )

    def _init_proprio_context(self, cfg, *, text_dim: int = 4096) -> None:
        """Initialize FastWAM-style proprio-as-context conditioning."""
        enabled = bool(self._cfg_get(cfg, "use_proprioception", False))
        self._use_proprioception_context = enabled
        self.proprio_encoder: Optional[nn.Module] = None
        self.proprio_dim = 0
        self.context_dim = int(text_dim)
        if not enabled:
            return
        state_dim = int(self._cfg_get(cfg, "state_dim", 0) or 0)
        if state_dim <= 0:
            raise ValueError("use_proprioception=True requires explicit state_dim for context-token proprio.")
        self.proprio_dim = state_dim
        self.proprio_encoder = nn.Linear(state_dim, self.context_dim)

    def _append_proprio_context_token(self, pipeline_inputs: dict, proprio_state: Optional[Tensor]) -> dict:
        """Append one proprio token to raw text context and extend context_mask."""
        if not bool(getattr(self, "_use_proprioception_context", False)):
            return pipeline_inputs
        if self.proprio_encoder is None:
            raise RuntimeError("proprio context is enabled but proprio_encoder is not initialized.")
        if proprio_state is None:
            raise ValueError("use_proprioception=True requires `proprio_state` from sample['proprio'] or obs['state'].")
        if proprio_state.ndim == 1:
            proprio_state = proprio_state.unsqueeze(0)
        elif proprio_state.ndim == 3 and proprio_state.shape[1] == 1:
            proprio_state = proprio_state[:, 0, :]
        if proprio_state.ndim != 2:
            raise ValueError(f"proprio_state must be [B, D] or [B, 1, D], got shape {tuple(proprio_state.shape)}")
        if proprio_state.shape[1] != self.proprio_dim:
            raise ValueError(f"proprio_state last dim must be {self.proprio_dim}, got {proprio_state.shape[1]}")

        context = pipeline_inputs["context"]
        if context.shape[0] != proprio_state.shape[0]:
            if proprio_state.shape[0] == 1 and context.shape[0] > 1:
                proprio_state = proprio_state.expand(context.shape[0], -1)
            else:
                raise ValueError(
                    f"Batch mismatch between context and proprio_state: {context.shape[0]} vs {proprio_state.shape[0]}"
                )
        proprio_token = (
            self.proprio_encoder(proprio_state.to(device=context.device, dtype=self.proprio_encoder.weight.dtype))
            .to(dtype=context.dtype)
            .unsqueeze(1)
        )

        context_mask = pipeline_inputs.get("context_mask")
        if context_mask is None:
            seq_lens = pipeline_inputs.get("seq_lens")
            if seq_lens is not None:
                seq_lens = seq_lens.to(device=context.device)
                positions = torch.arange(context.shape[1], device=context.device).unsqueeze(0)
                context_mask = positions < seq_lens.unsqueeze(1)
            else:
                context_mask = torch.ones((context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device)
        else:
            context_mask = context_mask.to(device=context.device, dtype=torch.bool)

        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        updated = dict(pipeline_inputs)
        updated["context"] = torch.cat([context, proprio_token], dim=1)
        updated["context_mask"] = torch.cat([context_mask, proprio_mask], dim=1)
        # The appended proprio token can sit after padded text tokens, so the
        # resulting valid tokens are not necessarily a contiguous prefix.
        # Keep the original text seq_lens and make context_mask authoritative.
        return updated

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
        proprio_encoder = getattr(self, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.to(dtype=dtype, device=device)
        for bb in self.backbones.values():
            bb.set_dtype_device(dtype, device)

    def attach_action_normalizer(self, normalizer) -> None:
        """Attach (or clear) an action normalizer used by ``generate``.

        Deployment paths build the same normalizer used by training from
        ``action_stats.npy``. ``generate`` uses it to return real-scale actions,
        while server-side proprio preprocessing uses it to normalize raw robot
        state into the model's training space. Pass ``None`` to clear.
        """
        self.action_normalizer = normalizer

    def normalize_deploy_proprio(self, proprio_state):
        """Normalize raw deploy proprio with the training action normalizer."""
        normalizer = getattr(self, "action_normalizer", None)
        if normalizer is None or proprio_state is None:
            return proprio_state

        import numpy as np
        import torch

        was_tensor = isinstance(proprio_state, torch.Tensor)
        device = proprio_state.device if was_tensor else None
        dtype = proprio_state.dtype if was_tensor and proprio_state.is_floating_point() else None
        arr = proprio_state.detach().cpu().numpy() if was_tensor else np.asarray(proprio_state, dtype=np.float32)
        norm = normalizer.normalize(arr.astype(np.float32, copy=False))
        if was_tensor:
            return torch.from_numpy(norm).to(device=device, dtype=dtype or torch.float32)
        return norm

    # --- Checkpoint save / load ---

    def save_checkpoint(self, path: str) -> None:
        """Save architecture state to safetensors.

        VLM backbone parameters are excluded — the VLM checkpoint is saved
        as a separate directory by the trainer. This avoids tied-weight
        deduplication complexity and keeps the file small.
        """
        from safetensors.torch import save_file

        state_dict = self.state_dict()
        if getattr(self, "vlm_backbone", None) is not None:
            state_dict = _exclude_vlm_from_state_dict(state_dict)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(state_dict, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        """Load architecture state from a safetensors checkpoint.

        VLM backbone weights are not stored in the safetensors file (they
        are saved as a separate directory). When a VLM backbone is present,
        missing ``vlm_backbone.*`` keys are tolerated; unexpected or missing
        non-VLM keys still raise under ``strict=True``.
        """
        from safetensors.torch import load_file

        state_dict = load_file(path)
        has_vlm = getattr(self, "vlm_backbone", None) is not None
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if strict and not has_vlm:
            if missing or unexpected:
                raise RuntimeError(f"Strict load failed: missing={missing}, unexpected={unexpected}")
        elif strict and has_vlm:
            non_vlm_missing = [k for k in missing if not k.startswith(VLM_STATE_DICT_PREFIX)]
            if non_vlm_missing or unexpected:
                raise RuntimeError(
                    f"Strict load failed (VLM keys excluded): missing={non_vlm_missing}, unexpected={unexpected}"
                )

    # --- Training: module management ---

    def init_training_schedulers(self, num_timesteps: int = 1000) -> None:
        """Initialize all backbone schedulers for training."""
        for bb in self.backbones.values():
            if hasattr(bb, "scheduler"):
                bb.scheduler.set_timesteps(num_timesteps, training=True)

    def freeze_modules(self, names: list[str]) -> list[str]:
        """Freeze named sub-modules by dotted path. Returns actually frozen names.

        Single-point freeze API. Two effects per frozen submodule:

        1. ``module.requires_grad_(False)`` — optimizer cannot update its params.
        2. ``module.forward`` is wrapped in ``torch.no_grad`` so the frozen
           subtree never saves activations for backward. This is the full
           semantic of "freeze" — neither the trainer nor any backbone needs to
           inspect freeze status separately.

        For text_encoder / vae, which are already called under the
        ``@torch.no_grad()`` ``prepare_inputs`` decorator, the wrapper is a
        no-op (nested ``no_grad``). For modules called inside the training
        forward graph (e.g. tri_system's frozen Qwen3-VL backbone), the
        wrapper is what actually saves activation memory.

        Uses ``nn.Module.get_submodule()`` so dotted paths like
        ``video_backbone._pipe.text_encoder`` work naturally; unknown names
        are silently skipped, so a freeze list mentioning modules absent on a
        given architecture (e.g. ``vlm_backbone.vlm_model`` on dual_system)
        is harmless.
        """
        frozen = []
        for name in names:
            try:
                module = self.get_submodule(name)
            except (AttributeError, KeyError):
                module = None
            if module is not None:
                module.requires_grad_(False)
                # Set eval mode on the frozen subtree. Use modules() instead
                # of .eval() to avoid infinite recursion when a submodule has
                # self-referential aliases (e.g. HF model.model = self).
                for sub in module.modules():
                    sub.training = False
                _wrap_forward_in_no_grad(module)
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
        all_proprios: list = []
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

            proprio = sample.get("proprio")
            if self.uses_proprioception:
                if proprio is None:
                    raise ValueError(
                        "use_proprioception=True requires sample['proprio']; action[0] fallback is disabled."
                    )
                if isinstance(proprio, np.ndarray):
                    proprio = torch.from_numpy(proprio)
                proprio = proprio.to(dtype=_dtype, device=_device)
                if proprio.ndim == 1:
                    pass
                elif proprio.ndim == 2 and proprio.shape[0] == 1:
                    proprio = proprio[0]
                else:
                    raise ValueError(f"sample['proprio'] must be [D] or [1, D], got shape {tuple(proprio.shape)}")
            all_proprios.append(proprio)

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

        if self.uses_proprioception:
            inputs["proprio_state"] = torch.stack(all_proprios, dim=0).contiguous()

        if all_action_masks[0] is not None:
            inputs["action_is_pad"] = torch.stack([~m for m in all_action_masks], dim=0).to(device=_device)
        if all_video_masks[0] is not None:
            latent_masks = [downsample_video_mask_to_latent(~m) for m in all_video_masks]
            inputs["video_is_pad"] = torch.stack(latent_masks, dim=0).to(device=_device)

        return inputs

    # --- ZeRO-3 external-parameter protocol ---
    #
    # The MoT driver reads several leaf ``nn.Parameter`` (e.g. ``block.modulation``,
    # ``block.wan_und_qkv``) directly inside the architecture forward, bypassing the
    # owning submodule's ``__call__``. Under DeepSpeed ZeRO-3 those leaves are
    # partitioned and the forward-pre-hook that would gather them never fires for
    # the owner. The fix is the standard external-parameter protocol: register
    # the leaves against ``self`` (the architecture) and call ``self(...)`` so the
    # architecture-level forward-pre-hook gathers them before the raw read.

    def _iter_zero3_external_params(self):
        """Yield each raw-access leaf ``nn.Parameter`` that the MoT path reads.

        Default: empty. Overridden by every architecture whose training
        forward pulls partitioned leaves outside the owner submodule's
        ``__call__`` — concretely, the MoT-driven variants:

        - ``DualSystemSelfAttnArchitecture`` — video + action ``block.modulation``
        - ``DualSystemIDMArchitecture`` — same as joint_self_attn; IDM's
          ``compute_loss`` override routes its 3-branch forward through
          ``self.__call__`` so the same protocol applies.
        - ``TriSystemJointSelfAttnArchitecture`` — also understanding
          ``block.wan_und_qkv``.

        Cross-attn and shared variants go through standard ``block.__call__``
        and don't need to override.
        """
        return ()

    def _register_zero3_externals(self) -> None:
        """Register raw-access leaves as DeepSpeed ZeRO-3 external params of ``self``.

        Required because the MoT driver reads these leaves directly, bypassing
        the owning submodule's ``__call__``. Registering them makes DeepSpeed
        gather them on ``self.__call__``'s forward-pre-hook and hold through
        backward — without this AccumulateGrad sees a size-0 leaf.

        Idempotent + no-op when deepspeed isn't importable or params lack
        ``ds_id`` (non-ZeRO-3 paths, CPU mock tests). The gate only seals after
        at least one successful register so a pre-``accelerator.prepare`` call
        (params still un-partitioned) can be retried post-prepare. Architectures
        whose iterator is empty by design (cross-attn, shared variants) seal
        immediately — they will never need to register anything.
        """
        if getattr(self, "_zero3_externals_registered", False):
            return
        leaves = list(self._iter_zero3_external_params())
        if not leaves:
            self._zero3_externals_registered = True
            return
        try:
            from deepspeed.runtime.zero import register_external_parameter
        except ImportError:
            self._zero3_externals_registered = True
            return
        registered_any = False
        for p in leaves:
            if getattr(p, "ds_id", None) is not None:
                register_external_parameter(self, p)
                registered_any = True
        if registered_any:
            self._zero3_externals_registered = True

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
        if action_timestep_per_token:
            raise ValueError(
                "action_timestep_per_token=True is not supported in the FastWAM-compatible path; "
                "action timestep must be per-sample [B]."
            )

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

        # Use ``self(...)`` (not ``self.forward(...)``) so ``nn.Module.__call__``
        # is invoked and the architecture-level forward-pre-hook fires. Under
        # DeepSpeed ZeRO-3 that hook gathers the leaves registered by
        # ``_register_zero3_externals`` (raw-access params read by the MoT
        # driver). On non-ZeRO-3 paths this is a no-op detour through the empty
        # hook chain.
        self._register_zero3_externals()
        video_noise_pred, action_noise_pred = self(
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

        num_clean_prefix = int(inputs.get("num_clean_prefix_frames", 0) or 0)
        video_is_pad = inputs.get("video_is_pad")

        n_skip = 0
        if num_clean_prefix > 0 or inputs.get("first_frame_latents") is not None:
            # Explicit signal from TI2V / VACE preprocess: trim ref-prefix
            # plus the first VAE-conditioning latent that
            # ``downsample_video_mask_to_latent`` also excludes from the mask.
            n_skip = num_clean_prefix + 1
        elif video_is_pad is not None and video_is_pad.shape[-1] < noise_pred.shape[2]:
            # Production tail-mask convention (no ref-prefix backbone such as
            # I2V): ``video_is_pad`` is sized to T_lat minus the leading
            # conditioning latents. Trim noise_pred / target to match.
            n_skip = noise_pred.shape[2] - video_is_pad.shape[-1]

        if n_skip > 0:
            noise_pred = noise_pred[:, :, n_skip:]
            target = target[:, :, n_skip:]

        vb = self.video_backbone
        tw = vb.scheduler.linear_timesteps_weights[timestep_ids].to(dtype=torch.float32, device=device)

        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        per_frame = per_element.mean(dim=(1, 3, 4))

        if video_is_pad is not None:
            if video_is_pad.shape[-1] != noise_pred.shape[2]:
                raise ValueError(
                    f"video_is_pad length {video_is_pad.shape[-1]} does not match "
                    f"trimmed noise_pred T={noise_pred.shape[2]} (n_skip={n_skip}). "
                    "Expected mask sized to T_lat minus leading conditioning latents."
                )
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
        if tw.ndim != 1:
            raise ValueError(f"action loss weights must be per-sample [B], got shape {tuple(tw.shape)}")
        per_element = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        per_step = per_element.mean(dim=2)

        action_is_pad = inputs.get("action_is_pad")

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=per_step.device, dtype=torch.bool)
            valid_mask = ~action_is_pad
            per_step = per_step * valid_mask.float()
            valid_count = valid_mask.float().sum(dim=1).clamp(min=1)
            per_sample = per_step.sum(dim=1) / valid_count
            return (per_sample * tw).mean()

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
        action_num_frames: Optional[int] = None,
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
        proprio_state: Optional[Tensor] = None,
        **extra_pipeline_inputs: Any,
    ) -> dict:
        """Execute joint video-action denoising driven by a schedule.

        This is the single entry point for inference. External code
        (engine) should call this instead of touching video_backbone directly.

        Args:
            num_frames: Video frame count passed to the video backbone. For
                RoboTwin this is the post-``video_stride`` count seen during
                training, not the raw action window length.
            action_num_frames: Raw state/action window length. Generated
                action chunk length is ``action_num_frames - 1``. Defaults to
                ``num_frames`` for datasets whose video/action rates match.

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

        action_num_frames = int(action_num_frames if action_num_frames is not None else num_frames)

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
        # Architecture-specific pipeline inputs (e.g. tri_system's vlm_inputs /
        # vlm_hidden / vlm_attention_mask) are forwarded as-is. Subclasses
        # extract what they recognize in their forward(); unrelated architectures
        # never see these keys because callers only pass them via super().generate.
        for key, value in extra_pipeline_inputs.items():
            if value is not None:
                inputs_shared[key] = value
        ref_latents = inputs_shared.get("first_frame_latents")
        if ref_latents is not None:
            latents = inputs_shared["latents"].clone()
            latents[:, :, : ref_latents.shape[2]] = ref_latents
            inputs_shared["latents"] = latents
        if self.uses_proprioception:
            if proprio_state is None:
                raise ValueError("use_proprioception=True requires `proprio_state` during generation.")
            inputs_shared["proprio_state"] = proprio_state.to(device=device, dtype=dtype)

        action_latents = torch.randn(
            1,
            action_num_frames - 1,
            self.action_dim,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

        num_train_ts_v = float(self.video_scheduler.num_train_timesteps)
        num_train_ts_a = float(self.action_scheduler.num_train_timesteps)

        t_loop = time.time()

        for i in tqdm(range(len(schedule) - 1), desc="Joint denoising"):
            t_v, t_a = schedule[i]
            t_v_next, t_a_next = schedule[i + 1]

            sigma_v = t_v / num_train_ts_v
            sigma_a = t_a / num_train_ts_a
            sigma_v_next = t_v_next / num_train_ts_v
            sigma_a_next = t_a_next / num_train_ts_a

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
                ref_latents = inputs_shared.get("first_frame_latents")
                if ref_latents is not None:
                    new_latents = new_latents.clone()
                    new_latents[:, :, : ref_latents.shape[2]] = ref_latents
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
            video_frames = vb.decode_video(inputs_shared["latents"], tiled=tiled)
        else:
            video_frames = None

        actions = action_latents.squeeze(0).float().cpu().numpy()
        normalizer = getattr(self, "action_normalizer", None)
        if normalizer is not None:
            actions = normalizer.unnormalize(actions)

        return {"video": video_frames, "actions": actions}

    # --- Deploy helpers (combine action module + video backbone) ---

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Apply architecture-specific deploy-time compile optimizations."""
        mode = compile_mode(compile_cfg, default="none", strict=True)
        if mode in (None, "auto", "none"):
            return
        logger.warning(
            "torch.compile mode '%s' is not implemented for %s; running eager.",
            mode,
            type(self).__name__,
        )

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
