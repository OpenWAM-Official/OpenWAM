"""Joint video-action trainer wrapping the legacy VideoActionTrainingModule."""

import sys
import os
from pathlib import Path
from typing import Optional

import logging
import torch
import numpy as np

from open_wam.training.base import BaseTrainer

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WAM_DIR = str(_PROJECT_ROOT / "examples" / "wanvideo" / "wam")
if _WAM_DIR not in sys.path:
    sys.path.insert(0, _WAM_DIR)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class JointTrainer(BaseTrainer):
    """Joint video-action trainer.

    Wraps the legacy ``VideoActionTrainingModule`` behind the
    :class:`BaseTrainer` interface. The legacy module handles model
    construction, forward pass, and loss computation internally.

    Args:
        cfg: Hydra DictConfig with model, training, data sections.
        accelerator: HuggingFace Accelerator instance.
        dataset: Training dataset (used for action stats loading).
    """

    def __init__(self, cfg, accelerator=None, dataset=None):
        super().__init__(cfg, model=None, dataset=dataset, accelerator=accelerator)
        self._build_legacy_module(cfg, accelerator, dataset)

    def _build_legacy_module(self, cfg, accelerator, dataset):
        """Construct the legacy VideoActionTrainingModule from Hydra config."""
        from train_video_action import VideoActionTrainingModule  # noqa: E402

        t = cfg.training
        m = cfg.model
        b = cfg.model.backbone
        d = cfg.data

        bridge_layers_str = ",".join(str(x) for x in m.bridge_layers)

        device = "cpu"
        if getattr(t, "initialize_model_on_cpu", True) is False and accelerator is not None:
            device = accelerator.device

        self._legacy_module = VideoActionTrainingModule(
            model_paths=t.model_paths,
            model_id_with_origin_paths=t.model_id_with_origin_paths,
            tokenizer_path=t.tokenizer_path,
            audio_processor_path=None,
            trainable_models=",".join(t.trainable_models) if t.trainable_models else None,
            lora_base_model=t.lora_base_model,
            lora_target_modules=t.lora_target_modules,
            lora_rank=int(t.lora_rank),
            lora_checkpoint=t.lora_checkpoint,
            preset_lora_path=t.preset_lora_path,
            preset_lora_model=t.preset_lora_model,
            use_gradient_checkpointing=bool(t.use_gradient_checkpointing),
            use_gradient_checkpointing_offload=bool(t.use_gradient_checkpointing_offload),
            extra_inputs=getattr(t, "extra_inputs", "vace_video,vace_reference_image,action_trajectory"),
            fp8_models=t.fp8_models,
            offload_models=t.offload_models,
            task="sft",
            device=device,
            max_timestep_boundary=float(t.max_timestep_boundary),
            min_timestep_boundary=float(t.min_timestep_boundary),
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
            action_stats_path=getattr(d, "action_stats_path", None),
        )

        # Load action stats into ActionDiT buffers from dataset
        if dataset is not None and float(t.lambda_action) > 0:
            stats = getattr(dataset, "action_stats", None)
            if callable(stats):
                stats = stats()
            elif hasattr(dataset, "_legacy") and hasattr(dataset._legacy, "action_stats"):
                stats = dataset._legacy.action_stats
            if stats is not None:
                self._legacy_module.action_dit.action_mean.copy_(
                    torch.from_numpy(stats["mean"].astype(np.float32))
                )
                self._legacy_module.action_dit.action_std.copy_(
                    torch.from_numpy(stats["std"].astype(np.float32))
                )

        # Decoupled training support (DreamZero-Flash inspired)
        decoupled_cfg = getattr(t, "decoupled", None)
        if decoupled_cfg is not None and getattr(decoupled_cfg, "enabled", False):
            from open_wam.training.decoupled_loss import DecoupledFlowMatchLoss
            self._legacy_module.decoupled_sampler = DecoupledFlowMatchLoss(
                video_beta_a=float(getattr(decoupled_cfg, "video_beta_a", 0.5)),
                video_beta_b=float(getattr(decoupled_cfg, "video_beta_b", 1.0)),
                warmup_steps=int(getattr(decoupled_cfg, "warmup_steps", 0)),
            )

        self.model = self._legacy_module

    def compute_loss(self, batch) -> dict:
        """Compute joint video-action loss via the legacy forward pass.

        Args:
            batch: A single sample dict or list of sample dicts from the dataset.

        Returns:
            dict with keys: ``total``, ``video``, ``action``, and additional
            breakdown keys from the legacy loss function.
        """
        loss_dict = self._legacy_module.forward(batch)

        # Map legacy keys to BaseTrainer convention
        return {
            "total": loss_dict["loss"],
            "video": loss_dict.get("loss_video", torch.tensor(0.0)),
            "action": loss_dict.get("loss_action", torch.tensor(0.0)),
            # Pass through all legacy keys for logging
            **{k: v for k, v in loss_dict.items() if k != "loss"},
        }

    def save_checkpoint(self, path: str):
        """Export trainable state dict to a safetensors file."""
        state_dict = self._legacy_module.state_dict()
        trainable_state = self._legacy_module.export_trainable_state_dict(state_dict)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".safetensors"):
            from safetensors.torch import save_file
            save_file(trainable_state, path)
        else:
            torch.save(trainable_state, path)

    def load_checkpoint(self, path: str):
        """Load a checkpoint into the legacy module."""
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu")

        # Handle action_dit prefix
        action_keys = {k: v for k, v in state_dict.items() if k.startswith("action_dit.")}
        other_keys = {k for k in state_dict if not k.startswith("action_dit.")}
        if other_keys:
            logger.warning(
                "Checkpoint contains %d non-ActionDiT keys that will be ignored: %s",
                len(other_keys),
                ", ".join(sorted(other_keys)[:5]) + ("..." if len(other_keys) > 5 else ""),
            )
        if action_keys:
            cleaned = {k.removeprefix("action_dit."): v for k, v in action_keys.items()}
            missing, unexpected = self._legacy_module.action_dit.load_state_dict(cleaned, strict=False)
            if missing:
                logger.warning("Missing keys in ActionDiT checkpoint: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys in ActionDiT checkpoint: %s", unexpected)
        else:
            # Try loading directly (no prefix)
            missing, unexpected = self._legacy_module.action_dit.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("Missing keys in ActionDiT checkpoint: %s", missing[:10])

    @property
    def pipe(self):
        """Access the underlying WanVideoPipeline."""
        return self._legacy_module.pipe

    @property
    def action_dit(self):
        """Access the underlying ActionDiT model."""
        return self._legacy_module.action_dit

    @property
    def action_scheduler(self):
        """Access the underlying action FlowMatchScheduler."""
        return self._legacy_module.action_scheduler

    def trainable_modules(self):
        """Return trainable parameters (delegates to legacy module)."""
        return self._legacy_module.trainable_modules()
