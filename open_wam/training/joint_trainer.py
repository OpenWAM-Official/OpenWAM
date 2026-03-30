"""Joint video-action trainer wrapping the supported training runtime."""

import os

import logging
import torch

from open_wam.training.base import BaseTrainer
from open_wam.training.runtime import build_training_module, cfg_to_flat_namespace

logger = logging.getLogger(__name__)


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
        """Construct the training module from the package-native runtime."""
        args = cfg_to_flat_namespace(cfg)
        self._legacy_module = build_training_module(
            args,
            accelerator=accelerator,
            dataset=dataset,
        )

        # Decoupled training support (DreamZero-Flash inspired)
        decoupled_cfg = getattr(cfg.training, "decoupled", None)
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
