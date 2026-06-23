"""OpenWAM trainer: assembles the architecture and implements the BaseTrainer contract.

The generic training loop lives in ``openwam.train.base.BaseTrainer``. This
subclass only adds OpenWAM-specific pieces:
  - __init__ assembly: seed, build_architecture, freeze, latent-action setup, stats
  - compute_loss: joint flow-matching MSE on video + action (delegates to the architecture)
  - build_optimizer / get_trainable_parameters: optimizer groups
  - save/load_checkpoint: architecture state (VLM excluded)
  - hooks: on_setup_run (deploy assets), _loss_labels (latent), _prepare_auxiliary_modules

Usage:
    trainer = OpenWAMTrainer(cfg, accelerator, dataset)
    trainer.train()
"""

import logging
import os
import random
import shutil

import numpy as np
import torch
from omegaconf import DictConfig

from openwam.train.base import BaseTrainer
from openwam.train.utils.checkpointing import save_config, save_normalization_stats
from openwam.train.utils.optimizer_groups import build_trainable_parameters
from openwam.train.utils.training_utils import log_parameter_counts

logger = logging.getLogger(__name__)


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _latent_action_enabled(cfg: DictConfig) -> bool:
    action_cfg = _cfg_get(getattr(cfg, "model", None), "action_backbone", None)
    return _cfg_get(action_cfg, "type", "explicit") == "latent"


class OpenWAMTrainer(BaseTrainer):
    """Joint video-action trainer for OpenWAM. See module docstring.

    Args:
        cfg: Hydra DictConfig with model, training, data, project sections.
        accelerator: HuggingFace Accelerator instance.
        dataset: Training dataset (used for action stats loading).
    """

    def __init__(self, cfg: DictConfig, accelerator=None, dataset=None):
        super().__init__(cfg, model=None, dataset=dataset, accelerator=accelerator)

        # ---- Reproducible seed (FastWAM-style, yaml-driven) ----
        # Seed before build_architecture so DiT/ActionDiT weight init lands on
        # deterministic RNG. Deliberately does NOT touch cudnn/cuBLAS determinism
        # (would cost autotuning + FSDP/fused-attention compat, not needed for
        # "same seed -> same initial weights").
        project_cfg = getattr(cfg, "project", None)
        yaml_seed = getattr(project_cfg, "seed", None) if project_cfg is not None else None
        self._rank = int(os.environ.get("RANK", 0))
        self._run_seed = int(yaml_seed) if yaml_seed is not None else None
        if self._run_seed is not None:
            process_seed = self._run_seed + self._rank
            random.seed(process_seed)
            np.random.seed(process_seed)
            torch.manual_seed(process_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(process_seed)
            if self._rank == 0:
                logger.info(
                    "Reproducible mode: cfg.project.seed=%d (process_seed=%d, rank=%d)",
                    self._run_seed,
                    process_seed,
                    self._rank,
                )

        t = cfg.training
        m = cfg.model

        # Build architecture (creates video_backbone internally from config).
        from openwam.model import build_architecture, resolve_architecture_config

        resolved_arch = resolve_architecture_config(m)
        self.architecture = build_architecture(resolved_arch.registry_name, resolved_arch.params)
        logger.info(
            "Architecture: %s (framework=%s variant=%s)",
            resolved_arch.registry_name,
            resolved_arch.canonical.framework,
            resolved_arch.canonical.variant,
        )

        # Device placement: skip .to(device) when initialize_model_on_cpu + DeepSpeed,
        # because DeepSpeed's prepare() will handle the move.
        _init_on_cpu = bool(t.get("initialize_model_on_cpu", False))
        _use_deepspeed = (
            accelerator is not None
            and hasattr(accelerator, "distributed_type")
            and str(accelerator.distributed_type).endswith("DEEPSPEED")
        )
        if not (_init_on_cpu and _use_deepspeed):
            self.architecture.set_dtype_device(self.architecture.dtype, self.architecture.device)

        # --- Freeze: declared per-architecture in the model yaml (freeze:);
        # freeze_modules silently skips paths absent on a given architecture.
        freeze_list = list(getattr(m, "freeze", []))
        for name in self.architecture.freeze_modules(freeze_list):
            logger.info("Frozen: %s", name)

        # External encoder freeze sanity-check: an irreversible encoder (V-JEPA 2.1 /
        # DINOv3, no pixel decode) is almost always pretrained-and-frozen. Warn if
        # the freeze list omits it, before a run wastes GPU on a trainable ViT.
        external_encoder = getattr(self.architecture, "external_encoder", None)
        if (
            external_encoder is not None
            and not external_encoder.properties.pixel_decode
            and not any(
                p == "video_backbone.video_encoder" or p.startswith("video_backbone.video_encoder.")
                for p in freeze_list
            )
        ):
            logger.warning(
                "external encoder %s is not in freeze_modules; ViT is fully trainable. "
                "Add 'video_backbone.video_encoder' to your model freeze list "
                "if you intended to freeze the ViT backbone.",
                type(external_encoder).__name__,
            )

        # Initialize all schedulers (video + action) inside architecture
        self.architecture.init_training_schedulers(1000)

        # Loss weights from the training config
        self.lambda_video = float(t.lambda_video)
        self.lambda_action = float(t.lambda_action)
        self.lambda_decoder = float(_cfg_get(t, "lambda_decoder", 0.0))
        self.latent_action_provider = None
        self.latent_action_enabled = _latent_action_enabled(cfg)

        if self.latent_action_enabled:
            self._setup_latent_action(cfg)

        # Load action stats
        if dataset is not None and self.lambda_action > 0 and not self.latent_action_enabled:
            self._load_normalization_stats(dataset)

        # Push forward-time training flags onto the architecture so prepare_inputs
        # is self-contained.
        self.architecture.set_training_runtime(
            use_gradient_checkpointing=bool(t.use_gradient_checkpointing),
            use_gradient_checkpointing_offload=bool(t.use_gradient_checkpointing_offload),
            max_timestep_boundary=float(t.max_timestep_boundary),
            min_timestep_boundary=float(t.min_timestep_boundary),
        )

        # Store reference for BaseTrainer interface
        self.model = self

        is_main = self.accelerator is None or self.accelerator.is_main_process
        log_parameter_counts(self.architecture, is_main=is_main)

    def _setup_latent_action(self, cfg: DictConfig) -> None:
        """Validate latent-action config and build the latent action provider."""
        latent_cfg = cfg.model.action_backbone.latent_encoder
        output_cfg = _cfg_get(latent_cfg, "output")
        action_dim = int(_cfg_get(output_cfg, "action_dim", 0) or 0)
        token_dim = int(_cfg_get(output_cfg, "token_dim", action_dim) or 0)
        arch_cfg = getattr(cfg.model, "architecture", None)
        cfg_uses_proprio = bool(_cfg_get(arch_cfg, "use_proprioception", False))
        if cfg_uses_proprio or bool(getattr(self.architecture, "uses_proprioception", False)):
            raise ValueError(
                "model.action_backbone.type=latent requires model.architecture.use_proprioception=false "
                "for latent-action pretraining."
            )
        if action_dim <= 0:
            raise ValueError("model.action_backbone.latent_encoder.output.action_dim must be a positive integer.")
        if token_dim != action_dim:
            raise ValueError(
                f"model.action_backbone.latent_encoder.output.token_dim={token_dim} must match "
                f"output.action_dim={action_dim}."
            )
        if int(self.architecture.action_dim) != action_dim:
            raise ValueError(
                f"model.action_backbone.latent_encoder.output.action_dim={action_dim} does not match "
                f"architecture.action_dim={self.architecture.action_dim}."
            )
        if self.lambda_action <= 0:
            raise ValueError("model.action_backbone.type=latent requires training.lambda_action > 0.")
        from openwam.model.action_backbone.latent_encoder import build_latent_action_provider

        self.latent_action_provider = build_latent_action_provider(
            latent_cfg,
            device=self.architecture.device,
            dtype=self.architecture.dtype,
        )

    def _load_normalization_stats(self, dataset):
        """Load action normalization stats from dataset into architecture buffers."""
        stats = getattr(dataset, "normalization_stats", None)
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

    def build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW over the optimizer parameter groups."""
        t = self.cfg.training
        lr = float(t.learning_rate)
        betas = tuple(getattr(t, "adam_betas", [0.9, 0.95]))
        params = self.get_trainable_parameters()
        return torch.optim.AdamW(params, lr=lr, weight_decay=float(t.weight_decay), betas=betas)

    def compute_loss(self, batch) -> dict:
        """Compute joint video-action loss. Returns dict: total/video/action/decoder."""
        if not isinstance(batch, list):
            batch = [batch]

        # Latent mode keeps the real action + action_mask (collected by
        # prepare_inputs) as the decoder's supervision; only ActionDiT's
        # ``actions`` is swapped to the latent target (which has no pad mask).
        inputs = self.architecture.prepare_inputs(batch)
        if self.latent_action_enabled:
            if self.latent_action_provider is None:
                raise RuntimeError("model.action_backbone.type=latent but latent_action_provider is not initialized.")
            inputs["decoder_target"] = inputs.get("actions")
            inputs["decoder_action_is_pad"] = inputs.get("action_is_pad")
            inputs["action_is_pad"] = None
            videos = [sample["video"] for sample in batch]
            inputs["actions"] = self.latent_action_provider(videos)
        elif self.lambda_action > 0 and inputs.get("actions") is None:
            raise ValueError("lambda_action > 0 but no action in data.")

        result = self.architecture.compute_loss(
            **inputs,
            lambda_video=self.lambda_video,
            lambda_action=self.lambda_action,
            lambda_decoder=self.lambda_decoder,
            current_step=self._current_step,
        )

        return {
            "total": result["loss"],
            "video": result.get("loss_video", torch.tensor(0.0)),
            "action": result.get("loss_action", torch.tensor(0.0)),
            "decoder": result.get("loss_decoder", torch.tensor(0.0)),
        }

    # ---- BaseTrainer hooks ----
    def on_setup_run(self, output_path: str) -> None:
        """Make the checkpoint self-contained for deploy. save_assets_for_deployment
        merges component specs + copies backbone artifacts; must run BEFORE
        save_config so config.yaml carries the merged specs."""
        self.architecture.save_assets_for_deployment(output_path, self.cfg)
        save_config(output_path, self.cfg)
        if self.dataset is not None:
            save_normalization_stats(output_path, self.dataset)
        # Copy VLM checkpoint so deploy is self-contained (tri_system).
        vlm_bb = getattr(self.architecture, "vlm_backbone", None)
        if vlm_bb is not None and getattr(vlm_bb, "_checkpoint_path", None):
            vlm_dest = os.path.join(output_path, "vlm_backbone")
            if not os.path.exists(vlm_dest):
                shutil.copytree(vlm_bb._checkpoint_path, vlm_dest)
                logger.info("Copied VLM checkpoint to %s", vlm_dest)

    def _loss_labels(self) -> tuple[str, str]:
        # Latent mode: action stream predicts the LATENT action (-> "latent_action");
        # the decoder MSE is the REAL action loss (-> "action").
        if self.latent_action_enabled:
            return "latent_action", "action"
        return "action", "decoder"

    def _prepare_auxiliary_modules(self, device) -> None:
        if self.latent_action_provider is not None:
            self.latent_action_provider.to(device)
            self.latent_action_provider.device = torch.device(device)

    def save_checkpoint(self, path: str):
        """Export architecture state to safetensors. Safe under ZeRO-1/2, DDP, single-process.

        ALL ranks must call this together (``get_state_dict`` is a DeepSpeed
        collective); only rank 0 writes the file. VLM backbone params are
        excluded — the VLM checkpoint is saved as a separate directory.
        """
        from safetensors.torch import save_file

        from openwam.model.architectures.base import _exclude_vlm_from_state_dict

        if self.accelerator is not None:
            state_dict = self.accelerator.get_state_dict(self.architecture)
            if not self.accelerator.is_main_process:
                return
        else:
            state_dict = self.architecture.state_dict()

        state_dict = _exclude_vlm_from_state_dict(state_dict)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(state_dict, path)

    def load_checkpoint(self, path: str, strict: bool = True):
        """Load a checkpoint into the *unwrapped* architecture (flat safetensors,
        not DeepSpeed's sharded layout). Tolerates missing vlm_backbone.* keys.
        ``strict`` defaults True so a renamed state-dict raises rather than
        dropping weights silently."""
        unwrapped = (
            self.accelerator.unwrap_model(self.architecture) if self.accelerator is not None else self.architecture
        )
        unwrapped.load_checkpoint(path, strict=strict)
