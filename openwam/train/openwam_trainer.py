"""OpenWAM trainer that consumes Hydra DictConfig directly.

Composes package-native components:
  - Loss: implemented inside ``BaseWAMArchitecture.compute_loss``
    (openwam/model/base.py) — joint flow-matching MSE on video and action.
    Optionally wrapped with ``DecoupledFlowMatchLoss`` for DreamZero-Flash
    style Beta-distributed video timestep sampling.
  - Optimizer groups: openwam.train.utils.optimizer_groups
  - Checkpointing: openwam.train.utils.checkpointing
  - Architecture: openwam.model.registry (DualSystem / MoE / SharedBackbone)

Usage:
    trainer = OpenWAMTrainer(cfg, accelerator, dataset)
    trainer.train()
"""

import logging
import math
import os

import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from openwam.train.base import BaseTrainer
from openwam.train.utils.checkpointing import (
    manage_checkpoints,
    save_action_stats,
    save_config,
)
from openwam.train.utils.optimizer_groups import build_trainable_parameters

logger = logging.getLogger(__name__)


class OpenWAMTrainer(BaseTrainer):
    """Joint video-action trainer for OpenWAM.

    Directly consumes Hydra DictConfig without argparse conversion.
    Builds all components from package-native modules, with no dependency
    on the old vendored training infrastructure.

    Args:
        cfg: Hydra DictConfig with model, training, data, project sections.
        accelerator: HuggingFace Accelerator instance.
        dataset: Training dataset (used for action stats loading).
    """

    def __init__(self, cfg: DictConfig, accelerator=None, dataset=None):
        super().__init__(cfg, model=None, dataset=dataset, accelerator=accelerator)

        t = cfg.training
        m = cfg.model

        # Build architecture (creates video_backbone internally from config).
        # Wrap construction in a ZeRO-3 init-disable scope: when the Accelerator
        # was built with ``zero3_init_flag=True``, DeepSpeed enters a global
        # ``zero.Init(enabled=True)`` context that auto-partitions every
        # nn.Parameter at allocation time. For OpenWAM that's actively harmful
        # — frozen modules (text_encoder ~13 GiB umt5-xxl, VAE) get partitioned
        # along with trainable DiT, and every forward then triggers a ~26 GiB
        # all-gather spike to materialize them (guaranteed OOM on forward 2 of
        # training). Wrapping construction in ``zero.Init(enabled=False)``
        # skips DeepSpeed's per-Parameter tracking (no ``ds_id`` / ``ds_status``
        # is attached). At ``initialize()`` time, untagged params stay
        # replicated; only params constructed under the outer
        # ``zero.Init(enabled=True)`` scope (the trainable DiT/VACE created
        # by ``build_architecture`` below) get partitioned. So frozen modules
        # never enter the shard table and never trigger an all-gather.
        from openwam.model import build_architecture, resolve_architecture_config

        resolved_arch = resolve_architecture_config(m)
        with self._zero3_init_disabled():
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
            self.architecture.action_backbone.to(dtype=self.architecture.dtype, device=self.architecture.device)

        # --- Freeze: apply after all models are built ---
        # Read from training_strategy config (e.g. joint.yaml / video_only.yaml)
        strategy = cfg.training_strategy
        freeze_list = list(getattr(strategy, "freeze", []))
        for name in self.architecture.freeze_modules(freeze_list):
            logger.info("Frozen: %s", name)

        # Initialize all schedulers (video + action) inside architecture
        self.architecture.init_training_schedulers(1000)

        # Loss weights from training_strategy config
        self.lambda_video = float(strategy.lambda_video)
        self.lambda_action = float(strategy.lambda_action)

        self.action_timestep_per_token = bool(getattr(t, "action_timestep_per_token", False))
        if self.action_timestep_per_token:
            logger.warning(
                "action_timestep_per_token=True: training samples per-token diffusion "
                "timesteps, but the inference path still broadcasts a per-sample "
                "a_timestep. Train/inference sampling will diverge until the inference "
                "side is updated — only use this flag for training-only ablations."
            )

        # Decoupled training support
        decoupled_cfg = getattr(t, "decoupled", None)
        self.decoupled_sampler = None
        if decoupled_cfg is not None and getattr(decoupled_cfg, "enabled", False):
            from openwam.train.loss.decoupled_loss import DecoupledFlowMatchLoss

            self.decoupled_sampler = DecoupledFlowMatchLoss(
                video_beta_a=float(getattr(decoupled_cfg, "video_beta_a", 0.5)),
                video_beta_b=float(getattr(decoupled_cfg, "video_beta_b", 1.0)),
                warmup_steps=int(getattr(decoupled_cfg, "warmup_steps", 0)),
            )

        # Load action stats
        if dataset is not None and self.lambda_action > 0:
            self._load_action_stats(dataset)

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

        # Step counter
        self._current_step = 0
        self._last_loss_components = {}

        # Print param counts (total + trainable, per backbone). Use print()
        # rather than logger so it survives Hydra's default logging filter,
        # and gate explicitly on rank-0 instead of relying on train.py's
        # global ``builtins.print = noop`` on non-main ranks (that suppression
        # is launch-flow specific; the explicit guard keeps this correct if
        # the trainer is ever invoked from a different launcher or subprocess).
        is_main = self.accelerator is None or self.accelerator.is_main_process
        if is_main:

            def _count(module):
                if module is None:
                    return 0, 0
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                return total, trainable

            vb_total, vb_train = _count(self.architecture.video_backbone)
            ab_total, ab_train = _count(self.architecture.action_backbone)
            print("=" * 60)
            print("Parameter counts")
            print(f"  VideoBackbone : total={vb_total / 1e6:7.1f}M  trainable={vb_train / 1e6:7.1f}M")
            print(f"  ActionBackbone: total={ab_total / 1e6:7.1f}M  trainable={ab_train / 1e6:7.1f}M")
            print(
                f"  Architecture  : total={(vb_total + ab_total) / 1e6:7.1f}M  "
                f"trainable={(vb_train + ab_train) / 1e6:7.1f}M"
            )
            print("=" * 60, flush=True)

    @staticmethod
    def _zero3_init_disabled():
        """Context that disables DeepSpeed ZeRO-3 construction-time partitioning.

        ``deepspeed.zero.Init(enabled=False)`` is the official way to nest a
        "do-not-partition" scope inside an outer ``zero.Init(enabled=True)``
        (the same mechanism HuggingFace transformers uses to load frozen
        adapters under ZeRO-3). Falls back to a no-op context when DeepSpeed
        isn't importable, so the DDP / single-GPU paths are unaffected.

        See the ``__init__`` rationale for why we disable this — short version:
        partitioning frozen modules (text_encoder, VAE) causes a ~26 GiB
        all-gather spike on every forward and OOMs by step 2.
        """
        from contextlib import nullcontext

        try:
            import deepspeed
        except ImportError:
            return nullcontext()
        return deepspeed.zero.Init(enabled=False)

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
        if not isinstance(batch, list):
            batch = [batch]

        inputs = self.architecture.prepare_inputs(batch)
        if self.lambda_action > 0 and inputs.get("actions") is None:
            raise ValueError("lambda_action > 0 but no action in data.")

        result = self.architecture.compute_loss(
            **inputs,
            lambda_video=self.lambda_video,
            lambda_action=self.lambda_action,
            current_step=self._current_step,
            decoupled_sampler=self.decoupled_sampler,
            action_timestep_per_token=self.action_timestep_per_token,
        )

        return {
            "total": result["loss"],
            "video": result.get("loss_video", torch.tensor(0.0)),
            "action": result.get("loss_action", torch.tensor(0.0)),
        }

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

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer. Override to use a different optimizer."""
        t = self.cfg.training
        lr = float(t.learning_rate)
        betas = tuple(getattr(t, "adam_betas", [0.9, 0.95]))
        params = self.get_trainable_parameters()
        return torch.optim.AdamW(params, lr=lr, weight_decay=float(t.weight_decay), betas=betas)

    def build_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Build the training DataLoader. Override for custom sampling."""
        t = self.cfg.training
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(t.dataset_num_workers),
            collate_fn=list,
            pin_memory=True,
        )

    def build_lr_scheduler(self, optimizer, total_opt_steps: int, debug: bool = False):
        """Build the LR scheduler. Returns scheduler or None. Override for custom schedules."""
        t = self.cfg.training
        lr = float(t.learning_rate)
        lr_scheduler_type = getattr(t, "lr_scheduler", None)
        if debug:
            return None
        if lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup_ratio = float(getattr(t, "warmup_ratio", 0.05))
            lr_min_ratio = float(getattr(t, "lr_min_ratio", 0.01))
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
            return scheduler
        return None

    def on_train_begin(self, *, output_path: str, total_steps: int, **ctx):
        """Hook called before the training loop starts. Override for custom setup."""

    def on_step_end(
        self,
        global_step: int,
        *,
        loss_total: float,
        loss_video: float,
        loss_action: float,
        grad_norm: float,
        lr: float,
        epoch: int,
        pbar=None,
        wandb_run=None,
        steps_per_sec: float = 0.0,
        batch_size: int = 1,
        **ctx,
    ):
        """Hook called after each training step. Override for custom logging.

        Default implementation updates the progress bar and logs to wandb.
        """
        if pbar is not None:
            pbar.set_postfix(
                loss=f"{loss_total:.4f}",
                video=f"{loss_video:.4f}",
                action=f"{loss_action:.4f}",
                lr=f"{lr:.2e}",
                epoch=epoch,
            )
            pbar.update(1)

        if wandb_run is not None:
            _num_procs = self.accelerator.num_processes if self.accelerator is not None else 1
            log_dict = {
                "train/loss": loss_total,
                "train/loss_video": loss_video,
                "train/loss_action": loss_action,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "performance/steps_per_sec": steps_per_sec,
                "performance/samples_per_sec": steps_per_sec * batch_size * _num_procs,
            }
            wandb_run.log(log_dict, step=global_step)

    def on_train_end(self, global_step: int, *, output_path: str, wandb_run=None, **ctx):
        """Hook called after training completes. Override for custom teardown."""
        if wandb_run is not None:
            wandb_run.finish()

    def should_save_checkpoint(self, global_step: int, save_steps: int | None) -> bool:
        """Whether to save a checkpoint at this step. Override for custom logic."""
        return save_steps is not None and global_step > 0 and global_step % save_steps == 0

    def train(self, num_epochs: int = None, max_steps: int = None):
        """Run the training loop.

        Uses HuggingFace Accelerate for distributed training.

        Args:
            num_epochs: Override for ``training.num_epochs``.
            max_steps: Override for ``training.max_steps``.
        """
        t = self.cfg.training
        num_epochs = num_epochs or int(t.num_epochs)
        max_steps = max_steps or getattr(t, "max_steps", None)
        batch_size = int(t.batch_size)
        grad_accum = int(t.gradient_accumulation_steps)

        # Debug mode: override to a short sanity-check run
        debug = bool(getattr(t, "debug", False))
        if debug:
            max_steps = 20
            save_steps_override = 5
            logger.info("DEBUG mode: max_steps=20, save@5, constant LR")

        # Build optimizer, dataloader, scheduler via overridable methods
        optimizer = self.build_optimizer()
        dataloader = self.build_dataloader(batch_size)

        # Gradient clipping
        max_grad_norm = float(t.max_grad_norm) if getattr(t, "max_grad_norm", None) else None

        # LR scheduler via overridable method
        steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
        total_opt_steps = steps_per_epoch * num_epochs
        if max_steps:
            total_opt_steps = min(total_opt_steps, max_steps)
        scheduler = self.build_lr_scheduler(optimizer, total_opt_steps, debug=debug)

        # Checkpoint intervals
        if debug:
            save_steps = save_steps_override
        else:
            save_steps = getattr(t, "save_steps", None)
            if save_steps is not None:
                save_steps = int(save_steps)
        keep_last_k = int(getattr(t, "keep_last_k_ckpts", 3))
        base_output_path = getattr(t, "output_path", "./models")

        # Create output directory on rank 0 only, then broadcast the path
        # so all ranks share the same directory (avoids duplicate dirs from
        # slightly different timestamps across processes).
        _is_main = self.accelerator is None or self.accelerator.is_main_process
        if _is_main:
            from datetime import datetime

            run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if debug:
                run_dir_name += "_debug"
            output_path = os.path.join(base_output_path, run_dir_name)
            os.makedirs(output_path, exist_ok=True)
            # Inject video backbone component specs into config for
            # self-contained deployment (no manifest.json needed).
            model_path = None
            try:
                model_path = str(self.cfg.model.video_backbone.model_path)
            except Exception:
                pass
            if model_path and os.path.isdir(model_path):
                from omegaconf import OmegaConf

                specs = self.architecture.get_component_specs(model_path)
                if specs is not None:
                    with open_dict(self.cfg):
                        if "components" not in self.cfg.model.video_backbone:
                            OmegaConf.update(self.cfg, "model.video_backbone.components", specs["components"])
                        if "tokenizer" in specs and "tokenizer" not in self.cfg.model.video_backbone:
                            OmegaConf.update(self.cfg, "model.video_backbone.tokenizer", specs["tokenizer"])
            save_config(output_path, self.cfg)
            if self.dataset is not None:
                save_action_stats(output_path, self.dataset)
        else:
            output_path = None

        if self.accelerator is not None:
            import torch.distributed as dist

            path_list = [output_path] if _is_main else [None]
            dist.broadcast_object_list(path_list, src=0)
            output_path = path_list[0]

        logger.info("Checkpoints will be saved to %s", output_path)

        # Detect DeepSpeed
        use_deepspeed = (
            self.accelerator is not None
            and hasattr(self.accelerator, "distributed_type")
            and str(self.accelerator.distributed_type).endswith("DEEPSPEED")
        )

        # Prepare with accelerator — wrap architecture directly (no intermediate
        # TrainableModuleWrapper). DeepSpeedEngine forwards attribute access
        # (.action_backbone / .compute_loss / .prepare_inputs / ...) to the
        # underlying module via __getattr__, and param-level grad hooks attached
        # during prepare ensure backward sync works regardless of which forward
        # path the trainer takes.
        if use_deepspeed:
            prepare_args = [self.architecture, optimizer, dataloader]
            if scheduler is not None:
                prepare_args.append(scheduler)
                self.architecture, optimizer, dataloader, scheduler = self.accelerator.prepare(*prepare_args)
            else:
                self.architecture, optimizer, dataloader = self.accelerator.prepare(*prepare_args)

            # Propagate device from accelerator down through architecture → backbones
            self.architecture.set_dtype_device(self.architecture.dtype, self.accelerator.device)
            # Frozen modules (T5, VAE) — idempotent defensive move
            self.architecture.move_frozen_to_device(self.accelerator.device)

            logger.info("DeepSpeed: architecture wrapped, device=%s", self.accelerator.device)
        elif self.accelerator is not None:
            # Plain DDP / single GPU — also prepare model so accelerator.accumulate works
            self.architecture, optimizer, dataloader = self.accelerator.prepare(
                self.architecture, optimizer, dataloader
            )

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

        import time as _time

        opt_step = 0
        global_step = 0
        _step_t0 = _time.monotonic()
        pbar = tqdm(total=total_steps, desc="Training", unit="step")

        self.on_train_begin(output_path=output_path, total_steps=total_steps)

        # Accelerator must exist (constructed unconditionally in scripts/train.py).
        # All paths (DDP / DeepSpeed) go through accelerator.accumulate(...) so
        # gradient accumulation is delegated to the framework — no manual gating.
        assert self.accelerator is not None, "OpenWAMTrainer requires an Accelerator"

        for epoch in range(num_epochs):
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            for batch in dataloader:
                with self.accelerator.accumulate(self.architecture):
                    losses = self.compute_loss(batch)
                    loss = losses["total"]
                    self.accelerator.backward(loss)

                    grad_norm = torch.tensor(0.0, device=loss.device)
                    if self.accelerator.sync_gradients:
                        if max_grad_norm is not None:
                            grad_norm_val = self.accelerator.clip_grad_norm_(all_params, max_grad_norm)
                            grad_norm = torch.tensor(float(grad_norm_val), device=loss.device)
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        opt_step += 1

                self._current_step = global_step
                global_step += 1

                # --- Gather losses across all ranks ---
                _device = loss.device
                if self.accelerator is not None and self.accelerator.num_processes > 1:
                    local_metrics = torch.tensor(
                        [
                            loss.detach().float().item(),
                            losses["video"].item()
                            if isinstance(losses["video"], torch.Tensor)
                            else float(losses["video"]),
                            losses["action"].item()
                            if isinstance(losses["action"], torch.Tensor)
                            else float(losses["action"]),
                            grad_norm.item(),
                        ],
                        device=_device,
                        dtype=torch.float32,
                    ).reshape(1, -1)
                    gathered = self.accelerator.gather(local_metrics)
                    global_metrics = gathered.mean(dim=0)
                    loss_total = global_metrics[0].item()
                    loss_video = global_metrics[1].item()
                    loss_action = global_metrics[2].item()
                    global_grad_norm = global_metrics[3].item()
                else:
                    loss_total = loss.detach().item()
                    loss_video = (
                        losses["video"].item() if isinstance(losses["video"], torch.Tensor) else float(losses["video"])
                    )
                    loss_action = (
                        losses["action"].item()
                        if isinstance(losses["action"], torch.Tensor)
                        else float(losses["action"])
                    )
                    global_grad_norm = grad_norm.item()

                # --- Step hook (logging, progress bar, wandb) ---
                current_lr = optimizer.param_groups[0]["lr"]
                _now = _time.monotonic()
                steps_per_sec = 1.0 / max(_now - _step_t0, 1e-9)
                _step_t0 = _now

                self.on_step_end(
                    global_step,
                    loss_total=loss_total,
                    loss_video=loss_video,
                    loss_action=loss_action,
                    grad_norm=global_grad_norm,
                    lr=current_lr,
                    epoch=epoch,
                    pbar=pbar,
                    wandb_run=wandb_run,
                    steps_per_sec=steps_per_sec,
                    batch_size=batch_size,
                )

                # Periodic checkpoint saving. ALL ranks must enter
                # ``save_checkpoint`` together because under ZeRO-3 it issues a
                # collective all-gather to consolidate sharded params; only the
                # rank-0 file IO is gated.
                _is_main = self.accelerator is None or self.accelerator.is_main_process
                if self.should_save_checkpoint(global_step, save_steps):
                    ckpt_path = os.path.join(output_path, f"checkpoint_step_{global_step}.safetensors")
                    if _is_main:
                        msg_start = f"[checkpoint] Saving step {global_step} -> {ckpt_path}"
                        logger.info(msg_start)
                        tqdm.write(msg_start)
                    self.save_checkpoint(ckpt_path)
                    if _is_main:
                        msg_done = f"[checkpoint] Saved: {ckpt_path}"
                        logger.info(msg_done)
                        tqdm.write(msg_done)
                        manage_checkpoints(output_path, keep_last_k)

                if max_steps and global_step >= max_steps:
                    pbar.close()
                    self.on_train_end(global_step, output_path=output_path, wandb_run=wandb_run)
                    return

        pbar.close()

        # Save final checkpoint. Same rule as periodic saves: ALL ranks enter
        # ``save_checkpoint`` (collective under ZeRO-3); only rank-0 writes IO.
        _is_main = self.accelerator is None or self.accelerator.is_main_process
        if save_steps:
            ckpt_path = os.path.join(output_path, f"checkpoint_step_{global_step}.safetensors")
            if _is_main:
                msg_start = f"[checkpoint] Saving final step {global_step} -> {ckpt_path}"
                logger.info(msg_start)
                tqdm.write(msg_start)
            self.save_checkpoint(ckpt_path)
            if _is_main:
                msg_done = f"[checkpoint] Saved final: {ckpt_path}"
                logger.info(msg_done)
                tqdm.write(msg_done)
                manage_checkpoints(output_path, keep_last_k)

        self.on_train_end(global_step, output_path=output_path, wandb_run=wandb_run)

    def save_checkpoint(self, path: str):
        """Export full architecture state to safetensors. Safe under ZeRO-1/2/3, DDP, and single-process.

        ALL ranks must call this together. Under ZeRO-3 ``Accelerator.get_state_dict``
        issues a collective all-gather to consolidate sharded params on rank 0; under
        ZeRO-1/2 / DDP / single-process it falls back to a local ``unwrap(model).state_dict()``.
        Only rank 0 writes the file.

        Note: ``self.architecture`` after ``accelerator.prepare()`` is a ``DeepSpeedEngine`` —
        calling its ``.save_checkpoint(path)`` directly would dispatch to DeepSpeed's own
        method (collective sharded checkpoint), so we route through ``get_state_dict`` and
        write the safetensors file ourselves.
        """
        from safetensors.torch import save_file

        if self.accelerator is not None:
            # Collective path. Under ZeRO-3 this gathers sharded params; under ZeRO-1/2
            # the params are already full on every rank and this is a local copy.
            state_dict = self.accelerator.get_state_dict(self.architecture)
            if not self.accelerator.is_main_process:
                return
        else:
            state_dict = self.architecture.state_dict()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        save_file(state_dict, path)

    def load_checkpoint(self, path: str, strict: bool = True):
        """Load a checkpoint into the architecture.

        Currently supports ZeRO-1 / ZeRO-2 / DDP / single-process. ZeRO-3 is NOT
        supported: after ``accelerator.prepare()`` each rank holds only a sharded
        ``ds_tensor`` slice of every parameter, so a naive ``load_state_dict``
        would either shape-mismatch or silently write a full tensor into a
        slice slot. Tracked as a TODO in README (resume-from-checkpoint under
        ZeRO-3 needs ``deepspeed.zero.GatheredParameters`` plumbing).

        Loads weights into the *unwrapped* underlying ``BaseWAMArchitecture`` so we don't
        invoke ``DeepSpeedEngine.load_checkpoint`` (which expects DeepSpeed's own sharded
        checkpoint layout, not our flat safetensors).

        ``strict`` defaults to ``True`` so a renamed state-dict (e.g. v1.0 → v1.1
        where ``moe_expert_dit.*`` became ``shared_moe.*``) raises explicitly
        rather than dropping weights silently.
        """
        from safetensors.torch import load_file

        # ZeRO-3 guard. Raise before doing anything destructive to the in-memory
        # sharded params; the caller has to either load before prepare() or wrap
        # the load in ``deepspeed.zero.GatheredParameters``.
        if self.accelerator is not None:
            ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
            if ds_plugin is not None:
                zero_stage = int(getattr(ds_plugin, "zero_stage", 0) or 0)
                if zero_stage >= 3:
                    raise RuntimeError(
                        "load_checkpoint does not support ZeRO-3: params are sharded "
                        "after accelerator.prepare(). Either call this before prepare(), "
                        "or wrap the load in deepspeed.zero.GatheredParameters("
                        "list(unwrapped.parameters()), modifier_rank=0). "
                        "See the resume-from-checkpoint TODO in README."
                    )

        unwrapped = (
            self.accelerator.unwrap_model(self.architecture) if self.accelerator is not None else self.architecture
        )
        sd = load_file(path)
        unwrapped.load_state_dict(sd, strict=strict)
