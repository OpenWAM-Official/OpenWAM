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
import random
import shutil

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

        # ---- Reproducible seed (FastWAM-style, yaml-driven) ----
        # Reads seed from ``cfg.project.seed`` and seeds Python random, numpy,
        # torch CPU and torch CUDA RNGs. Must run before ``build_architecture``
        # so any randomness during model construction (DiT/ActionDiT weight
        # init, including a future ``video_backbone.from_scratch`` reinit path)
        # lands on deterministic RNG.
        #
        # Mirrors FastWAM's set_global_seed
        # (references/FastWAM/src/fastwam/utils/pytorch_utils.py:17). We
        # deliberately do NOT touch cudnn.deterministic, cudnn.benchmark,
        # CUBLAS_WORKSPACE_CONFIG, or torch.use_deterministic_algorithms:
        # they would gain bit-exact loss reproducibility at the cost of cuDNN
        # autotuning and FSDP/fused-attention compatibility, and they are not
        # needed for "same seed -> same initial DiT weights".
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
        # Wrap construction in a ZeRO-3 init-disable scope: when the Accelerator
        # was built with ``zero3_init_flag=True``, DeepSpeed enters a global
        # ``zero.Init(enabled=True)`` context that auto-partitions every
        # nn.Parameter at allocation time. See ``_zero3_init_disabled`` below
        # for the actual deepspeed 0.18.5 behavior — short version: the
        # construction-time skip avoids OOM-prone overhead on huge frozen
        # modules (text_encoder, VAE), but ``deepspeed.initialize`` still
        # partitions every trainable param post-prepare.
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
            self.architecture.set_dtype_device(self.architecture.dtype, self.architecture.device)

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
            raise ValueError(
                "action_timestep_per_token=True is not supported by the current OpenWAM "
                "training path. Use per-sample action timesteps."
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

            bb_counts = {name: _count(module) for name, module in self.architecture.backbones.items()}
            extra_counts = {}
            for name, module in self.architecture.named_children():
                if name in bb_counts:
                    continue
                total, trainable = _count(module)
                if total:
                    extra_counts[name] = (total, trainable)
            arch_total = sum(total for total, _ in bb_counts.values()) + sum(
                total for total, _ in extra_counts.values()
            )
            arch_train = sum(train for _, train in bb_counts.values()) + sum(
                train for _, train in extra_counts.values()
            )
            print("=" * 60)
            print("Parameter counts")
            for name, (total, trainable) in {**bb_counts, **extra_counts}.items():
                print(f"  {name:<15}: total={total / 1e6:7.1f}M  trainable={trainable / 1e6:7.1f}M")
            print(f"  Architecture  : total={arch_total / 1e6:7.1f}M  trainable={arch_train / 1e6:7.1f}M")
            print("=" * 60, flush=True)

    @staticmethod
    def _wire_sampler_seed(dataloader, run_seed: int) -> None:
        """Tie the (possibly wrapped) DistributedSampler's ``seed`` attribute to
        ``run_seed`` so per-epoch shuffle order varies with ``cfg.project.seed``.

        Without this, ``accelerator.prepare`` keeps the auto-wrapped
        ``DistributedSampler`` at its upstream default ``seed=0`` and shuffle
        order is identical regardless of ``cfg.project.seed``. The trainer's
        per-epoch ``dataloader.set_epoch(epoch)`` call then combines this seed
        with the epoch number so each epoch still gets its own permutation.

        Walks both ``dataloader.sampler`` and ``dataloader.batch_sampler.sampler``
        — accelerate's wrapping can place the underlying sampler in either spot.

        If no sampler with a ``.seed`` attribute is reachable (e.g. unusual
        accelerate wrapper, IterableDataset path, or shuffle=False loader),
        logs a WARNING so the user doesn't silently get the upstream default
        while expecting ``cfg.project.seed`` to control shuffle order.
        """
        sampler = getattr(dataloader, "sampler", None)
        if sampler is None:
            batch_sampler = getattr(dataloader, "batch_sampler", None)
            sampler = getattr(batch_sampler, "sampler", None) if batch_sampler is not None else None
        if sampler is not None and hasattr(sampler, "seed"):
            old = sampler.seed
            sampler.seed = int(run_seed)
            logger.info(
                "%s.seed wired to cfg.project.seed: %s -> %d",
                type(sampler).__name__,
                old,
                run_seed,
            )
        else:
            logger.warning(
                "cfg.project.seed=%d is set but the prepared dataloader has no sampler with a "
                "``.seed`` attribute (found %s). Per-epoch shuffle order will fall back to the "
                "library default (typically seed=0) and will NOT vary with cfg.project.seed. "
                "Other seeded paths (model init, worker_init_fn, training-loop noise) are "
                "unaffected.",
                run_seed,
                type(sampler).__name__ if sampler is not None else "None",
            )

    @staticmethod
    def _zero3_init_disabled():
        """Context that skips DeepSpeed ZeRO-3 *construction-time* partitioning.

        On deepspeed 0.18.5 ``zero.Init(enabled=False)`` is a no-op context
        manager (``partition_parameters.py:344-358``) — it merely suppresses
        the ``zero.Init`` constructor's per-Parameter hooks while the scope is
        active, so newly-allocated params have no ``ds_id`` / ``ds_status``
        attached at allocation time. It does **not** make the resulting params
        "stay replicated": once ``deepspeed.initialize`` runs,
        ``_convert_to_zero_parameters`` (``parameter_offload.py:205-226``) walks
        the whole model and partitions every trainable param it finds — frozen
        params included if they're still on the trainable graph.

        What this scope actually buys: avoiding the construction-time overhead
        of running ``zero.Init`` hooks on every leaf as huge frozen modules
        (text_encoder ~13 GiB umt5-xxl, VAE) are built. Frozen modules that
        the trainer later marks ``requires_grad_(False)`` and removes from the
        optimizer's parameter groups stay un-partitioned in practice because
        DeepSpeed's prepare only partitions params it actually owns; that's a
        separate concern from this context manager. The DDP / single-GPU path
        is unaffected because deepspeed isn't importable there — the
        ``ImportError`` branch returns a ``nullcontext``.
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
        """Build the training DataLoader. Override for custom sampling.

        When ``cfg.project.seed`` is configured, hooks in two seeding pieces
        so dataset-side randomness becomes reproducible across runs while
        retaining within-run diversity:

        * ``generator`` — DataLoader's own RNG, used to derive each worker's
          ``base_seed`` at every ``__iter__``. The generator's state advances
          naturally per epoch (each epoch consumes one random draw), so
          workers spawned in epoch N see a different ``base_seed`` from
          workers spawned in epoch M.
        * ``worker_init_fn`` — ``dataloader_worker_init_fn`` reads PyTorch's
          auto-derived ``info.seed`` (which carries the per-epoch / per-worker
          variation above) and seeds Python ``random`` + NumPy with it. This
          makes the dataset transforms in
          ``openwam/dataloader/transforms/video.py`` (random crop, brightness,
          flip) reproducible across runs **without** repeating the same
          augmentation in every epoch.
        """
        t = self.cfg.training
        kwargs: dict = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(t.dataset_num_workers),
            collate_fn=list,
            pin_memory=True,
        )
        if self._run_seed is not None:
            from openwam.train.utils.seeding import dataloader_worker_init_fn, make_dataloader_generator

            kwargs["generator"] = make_dataloader_generator(self._run_seed, rank=self._rank)
            kwargs["worker_init_fn"] = dataloader_worker_init_fn
        return torch.utils.data.DataLoader(self.dataset, **kwargs)

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
        # Run-level peak VRAM trackers. ``_record_step_memory`` updates these on
        # every step and (when ``need_detail=True``) resets CUDA's internal
        # peak so the next step is measured cleanly; the run-level max survives
        # the reset and is logged in ``on_train_end``.
        #
        # The reset below runs AFTER ``accelerator.prepare(...)`` (called by
        # ``train()`` before invoking this hook), so init-time allocations
        # (DiT params, optimizer state, gradient buffers, sharded ZeRO state)
        # are NOT counted in the run-level peak — the numbers reported in
        # ``memory_summary.csv`` reflect training-loop peak only.
        self._run_peak_alloc_gb = 0.0
        self._run_peak_reserved_gb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _record_step_memory(self, need_detail: bool = False) -> dict:
        """Snapshot VRAM peaks, optionally returning per-step detail.

        Always: reads ``max_memory_allocated`` / ``max_memory_reserved`` and
        updates the Python-side run-level max so ``on_train_end``'s
        ``memory_summary.csv`` is always populated.

        ``need_detail=True``: additionally reads current live alloc + reserved,
        resets CUDA's per-step peak counter, and returns a full dict for wandb
        / debug-CSV consumption. ``reset_peak_memory_stats`` is gated behind
        this flag because it is the only call here that could pollute
        ``steps_per_sec`` measurements when the result is unused; the peak
        reads themselves are host-side counter lookups on the CUDA caching
        allocator and do not synchronize.

        Returns an empty dict on CPU or when ``need_detail=False``.
        """
        if not torch.cuda.is_available():
            return {}
        peak_alloc_gb = float(torch.cuda.max_memory_allocated()) / 1e9
        peak_reserved_gb = float(torch.cuda.max_memory_reserved()) / 1e9
        self._run_peak_alloc_gb = max(getattr(self, "_run_peak_alloc_gb", 0.0), peak_alloc_gb)
        self._run_peak_reserved_gb = max(getattr(self, "_run_peak_reserved_gb", 0.0), peak_reserved_gb)
        if not need_detail:
            return {}
        alloc_gb = float(torch.cuda.memory_allocated()) / 1e9
        reserved_gb = float(torch.cuda.memory_reserved()) / 1e9
        torch.cuda.reset_peak_memory_stats()
        return {
            "mem_alloc_gb": alloc_gb,
            "mem_reserved_gb": reserved_gb,
            "step_peak_alloc_gb": peak_alloc_gb,
            "step_peak_reserved_gb": peak_reserved_gb,
            "run_peak_alloc_gb": self._run_peak_alloc_gb,
            "run_peak_reserved_gb": self._run_peak_reserved_gb,
        }

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

        mem_stats = ctx.get("mem_stats") or {}

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
            for k, v in mem_stats.items():
                log_dict[f"memory/{k}"] = v
            wandb_run.log(log_dict, step=global_step)

        if bool(ctx.get("debug", False)):
            is_main = self.accelerator is None or self.accelerator.is_main_process
            if not is_main:
                return
            opt_step = int(ctx.get("opt_step", global_step))
            mem_suffix = ""
            if mem_stats:
                mem_suffix = (
                    f" peak_alloc={mem_stats.get('step_peak_alloc_gb', 0):.2f}GB"
                    f" peak_res={mem_stats.get('step_peak_reserved_gb', 0):.2f}GB"
                )
            msg = (
                f"[debug][step {global_step:04d} opt {opt_step:04d}] "
                f"loss={loss_total:.6f} video={loss_video:.6f} action={loss_action:.6f} "
                f"grad_norm={grad_norm:.6f} lr={lr:.3e} epoch={epoch} "
                f"steps_per_sec={steps_per_sec:.3f}{mem_suffix}"
            )
            logger.info(msg)
            if pbar is not None:
                pbar.write(msg)
            else:
                print(msg, flush=True)

            output_path = ctx.get("output_path")
            if output_path:
                loss_log_path = os.path.join(output_path, "debug_loss_history.csv")
                write_header = not os.path.exists(loss_log_path)
                with open(loss_log_path, "a", encoding="utf-8") as f:
                    if write_header:
                        f.write(
                            "step,opt_step,epoch,loss,loss_video,loss_action,grad_norm,lr,steps_per_sec,"
                            "mem_alloc_gb,mem_reserved_gb,step_peak_alloc_gb,step_peak_reserved_gb,"
                            "run_peak_alloc_gb,run_peak_reserved_gb\n"
                        )
                    f.write(
                        f"{global_step},{opt_step},{epoch},{loss_total:.10g},{loss_video:.10g},"
                        f"{loss_action:.10g},{grad_norm:.10g},{lr:.10g},{steps_per_sec:.10g},"
                        f"{mem_stats.get('mem_alloc_gb', float('nan')):.6g},"
                        f"{mem_stats.get('mem_reserved_gb', float('nan')):.6g},"
                        f"{mem_stats.get('step_peak_alloc_gb', float('nan')):.6g},"
                        f"{mem_stats.get('step_peak_reserved_gb', float('nan')):.6g},"
                        f"{mem_stats.get('run_peak_alloc_gb', float('nan')):.6g},"
                        f"{mem_stats.get('run_peak_reserved_gb', float('nan')):.6g}\n"
                    )

    def on_train_end(self, global_step: int, *, output_path: str, wandb_run=None, **ctx):
        """Hook called after training completes. Override for custom teardown."""
        run_peak_alloc = float(getattr(self, "_run_peak_alloc_gb", 0.0))
        run_peak_reserved = float(getattr(self, "_run_peak_reserved_gb", 0.0))
        if torch.cuda.is_available() and (run_peak_alloc > 0.0 or run_peak_reserved > 0.0):
            is_main = self.accelerator is None or self.accelerator.is_main_process
            if is_main:
                summary = (
                    f"[memory] run peak alloc={run_peak_alloc:.2f}GB "
                    f"reserved={run_peak_reserved:.2f}GB (rank0)"
                )
                logger.info(summary)
                print(summary, flush=True)
                if output_path:
                    summary_path = os.path.join(output_path, "memory_summary.csv")
                    write_header = not os.path.exists(summary_path)
                    with open(summary_path, "a", encoding="utf-8") as f:
                        if write_header:
                            f.write("global_step,run_peak_alloc_gb,run_peak_reserved_gb\n")
                        f.write(f"{global_step},{run_peak_alloc:.6g},{run_peak_reserved:.6g}\n")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "memory/run_peak_alloc_gb": run_peak_alloc,
                        "memory/run_peak_reserved_gb": run_peak_reserved,
                    },
                    step=global_step,
                )
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
            save_steps_override = 10
            logger.info("DEBUG mode: max_steps=20, save@10, constant LR")

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
            # Inject video backbone component specs into config for deployment.
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
            # Copy tokenizer so component-spec deployment does not depend on
            # ``model.video_backbone.model_path`` being reachable.
            from openwam.model.video_backbone.wan.component_specs import copy_video_backbone_tokenizer

            copy_video_backbone_tokenizer(output_path, self.cfg)
            # Copy VLM checkpoint so deploy is self-contained (tri_system).
            vlm_bb = getattr(self.architecture, "vlm_backbone", None)
            if vlm_bb is not None and getattr(vlm_bb, "_checkpoint_path", None):
                vlm_dest = os.path.join(output_path, "vlm_backbone")
                if not os.path.exists(vlm_dest):
                    shutil.copytree(vlm_bb._checkpoint_path, vlm_dest)
                    logger.info("Copied VLM checkpoint to %s", vlm_dest)
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

        # Wire the (possibly wrapped) DistributedSampler's ``seed`` to
        # ``cfg.project.seed``. Without this, accelerator.prepare's auto-wrapped
        # DistributedSampler keeps the default ``seed=0`` and the per-epoch
        # shuffle order is identical regardless of cfg.project.seed.
        # set_epoch (called every epoch in the training loop) combines this
        # with the epoch number, so each epoch still gets its own permutation.
        if self._run_seed is not None:
            self._wire_sampler_seed(dataloader, int(self._run_seed))

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

        # Per-step manual_seed makes timestep + noise sampling in
        # ``base.compute_loss`` reproducible across runs (and across ZeRO stages).
        # Without this, the cumulative global-RNG state diverges between ds2/ds3
        # because each forward consumes a slightly different amount of RNG (e.g.
        # all-gather vs reduce-scatter ordering), so step-N timestep / noise
        # drift apart even with identical ``set_seed`` at process start.
        # Re-seeding before every step neutralizes that drift.
        #
        # Gated on ``self._run_seed`` (set by ``__init__`` from
        # ``cfg.project.seed``) so production runs without a configured seed
        # keep their full stochasticity — the per-step re-seed only kicks in
        # under the same opt-in switch that controls model-init determinism.
        #
        # ``per_step_seed(seed, rank=R, step=S)`` makes the (rank, step) pair
        # the full RNG identity. Same (rank, step) across ZeRO stages →
        # identical timestep + noise, so parity-able. Different ranks at the
        # same step → different timestep + noise, so the effective in-batch
        # timestep diversity that production training relies on is preserved.
        from openwam.train.utils.seeding import per_step_seed

        for epoch in range(num_epochs):
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            for batch in dataloader:
                if self._run_seed is not None:
                    step_seed = per_step_seed(self._run_seed, rank=self._rank, step=global_step)
                    torch.manual_seed(step_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(step_seed)
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

                # Peak-VRAM snapshot: read after backward + step + zero_grad
                # have all run for this iteration. ``need_detail`` is only set
                # when there is a consumer for the per-step dict (wandb log or
                # debug CSV) — see ``_record_step_memory`` for the gated reset.
                need_mem_detail = (wandb_run is not None) or bool(debug)
                mem_stats = self._record_step_memory(need_detail=need_mem_detail)

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
                    debug=debug,
                    output_path=output_path,
                    opt_step=opt_step,
                    mem_stats=mem_stats,
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
        """Export architecture state to safetensors. Safe under ZeRO-1/2/3, DDP, and single-process.

        ALL ranks must call this together. Under ZeRO-3 ``Accelerator.get_state_dict``
        issues a collective all-gather to consolidate sharded params on rank 0; under
        ZeRO-1/2 / DDP / single-process it falls back to a local ``unwrap(model).state_dict()``.
        Only rank 0 writes the file.

        VLM backbone parameters (tri_system's Qwen3-VL) are excluded from the
        safetensors file — the VLM checkpoint is saved as a separate directory
        (see ``train()``). This avoids tied-weight deduplication complexity and
        keeps the safetensors file small.
        """
        from safetensors.torch import save_file

        from openwam.model.base import _exclude_vlm_from_state_dict

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
        # Delegate to BaseWAMArchitecture.load_checkpoint which tolerates
        # missing vlm_backbone.* keys (VLM is saved as a separate directory,
        # not inside the safetensors file).
        unwrapped.load_checkpoint(path, strict=strict)
