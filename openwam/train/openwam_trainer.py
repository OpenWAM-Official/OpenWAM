"""OpenWAM joint video-action trainer (self-contained).

Call order — core skeleton only:

  __init__():  seed -> build_architecture -> freeze -> init_schedulers
               -> [latent setup] -> [load action stats] -> log param counts
  train():     build optimizer/dataloader/scheduler -> setup output dir
               -> accelerate prepare -> loop{ compute_loss -> backward/clip/step
               -> reduce metrics -> record vram -> log step -> maybe save ckpt }
               -> save final ckpt -> vram summary

Stateless helpers live in ``openwam.train.utils`` (config / param report / LR /
wandb / metric reduction / debug-CSV / VRAM tracker / checkpoint mgmt /
optimizer groups / seeding).

Usage:
    trainer = OpenWAMTrainer(cfg, accelerator, dataset)
    trainer.train()
"""

import logging
import math
import os

import numpy as np
import torch
from omegaconf import DictConfig

from openwam.train.utils.checkpointing import manage_checkpoints, save_config, save_normalization_stats
from openwam.train.utils.optimizer_groups import build_trainable_parameters
from openwam.train.utils.seeding import per_step_seed, seed_process, wire_sampler_seed
from openwam.train.utils.training_utils import (
    VramTracker,
    build_cosine_scheduler,
    cfg_get,
    init_wandb,
    latent_action_enabled,
    log_parameter_counts,
    reduce_step_metrics,
    write_debug_loss_row,
)

logger = logging.getLogger(__name__)


class OpenWAMTrainer:
    """Joint video-action trainer for OpenWAM. See module docstring for call order.

    Args:
        cfg: Hydra DictConfig with model, training, data, project sections.
        accelerator: HuggingFace Accelerator instance.
        dataset: Training dataset (used for action stats loading).
    """

    def __init__(self, cfg: DictConfig, accelerator=None, dataset=None):
        self.cfg = cfg
        self.dataset = dataset
        self.accelerator = accelerator
        self._current_step = 0
        self._vram = VramTracker()

        # ---- Reproducible seed (FastWAM-style, yaml-driven) ----
        # Seed before build_architecture so DiT/ActionDiT weight init is
        # deterministic. seed_process uses the same RANK_OFFSET rank stride as
        # per_step_seed and the launcher's seed_everything, so a process's init
        # and per-step seeds share one rank window (cudnn left to the launcher).
        project_cfg = getattr(cfg, "project", None)
        yaml_seed = getattr(project_cfg, "seed", None) if project_cfg is not None else None
        self._rank = int(os.environ.get("RANK", 0))
        self._run_seed = int(yaml_seed) if yaml_seed is not None else None
        if self._run_seed is not None:
            seed_process(self._run_seed, rank=self._rank)
            if self._rank == 0:
                logger.info("Reproducible mode: cfg.project.seed=%d (rank=%d)", self._run_seed, self._rank)

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
        self._warn_unfrozen_external_encoder(freeze_list)

        # Initialize all schedulers (video + action) inside architecture
        self.architecture.init_training_schedulers(1000)

        # Loss weights from the training config
        self.lambda_video = float(t.lambda_video)
        self.lambda_action = float(t.lambda_action)
        self.lambda_decoder = float(cfg_get(t, "lambda_decoder", 0.0))
        self.latent_action_provider = None
        self.latent_action_enabled = latent_action_enabled(cfg)

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

        self.model = self  # self-reference some external callers expect

        is_main = self.accelerator is None or self.accelerator.is_main_process
        log_parameter_counts(self.architecture, is_main=is_main)

    def train(self, num_epochs: int = None, max_steps: int = None):
        """Run the training loop (HuggingFace Accelerate distributed)."""
        t = self.cfg.training
        num_epochs = num_epochs or int(t.num_epochs)
        max_steps = max_steps or getattr(t, "max_steps", None)
        batch_size = int(t.batch_size)
        grad_accum = int(t.gradient_accumulation_steps)

        debug = bool(getattr(t, "debug", False))
        if debug:
            max_steps = 20
            save_steps_override = 10
            logger.info("DEBUG mode: max_steps=20, save@10, constant LR")

        optimizer = self.build_optimizer()
        dataloader = self.build_dataloader(batch_size)
        max_grad_norm = float(t.max_grad_norm) if getattr(t, "max_grad_norm", None) else None

        steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
        total_opt_steps = steps_per_epoch * num_epochs
        if max_steps:
            total_opt_steps = min(total_opt_steps, max_steps)
        scheduler = self.build_lr_scheduler(optimizer, total_opt_steps, debug=debug)

        if debug:
            save_steps = save_steps_override
        else:
            save_steps = getattr(t, "save_steps", None)
            if save_steps is not None:
                save_steps = int(save_steps)
        keep_last_k = int(getattr(t, "keep_last_k_ckpts", 3))

        output_path = self._setup_output_dir(debug)
        optimizer, dataloader, scheduler = self._prepare_accelerate(optimizer, dataloader, scheduler)

        if self._run_seed is not None:
            wire_sampler_seed(dataloader, int(self._run_seed))

        all_params = [p for group in optimizer.param_groups for p in group["params"]]

        is_main = self.accelerator is None or self.accelerator.is_main_process
        wandb_run = None if (debug or not is_main) else init_wandb(self.cfg)

        from tqdm import tqdm

        total_steps = len(dataloader) * num_epochs
        if max_steps:
            total_steps = min(total_steps, max_steps)

        import time as _time

        opt_step = 0
        global_step = 0
        _step_t0 = _time.monotonic()
        pbar = tqdm(total=total_steps, desc="Training", unit="step")

        self._vram.begin()
        assert self.accelerator is not None, "OpenWAMTrainer requires an Accelerator"

        # Per-step manual_seed makes timestep + noise sampling in compute_loss
        # reproducible across runs and ZeRO stages: same (rank, step) -> same RNG,
        # different ranks at the same step keep in-batch timestep diversity.
        # Gated on _run_seed so unseeded production runs stay fully stochastic.
        for epoch in range(num_epochs):
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(epoch)
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

                metrics = reduce_step_metrics(self.accelerator, losses, grad_norm)

                current_lr = optimizer.param_groups[0]["lr"]
                _now = _time.monotonic()
                steps_per_sec = 1.0 / max(_now - _step_t0, 1e-9)
                _step_t0 = _now

                need_mem_detail = (wandb_run is not None) or bool(debug)
                mem_stats = self._vram.record(need_detail=need_mem_detail)

                self._log_step(
                    metrics=metrics,
                    global_step=global_step,
                    opt_step=opt_step,
                    epoch=epoch,
                    lr=current_lr,
                    steps_per_sec=steps_per_sec,
                    batch_size=batch_size,
                    pbar=pbar,
                    wandb_run=wandb_run,
                    mem_stats=mem_stats,
                    debug=debug,
                    output_path=output_path,
                )

                if save_steps is not None and global_step > 0 and global_step % save_steps == 0:
                    self._save_checkpoint_files(output_path, global_step, keep_last_k, final=False)

                if max_steps and global_step >= max_steps:
                    pbar.close()
                    self._vram.write_summary(
                        global_step=global_step, output_path=output_path, is_main=is_main, wandb_run=wandb_run
                    )
                    return

        pbar.close()
        if save_steps:
            self._save_checkpoint_files(output_path, global_step, keep_last_k, final=True)
        self._vram.write_summary(global_step=global_step, output_path=output_path, is_main=is_main, wandb_run=wandb_run)

    # ---- Contract: loss / optimizer / checkpoint I/O ----
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

    # ---- Construction helpers ----
    def _setup_latent_action(self, cfg: DictConfig) -> None:
        """Validate latent-action config and build the latent action provider."""
        latent_cfg = cfg.model.action_backbone.latent_encoder
        output_cfg = cfg_get(latent_cfg, "output")
        action_dim = int(cfg_get(output_cfg, "action_dim", 0) or 0)
        token_dim = int(cfg_get(output_cfg, "token_dim", action_dim) or 0)
        arch_cfg = getattr(cfg.model, "architecture", None)
        cfg_uses_proprio = bool(cfg_get(arch_cfg, "use_proprioception", False))
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

    def _warn_unfrozen_external_encoder(self, freeze_list: list[str]) -> None:
        """Warn if an irreversible external encoder (V-JEPA 2.1 / DINOv3, no pixel
        decode) is left trainable. Such encoders are almost always pretrained-and-
        frozen; a missing freeze entry silently wastes GPU on a trainable ViT."""
        external_encoder = getattr(self.architecture, "external_encoder", None)
        if external_encoder is None or external_encoder.properties.pixel_decode:
            return
        if any(
            p == "video_backbone.video_encoder" or p.startswith("video_backbone.video_encoder.") for p in freeze_list
        ):
            return
        logger.warning(
            "external encoder %s is not in freeze_modules; ViT is fully trainable. "
            "Add 'video_backbone.video_encoder' to your model freeze list "
            "if you intended to freeze the ViT backbone.",
            type(external_encoder).__name__,
        )

    # ---- Training-loop helpers (in call order) ----
    def build_dataloader(self, batch_size: int) -> torch.utils.data.DataLoader:
        """Build the training DataLoader.

        When ``cfg.project.seed`` is set, wires a per-rank ``generator`` and a
        ``worker_init_fn`` so dataset-side randomness is reproducible across
        runs while keeping per-epoch / per-worker variation.
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
        """Linear-warmup + cosine schedule, or None (constant LR / debug)."""
        t = self.cfg.training
        if debug:
            return None
        if getattr(t, "lr_scheduler", None) == "cosine":
            return build_cosine_scheduler(optimizer, total_opt_steps=total_opt_steps, cfg=self.cfg)
        return None

    def _setup_output_dir(self, debug: bool) -> str:
        """Create the rank-0 run dir (timestamped), save deploy assets, broadcast the path."""
        base_output_path = getattr(self.cfg.training, "output_path", "./models")
        is_main = self.accelerator is None or self.accelerator.is_main_process
        if is_main:
            from datetime import datetime

            run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if debug:
                run_dir_name += "_debug"
            output_path = os.path.join(base_output_path, run_dir_name)
            os.makedirs(output_path, exist_ok=True)
            # Make the checkpoint self-contained for deploy: each backbone saves its
            # own assets, then config + action stats. Runs BEFORE save_config so
            # config.yaml carries the backbones' merged reconstruction specs.
            self.architecture.save_assets_for_deployment(output_path, self.cfg)
            save_config(output_path, self.cfg)
            if self.dataset is not None:
                save_normalization_stats(output_path, self.dataset)
        else:
            output_path = None
        if self.accelerator is not None:
            import torch.distributed as dist

            path_list = [output_path] if is_main else [None]
            dist.broadcast_object_list(path_list, src=0)
            output_path = path_list[0]
        logger.info("Checkpoints will be saved to %s", output_path)
        return output_path

    def _prepare_accelerate(self, optimizer, dataloader, scheduler):
        """Wrap architecture/optimizer/dataloader with accelerate (DeepSpeed or DDP)."""
        use_deepspeed = (
            self.accelerator is not None
            and hasattr(self.accelerator, "distributed_type")
            and str(self.accelerator.distributed_type).endswith("DEEPSPEED")
        )
        if use_deepspeed:
            prepare_args = [self.architecture, optimizer, dataloader]
            if scheduler is not None:
                prepare_args.append(scheduler)
                self.architecture, optimizer, dataloader, scheduler = self.accelerator.prepare(*prepare_args)
            else:
                self.architecture, optimizer, dataloader = self.accelerator.prepare(*prepare_args)
            # Propagate device down through architecture; frozen modules (T5/VAE) idempotent move.
            self.architecture.set_dtype_device(self.architecture.dtype, self.accelerator.device)
            self.architecture.move_frozen_to_device(self.accelerator.device)
            self._move_latent_to_device(self.accelerator.device)
            logger.info("DeepSpeed: architecture wrapped, device=%s", self.accelerator.device)
        elif self.accelerator is not None:
            self.architecture, optimizer, dataloader = self.accelerator.prepare(
                self.architecture, optimizer, dataloader
            )
            self._move_latent_to_device(self.accelerator.device)
        return optimizer, dataloader, scheduler

    def _move_latent_to_device(self, device) -> None:
        if self.latent_action_provider is not None:
            self.latent_action_provider.to(device)
            self.latent_action_provider.device = torch.device(device)

    def _log_step(
        self,
        *,
        metrics,
        global_step,
        opt_step,
        epoch,
        lr,
        steps_per_sec,
        batch_size,
        pbar,
        wandb_run,
        mem_stats,
        debug,
        output_path,
    ) -> None:
        """Update progress bar, log to wandb, and (debug) write the loss-history CSV row.

        Loss streams come from ``_loss_labels()`` so non-latent mode shows only the
        real action loss (no dead decoder column).
        """
        labels = self._loss_labels()
        loss_total = metrics["loss_total"]
        loss_video = metrics["loss_video"]
        grad_norm = metrics["grad_norm"]

        if pbar is not None:
            postfix = {"loss": f"{loss_total:.4f}", "video": f"{loss_video:.4f}"}
            for name, key in labels:
                postfix[name] = f"{metrics[key]:.4f}"
            postfix["lr"] = f"{lr:.2e}"
            postfix["epoch"] = epoch
            pbar.set_postfix(postfix)
            pbar.update(1)

        mem_stats = mem_stats or {}

        if wandb_run is not None:
            num_procs = self.accelerator.num_processes if self.accelerator is not None else 1
            log_dict = {
                "train/loss": loss_total,
                "train/loss_video": loss_video,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "performance/steps_per_sec": steps_per_sec,
                "performance/samples_per_sec": steps_per_sec * batch_size * num_procs,
            }
            for name, key in labels:
                log_dict[f"train/loss_{name}"] = metrics[key]
            for k, v in mem_stats.items():
                log_dict[f"memory/{k}"] = v
            wandb_run.log(log_dict, step=global_step)

        if not debug:
            return
        is_main = self.accelerator is None or self.accelerator.is_main_process
        if not is_main:
            return
        mem_suffix = ""
        if mem_stats:
            mem_suffix = (
                f" peak_alloc={mem_stats.get('step_peak_alloc_gb', 0):.2f}GB"
                f" peak_res={mem_stats.get('step_peak_reserved_gb', 0):.2f}GB"
            )
        loss_parts = " ".join(f"{name}={metrics[key]:.6f}" for name, key in labels)
        msg = (
            f"[debug][step {global_step:04d} opt {opt_step:04d}] "
            f"loss={loss_total:.6f} video={loss_video:.6f} {loss_parts} "
            f"grad_norm={grad_norm:.6f} lr={lr:.3e} epoch={epoch} "
            f"steps_per_sec={steps_per_sec:.3f}{mem_suffix}"
        )
        logger.info(msg)
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg, flush=True)

        if output_path:
            write_debug_loss_row(
                output_path,
                labels=labels,
                metrics=metrics,
                global_step=global_step,
                opt_step=opt_step,
                epoch=epoch,
                lr=lr,
                steps_per_sec=steps_per_sec,
                mem_stats=mem_stats,
            )

    def _loss_labels(self) -> list[tuple[str, str]]:
        """(display_name, metrics_key) pairs for the action/decoder loss streams.

        Latent mode: the action stream predicts the LATENT action, and the decoder
        MSE is the REAL action loss. Non-latent: only the real action stream (the
        decoder is unused, so it gets no label/column).
        """
        if self.latent_action_enabled:
            return [("latent_action", "loss_action"), ("action", "loss_decoder")]
        return [("action", "loss_action")]

    def _save_checkpoint_files(self, output_path: str, global_step: int, keep_last_k: int, *, final: bool) -> None:
        """ALL ranks enter save_checkpoint (DeepSpeed collective); only rank-0 writes IO + prunes."""
        from tqdm import tqdm

        is_main = self.accelerator is None or self.accelerator.is_main_process
        ckpt_path = os.path.join(output_path, f"checkpoint_step_{global_step}.safetensors")
        tag = "final " if final else ""
        if is_main:
            msg = f"[checkpoint] Saving {tag}step {global_step} -> {ckpt_path}"
            logger.info(msg)
            tqdm.write(msg)
        self.save_checkpoint(ckpt_path)
        if is_main:
            msg = f"[checkpoint] Saved{' final' if final else ''}: {ckpt_path}"
            logger.info(msg)
            tqdm.write(msg)
            manage_checkpoints(output_path, keep_last_k)
