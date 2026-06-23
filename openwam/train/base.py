"""Generic training-loop template for OpenWAM trainers.

``BaseTrainer`` owns the model-agnostic training loop skeleton: epoch/step
loop, distributed prepare, cross-rank metric reduction, logging, VRAM
tracking and checkpoint cadence. It assumes ``self.architecture`` exposes the
``BaseWAMArchitecture`` contract (set_dtype_device / move_frozen_to_device /
compute_loss via the subclass).

Subclasses implement the contract (compute_loss / build_optimizer /
save_checkpoint / load_checkpoint) and may override the hooks
(on_setup_run / _loss_labels / _prepare_auxiliary_modules) to inject
model-specific behaviour.
"""

import logging
import math
import os
from abc import ABC, abstractmethod

import torch

from openwam.train.utils.checkpointing import manage_checkpoints
from openwam.train.utils.seeding import per_step_seed
from openwam.train.utils.training_utils import build_cosine_scheduler, init_wandb

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Model-agnostic training-loop template. See module docstring."""

    def __init__(self, cfg, model=None, dataset=None, accelerator=None):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.accelerator = accelerator
        self._rank = int(os.environ.get("RANK", 0))
        self._run_seed = None
        self._current_step = 0
        self._run_peak_alloc_gb = 0.0
        self._run_peak_reserved_gb = 0.0

    # ---- Contract (subclass implements) ----
    @abstractmethod
    def compute_loss(self, batch) -> dict:
        """Return a dict with at least ``total`` plus optional breakdown."""
        ...

    @abstractmethod
    def build_optimizer(self) -> torch.optim.Optimizer: ...

    @abstractmethod
    def save_checkpoint(self, path: str): ...

    @abstractmethod
    def load_checkpoint(self, path: str, strict: bool = True): ...

    # ---- Hooks (subclass overrides to inject model-specific behaviour) ----
    def on_setup_run(self, output_path: str) -> None:
        """Rank-0 one-time run setup (default no-op). Subclass saves deploy assets / config / stats here."""

    def _loss_labels(self) -> tuple[str, str]:
        """(action_label, decoder_label) used for logging. Default literal names."""
        return "action", "decoder"

    def _prepare_auxiliary_modules(self, device) -> None:
        """Move subclass-owned auxiliary modules onto ``device`` after prepare (default no-op)."""

    # ---- Builders (generic defaults; subclass may override) ----
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

    def should_save_checkpoint(self, global_step: int, save_steps: int | None) -> bool:
        return save_steps is not None and global_step > 0 and global_step % save_steps == 0

    # ---- Loop helpers ----
    @staticmethod
    def _wire_sampler_seed(dataloader, run_seed: int) -> None:
        """Tie the (possibly wrapped) DistributedSampler's ``seed`` to ``run_seed``.

        Without this, ``accelerator.prepare``'s auto-wrapped DistributedSampler
        keeps the upstream default ``seed=0`` and per-epoch shuffle order is
        identical regardless of ``cfg.project.seed``. Walks both
        ``dataloader.sampler`` and ``dataloader.batch_sampler.sampler``; warns
        if no seedable sampler is reachable.
        """
        sampler = getattr(dataloader, "sampler", None)
        if sampler is None:
            batch_sampler = getattr(dataloader, "batch_sampler", None)
            sampler = getattr(batch_sampler, "sampler", None) if batch_sampler is not None else None
        if sampler is not None and hasattr(sampler, "seed"):
            old = sampler.seed
            sampler.seed = int(run_seed)
            logger.info("%s.seed wired to cfg.project.seed: %s -> %d", type(sampler).__name__, old, run_seed)
        else:
            logger.warning(
                "cfg.project.seed=%d is set but the prepared dataloader has no sampler with a "
                "``.seed`` attribute (found %s). Per-epoch shuffle order falls back to the library "
                "default and will NOT vary with cfg.project.seed.",
                run_seed,
                type(sampler).__name__ if sampler is not None else "None",
            )

    def _setup_output_dir(self, debug: bool) -> str:
        """Create the rank-0 run dir (timestamped), run on_setup_run, broadcast the path."""
        base_output_path = getattr(self.cfg.training, "output_path", "./models")
        is_main = self.accelerator is None or self.accelerator.is_main_process
        if is_main:
            from datetime import datetime

            run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if debug:
                run_dir_name += "_debug"
            output_path = os.path.join(base_output_path, run_dir_name)
            os.makedirs(output_path, exist_ok=True)
            self.on_setup_run(output_path)
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
            self._prepare_auxiliary_modules(self.accelerator.device)
            logger.info("DeepSpeed: architecture wrapped, device=%s", self.accelerator.device)
        elif self.accelerator is not None:
            self.architecture, optimizer, dataloader = self.accelerator.prepare(
                self.architecture, optimizer, dataloader
            )
            self._prepare_auxiliary_modules(self.accelerator.device)
        return optimizer, dataloader, scheduler

    def _reduce_step_metrics(self, losses: dict, grad_norm) -> dict:
        """Reduce loss/grad_norm across ranks (mean); single-process fast path."""
        loss = losses["total"]

        def _f(v):
            return v.item() if isinstance(v, torch.Tensor) else float(v)

        if self.accelerator is not None and self.accelerator.num_processes > 1:
            local = torch.tensor(
                [
                    loss.detach().float().item(),
                    _f(losses["video"]),
                    _f(losses["action"]),
                    _f(losses["decoder"]),
                    grad_norm.item(),
                ],
                device=loss.device,
                dtype=torch.float32,
            ).reshape(1, -1)
            g = self.accelerator.gather(local).mean(dim=0)
            return {
                "loss_total": g[0].item(),
                "loss_video": g[1].item(),
                "loss_action": g[2].item(),
                "loss_decoder": g[3].item(),
                "grad_norm": g[4].item(),
            }
        return {
            "loss_total": loss.detach().item(),
            "loss_video": _f(losses["video"]),
            "loss_action": _f(losses["action"]),
            "loss_decoder": _f(losses["decoder"]),
            "grad_norm": grad_norm.item(),
        }

    def _vram_begin(self) -> None:
        """Reset peak counters AFTER prepare so init-time allocations aren't counted."""
        self._run_peak_alloc_gb = 0.0
        self._run_peak_reserved_gb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _vram_record(self, need_detail: bool = False) -> dict:
        """Update run-level peak; with need_detail, also read live usage, reset the
        per-step peak and return a dict for wandb / debug-CSV. Empty dict on CPU
        or when need_detail=False."""
        if not torch.cuda.is_available():
            return {}
        peak_alloc_gb = float(torch.cuda.max_memory_allocated()) / 1e9
        peak_reserved_gb = float(torch.cuda.max_memory_reserved()) / 1e9
        self._run_peak_alloc_gb = max(self._run_peak_alloc_gb, peak_alloc_gb)
        self._run_peak_reserved_gb = max(self._run_peak_reserved_gb, peak_reserved_gb)
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

    def _vram_summary(self, global_step: int, output_path: str, wandb_run) -> None:
        """Log run-level peak VRAM to memory_summary.csv + wandb; finish wandb run."""
        run_peak_alloc = float(self._run_peak_alloc_gb)
        run_peak_reserved = float(self._run_peak_reserved_gb)
        if torch.cuda.is_available() and (run_peak_alloc > 0.0 or run_peak_reserved > 0.0):
            is_main = self.accelerator is None or self.accelerator.is_main_process
            if is_main:
                summary = f"[memory] run peak alloc={run_peak_alloc:.2f}GB reserved={run_peak_reserved:.2f}GB (rank0)"
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
                    {"memory/run_peak_alloc_gb": run_peak_alloc, "memory/run_peak_reserved_gb": run_peak_reserved},
                    step=global_step,
                )
        if wandb_run is not None:
            wandb_run.finish()

    def _log_step(
        self,
        *,
        global_step,
        opt_step,
        epoch,
        loss_total,
        loss_video,
        loss_action,
        loss_decoder,
        grad_norm,
        lr,
        steps_per_sec,
        batch_size,
        pbar,
        wandb_run,
        mem_stats,
        debug,
        output_path,
    ) -> None:
        """Update progress bar, log to wandb, and (debug) write the loss-history CSV row."""
        action_label, decoder_label = self._loss_labels()
        if pbar is not None:
            pbar.set_postfix(
                {
                    "loss": f"{loss_total:.4f}",
                    "video": f"{loss_video:.4f}",
                    action_label: f"{loss_action:.4f}",
                    decoder_label: f"{loss_decoder:.4f}",
                    "lr": f"{lr:.2e}",
                    "epoch": epoch,
                }
            )
            pbar.update(1)

        mem_stats = mem_stats or {}

        if wandb_run is not None:
            _num_procs = self.accelerator.num_processes if self.accelerator is not None else 1
            log_dict = {
                "train/loss": loss_total,
                "train/loss_video": loss_video,
                f"train/loss_{action_label}": loss_action,
                f"train/loss_{decoder_label}": loss_decoder,
                "train/grad_norm": grad_norm,
                "train/lr": lr,
                "performance/steps_per_sec": steps_per_sec,
                "performance/samples_per_sec": steps_per_sec * batch_size * _num_procs,
            }
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
        msg = (
            f"[debug][step {global_step:04d} opt {opt_step:04d}] "
            f"loss={loss_total:.6f} video={loss_video:.6f} {action_label}={loss_action:.6f} "
            f"{decoder_label}={loss_decoder:.6f} "
            f"grad_norm={grad_norm:.6f} lr={lr:.3e} epoch={epoch} "
            f"steps_per_sec={steps_per_sec:.3f}{mem_suffix}"
        )
        logger.info(msg)
        if pbar is not None:
            pbar.write(msg)
        else:
            print(msg, flush=True)

        if output_path:
            loss_log_path = os.path.join(output_path, "debug_loss_history.csv")
            write_header = not os.path.exists(loss_log_path)
            with open(loss_log_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(
                        f"step,opt_step,epoch,loss,loss_video,loss_{action_label},loss_{decoder_label},"
                        "grad_norm,lr,steps_per_sec,"
                        "mem_alloc_gb,mem_reserved_gb,step_peak_alloc_gb,step_peak_reserved_gb,"
                        "run_peak_alloc_gb,run_peak_reserved_gb\n"
                    )
                f.write(
                    f"{global_step},{opt_step},{epoch},{loss_total:.10g},{loss_video:.10g},"
                    f"{loss_action:.10g},{loss_decoder:.10g},{grad_norm:.10g},{lr:.10g},{steps_per_sec:.10g},"
                    f"{mem_stats.get('mem_alloc_gb', float('nan')):.6g},"
                    f"{mem_stats.get('mem_reserved_gb', float('nan')):.6g},"
                    f"{mem_stats.get('step_peak_alloc_gb', float('nan')):.6g},"
                    f"{mem_stats.get('step_peak_reserved_gb', float('nan')):.6g},"
                    f"{mem_stats.get('run_peak_alloc_gb', float('nan')):.6g},"
                    f"{mem_stats.get('run_peak_reserved_gb', float('nan')):.6g}\n"
                )

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

    # ---- Skeleton ----
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
            self._wire_sampler_seed(dataloader, int(self._run_seed))

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

        self._vram_begin()
        assert self.accelerator is not None, "BaseTrainer requires an Accelerator"

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

                metrics = self._reduce_step_metrics(losses, grad_norm)

                current_lr = optimizer.param_groups[0]["lr"]
                _now = _time.monotonic()
                steps_per_sec = 1.0 / max(_now - _step_t0, 1e-9)
                _step_t0 = _now

                need_mem_detail = (wandb_run is not None) or bool(debug)
                mem_stats = self._vram_record(need_detail=need_mem_detail)

                self._log_step(
                    global_step=global_step,
                    opt_step=opt_step,
                    epoch=epoch,
                    loss_total=metrics["loss_total"],
                    loss_video=metrics["loss_video"],
                    loss_action=metrics["loss_action"],
                    loss_decoder=metrics["loss_decoder"],
                    grad_norm=metrics["grad_norm"],
                    lr=current_lr,
                    steps_per_sec=steps_per_sec,
                    batch_size=batch_size,
                    pbar=pbar,
                    wandb_run=wandb_run,
                    mem_stats=mem_stats,
                    debug=debug,
                    output_path=output_path,
                )

                if self.should_save_checkpoint(global_step, save_steps):
                    self._save_checkpoint_files(output_path, global_step, keep_last_k, final=False)

                if max_steps and global_step >= max_steps:
                    pbar.close()
                    self._vram_summary(global_step, output_path, wandb_run)
                    return

        pbar.close()
        if save_steps:
            self._save_checkpoint_files(output_path, global_step, keep_last_k, final=True)
        self._vram_summary(global_step, output_path, wandb_run)
