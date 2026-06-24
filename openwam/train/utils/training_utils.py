"""Stateless trainer helpers + VRAM tracker.

纯计算/IO,无训练状态(config 取值、参数报告、LR 调度、wandb、跨 rank 规约、debug CSV 写);
``VramTracker`` 持有 run 级显存峰值状态。
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def cfg_get(cfg, key: str, default=None):
    """Read ``key`` from a dict or attr-style config, with default."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def latent_action_enabled(cfg) -> bool:
    action_cfg = cfg_get(getattr(cfg, "model", None), "action_backbone", None)
    return cfg_get(action_cfg, "type", "explicit") == "latent"


def log_parameter_counts(architecture, *, is_main: bool) -> None:
    """Print per-backbone total/trainable param counts (rank-0 only).

    Uses print() not logger so it survives Hydra's default logging filter;
    the is_main gate keeps it correct regardless of the launcher's stdout
    suppression on non-main ranks.
    """
    if not is_main:
        return

    def _count(module):
        if module is None:
            return 0, 0
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    bb_counts = {name: _count(module) for name, module in architecture.backbones.items()}
    extra_counts = {}
    for name, module in architecture.named_children():
        if name in bb_counts:
            continue
        total, trainable = _count(module)
        if total:
            extra_counts[name] = (total, trainable)
    arch_total = sum(total for total, _ in bb_counts.values()) + sum(total for total, _ in extra_counts.values())
    arch_train = sum(train for _, train in bb_counts.values()) + sum(train for _, train in extra_counts.values())
    print("=" * 60)
    print("Parameter counts")
    for name, (total, trainable) in {**bb_counts, **extra_counts}.items():
        print(f"  {name:<15}: total={total / 1e6:7.1f}M  trainable={trainable / 1e6:7.1f}M")
    print(f"  Architecture  : total={arch_total / 1e6:7.1f}M  trainable={arch_train / 1e6:7.1f}M")
    print("=" * 60, flush=True)


def build_cosine_scheduler(optimizer, *, total_opt_steps: int, cfg):
    """Linear-warmup + cosine-anneal LR schedule."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    t = cfg.training
    lr = float(t.learning_rate)
    warmup_ratio = float(getattr(t, "warmup_ratio", 0.05))
    lr_min_ratio = float(getattr(t, "lr_min_ratio", 0.01))
    warmup_steps = int(total_opt_steps * warmup_ratio)
    cosine_steps = max(total_opt_steps - warmup_steps, 1)
    warmup_sched = LinearLR(optimizer, start_factor=1.0 / max(warmup_steps, 1), total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=lr * lr_min_ratio)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
    logger.info(
        "LR scheduler: cosine | total_opt_steps=%d warmup=%d eta_min=%.2e",
        total_opt_steps,
        warmup_steps,
        lr * lr_min_ratio,
    )
    return scheduler


def init_wandb(cfg):
    """Init a wandb run from cfg.project.wandb. Returns the run or None."""
    wandb_cfg = cfg.project.get("wandb", None)
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
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
    logger.info("wandb initialized: %s/%s", project, run.name)
    return run


def reduce_step_metrics(accelerator, losses: dict, grad_norm) -> dict:
    """Reduce loss/grad_norm across ranks (mean); single-process fast path."""
    loss = losses["total"]

    def _f(v):
        return v.item() if isinstance(v, torch.Tensor) else float(v)

    if accelerator is not None and accelerator.num_processes > 1:
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
        g = accelerator.gather(local).mean(dim=0)
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


def write_debug_loss_row(
    output_path,
    *,
    labels,
    metrics,
    global_step,
    opt_step,
    epoch,
    lr,
    steps_per_sec,
    mem_stats,
) -> None:
    """Append one row to debug_loss_history.csv (writes the header on first call).

    Loss columns follow ``labels`` = ``[(display_name, metrics_key)]``, so they
    vary with mode (e.g. latent adds a ``loss_latent_action`` column).
    """
    loss_log_path = os.path.join(output_path, "debug_loss_history.csv")
    write_header = not os.path.exists(loss_log_path)
    label_cols = ",".join(f"loss_{name}" for name, _ in labels)
    label_vals = ",".join(f"{metrics[key]:.10g}" for _, key in labels)
    with open(loss_log_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(
                f"step,opt_step,epoch,loss,loss_video,{label_cols},"
                "grad_norm,lr,steps_per_sec,"
                "mem_alloc_gb,mem_reserved_gb,step_peak_alloc_gb,step_peak_reserved_gb,"
                "run_peak_alloc_gb,run_peak_reserved_gb\n"
            )
        f.write(
            f"{global_step},{opt_step},{epoch},{metrics['loss_total']:.10g},{metrics['loss_video']:.10g},{label_vals},"
            f"{metrics['grad_norm']:.10g},{lr:.10g},{steps_per_sec:.10g},"
            f"{mem_stats.get('mem_alloc_gb', float('nan')):.6g},"
            f"{mem_stats.get('mem_reserved_gb', float('nan')):.6g},"
            f"{mem_stats.get('step_peak_alloc_gb', float('nan')):.6g},"
            f"{mem_stats.get('step_peak_reserved_gb', float('nan')):.6g},"
            f"{mem_stats.get('run_peak_alloc_gb', float('nan')):.6g},"
            f"{mem_stats.get('run_peak_reserved_gb', float('nan')):.6g}\n"
        )


class VramTracker:
    """Tracks run-level peak VRAM across the training loop.

    ``begin`` resets counters AFTER accelerate.prepare so init-time allocations
    (params / optimizer state / ZeRO shards) aren't counted in the run-level peak.
    """

    def __init__(self):
        self.run_peak_alloc_gb = 0.0
        self.run_peak_reserved_gb = 0.0

    def begin(self) -> None:
        self.run_peak_alloc_gb = 0.0
        self.run_peak_reserved_gb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def record(self, *, need_detail: bool) -> dict:
        """Update run-level peak; with need_detail also read live usage, reset the
        per-step peak, and return a dict for wandb / debug-CSV. Empty dict on CPU
        or when need_detail=False."""
        if not torch.cuda.is_available():
            return {}
        peak_alloc_gb = float(torch.cuda.max_memory_allocated()) / 1e9
        peak_reserved_gb = float(torch.cuda.max_memory_reserved()) / 1e9
        self.run_peak_alloc_gb = max(self.run_peak_alloc_gb, peak_alloc_gb)
        self.run_peak_reserved_gb = max(self.run_peak_reserved_gb, peak_reserved_gb)
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
            "run_peak_alloc_gb": self.run_peak_alloc_gb,
            "run_peak_reserved_gb": self.run_peak_reserved_gb,
        }

    def write_summary(self, *, global_step, output_path, is_main, wandb_run) -> None:
        """Log run-level peak to memory_summary.csv + wandb; finish the wandb run."""
        run_peak_alloc = self.run_peak_alloc_gb
        run_peak_reserved = self.run_peak_reserved_gb
        if torch.cuda.is_available() and (run_peak_alloc > 0.0 or run_peak_reserved > 0.0):
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
