"""Stateless trainer helpers: parameter report, LR schedule, wandb init.

纯计算/IO,无训练状态;循环编排留在 trainer。
"""

import logging

logger = logging.getLogger(__name__)


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
