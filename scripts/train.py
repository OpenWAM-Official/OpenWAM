"""Hydra entry point for OpenWAM training.

Launch with torchrun (DeepSpeed via HuggingFace Accelerate):
    torchrun --nproc_per_node=4 scripts/train.py

World size comes from torchrun; everything else (mixed precision, ZeRO stage,
gradient accumulation/clipping, optimizer offload) from cfg.training
(e.g. select the ZeRO stage via ``training.zero_stage=1``).
"""

import faulthandler
import logging
import os
import sys
import traceback
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Force tracebacks to flush even when an exception fires inside DataLoader
# workers / Hydra's own try-except wrapper. Without this, a silent failure
# on rank 0 leaves the other ranks deadlocked at FSDP all-gather with no
# clue what went wrong (observed during a mixture smoke run).
faulthandler.enable(file=sys.stderr, all_threads=True)


def _force_flush_excepthook(exc_type, exc_value, exc_tb):
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    sys.stderr.write(f"\n===== UNHANDLED EXCEPTION ON RANK {rank} =====\n")
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()


sys.excepthook = _force_flush_excepthook

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _StartupNoiseFilter(logging.Filter):
    # Hide optional compiler probes while keeping warnings and errors visible.
    _HIDDEN_PREFIXES = ("gcc -pthread ", "NCCL version ")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (record.name == "root" and message.startswith(self._HIDDEN_PREFIXES))


def _install_startup_noise_filter() -> None:
    # Suppress verbose optional-op probes emitted by DeepSpeed/distutils.
    noise_filter = _StartupNoiseFilter()
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(noise_filter)


logger = logging.getLogger(__name__)


def _build_accelerator(cfg: DictConfig):
    """Build a DeepSpeed Accelerator for the torchrun launch path.

    torchrun has already initialised torch.distributed (``RANK``, ``WORLD_SIZE``,
    ``LOCAL_RANK`` are set); we construct the ``DeepSpeedPlugin`` from cfg.training
    so DeepSpeed is activated without ``accelerate launch``.
    """
    import accelerate

    t = cfg.training
    grad_accum = int(t.gradient_accumulation_steps)
    max_grad_norm = getattr(t, "max_grad_norm", None)
    mixed_precision = str(t.mixed_precision)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        logger.info("mixed_precision = %s (from cfg.training.mixed_precision)", mixed_precision)

    plugin = accelerate.DeepSpeedPlugin(
        zero_stage=int(t.zero_stage),
        gradient_accumulation_steps=grad_accum,
        # DeepSpeed clips internally with this value (the loop's clip_grad_norm_ only reads
        # back the norm under DeepSpeed); single source = training.max_grad_norm, 0.0 = off.
        gradient_clipping=float(max_grad_norm) if max_grad_norm else 0.0,
        offload_optimizer_device=str(t.offload_optimizer_device),
    )

    return accelerate.Accelerator(
        gradient_accumulation_steps=grad_accum,
        deepspeed_plugin=plugin,
        mixed_precision=mixed_precision,
    )


def _inject_project_seed(cfg: DictConfig) -> None:
    """Propagate ``cfg.project.seed`` down to ``cfg.dataloader.seed``.

    Dataloader yamls no longer carry their own ``seed`` field; the
    authoritative source is ``project.seed`` in train.yaml. When
    ``project.seed`` is null (production stochastic runs), the dataset
    ctor's ``seed=42`` default kicks in.
    """
    project_seed = OmegaConf.select(cfg, "project.seed", default=None)
    if project_seed is None:
        return
    dl = cfg.get("dataloader", None)
    if dl is None:
        return
    OmegaConf.update(dl, "seed", int(project_seed), force_add=True)


def _train(cfg: DictConfig) -> None:
    """Package-native training path."""
    _inject_project_seed(cfg)
    _train_openwam(cfg)


def _train_openwam(cfg: DictConfig) -> None:
    """Original OpenWAM training path."""
    from openwam.dataloader.registry import build_dataset
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.seeding import seed_everything

    # Seed Python random / numpy / torch BEFORE dataset construction so that
    # any reader-time randomness (e.g. MixtureDataset index_map shuffle when
    # seed isn't explicitly set, lerobot splits, etc.) is reproducible.
    # OpenWAMTrainer.__init__ re-seeds via seed_process for model init using the
    # same RANK_OFFSET rank stride (cudnn stays as configured here). Null
    # cfg.project.seed = production stochastic run, so we skip seeding here.
    project_seed = OmegaConf.select(cfg, "project.seed", default=None)
    if project_seed is not None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        seed_everything(int(project_seed), rank=rank)

    accelerator = _build_accelerator(cfg)

    # Build dataset via registry
    dataset = build_dataset(cfg.dataloader, split="train")

    # Build trainer and run
    trainer = OpenWAMTrainer(cfg, accelerator=accelerator, dataset=dataset)
    trainer.train()


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
    _install_startup_noise_filter()

    # Only print on rank 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("=" * 60)
        print("OpenWAM Training")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)
    else:
        # Silence stray ``print(...)`` calls in vendored loaders (e.g. diffsynth's
        # ``model_loader.py``) on non-main ranks — they otherwise print "Loading
        # models from: ..." / "Loaded model: { ... }" once per rank, doubling
        # the startup log. Logging-based output is unaffected.
        import builtins

        builtins.print = lambda *a, **kw: None

    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        _train(cfg)
    except BaseException:
        # destroy_process_group below is a collective. If only this rank
        # raised, the other ranks are still in mid-training collectives
        # (e.g. accelerate's RNG-state broadcast inside dataloader.__iter__),
        # and destroy will block forever waiting for them — masking the
        # actual rank-0 exception. Mirror the alternate path's pre-destroy
        # traceback print so the real error survives the deadlock.
        import traceback as _tb

        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
        sys.stderr.write(f"\n===== RANK {rank} EXCEPTION (pre-destroy) =====\n")
        _tb.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        raise
    finally:
        # Avoid `destroy_process_group() was not called before program exit`
        # warning on shutdown by tearing down the NCCL process group cleanly.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
