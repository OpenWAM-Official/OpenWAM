"""Hydra entry point for OpenWAM training.

Supports two launch modes:

1. torchrun (recommended for cloud / multi-node):
    torchrun --nproc_per_node=2 scripts/train.py

   DeepSpeed stage is controlled by the ``DEEPSPEED_ZERO_STAGE`` env var
   (default: 2) or by the Hydra config ``accelerate.deepspeed_config.zero_stage``.

2. accelerate launch (local convenience):
    accelerate launch --config_file configs/accelerate/deepspeed_zero2.yaml scripts/train.py

Both modes use HuggingFace Accelerate internally for DeepSpeed integration.
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

logger = logging.getLogger(__name__)


def _build_accelerator(cfg: DictConfig):
    """Build an Accelerator, optionally with DeepSpeed plugin.

    When launched via ``torchrun``, torch.distributed is already initialised
    (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK`` are set).  We construct a
    ``DeepSpeedPlugin`` from the Hydra accelerate config and pass it to
    ``Accelerator`` so that DeepSpeed is activated without needing
    ``accelerate launch``.

    When launched via ``accelerate launch``, the accelerate config file
    already provides the DeepSpeed settings, so we just create a plain
    ``Accelerator``.
    """
    import accelerate

    t = cfg.training
    grad_accum = int(t.gradient_accumulation_steps)
    mixed_precision = str(cfg.accelerate.mixed_precision)
    logger.info("mixed_precision = %s (from cfg.accelerate.mixed_precision)", mixed_precision)

    # Detect if we were launched by accelerate (it sets ACCELERATE_MIXED_PRECISION etc.)
    launched_by_accelerate = os.environ.get("ACCELERATE_MIXED_PRECISION") is not None

    if launched_by_accelerate:
        # accelerate launch already configured everything
        return accelerate.Accelerator(gradient_accumulation_steps=grad_accum)

    # torchrun path: build DeepSpeed plugin from Hydra config
    ds_cfg = cfg.accelerate.deepspeed_config
    zero_stage = int(os.environ.get("DEEPSPEED_ZERO_STAGE", ds_cfg.zero_stage))

    plugin = accelerate.DeepSpeedPlugin(
        zero_stage=zero_stage,
        gradient_accumulation_steps=grad_accum,
        gradient_clipping=float(ds_cfg.gradient_clipping),
        offload_optimizer_device=str(ds_cfg.offload_optimizer_device),
        offload_param_device=str(ds_cfg.offload_param_device),
        zero3_init_flag=bool(ds_cfg.zero3_init_flag),
        zero3_save_16bit_model=bool(ds_cfg.zero3_save_16bit_model),
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
    from openwam.train.utils.temporal_contract import apply_temporal_contract_bridge

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

    # Bridge encoder temporal contract from model yaml to dataloader cfg before
    # the dataset is built (Wan VAE = (4, True), V-JEPA 2.1 = (2, True),
    # future non-causal encoders = (tc, False)). See
    # openwam/train/utils/temporal_contract.py.
    apply_temporal_contract_bridge(cfg)

    # Build dataset via registry
    dataset = build_dataset(cfg.dataloader, split="train")

    # Build trainer and run
    trainer = OpenWAMTrainer(cfg, accelerator=accelerator, dataset=dataset)
    trainer.train()


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
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
