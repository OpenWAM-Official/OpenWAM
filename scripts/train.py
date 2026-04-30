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

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
        mixed_precision=str(cfg.accelerate.mixed_precision),
    )


def _train(cfg: DictConfig) -> None:
    """Package-native training path."""
    from openwam.dataloader.registry import build_dataset
    from openwam.train.openwam_trainer import OpenWAMTrainer

    accelerator = _build_accelerator(cfg)

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
    finally:
        # Avoid `destroy_process_group() was not called before program exit`
        # warning on shutdown by tearing down the NCCL process group cleanly.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
