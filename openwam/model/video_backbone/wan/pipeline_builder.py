"""Pipeline construction factory for training.

Builds a WanVideoPipeline from Hydra config by auto-discovering model files
(sharded safetensors, standalone safetensors, .pth), tokenizer, and optionally
applying LoRA and gradient checkpointing.
"""

import glob as _glob
import importlib
import logging
import os
import re
from collections import defaultdict

import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def discover_model_files(model_dir: str):
    """Auto-discover model files and tokenizer config from a model directory.

    Groups sharded safetensors by prefix, collects standalone safetensors
    and .pth files, and detects the tokenizer.

    Returns:
        (model_configs, tokenizer_config) — lists of ``ModelConfig`` objects.
    """
    from openwam.model.video_backbone.wan.shared.core.loader import ModelConfig

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_path does not exist: {model_dir}")

    safetensors = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
    pth_files = sorted(_glob.glob(os.path.join(model_dir, "*.pth")))
    if not safetensors and not pth_files:
        raise FileNotFoundError(f"No *.safetensors or *.pth files found in {model_dir}")

    shard_groups = defaultdict(list)
    standalone = []
    for f in safetensors:
        basename = os.path.basename(f)
        m = re.match(r"^(.+)-\d{5}-of-\d{5}\.safetensors$", basename)
        if m:
            shard_groups[m.group(1)].append(f)
        else:
            standalone.append(f)

    model_paths = []
    for prefix in sorted(shard_groups):
        shards = sorted(shard_groups[prefix])
        model_paths.append(shards)
        logger.info("Grouped %d shards as one model: %s-*", len(shards), prefix)
    for f in standalone:
        model_paths.append(f)
    for f in pth_files:
        model_paths.append(f)
    logger.info("Auto-discovered %d model entries from %s", len(model_paths), model_dir)

    model_configs = [ModelConfig(p) for p in model_paths]

    tokenizer_dir = os.path.join(model_dir, "google", "umt5-xxl")
    if os.path.isdir(tokenizer_dir):
        tokenizer_config = ModelConfig(tokenizer_dir)
        logger.info("Auto-detected tokenizer at %s", tokenizer_dir)
    else:
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")

    return model_configs, tokenizer_config


def _import_class(dotted: str):
    """Import ``pkg.mod.Class`` dotted path."""
    module_path, cls_name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), cls_name)


def _build_tokenizer(tok_cfg: dict, base_dir: str):
    """Instantiate a tokenizer described by a config ``tokenizer`` block.

    Two construction modes:
      - ``method`` absent / ``"__init__"`` → ``cls(**{path_kwarg: path, **kwargs})``
      - ``method: "from_pretrained"``      → ``cls.from_pretrained(path, **kwargs)``

    ``subdir`` is resolved relative to ``base_dir``.
    """
    cls = _import_class(tok_cfg["class"])
    subdir = tok_cfg.get("subdir")
    path = os.path.join(base_dir, subdir) if subdir else None
    if path is not None and not os.path.isdir(path):
        raise FileNotFoundError(f"tokenizer subdir does not exist: {path}")

    kwargs = dict(tok_cfg.get("kwargs", {}) or {})
    method = tok_cfg.get("method")
    if method == "from_pretrained":
        return cls.from_pretrained(path, **kwargs) if path else cls.from_pretrained(**kwargs)
    path_kwarg = tok_cfg.get("path_kwarg", "name")
    if path is not None:
        kwargs[path_kwarg] = path
    return cls(**kwargs)


def build_training_pipeline(cfg: DictConfig):
    """Build WanVideoPipeline from Hydra config.

    Auto-discovers model files in the directory specified by
    ``cfg.model.video_backbone.model_path``, groups sharded safetensors,
    detects the tokenizer, applies LoRA if configured, and enables
    gradient checkpointing.

    Args:
        cfg: Full Hydra config (reads ``cfg.training`` and
            ``cfg.model.video_backbone``).

    Returns:
        Initialized WanVideoPipeline ready for training.
    """
    from openwam.model.video_backbone.wan.pipeline import WanVideoPipeline

    t = cfg.training
    backbone_cfg = cfg.model.video_backbone

    device = "cpu" if bool(t.initialize_model_on_cpu) else "cuda"
    model_dir = str(backbone_cfg.model_path)

    model_configs, tokenizer_config = discover_model_files(model_dir)

    # Load pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    # Apply LoRA if configured
    lora_base_model = getattr(t, "lora_base_model", None)
    if lora_base_model:
        pipe = setup_lora(
            pipe,
            lora_base_model,
            getattr(t, "lora_target_modules", None),
            int(getattr(t, "lora_rank", 32)),
        )

    # Gradient checkpointing
    if bool(t.use_gradient_checkpointing):
        for module in pipe.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()

    return pipe


def setup_lora(pipe, lora_base_model, lora_target_modules, lora_rank):
    """Apply LoRA to a pipeline model via PEFT.

    Args:
        pipe: WanVideoPipeline instance.
        lora_base_model: Name of the pipeline sub-module to wrap (e.g. ``"dit"``).
        lora_target_modules: Comma-separated target module names.
        lora_rank: LoRA rank.

    Returns:
        The pipeline with LoRA applied.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        logger.warning("PEFT not available, skipping LoRA setup")
        return pipe

    if lora_base_model and hasattr(pipe, lora_base_model):
        base_model = getattr(pipe, lora_base_model)
        target_modules = lora_target_modules.split(",") if lora_target_modules else None
        lora_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
        )
        setattr(pipe, lora_base_model, get_peft_model(base_model, lora_config))

    return pipe
