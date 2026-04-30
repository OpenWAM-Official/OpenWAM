"""Pipeline construction factory for training.

Builds a WanVideoPipeline from Hydra config by auto-discovering model files
(sharded safetensors, standalone safetensors, .pth), tokenizer, and optionally
applying LoRA and gradient checkpointing.
"""

import glob as _glob
import importlib
import json
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


def _build_tokenizer(tok_cfg: dict, manifest_dir: str):
    """Instantiate a tokenizer described by a manifest ``tokenizer`` block.

    Two construction modes:
      - ``method`` absent / ``"__init__"`` → ``cls(**{path_kwarg: path, **kwargs})``
      - ``method: "from_pretrained"``      → ``cls.from_pretrained(path, **kwargs)``

    ``subdir`` is resolved relative to ``manifest_dir``.
    """
    cls = _import_class(tok_cfg["class"])
    subdir = tok_cfg.get("subdir")
    path = os.path.join(manifest_dir, subdir) if subdir else None
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


def build_video_backbone_from_manifest(manifest_path: str, device: str = "cpu"):
    """Build an empty pipeline from a ``video_backbone_manifest.json``.

    Self-contained inference: the checkpoint's safetensors will overwrite all
    weights, so we only need (1) architecture — class + extra_kwargs for each
    sub-module, (2) a local tokenizer. No backbone-source directory required.

    Manifest schema (backbone-agnostic)::

        {
          "pipeline": {
            "class": "openwam.model.video_backbone.wan.pipeline.WanVideoPipeline",
            "kwargs": {}               # extra ctor kwargs (beyond device/torch_dtype)
          },
          "tokenizer": {                # optional
            "class":      "<dotted.path.ClassName>",
            "attr":       "tokenizer", # where to attach on pipe
            "subdir":     "tokenizer/...",
            "method":     "__init__",  # or "from_pretrained"
            "path_kwarg": "name",      # kwarg name receiving the resolved subdir path
            "kwargs":     {...}        # extra construction kwargs
          },
          "vae_division_factor_scale": 2,   # optional: sets pipe.height/width_division_factor
          "models": [
            {"attr": "text_encoder|dit|vae|...",
             "model_class": "<dotted.path>",
             "extra_kwargs": {...}},
            ...
          ]
        }

    Args:
        manifest_path: Absolute path to ``video_backbone_manifest.json``.
        device: Device on which empty model instances are constructed.

    Returns:
        A pipeline instance with models + tokenizer attached, ready for
        ``pipe.load_state_dict`` to fill in real weights.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

    pipeline_cfg = manifest["pipeline"]
    pipeline_cls = _import_class(pipeline_cfg["class"])
    pipeline_kwargs = dict(pipeline_cfg.get("kwargs", {}) or {})
    pipeline_kwargs.setdefault("device", device)
    pipeline_kwargs.setdefault("torch_dtype", torch.bfloat16)
    pipe = pipeline_cls(**pipeline_kwargs)

    for entry in manifest.get("models", []):
        cls = _import_class(entry["model_class"])
        kwargs = entry.get("extra_kwargs", {}) or {}
        logger.info(
            "Instantiating %s as pipe.%s (extra_kwargs keys=%s)",
            entry["model_class"],
            entry["attr"],
            list(kwargs.keys()),
        )
        with torch.device(device):
            model = cls(**kwargs)
        model.to(dtype=torch.bfloat16)
        setattr(pipe, entry["attr"], model)

    scale = manifest.get("vae_division_factor_scale")
    if scale and getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "upsampling_factor"):
        pipe.height_division_factor = pipe.vae.upsampling_factor * int(scale)
        pipe.width_division_factor = pipe.vae.upsampling_factor * int(scale)

    tok_cfg = manifest.get("tokenizer")
    if tok_cfg:
        tokenizer = _build_tokenizer(tok_cfg, manifest_dir)
        attr = tok_cfg.get("attr", "tokenizer")
        setattr(pipe, attr, tokenizer)
        logger.info("Loaded tokenizer %s → pipe.%s", tok_cfg["class"], attr)

    return pipe


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
