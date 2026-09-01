"""Pipeline construction factory for training.

Builds a WanVideoPipeline from Hydra config by auto-discovering model files
(sharded safetensors, standalone safetensors, .pth), tokenizer, and optionally
enabling gradient checkpointing.
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


def _normalize_backbone_token(s) -> str:
    """Lowercase + strip non-alphanumeric, so ``Wan2.2-TI2V-5B`` and
    ``wan22_ti2v_5b`` both collapse to ``wan22ti2v5b``."""
    if s is None:
        return ""
    return "".join(c for c in str(s).lower() if c.isalnum())


def _warn_on_name_path_mismatch(name, model_dir: str) -> None:
    # All Wan variants share one adapter class, so video_backbone.name only
    # drives registry dispatch — the loaded weights are decided entirely by
    # model_path. A half-override on the CLI (only `name`, not `model_path`)
    # silently loads the wrong backbone. Soft normalize-and-match cross-check;
    # WARN but don't abort — intentional name/path ablations are allowed.
    if not name:
        return
    path_stem = os.path.basename(os.path.normpath(model_dir))
    norm_name = _normalize_backbone_token(name)
    norm_path = _normalize_backbone_token(path_stem)
    if not norm_name or not norm_path:
        return
    if norm_name in norm_path or norm_path in norm_name:
        return
    logger.warning(
        "video_backbone.name=%r looks inconsistent with model_path=%r "
        "(stem=%r). The actual backbone loaded is determined by model_path, "
        "not name; if you intended to switch backbones on the CLI, override "
        "BOTH model.video_backbone.name AND model.video_backbone.model_path. "
        "Suppress this warning by aligning the two fields, or ignore it for "
        "intentional name/path ablations.",
        name,
        model_dir,
        path_stem,
    )


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


def _filter_native_vae_configs(model_configs):
    """Drop any ``ModelConfig`` whose path/pattern matches a Wan VAE weight
    file. Used by callers that supply an external :class:`VideoEncoder` and
    therefore want the native VAE to never materialize in RAM.

    Match is case-insensitive on ``"vae"`` substring against ``path`` (str
    or first entry of a sharded list) AND ``origin_file_pattern``; either
    hit drops the config. Other model files (DiT, T5, CLIP) are untouched.
    """
    kept = []
    for c in model_configs:
        candidates = []
        if isinstance(c.path, list):
            candidates.extend(str(p) for p in c.path if p)
        elif c.path is not None:
            candidates.append(str(c.path))
        if c.origin_file_pattern is not None:
            candidates.append(str(c.origin_file_pattern))
        if any("vae" in os.path.basename(s).lower() for s in candidates):
            continue
        kept.append(c)
    return kept


def build_training_pipeline(cfg: DictConfig, *, skip_native_vae: bool = False):
    """Build WanVideoPipeline from Hydra config.

    Auto-discovers model files in the directory specified by
    ``cfg.model.video_backbone.model_path``, groups sharded safetensors,
    detects the tokenizer, and enables gradient checkpointing.

    Args:
        cfg: Full Hydra config (reads ``cfg.training`` and
            ``cfg.model.video_backbone``).
        skip_native_vae: When True, filter the discovered ``ModelConfig``
            list to drop the native VAE weight file before it is loaded.
            Used by :meth:`Wan22Ti2v.from_pretrained` on the
            irreversible external-encoder path so the ~1.5GB Wan2.2 VAE
            never materializes on CPU only to be released immediately.

    Returns:
        Initialized WanVideoPipeline ready for training.
    """
    from openwam.model.video_backbone.wan.loader import load_wan_components

    t = cfg.training
    backbone_cfg = cfg.model.video_backbone

    device = "cpu" if bool(t.initialize_model_on_cpu) else "cuda"
    model_dir = str(backbone_cfg.model_path)

    _warn_on_name_path_mismatch(backbone_cfg.get("name", None), model_dir)

    model_configs, tokenizer_config = discover_model_files(model_dir)
    if skip_native_vae:
        model_configs = _filter_native_vae_configs(model_configs)

    # Load components
    pipe = load_wan_components(
        model_configs,
        tokenizer_config,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    # Gradient checkpointing — the holder is not an nn.Module, so walk each
    # loaded sub-module's own tree (was ``pipe.modules()`` on the old pipeline).
    if bool(t.use_gradient_checkpointing):
        from openwam.model.video_backbone.wan.loader import _MODULE_SLOTS

        for _name in _MODULE_SLOTS:
            comp = getattr(pipe, _name, None)
            if comp is None:
                continue
            for module in comp.modules():
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable()

    return pipe
