"""Construct Wan backbone components without a pipeline container.

``load_wan_components`` replaces the former ``WanVideoPipeline.from_pretrained``:
it loads the Wan modules from a ``ModelConfig`` list and returns a plain holder
(``SimpleNamespace``) that :meth:`WanBase.__init__` drains into itself.
No ``BasePipeline`` / ``WanVideoPipeline`` is involved. ``new_components`` builds
the empty holder used by the config-driven (``components``) deploy path.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

from openwam.model.video_backbone.wan.models.text_encoder import HuggingfaceTokenizer
from openwam.model.video_backbone.wan.shared.core.device.torch_device import get_device_type
from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler
from openwam.model.video_backbone.wan.shared.models.model_loader import ModelPool

logger = logging.getLogger(__name__)

# Names of the module slots a holder carries (Module → backbone named child;
# others stay plain attributes). Mirrors WanVideoPipeline's old attribute set.
_MODULE_SLOTS = (
    "text_encoder",
    "image_encoder",
    "dit",
    "dit2",
    "vae",
    "motion_controller",
    "vace",
    "vace2",
    "vap",
    "animate_adapter",
    "audio_encoder",
)

# .pth → converted-safetensors redirect to avoid re-downloading shared weights.
_REDIRECT_DICT = {
    "models_t5_umt5-xxl-enc-bf16.pth": (
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
    ),
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": (
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
    ),
    "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
    "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
}


def new_components(device=get_device_type(), torch_dtype=torch.bfloat16) -> SimpleNamespace:
    """Empty Wan component holder with Wan defaults (scheduler + division factors)."""
    c = SimpleNamespace(
        device=device,
        torch_dtype=torch_dtype,
        scheduler=FlowMatchScheduler("Wan"),
        tokenizer=None,
        audio_processor=None,
        height_division_factor=16,
        width_division_factor=16,
        time_division_factor=4,
        time_division_remainder=1,
    )
    for name in _MODULE_SLOTS:
        setattr(c, name, None)
    return c


def _apply_redirect(model_configs, redirect_common_files=True):
    if not redirect_common_files:
        return
    for mc in model_configs:
        if mc.origin_file_pattern is None or mc.model_id is None:
            continue
        if mc.origin_file_pattern in _REDIRECT_DICT and mc.model_id != _REDIRECT_DICT[mc.origin_file_pattern][0]:
            print(
                f"To avoid repeatedly downloading model files, ({mc.model_id}, {mc.origin_file_pattern}) is "
                f"redirected to {_REDIRECT_DICT[mc.origin_file_pattern]}. You can use "
                f"`redirect_common_files=False` to disable file redirection."
            )
            mc.model_id, mc.origin_file_pattern = _REDIRECT_DICT[mc.origin_file_pattern]


def load_wan_components(
    model_configs,
    tokenizer_config=None,
    *,
    device=get_device_type(),
    torch_dtype=torch.bfloat16,
    vram_limit=None,
    redirect_common_files=True,
) -> SimpleNamespace:
    """Load Wan modules from ``model_configs`` into a holder (was ``WanVideoPipeline.from_pretrained``)."""
    _apply_redirect(model_configs, redirect_common_files)
    c = new_components(device=device, torch_dtype=torch_dtype)

    # (was BasePipeline.download_and_load_models, inlined)
    model_pool = ModelPool()
    for mc in model_configs:
        mc.download_if_necessary()
        vram_config = mc.vram_config()
        vram_config["computation_dtype"] = vram_config["computation_dtype"] or torch_dtype
        vram_config["computation_device"] = vram_config["computation_device"] or device
        model_pool.auto_load_model(
            mc.path,
            vram_config=vram_config,
            vram_limit=vram_limit,
            clear_parameters=mc.clear_parameters,
            state_dict=mc.state_dict,
        )

    c.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
    dit = model_pool.fetch_model("wan_video_dit", index=2)
    if isinstance(dit, list):
        c.dit, c.dit2 = dit
    else:
        c.dit = dit
    c.vae = model_pool.fetch_model("wan_video_vae")
    c.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
    c.motion_controller = model_pool.fetch_model("wan_video_motion_controller")
    vace = model_pool.fetch_model("wan_video_vace", index=2)
    if isinstance(vace, list):
        c.vace, c.vace2 = vace
    else:
        c.vace = vace
    c.vap = model_pool.fetch_model("wan_video_vap")
    c.audio_encoder = model_pool.fetch_model("wans2v_audio_encoder")
    c.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")

    if c.vae is not None:
        c.height_division_factor = c.vae.upsampling_factor * 2
        c.width_division_factor = c.vae.upsampling_factor * 2

    if tokenizer_config is not None:
        tokenizer_config.download_if_necessary()
        c.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean="whitespace")

    return c


def resolve_cfg_shift_video(source) -> Optional[float]:
    """Extract ``shift_video`` from the various ``from_pretrained`` source
    shapes; ``None`` when unset or the source has no cfg context.
    """
    from omegaconf import DictConfig

    vb_cfg = None
    if isinstance(source, DictConfig):
        vb_cfg = source.get("video_backbone") if "video_backbone" in source else None
    elif isinstance(source, dict):
        # model_loader hands a dict that either IS or contains video_backbone.
        vb_cfg = source.get("video_backbone", source) if "video_backbone" in source else source
    if vb_cfg is None:
        return None
    if isinstance(vb_cfg, dict):
        raw = vb_cfg.get("shift_video")
    else:
        raw = getattr(vb_cfg, "shift_video", None)
    return None if raw is None else float(raw)


def infer_text_dim(dit) -> Optional[int]:
    """Wan DiT text_embedding input width (the raw context dim it expects)."""
    text_embedding = getattr(dit, "text_embedding", None)
    if isinstance(text_embedding, nn.Linear):
        return int(text_embedding.in_features)
    if isinstance(text_embedding, nn.Module):
        for module in text_embedding.modules():
            if isinstance(module, nn.Linear):
                return int(module.in_features)
    return None


def build_holder_from_components(
    components: list,
    tokenizer: dict = None,
    device: str = "cpu",
    ckpt_dir: str = None,
    model_path: str = None,
    *,
    skip_native_vae: bool = False,
):
    """Build an empty component holder from specs. Weights are NOT
    loaded here (``load_checkpoint`` does that). Tokenizer resolves
    ckpt-local first, then falls back to the ``model_path`` layout.

    ``skip_native_vae`` drops the ``vae`` entry so it never allocates CPU
    tensors (irreversible external-encoder path).
    """
    from openwam.model.video_backbone.wan.pipeline_builder import _build_tokenizer, _import_class

    holder = new_components(device=device, torch_dtype=torch.bfloat16)

    for entry in components:
        if skip_native_vae and entry.get("attr") == "vae":
            continue
        cls = _import_class(entry["model_class"])
        kwargs = entry.get("extra_kwargs", {}) or {}
        logger.info(
            "Instantiating %s as holder.%s (extra_kwargs keys=%s)",
            entry["model_class"],
            entry["attr"],
            list(kwargs.keys()),
        )
        with torch.device(device):
            model = cls(**kwargs)
        model.to(dtype=torch.bfloat16)
        setattr(holder, entry["attr"], model)

    if getattr(holder, "vae", None) is not None and hasattr(holder.vae, "upsampling_factor"):
        holder.height_division_factor = holder.vae.upsampling_factor * 2
        holder.width_division_factor = holder.vae.upsampling_factor * 2

    if tokenizer:
        tok = None
        subdir = tokenizer.get("subdir", "")
        if ckpt_dir and subdir and os.path.isdir(os.path.join(ckpt_dir, subdir)):
            tok = _build_tokenizer(tokenizer, ckpt_dir)
        elif model_path and os.path.isdir(model_path):
            # ckpt-local specs prefix ``tokenizer/``; upstream dirs don't.
            fallback_subdir = subdir
            if fallback_subdir.startswith("tokenizer/"):
                fallback_subdir = fallback_subdir[len("tokenizer/") :]
            if fallback_subdir and os.path.isdir(os.path.join(model_path, fallback_subdir)):
                fallback_cfg = dict(tokenizer)
                fallback_cfg["subdir"] = fallback_subdir
                logger.info(
                    "Tokenizer not found under ckpt_dir; falling back to model_path/%s",
                    fallback_subdir,
                )
                tok = _build_tokenizer(fallback_cfg, model_path)
        if tok is None:
            raise FileNotFoundError(
                f"Tokenizer subdir {subdir!r} not found under ckpt_dir={ckpt_dir!r} "
                f"nor under model_path={model_path!r} (with 'tokenizer/' prefix stripped). "
                "Either copy the tokenizer into the checkpoint dir, or ensure model_path is reachable."
            )
        setattr(holder, tokenizer.get("attr", "tokenizer"), tok)

    return holder


def build_holder_from_model_path(model_path: str, device: str = "cpu", *, skip_native_vae: bool = False):
    """Build a component holder from a model dir without full Hydra config.
    ``skip_native_vae`` drops the native VAE weight file before it
    materializes (irreversible external-encoder path).
    """
    from openwam.model.video_backbone.wan.pipeline_builder import (
        _filter_native_vae_configs,
        discover_model_files,
    )

    model_configs, tokenizer_config = discover_model_files(model_path)
    if skip_native_vae:
        model_configs = _filter_native_vae_configs(model_configs)
    return load_wan_components(
        model_configs,
        tokenizer_config,
        device=device,
        torch_dtype=torch.bfloat16,
    )


def build_holder(source, *, skip_native_vae: bool = False, **kw):
    """Resolve a ``from_pretrained`` source into a transient component holder.

    Sources: ``DictConfig`` (full Hydra cfg → pipeline builder), ``str`` dir
    path / ``dict`` with ``model_path`` (lightweight build via loader), else
    an already-built component holder. ``__init__`` drains the holder.
    """
    from omegaconf import DictConfig

    if isinstance(source, DictConfig):
        from openwam.model.video_backbone.wan.pipeline_builder import build_training_pipeline

        return build_training_pipeline(source, skip_native_vae=skip_native_vae)
    if isinstance(source, str):
        if os.path.isdir(source):
            return build_holder_from_model_path(source, device=kw.get("device", "cpu"), skip_native_vae=skip_native_vae)
        raise ValueError(f"from_pretrained(str) expects a directory path, got: {source!r}.")
    if isinstance(source, dict):
        vb_cfg = source.get("video_backbone", source)
        if isinstance(vb_cfg, dict) and "components" in vb_cfg:
            return build_holder_from_components(
                vb_cfg["components"],
                tokenizer=vb_cfg.get("tokenizer"),
                device=kw.get("device", "cpu"),
                ckpt_dir=kw.get("ckpt_dir"),
                model_path=vb_cfg.get("model_path"),
                skip_native_vae=skip_native_vae,
            )
        model_path = vb_cfg.get("model_path") if isinstance(vb_cfg, dict) else getattr(vb_cfg, "model_path", None)
        if model_path is None:
            raise ValueError("dict source must contain 'video_backbone.components' or 'video_backbone.model_path'")
        return build_holder_from_model_path(
            str(model_path), device=kw.get("device", "cpu"), skip_native_vae=skip_native_vae
        )
    return source
