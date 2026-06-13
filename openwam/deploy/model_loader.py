"""Model-loading helpers for inference and deployment.

Provides the checkpoint-first loading path used by OpenWAM deployment.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from openwam.model.architectures.base import BaseWAMArchitecture

logger = logging.getLogger(__name__)


def _find_latest_checkpoint(ckpt_dir: str) -> str:
    """Return the path to the highest-step ``checkpoint_step_N.safetensors`` in *ckpt_dir*.

    Malformed filenames that glob-match but don't carry a numeric step are
    skipped (instead of silently getting step=0 and competing for latest).
    If every remaining file has step == 0 we warn — typically that means
    training crashed before the first save_steps interval and the caller is
    about to deploy uninitialized weights.
    """
    pattern = os.path.join(ckpt_dir, "checkpoint_step_*.safetensors")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint_step_*.safetensors found in {ckpt_dir}")

    step_re = re.compile(r"checkpoint_step_(\d+)\.safetensors$")
    numbered: list[tuple[int, str]] = []
    for f in files:
        m = step_re.search(os.path.basename(f))
        if m is not None:
            numbered.append((int(m.group(1)), f))
        else:
            logger.warning("Skipping malformed checkpoint name: %s", f)

    if not numbered:
        raise FileNotFoundError(f"No checkpoint file in {ckpt_dir} matches checkpoint_step_<int>.safetensors")

    numbered.sort(key=lambda p: p[0])
    latest_step, latest_path = numbered[-1]
    if latest_step == 0:
        logger.warning(
            "Latest checkpoint in %s is step 0 (%s) — this usually means training "
            "crashed before completing its first save_steps interval. Verify before deploying.",
            ckpt_dir,
            os.path.basename(latest_path),
        )
    return latest_path


def load_from_checkpoint_dir(
    ckpt_dir: str,
    device: str = "cuda",
    ckpt_name: Optional[str] = None,
) -> Tuple[DictConfig, BaseWAMArchitecture]:
    """Load full model from a self-contained checkpoint directory.

    The directory must contain:
      - ``config.yaml`` — Hydra config saved during training.
      - One or more ``checkpoint_step_*.safetensors`` files.

    Args:
        ckpt_dir: Path to the checkpoint directory.
        device: Target device (e.g. ``"cuda"`` or ``"cuda:0"``).
        ckpt_name: Specific checkpoint filename.  If *None*, the latest
            (highest step number) checkpoint is used.

    Returns:
        ``(cfg, architecture)`` — the resolved config and architecture
        with all weights restored.
    """
    # 1. Load config
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in {ckpt_dir}")
    cfg = OmegaConf.load(config_path)

    # 2. Resolve checkpoint file
    if ckpt_name is not None:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    else:
        ckpt_path = _find_latest_checkpoint(ckpt_dir)
    logger.info("Loading checkpoint: %s", ckpt_path)

    # 3. Build architecture (creates video backbone internally from config).
    from openwam.model import build_architecture, resolve_architecture_config

    m = cfg.model
    resolved_arch = resolve_architecture_config(m)
    params = dict(resolved_arch.params)

    # ``resolve_architecture_config`` may hand back ``params["video_backbone"]``
    # as a ``DictConfig``. Subsequent ``vb_params["_source"] = <dict>``
    # assignments would then be auto-promoted by OmegaConf to a ``DictConfig``,
    # which causes ``WanVideoBackbone.from_pretrained`` to mis-dispatch the
    # deploy source into the training path (``build_training_pipeline``
    # requires ``cfg.training``, which is absent from a video_backbone-only
    # subtree). Pinning to a plain ``dict`` here keeps the deploy dispatch
    # honest.
    vb_params = params.get("video_backbone")
    if vb_params is None:
        vb_params = {}
    elif not isinstance(vb_params, dict) or isinstance(vb_params, DictConfig):
        vb_params = OmegaConf.to_container(vb_params, resolve=True) or {}
    params["video_backbone"] = vb_params

    # Provide video backbone source from config components or model_path.
    vb_components = OmegaConf.select(cfg, "model.video_backbone.components", default=None)
    if vb_components is not None:
        logger.info("Using config-embedded component specs for video-backbone construction")
        vb_cfg_dict = OmegaConf.to_container(cfg.model.video_backbone, resolve=True)
        # Cosmos25 Reason1 self-containment: when the ckpt was saved with the
        # ``_reason1_inner`` registration enabled, its weights live in the
        # unified safetensors and the small structural artifacts
        # (config.json + tokenizer.json) live under ``<ckpt_dir>/reason1/``.
        # Clearing ``text_encoder_path`` on that branch makes
        # ``build_cosmos25_pipeline`` take its deploy/empty-shell path
        # (``pipeline_builder.py`` Reason1 construction site). For old ckpts
        # without the ``reason1/`` artifact dir we leave the original
        # ``text_encoder_path`` intact so the live encoder still loads from
        # the external Cosmos-Reason1 bundle (backward compat).
        reason1_artifact_dir = os.path.join(ckpt_dir, "reason1")
        # Iterate the plain-dict copy. ``vb_components`` is an OmegaConf
        # ``ListConfig`` whose entries are ``DictConfig`` (NOT a ``dict``
        # subclass) — so ``isinstance(c, dict)`` would always be False
        # against the real saved config, silently skipping the marker.
        has_reason1_state_component = any(
            isinstance(c, dict) and c.get("attr") == "text_encoder" for c in (vb_cfg_dict.get("components") or [])
        )
        if os.path.isdir(reason1_artifact_dir) and (
            vb_cfg_dict.get("text_encoder") == "reason1_live" or has_reason1_state_component
        ):
            prev_path = vb_cfg_dict.get("text_encoder_path")
            vb_cfg_dict["text_encoder"] = "reason1_live"
            vb_cfg_dict["text_encoder_path"] = None
            logger.info(
                "Using self-contained Reason1 artifacts from %s (clearing external text_encoder_path=%r)",
                reason1_artifact_dir,
                prev_path,
            )
        vb_params["_source"] = vb_cfg_dict
        vb_params["_ckpt_dir"] = ckpt_dir
    else:
        model_path = OmegaConf.select(cfg, "model.video_backbone.model_path", default=None)
        if model_path is not None:
            vb_name = str(OmegaConf.select(cfg, "model.video_backbone.name", default=""))
            if vb_name.startswith("cosmos25_") or vb_name.startswith("sana_video_"):
                # Cosmos / SANA carry fields the path-only string source
                # would lose:
                # - Cosmos: `flow_shift`, `model_variant`, `text_encoder`
                # - SANA: `text_encoder_name`, `attn_kernel`, `flow_shift`,
                #   `model_kwargs`
                # Both backbones' `from_pretrained` accepts a dict natively
                # (Cosmos via `_video_backbone_cfg`, SANA via
                # `_spec_from_dict`). Wan never hits this branch.
                vb_params["_source"] = {k: v for k, v in vb_params.items() if not str(k).startswith("_")}
            else:
                vb_params["_source"] = str(model_path)
        else:
            logger.warning(
                "No components or model_path in config; architecture __init__ will attempt to build from config."
            )

    # tri_system: use VLM checkpoint saved in the checkpoint dir instead of
    # the external training-time path. The trainer copies the VLM directory
    # to ``<ckpt_dir>/vlm_backbone/`` so deploy is self-contained.
    if resolved_arch.canonical.framework == "tri_system":
        vlm_cfg = params.get("vlm_backbone", {})
        if not isinstance(vlm_cfg, dict):
            vlm_cfg = OmegaConf.to_container(vlm_cfg, resolve=True) or {}
            params["vlm_backbone"] = vlm_cfg
        vlm_dir = os.path.join(ckpt_dir, "vlm_backbone")
        if os.path.isdir(vlm_dir):
            vlm_cfg["checkpoint_path"] = vlm_dir
            logger.info("Using self-contained VLM checkpoint from %s", vlm_dir)

    architecture = build_architecture(resolved_arch.registry_name, params)
    logger.info(
        "Architecture: %s (framework=%s variant=%s)",
        resolved_arch.registry_name,
        resolved_arch.canonical.framework,
        resolved_arch.canonical.variant,
    )

    # 4. Load all weights from checkpoint
    architecture.load_checkpoint(ckpt_path)

    # 5. Move to device and set eval mode — top-down: architecture → video_backbone → submodules.
    _mp = OmegaConf.select(cfg, "accelerate.mixed_precision", default="bf16")
    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    model_dtype = _DTYPE_MAP.get(str(_mp).strip().lower(), torch.bfloat16)
    logger.info("Loading all models with dtype=%s (mixed_precision=%s)", model_dtype, _mp)

    architecture.set_dtype_device(model_dtype, torch.device(device))
    architecture.eval()

    # 6. Attach the action normalizer built from saved normalization_stats.npy + config.
    architecture.attach_normalizer(_build_normalizer(cfg, ckpt_dir))

    logger.info("Model loaded successfully on %s", device)
    return cfg, architecture


def _build_normalizer(cfg: DictConfig, ckpt_dir: str):
    """Build the normalizer for both deploy directions, or ``None`` if disabled.

    The returned ``Normalizer`` serves both: ``normalize`` maps the input
    proprio state into training space, ``unnormalize`` maps the output action
    back to physical units. proprio is a single-frame action
    (``raw_actions[0:1]``), so both share one set of stats.

    Reads ``dataloader.normalize_mode`` / ``action_mode`` from the saved config;
    when enabled, loads ``normalization_stats.npy`` and wraps the requested stats
    sub-dict. When disabled, returns ``None`` (no stats file required).
    """
    logger.info("[normalizer] Resolving deployment action normalizer from checkpoint dir: %s", ckpt_dir)

    dl = OmegaConf.select(cfg, "dataloader", default=None)
    norm_mode = OmegaConf.select(cfg, "dataloader.normalize_mode", default=None)
    action_mode = OmegaConf.select(cfg, "dataloader.action_mode", default="joint")
    if dl is None or norm_mode in (None, "", "none", "null"):
        logger.info(
            "[normalizer] normalize_mode=%r disabled in saved config; action normalizer INACTIVE "
            "(actions and deploy proprio will be returned/used as-is).",
            norm_mode,
        )
        return None
    logger.info(
        "[normalizer] Saved config: normalize_mode=%s, action_mode=%s",
        norm_mode,
        action_mode,
    )

    stats_path = os.path.join(ckpt_dir, "normalization_stats.npy")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Missing required normalization_stats.npy in checkpoint dir: {stats_path}. "
            "Checkpoints with active action normalization must include it "
            "(older action_stats.npy checkpoints: rename the file)."
        )
    logger.info("[normalizer] Found pre-computed stats file: %s (exists ✓)", stats_path)

    from openwam.dataloader.transforms.normalize import (
        YAML_TO_NORM_MODE,
        Normalizer,
        load_mode_stats,
    )

    if norm_mode not in YAML_TO_NORM_MODE:
        logger.warning(
            "[normalizer] Unknown normalize_mode %r in checkpoint config; action normalizer DISABLED.",
            norm_mode,
        )
        return None

    mode_stats = load_mode_stats(stats_path, action_mode)
    if mode_stats is None:
        logger.warning(
            "[normalizer] Stats file %s has no '%s' entry; action normalizer DISABLED.",
            stats_path,
            action_mode,
        )
        return None

    normalizer = Normalizer(mode=YAML_TO_NORM_MODE[norm_mode], stats=mode_stats)
    logger.info(
        "[normalizer] Active: mode=%s action_mode=%s dim=%d stats=%s",
        norm_mode,
        action_mode,
        len(mode_stats["mean"]),
        stats_path,
    )
    return normalizer
