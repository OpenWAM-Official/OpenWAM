"""Model-loading helpers for inference and deployment.

Provides the checkpoint-first loading path used by OpenWAM deployment.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Optional, Tuple

import numpy as np
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
        # CosmosPredict25 Reason1 self-containment: when the ckpt was saved with the
        # ``reason1`` registration enabled, its weights live in the
        # unified safetensors and the small structural artifacts
        # (config.json + tokenizer.json) live under ``<ckpt_dir>/reason1/``.
        # Clearing ``text_encoder_path`` on that branch makes
        # ``build_cosmos_predict25_pipeline`` take its deploy/empty-shell path
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
        # Symmetric with the Reason1 clearing above: when the ckpt carries a VAE
        # state component, its weights live in the unified safetensors (via
        # ``vae``), so any training-time ``vae_path`` must be cleared.
        # Otherwise ``build_cosmos_predict25_pipeline``'s deploy guard
        # (``ckpt_dir is not None and vae_path_override is None``) stays False and
        # ``_resolve_vae_path`` raises FileNotFoundError on a host lacking the
        # original path, before weights ever load.
        has_vae_state_component = any(
            isinstance(c, dict) and c.get("attr") == "vae" for c in (vb_cfg_dict.get("components") or [])
        )
        if has_vae_state_component and vb_cfg_dict.get("vae_path") is not None:
            prev_vae_path = vb_cfg_dict.get("vae_path")
            vb_cfg_dict["vae_path"] = None
            logger.info(
                "Using self-contained VAE weights from checkpoint (clearing external vae_path=%r)",
                prev_vae_path,
            )
        vb_params["_source"] = vb_cfg_dict
        vb_params["_ckpt_dir"] = ckpt_dir
    else:
        model_path = OmegaConf.select(cfg, "model.video_backbone.model_path", default=None)
        if model_path is not None:
            vb_name = str(OmegaConf.select(cfg, "model.video_backbone.name", default=""))
            if vb_name.startswith("cosmos_predict25_"):
                # Cosmos carries fields the path-only string source would
                # lose: `shift_video`, `model_variant`, `text_encoder`.
                # `from_pretrained` accepts a dict natively (via
                # `_video_backbone_cfg`). Wan never hits this branch.
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
    _mp = OmegaConf.select(cfg, "training.mixed_precision", default="bf16")
    _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    model_dtype = _DTYPE_MAP.get(str(_mp).strip().lower(), torch.bfloat16)
    logger.info("Loading all models with dtype=%s (mixed_precision=%s)", model_dtype, _mp)

    architecture.set_dtype_device(model_dtype, torch.device(device))
    architecture.eval()

    # 6. Attach the action normalizer built from saved normalization_stats.npy + config.
    architecture.attach_normalizer(_build_normalizer(cfg, ckpt_dir))

    logger.info("Model loaded successfully on %s", device)
    return cfg, architecture


def _build_inner_normalizer(cfg: DictConfig, ckpt_dir: str):
    """Build the RAW-space normalizer for both deploy directions, or ``None`` if disabled.

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


class _UnifyAwareNormalizer:
    """Deploy-time normalizer that inverts the train-time ``normalize -> map_to_unify``.

    ckpts trained with ``dataloader.unify_action=true`` emit/consume UNIFY_DIM
    (e.g. 80-D) vectors, but the normalization stats live in RAW action space
    (the Normalizer ran BEFORE the unify scatter — robotwin.py). So the correct
    deploy directions are, mirroring ``RoboTwinDataset.denormalize_action``:

      * ``unnormalize`` (model action OUT): gather UNIFY_DIM → raw, THEN unnormalize.
      * ``normalize``   (proprio IN):       normalize raw, THEN scatter raw → UNIFY_DIM.

    ``inner`` may be ``None`` (unify on but ``normalize_mode=null``): then only the
    gather/scatter is applied (no (un)normalize) — still required, since the model
    is in unified space regardless of whether normalization was on.

    Duck-typed to the ``Normalizer`` surface ``base.py`` uses (``.unnormalize`` /
    ``.normalize``), so ``base.py`` needs no change.
    """

    def __init__(self, inner, dst_index: np.ndarray, unify_dim: int, base_slice=None, base_normalizer=None):
        from openwam.dataloader.utils.unify_action import map_to_unify, unmap_from_unify

        self._inner = inner
        self._dst_index = np.asarray(dst_index, dtype=np.int64)
        self._unify_dim = int(unify_dim)
        self._map_to_unify = map_to_unify
        self._unmap_from_unify = unmap_from_unify
        # Mobile base (robocasa365): the base command lives in a contiguous reserved span [68:73)
        # (action-only) with its OWN stats. When set, unnormalize gathers it, un-normalizes with
        # base_normalizer, and appends it → the returned raw action is [arm_raw, base_raw]. proprio
        # (normalize path) has no base, so it is unaffected.
        self._base_slice = base_slice
        self._base_normalizer = base_normalizer

    # action OUT: model emits (..., unify_dim) normalized-unified → physical raw.
    def unnormalize(self, x):
        arr = np.asarray(x)
        if arr.shape[-1] != self._unify_dim:
            # Defensive: already raw width (e.g. a non-unified head) → don't gather.
            logger.warning(
                "[normalizer/unify] unnormalize got last-dim %d != unify_dim %d; "
                "skipping gather (passing through).",
                arr.shape[-1], self._unify_dim,
            )
            return arr.copy() if self._inner is None else self._inner.unnormalize(arr)
        base_phys = None
        if self._base_slice is not None:
            # Gather base BEFORE un-unifying the arm (unmap collapses to arm width, dropping [68:73)).
            base = arr[..., self._base_slice]
            base_phys = base.copy() if self._base_normalizer is None else self._base_normalizer.unnormalize(base)
        raw = self._unmap_from_unify(arr, self._dst_index)  # (..., unify_dim) -> (..., arm_raw)
        arm = raw if self._inner is None else self._inner.unnormalize(raw)
        if base_phys is not None:
            return np.concatenate([np.asarray(arm), np.asarray(base_phys)], axis=-1)  # [arm_raw, base_raw]
        return arm

    # proprio IN: physical raw → normalized-unified (..., unify_dim) the model wants.
    def normalize(self, x):
        arr = np.asarray(x)
        if self._inner is not None:
            arr = self._inner.normalize(arr)
        unified, _mask = self._map_to_unify(arr, self._dst_index, self._unify_dim)
        return unified

    # Expose inner stats for callers that introspect (best-effort).
    @property
    def stats(self):
        return getattr(self._inner, "stats", {}) if self._inner is not None else {}


def _infer_raw_dim(inner) -> Optional[int]:
    """Best-effort raw action width from an inner Normalizer's stats (for identity maps)."""
    if inner is None:
        return None
    stats = getattr(inner, "stats", {}) or {}
    for k in ("mean", "q01", "min"):
        v = stats.get(k)
        if v is not None:
            return int(np.asarray(v).shape[0])
    return None


def _build_key_normalizer(cfg: DictConfig, ckpt_dir: str, key: str):
    """Build a Normalizer from the ``key`` stats block of normalization_stats.npy (mobile base uses
    key='base'). Returns None when normalization is disabled; raises if the block is missing."""
    norm_mode = OmegaConf.select(cfg, "dataloader.normalize_mode", default=None)
    if norm_mode in (None, "", "none", "null"):
        return None
    from openwam.dataloader.transforms.normalize import YAML_TO_NORM_MODE, Normalizer, load_mode_stats

    if norm_mode not in YAML_TO_NORM_MODE:
        return None
    stats_path = os.path.join(ckpt_dir, "normalization_stats.npy")
    stats = load_mode_stats(stats_path, key)
    if stats is None:
        raise ValueError(
            f"[normalizer/unify] mobile_base=True but normalization_stats.npy has no '{key}' block "
            f"({stats_path}); retrain — the dataloader persists a 'base' block when mobile_base=true."
        )
    return Normalizer(mode=YAML_TO_NORM_MODE[norm_mode], stats=stats)


def _build_normalizer(cfg: DictConfig, ckpt_dir: str):
    """Build the deploy normalizer, wrapping for ``unify_action`` when the ckpt used it.

    Non-unify ckpts: identical to upstream (returns the raw-space Normalizer or None).
    Unify ckpts: wrap in :class:`_UnifyAwareNormalizer` so the model's UNIFY_DIM output
    is gathered back to raw dims BEFORE unnormalize (and proprio scattered AFTER
    normalize) — the exact inverse of the train-time transform. With ``mobile_base``,
    the base command span [68:73) is un-normalized separately and appended (→ [arm_raw, base_raw]).
    """
    inner = _build_inner_normalizer(cfg, ckpt_dir)

    unify_on = bool(OmegaConf.select(cfg, "dataloader.unify_action", default=False))
    if not unify_on:
        return inner

    from openwam.dataloader.utils.unify_action import UNIFY_DIM, parse_unify_spec

    # Use the shared UNIFY_DIM constant — the train-side reader hardcodes it too and never
    # reads a `dataloader.unify_dim` key, so deploy must not either (a config-only value would
    # silently diverge from train's hardcoded width).
    unify_dim = UNIFY_DIM
    spec = OmegaConf.select(cfg, "dataloader.unify_action_map", default=None)
    if spec is None:
        raw_dim = _infer_raw_dim(inner)
        if raw_dim is None:
            raise ValueError(
                "[normalizer/unify] unify_action=True but dataloader.unify_action_map is missing "
                "AND raw dim can't be inferred (no normalization stats). Add unify_action_map to "
                "config.yaml (mirrors the reader's identity-map fallback)."
            )
        spec = list(range(raw_dim))  # identity, mirrors robotwin.py reader fallback

    dst_index = parse_unify_spec(spec, unify_dim)
    logger.info(
        "[normalizer/unify] unify_action ON: model emits %d-D unified actions → deploy gathers back "
        "to %d raw dims (dst_index len=%d) %s unnormalize. Mirrors RoboTwinDataset.denormalize_action.",
        unify_dim,
        dst_index.shape[0],
        dst_index.shape[0],
        "then" if inner is not None else "(no stats, gather-only:)",
    )
    # Mobile base (robocasa365): un-normalize the base command span [68:73) separately from its
    # own "base" stats block, so the returned raw action is [arm_raw, base_raw] (25-D). The client
    # then bridges arm (→OSC) and sends base raw.
    base_slice = base_normalizer = None
    if bool(OmegaConf.select(cfg, "dataloader.mobile_base", default=False)):
        from openwam.dataloader.robocasa365 import _UNIFY_BASE

        base_slice = _UNIFY_BASE
        base_normalizer = _build_key_normalizer(cfg, ckpt_dir, "base")
        logger.info(
            "[normalizer/unify] mobile_base ON: base command span [%d:%d) un-normalized separately "
            "→ deploy returns [arm_raw, base_raw].",
            base_slice.start,
            base_slice.stop,
        )
    return _UnifyAwareNormalizer(inner, dst_index, unify_dim, base_slice=base_slice, base_normalizer=base_normalizer)
