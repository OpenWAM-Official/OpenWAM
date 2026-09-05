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
        # CosmosPredict25 Reason1 self-containment: the ckpt's Reason1 weights
        # live in the unified safetensors and the small structural artifacts
        # (config.json + tokenizer.json) under ``<ckpt_dir>/reason1/`` —
        # ``save_deploy_assets`` writes the component marker and the artifact
        # dir together. Clearing ``text_encoder_path`` makes
        # ``build_cosmos_predict25_pipeline`` take its deploy/empty-shell path;
        # a missing ``reason1/`` dir then hard-errors inside ``from_empty``.
        # Iterate the plain-dict copy. ``vb_components`` is an OmegaConf
        # ``ListConfig`` whose entries are ``DictConfig`` (NOT a ``dict``
        # subclass) — so ``isinstance(c, dict)`` would always be False
        # against the real saved config, silently skipping the marker.
        has_reason1_state_component = any(
            isinstance(c, dict) and c.get("attr") == "text_encoder" for c in (vb_cfg_dict.get("components") or [])
        )
        if has_reason1_state_component:
            prev_path = vb_cfg_dict.get("text_encoder_path")
            vb_cfg_dict["text_encoder_path"] = None
            logger.info(
                "Using self-contained Reason1 artifacts from %s (clearing external text_encoder_path=%r)",
                os.path.join(ckpt_dir, "reason1"),
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
            if vb_name.startswith(("cosmos_predict25_", "cosmos3_")):
                # Cosmos families carry fields the path-only string source would
                # lose (predict2.5: `shift_video` / `model_variant` /
                # `text_encoder_path`; cosmos3: `shift_video` / prompt knobs).
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

    # 7. Binary command dims for the final legality projection at WAMPolicy.predict_action.
    # Authoritative source is the CKPT's dataloader config, never the deploy yaml: the training data
    # decided which dims are two-point commands, deploy config must not be able to alter that.
    _bd = OmegaConf.select(cfg, "dataloader.binary_action_dims", default=None)
    architecture.binary_command_dims = tuple(int(d) for d in (_bd or ()))

    # 8. Representation contract advertised to eval clients (PolicyServer PONG). Bound HERE — to the
    # TRAINING cfg, before merge_deploy_cfg lets deploy overrides win — so a deploy yaml carrying
    # stray dataloader.* keys can never make the PONG advertise values the architecture isn't
    # actually using (normalizer stats selection, binary dims, torso masking all derive from the
    # training cfg at this point).
    architecture.repr_contract = repr_contract_from_cfg(cfg)

    logger.info("Model loaded successfully on %s", device)
    return cfg, architecture


def repr_contract_from_cfg(cfg: DictConfig) -> dict:
    """Checkpoint representation fields that an eval client must match.

    Optional legacy markers are advertised only when the training config
    actually contains them.  Compact RoboCasa365 now defines gripper and mode
    semantics as part of its representation and performs both projections in
    the evaluation bridge, so its checkpoint contract needs only the
    representation name.
    """
    dims = OmegaConf.select(cfg, "dataloader.binary_action_dims", default=None)
    gripper_convention = OmegaConf.select(cfg, "dataloader.gripper_convention", default=None)
    contract = {
        "representation": str(OmegaConf.select(cfg, "dataloader.action_mode", default="joint")),
    }
    if dims:
        contract["binary_action_dims"] = [int(d) for d in dims]
    if gripper_convention is not None:
        contract["gripper_convention"] = gripper_convention
    return contract


def _build_inner_normalizer(cfg: DictConfig, ckpt_dir: str):
    """Build the RAW-space normalizer for both deploy directions, or ``None`` if disabled.

    The returned ``Normalizer`` serves both directions: ``normalize`` maps the
    input proprio state into training space and ``unnormalize`` maps model
    actions back to physical units.

    Reads ``dataloader.normalize_mode`` / ``action_mode`` from the saved config;
    when enabled, loads ``normalization_stats.npy`` and wraps the requested stats
    sub-dict. If a sibling ``<action_mode>_state`` block exists, the
    ``Normalizer`` uses it for proprio while retaining the action block for
    output unnormalization. When disabled, returns ``None``.
    """
    logger.info("[normalizer] Resolving deployment action normalizer from checkpoint dir: %s", ckpt_dir)

    dl = OmegaConf.select(cfg, "dataloader", default=None)
    norm_mode = OmegaConf.select(cfg, "dataloader.normalize_mode", default=None)
    action_mode = OmegaConf.select(cfg, "dataloader.action_mode", default="joint")
    if dl is None or norm_mode in (None, "", "none", "null"):
        # binary_action_dims / base_proprio='global_pose' stay fully consistent here: with
        # normalization off the model trains and serves in RAW space for every dim (binary targets
        # are raw ±1 by construction; the pose proprio is raw meters, no stats block needed), and the
        # binary legality projection runs in WAMPolicy.predict_action independently of the normalizer.
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

    # Past this point normalization is ACTIVE (dl present + a real normalize_mode +
    # the stats file exists). Any failure to actually build the normalizer must be a
    # HARD error, not a silent disable: a disabled normalizer would hand the model's
    # normalized [-1, 1] outputs straight to the robot as physical poses/velocities.
    if norm_mode not in YAML_TO_NORM_MODE:
        raise ValueError(
            f"[normalizer] Checkpoint config sets an active normalize_mode={norm_mode!r} that is not a "
            f"recognized mode {sorted(YAML_TO_NORM_MODE)}. Refusing to deploy with normalization silently "
            "disabled (the model emits normalized actions; unnormalize would be skipped). Fix the checkpoint "
            "config, or add the mode to YAML_TO_NORM_MODE."
        )

    mode_stats = load_mode_stats(stats_path, action_mode)
    if mode_stats is None:
        raise KeyError(
            f"[normalizer] Stats file {stats_path} has no '{action_mode}' entry (its keys are written from the "
            f"training reader's DEPLOY_ACTION_MODE). Refusing to deploy with normalization silently disabled: "
            "the model emits normalized actions for this action_mode and unnormalize would be skipped, sending "
            "normalized [-1, 1] values to the robot as physical poses/velocities. Regenerate "
            "normalization_stats.npy so it contains the checkpoint's action_mode key."
        )

    normalizer = Normalizer(mode=YAML_TO_NORM_MODE[norm_mode], stats=mode_stats)
    logger.info(
        "[normalizer] Active: mode=%s action_mode=%s dim=%d stats=%s",
        norm_mode,
        action_mode,
        len(mode_stats["mean"]),
        stats_path,
    )

    # Legacy command-aware extras (keys absent = ordinary normalization):
    #   binary_action_dims — optional compatibility support for older checkpoints whose
    #     two-point commands bypassed normalization. New benchmark adapters own their
    #     command projection at the environment boundary.
    #   base_proprio="global_pose" — the proprio normalizes with its own 'eef_base_pose_proprio'
    #     stats block (pose is meters/unit-circle, not command space), while actions keep action_mode's.
    binary_dims = OmegaConf.select(cfg, "dataloader.binary_action_dims", default=None)
    base_proprio = OmegaConf.select(cfg, "dataloader.base_proprio", default="velocity")
    if base_proprio not in ("velocity", "global_pose"):
        raise ValueError(
            f"[normalizer] dataloader.base_proprio must be 'velocity' or 'global_pose', got {base_proprio!r}"
        )
    if binary_dims or base_proprio == "global_pose":
        proprio_inner = normalizer
        if base_proprio == "global_pose":
            # Same literal as the reader's _PROPRIO_POSE_STATS_KEY (not imported: the reader module
            # drags pandas/video deps into the serving process).
            pose_stats = load_mode_stats(stats_path, "eef_base_pose_proprio")
            if pose_stats is None:
                raise KeyError(
                    f"[normalizer] base_proprio='global_pose' but {stats_path} has no "
                    "'eef_base_pose_proprio' block. Regenerate normalization_stats.npy "
                    "(robocasa365_stats_computation --mobile-base emits it)."
                )
            proprio_inner = Normalizer(mode=YAML_TO_NORM_MODE[norm_mode], stats=pose_stats)
        normalizer = _CommandAwareNormalizer(
            action_inner=normalizer,
            proprio_inner=proprio_inner,
            binary_dims=tuple(int(d) for d in (binary_dims or ())),
        )
        logger.info(
            "[normalizer] command-aware: base_proprio=%s binary_action_dims=%s",
            base_proprio,
            list(normalizer._binary_dims),
        )
    return normalizer


class _CommandAwareNormalizer:
    """Raw-space normalizer wrapper for command-dim semantics the plain Normalizer can't express.

    * ``unnormalize`` (action OUT): continuous dims go through the inner stats inverse; each
      ``binary_dims`` dim is judged in the PRE-unnormalize space — the raw ±1 space the model was
      trained in, since those legacy targets bypassed normalization — and
      overridden to exact {-1, +1}. Judging the post-inverse value instead would corrupt the class
      under any non-identity stats (z-score: a correct +1 becomes ~-0.35 and snaps to -1).
      Threshold 0.5 — NOT the ±1 midpoint 0 — deliberately preserves the downstream conservative
      boundaries byte-for-byte (bridge confident-close ``>0.5``, env control_mode ``>=0.5``): an
      uncertain mid-range output still lands on the safe side (open / arm mode).
    * ``normalize`` (proprio IN): delegates to ``proprio_inner``. A plain
      ``Normalizer`` may itself carry a directional ``<action_mode>_state``
      stats block; legacy ``base_proprio='global_pose'`` uses a separate object.

    Executors do not mix overlapping chunks. The final projection in
    ``WAMPolicy.predict_action`` remains a defense-in-depth wire-contract check
    for raw or legacy engine outputs that land between the two legal commands.

    Duck-typed to the ``Normalizer`` surface (``.unnormalize`` / ``.normalize`` / ``.stats``), so it
    composes under :class:`_UnifyAwareNormalizer` unchanged (gather → unnormalize+snap in raw space;
    normalize in raw space → scatter).
    """

    def __init__(self, action_inner, proprio_inner, binary_dims: tuple):
        self._action_inner = action_inner
        self._proprio_inner = proprio_inner
        self._binary_dims = tuple(int(d) for d in binary_dims)

    def unnormalize(self, x):
        x = np.asarray(x)
        for d in self._binary_dims:
            if d >= x.shape[-1]:
                raise ValueError(
                    f"binary dim {d} out of range for a {x.shape[-1]}-D action — the input is not "
                    "the raw-width vector this normalizer was built for (unify gather skipped?)."
                )
        y = np.array(self._action_inner.unnormalize(x))
        for d in self._binary_dims:
            y[..., d] = np.where(x[..., d] > 0.5, 1.0, -1.0)
        return y

    def normalize(self, x):
        return self._proprio_inner.normalize(x)

    @property
    def stats(self):
        return getattr(self._action_inner, "stats", {})


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

    def __init__(
        self,
        inner,
        action_dst_index: np.ndarray,
        unify_dim: int,
        state_dst_index: Optional[np.ndarray] = None,
    ):
        from openwam.dataloader.utils.unify_action import map_to_unify, unmap_from_unify

        self._inner = inner
        self._action_dst_index = np.asarray(action_dst_index, dtype=np.int64)
        self._state_dst_index = np.asarray(
            state_dst_index if state_dst_index is not None else action_dst_index,
            dtype=np.int64,
        )
        # Backward-compatible introspection name: this has always meant the
        # action gather map.
        self._dst_index = self._action_dst_index
        self._unify_dim = int(unify_dim)
        self._map_to_unify = map_to_unify
        self._unmap_from_unify = unmap_from_unify

    # action OUT: model emits (..., unify_dim) normalized-unified → physical raw.
    def unnormalize(self, x):
        arr = np.asarray(x)
        if arr.shape[-1] != self._unify_dim:
            # Defensive: already raw width (e.g. a non-unified head) → don't gather.
            logger.warning(
                "[normalizer/unify] unnormalize got last-dim %d != unify_dim %d; skipping gather (passing through).",
                arr.shape[-1],
                self._unify_dim,
            )
            return arr.copy() if self._inner is None else self._inner.unnormalize(arr)
        raw = self._unmap_from_unify(arr, self._action_dst_index)
        return raw if self._inner is None else self._inner.unnormalize(raw)

    # proprio IN: physical raw → normalized-unified (..., unify_dim) the model wants.
    def normalize(self, x):
        arr = np.asarray(x)
        if self._inner is not None:
            arr = self._inner.normalize(arr)
        unified, _mask = self._map_to_unify(arr, self._state_dst_index, self._unify_dim)
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


def _build_normalizer(cfg: DictConfig, ckpt_dir: str):
    """Build the deploy normalizer, wrapping for ``unify_action`` when the ckpt used it.

    Non-unify ckpts: identical to upstream (returns the raw-space Normalizer or None).
    Unify ckpts: wrap in :class:`_UnifyAwareNormalizer` so the model's UNIFY_DIM output is gathered
    back to the raw width BEFORE unnormalize (and proprio scattered AFTER normalize) — the exact
    inverse of the train-time transform. FULLY GENERIC: the raw width and its stats come from the
    reader's ``action_mode`` statistics. An optional ``unify_state_map`` allows
    compact RoboCasa365's state19 and action15 to use distinct raw widths while
    preserving the symmetric map used by older datasets.
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

    action_dst_index = parse_unify_spec(spec, unify_dim)
    # Most datasets use one symmetric raw layout. RoboCasa365 compact is
    # intentionally asymmetric (action15 vs state19), so it declares a second
    # map. Absence preserves every legacy checkpoint's behavior.
    state_spec = OmegaConf.select(cfg, "dataloader.unify_state_map", default=None)
    state_dst_index = action_dst_index if state_spec is None else parse_unify_spec(state_spec, unify_dim)
    logger.info(
        "[normalizer/unify] ON: action %dD -> raw %dD; raw state %dD -> %dD unified; %s unnormalize.",
        unify_dim,
        action_dst_index.shape[0],
        state_dst_index.shape[0],
        unify_dim,
        "then" if inner is not None else "(no stats, gather-only:)",
    )
    return _UnifyAwareNormalizer(
        inner,
        action_dst_index,
        unify_dim,
        state_dst_index=state_dst_index,
    )
