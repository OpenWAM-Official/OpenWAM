"""Lazy loader for the Cosmos-Predict2.5 pipeline.

Builds a :class:`CosmosPredict25VideoBackbone` from the ``cosmos_predict2`` package
installed off the ``third_party/cosmos-predict2.5`` submodule, hides the
upstream ``MinimalV1LVGDiT`` config
behind a stable signature, and lets :meth:`CosmosPredict25VideoBackbone.from_pretrained`
treat the result as a duck-typed pipeline. The whole upstream import graph is
deferred until the function is called so CPU-only CI keeps working.

Currently supports the 2B base / post-trained / distilled variants. The 14B
variant is registered for naming but does not have its config baked in yet
(probe its checkpoint with ``scripts/install_cosmos_predict25.sh``-installed Python
when the weights land).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

_COSMOS_INSTALL_HINT = (
    "Cosmos-Predict2.5 dependencies are not installed. Initialise the submodule "
    "(`git submodule update --init third_party/cosmos-predict2.5`) and then run "
    "`bash scripts/install_cosmos_predict25.sh`."
)

# Phase 4 VAE wiring. The Cosmos-Predict2.5 bundle ships its tokenizer weights
# at `<model_path>/tokenizer.pth` (485 MB, raw Wan2pt1 state dict). The
# upstream interface used to load it is `Wan2pt1VAEInterface`; it is NOT an
# `nn.Module`, so we manage device/dtype movement ourselves (see
# `adapter.py::_move_cosmos_vae`).
_COSMOS25_VAE_FILENAME = "tokenizer.pth"
_VAE_CHOICES = {"none", "wan2pt1"}


# Concrete geometry for the 2B 720p network (matches probed
# `*_ema_bf16.pt` checkpoints: 28 blocks, dim=2048, head_dim=128, context_dim=1024).
# Mirrors `cosmos_predict2/_src/predict2/configs/video2world/defaults/net.py::COSMOS_V1_2B_NET_MININET`
# plus the Stage-c experiment overrides
# (`crossattn_proj_in_channels=100352`, `rope_*_extrapolation_ratio=3.0`, ...).
_COSMOS25_2B_NET_KWARGS: dict = dict(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,  # MinimalV1LVGDiT bumps this by +1 internally for the condition mask
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    crossattn_emb_channels=1024,
    use_crossattn_projection=True,
    crossattn_proj_in_channels=100352,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="transformer_engine",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False,
    # NB: SACConfig defaults to mode="mm_only" upstream, which wraps every
    # block in a `_checkpoint_wrapped_module` shim and renames every state-dict
    # key (`blocks.0._checkpoint_wrapped_module.self_attn.q_proj.weight`). The
    # released checkpoints were saved WITHOUT the wrapper, so wrapping at
    # __init__ time produces 56 missing `_extra_state` keys on strict=False
    # load. We always construct with mode=NONE and then re-apply SAC after
    # weight loading via the `sac_mode` config knob (see build_cosmos_predict25_pipeline).
    # SACConfig objects are constructed lazily inside the builder so this
    # module stays importable on CPU CI.
)

# Exposed geometry that build_cosmos_predict25_pipeline attaches to the wrapper so
# CosmosPredict25VideoBackbone._probe_pipeline_geometry can read them off.
_COSMOS25_2B_GEOMETRY = dict(
    dim=2048,
    num_layers=28,
    num_heads=16,
    head_dim=128,
    context_dim=1024,
)

# The geometry the backbone probes (above) and the net actually constructed
# from ``_COSMOS25_2B_NET_KWARGS`` are two independent hard-coded specs; guard
# against silent drift between them at import time (cheap; both are plain dicts).
assert _COSMOS25_2B_GEOMETRY["dim"] == _COSMOS25_2B_NET_KWARGS["model_channels"], (
    "cosmos_predict25 2B geometry dim != net model_channels"
)
assert _COSMOS25_2B_GEOMETRY["num_heads"] == _COSMOS25_2B_NET_KWARGS["num_heads"], (
    "cosmos_predict25 2B geometry num_heads != net num_heads"
)
assert (
    _COSMOS25_2B_GEOMETRY["head_dim"]
    == _COSMOS25_2B_NET_KWARGS["model_channels"] // _COSMOS25_2B_NET_KWARGS["num_heads"]
), "cosmos_predict25 2B geometry head_dim != model_channels // num_heads"
assert _COSMOS25_2B_GEOMETRY["num_layers"] == _COSMOS25_2B_NET_KWARGS["num_blocks"], (
    "cosmos_predict25 2B geometry num_layers != net num_blocks"
)


def import_cosmos_predict2():
    """Import the upstream ``cosmos_predict2`` package, with a clear error message."""
    try:
        import cosmos_predict2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only when extra is missing
        raise ImportError(_COSMOS_INSTALL_HINT) from exc
    return cosmos_predict2


def _resolve_vae_path(model_path: Path, override: Optional[str]) -> Path:
    """Locate the Cosmos VAE checkpoint (``tokenizer.pth``) for the **training**
    bootstrap path.

    If ``override`` is set we use it verbatim; otherwise we default to
    ``<model_path>/tokenizer.pth``. Raises ``FileNotFoundError`` with a clear
    "either set vae_path or use vae: none" exit message if the resolved path
    is missing.

    Deploy-time builds (signalled by a non-``None`` ``ckpt_dir`` in
    :func:`build_cosmos_predict25_pipeline`) skip this helper entirely and build an
    empty VAE shell via ``vae_pth=None`` — weights then load from the
    architecture's unified safetensors. See :func:`_build_cosmos_predict25_vae`.
    """
    if override is not None:
        candidate = Path(override)
    else:
        candidate = model_path / _COSMOS25_VAE_FILENAME
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Cosmos VAE checkpoint not found at {candidate}. Either: "
            f"(a) place `{_COSMOS25_VAE_FILENAME}` at {model_path}/, "
            f"(b) point `video_backbone.vae_path` at the actual file, or "
            "(c) set `video_backbone.vae: none` (caller must supply pre-encoded latents)."
        )
    return candidate


def _build_cosmos_predict25_vae(vae_pth: Optional[Path], *, device, dtype):
    """Construct a frozen ``Wan2pt1VAEInterface``.

    Two modes:

    - ``vae_pth`` is a real path → load weights from disk (training bootstrap).
    - ``vae_pth is None`` → build an empty shell via upstream's
      ``_video_vae(pretrained_path=None)`` → ``WanVAE_.to_empty()`` path
      (``wan2pt1.py:619-623``). The shell's inner ``WanVAE_`` is registered
      as a sub-module of ``CosmosPredict25VideoBackbone`` (see its ``__init__``),
      so the architecture's ``load_checkpoint`` populates its weights from the
      unified safetensors. Used at deploy time when ``ckpt_dir`` carries the
      saved state.

    The inner ``WanVAE`` already does ``model.eval().requires_grad_(False)``
    (``wan2pt1.py:788``) so the VAE is frozen by construction.
    """
    import torch
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import (  # type: ignore[import-not-found]
        Wan2pt1VAEInterface,
    )

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    iface = Wan2pt1VAEInterface(
        vae_pth=str(vae_pth) if vae_pth is not None else None,
        s3_credential_path="",
        temporal_window=4,
        is_parallel=False,
        load_mean_std=False,
    )
    # Upstream defaults to device="cuda" in WanVAE.__init__ (line 710); even
    # if we're targeting a different device the mean/std tensors and inner
    # nn.Module need to be moved explicitly. Reuse the shared VAE helper.
    from openwam.model.video_backbone.cosmos_predict25._vae_utils import _move_cosmos_vae

    _move_cosmos_vae(iface, dtype=dtype, device=target_device)
    return iface


def _resolve_checkpoint_path(model_path: Path, variant: str) -> Path:
    """Glob the single ``*_ema_bf16.pt`` checkpoint under ``model_path/variant``.

    The Cosmos-Predict2.5 bundle layout stores
    EMA inference weights as a single UUID-named file inside each variant
    sub-directory (``base/post-trained/<uuid>_ema_bf16.pt``). We refuse to
    guess when more than one file matches; the variant under
    ``base/pre-trained/`` also contains a 12 GB optimizer-state checkpoint we
    must NOT load.
    """
    variant_dir = model_path / variant
    if not variant_dir.is_dir():
        raise FileNotFoundError(
            f"Cosmos variant directory does not exist: {variant_dir}. "
            f"Valid choices under {model_path}: base/pre-trained, base/post-trained, "
            f"base/distilled, auto/multiview, robot/multiview-agibot, robot/action-cond."
        )
    candidates = sorted(p for p in variant_dir.glob("*_ema_bf16.pt") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No *_ema_bf16.pt found under {variant_dir}. Cosmos-Predict2.5 EMA weights "
            "are named `<uuid>_ema_bf16.pt` — point `model_path` at the bundle root."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple *_ema_bf16.pt candidates under {variant_dir}: {candidates}. "
            "Refusing to pick one; rename or move so exactly one remains."
        )
    return candidates[0]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _video_backbone_cfg(source: Any) -> Any:
    """Extract the ``video_backbone`` sub-config from a Hydra cfg / path / dict."""
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        return {"model_path": str(source)}
    if isinstance(source, dict):
        model = source.get("model") if "model" in source else source
        vb = model.get("video_backbone") if isinstance(model, dict) else None
        return vb if vb is not None else source
    model = getattr(source, "model", source)
    vb = getattr(model, "video_backbone", None)
    return vb if vb is not None else source


def _load_state_dict_into_net(net, ckpt_path: Path) -> Tuple[list, list]:
    """Load weights from a Cosmos-native ``*_ema_bf16.pt`` into ``net``.

    The checkpoint is a flat ``dict[str, Tensor]`` with every key prefixed by
    ``net.`` (689 keys for the 2B). TE's ``RMSNorm`` modules carry an
    ``_extra_state`` entry containing RNG / FP8 metadata — these load fine via
    TE's deserializer when the real ``transformer_engine`` is installed (see
    ``scripts/install_cosmos_predict25.sh``).
    """
    import torch

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cleaned = {}
    skipped_prefix = 0
    for k, v in raw.items():
        if k.startswith("net."):
            cleaned[k[len("net.") :]] = v
        else:
            skipped_prefix += 1
    if skipped_prefix:
        logger.info("Cosmos checkpoint had %d keys outside the `net.` namespace (skipped)", skipped_prefix)
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning(
            "Cosmos MinimalV1LVGDiT load: %d missing keys (first 3: %s)",
            len(missing),
            missing[:3],
        )
    if unexpected:
        logger.warning(
            "Cosmos MinimalV1LVGDiT load: %d unexpected keys (first 3: %s)",
            len(unexpected),
            unexpected[:3],
        )
    return missing, unexpected


def build_cosmos_predict25_pipeline(
    source: Any,
    *,
    device: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    **_unused,
):
    """Construct a :class:`CosmosPredict25VideoBackbone` from *source*.

    ``source`` is either:
      - a Hydra ``DictConfig`` carrying ``video_backbone.{model_path, model_variant,
        text_encoder_path, shift_video}``,
      - a plain dict with the same shape,
      - a model directory ``str`` / ``Path`` — deploy path only (``ckpt_dir``
        non-None); the training path requires ``text_encoder_path``, which a
        bare path cannot carry.

    Returns the wrapper with ``dim`` / ``num_layers`` / ``num_heads`` / ``head_dim``
    / ``context_dim`` attached as attributes so that
    :meth:`CosmosPredict25VideoBackbone._probe_pipeline_geometry` succeeds.
    """
    # Validate user-facing config before importing the upstream package so
    # `sac_mode=foo` raises a clear ValueError on CPU CI too, not just on hosts
    # that have `cosmos_predict2` installed.
    vb_cfg = _video_backbone_cfg(source)
    model_path_raw = _cfg_get(vb_cfg, "model_path")
    # ``model_path`` is required for the *training* bootstrap (DiT
    # ``*_ema_bf16.pt`` + VAE ``tokenizer.pth``). On the deploy path
    # (``ckpt_dir`` non-None) both come from the unified safetensors instead,
    # so ``model_path`` is optional — this is what makes a cosmos_predict25 checkpoint
    # truly portable to a host without the training-time asset paths.
    if model_path_raw is None and ckpt_dir is None:
        raise ValueError(
            "build_cosmos_predict25_pipeline requires `video_backbone.model_path` to be set "
            "(point it at the Cosmos-Predict2.5 bundle root, e.g. "
            "<assets-root>/Cosmos-Predict2.5-2B)."
        )
    model_path = Path(model_path_raw) if model_path_raw is not None else None
    variant = str(_cfg_get(vb_cfg, "model_variant", "base/post-trained"))
    shift_video = float(_cfg_get(vb_cfg, "shift_video", 5.0))
    sac_mode_raw = str(_cfg_get(vb_cfg, "sac_mode", "none")).lower()
    _valid_sac_modes = {"none", "mm_only", "block_wise"}
    if sac_mode_raw not in _valid_sac_modes:
        raise ValueError(
            f"video_backbone.sac_mode={sac_mode_raw!r} is not valid. "
            f"Choose one of: {sorted(_valid_sac_modes)} "
            "(see cosmos_predict2/_src/predict2/networks/selective_activation_checkpoint.py)."
        )
    vae_choice = str(_cfg_get(vb_cfg, "vae", "wan2pt1")).lower()
    if vae_choice not in _VAE_CHOICES:
        raise ValueError(
            f"video_backbone.vae={vae_choice!r} is not valid. "
            f"Choose one of: {sorted(_VAE_CHOICES)} "
            "(wan2pt1 = real Cosmos-Predict2.5 tokenizer; none = caller supplies pre-encoded latents)."
        )
    # `text_encoder_path` is checked here (pre-import / pre-checkpoint-load)
    # so an UNSET path fails fast before the 5 GB DiT load; a mistyped path
    # value still only surfaces at Reason1 construction, after the DiT load.
    # On the training path (``ckpt_dir is None``) it tells the builder where
    # to load Reason1 so it can be registered under ``reason1`` and saved into
    # the unified safetensors. On the deploy path (``ckpt_dir`` non-None),
    # weights flow from safetensors and the loader points ``from_empty`` at
    # ``<ckpt_dir>/reason1/`` structural artifacts instead.
    text_encoder_path_raw = _cfg_get(vb_cfg, "text_encoder_path", None)
    if text_encoder_path_raw is None and ckpt_dir is None:
        raise ValueError(
            "video_backbone.text_encoder_path is required on the training path. "
            "Point it at the Cosmos-Reason1-7B bundle root, e.g. "
            "<assets-root>/Cosmos-Reason1-7B."
        )
    # §14.7 — CFG dropout. The actual substitution happens in
    # `CosmosPredict25VideoBackbone._preprocess_input`; validate the range here
    # so a user misconfiguration fails before any 5 GB DiT load.
    text_dropout_p_raw = _cfg_get(vb_cfg, "text_encoder_dropout", 0.0)
    try:
        text_dropout_p = float(text_dropout_p_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"video_backbone.text_encoder_dropout={text_dropout_p_raw!r} is not a float in [0, 1]."
        ) from exc
    if not 0.0 <= text_dropout_p <= 1.0:
        raise ValueError(f"video_backbone.text_encoder_dropout={text_dropout_p} is out of range; must be in [0, 1].")
    text_dropout_seed = _cfg_get(vb_cfg, "text_encoder_dropout_seed", None)
    if text_dropout_seed is not None:
        try:
            text_dropout_seed = int(text_dropout_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"video_backbone.text_encoder_dropout_seed={text_dropout_seed!r} must be an integer or null."
            ) from exc
    vae_path_override = _cfg_get(vb_cfg, "vae_path", None)
    name = str(_cfg_get(vb_cfg, "name", "cosmos_predict25_2b"))

    import torch

    import_cosmos_predict2()
    # Late imports keep CPU CI green without the [cosmos_predict25] extra.
    import types

    from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import (  # type: ignore[import-not-found]
        MinimalV1LVGDiT,
    )
    from cosmos_predict2._src.predict2.networks.minimal_v4_dit import (  # type: ignore[import-not-found]
        CheckpointMode,
        SACConfig,
    )

    if name not in ("cosmos_predict25_2b",):
        raise NotImplementedError(
            f"build_cosmos_predict25_pipeline only handles `cosmos_predict25_2b` today; "
            f"got name={name!r}. The 14B config is registered but not wired up yet."
        )

    # Always construct with SAC disabled — the released `*_ema_bf16.pt`
    # checkpoints were saved without the `_checkpoint_wrapped_module` prefix,
    # so wrapping at __init__ time would surface 56 missing `_extra_state`
    # keys on strict=False load. We re-apply SAC after the weights are in
    # place; `MinimalV1LVGDiT.enable_selective_checkpoint` is idempotent and
    # public (see third_party/cosmos-predict2.5/cosmos_predict2/_src/predict2/
    # networks/minimal_v4_dit.py:1800).
    net = MinimalV1LVGDiT(
        sac_config=SACConfig(mode=CheckpointMode.NONE),
        **_COSMOS25_2B_NET_KWARGS,
    )
    # Deploy path (`ckpt_dir` non-None): the DiT params live inside the saved
    # safetensors (`CosmosPredict25VideoBackbone.net` is a registered nn.Module
    # child of the wrapper, so its state goes through OpenWAM's unified
    # save/load like any other dual_system / single_system weight). We
    # therefore skip the eager `*_ema_bf16.pt` load and let
    # `arch.load_checkpoint` populate `net` from the safetensors. Training
    # path keeps the eager bootstrap so a fresh run starts from the released
    # EMA weights.
    if ckpt_dir is None:
        ckpt_path = _resolve_checkpoint_path(model_path, variant)
        logger.info("Cosmos checkpoint: %s", ckpt_path)
        _load_state_dict_into_net(net, ckpt_path)
    else:
        logger.info("Cosmos DiT: skipping `*_ema_bf16.pt` bootstrap (deploy path; weights from unified safetensors)")
    if sac_mode_raw != "none":
        logger.info("Enabling Cosmos SAC selective-checkpoint (mode=%s) post-load", sac_mode_raw)
        net.enable_selective_checkpoint(SACConfig(mode=CheckpointMode(sac_mode_raw)), net.blocks)
    net = net.to(dtype=torch.bfloat16)

    from openwam.model.video_backbone.cosmos_predict25.text_encoder import Reason1LiveTextEncoder

    # Deploy path (``ckpt_dir`` non-None and no explicit path override):
    # build a meta-device shell — the inner Qwen module is registered as
    # ``reason1`` on the backbone, so ``arch.load_checkpoint`` will populate
    # its weights from the unified safetensors. Only the small structural
    # files (``config.json`` + ``tokenizer.json``) need to exist on the deploy
    # host, under ``<ckpt_dir>/reason1/``. Training path keeps the eager load
    # from the full Cosmos-Reason1 bundle.
    if ckpt_dir is not None and text_encoder_path_raw is None:
        artifact_dir = Path(ckpt_dir) / "reason1"
        logger.info(
            "Cosmos Reason1: building empty shell (deploy path; weights from unified safetensors, "
            "structural artifacts from %s)",
            artifact_dir,
        )
        text_encoder_obj = Reason1LiveTextEncoder.from_empty(artifact_dir, dtype=torch.bfloat16, device=device)
    else:
        logger.info("Cosmos Reason1 live text encoder: loading from %s", text_encoder_path_raw)
        text_encoder_obj = Reason1LiveTextEncoder(Path(text_encoder_path_raw), dtype=torch.bfloat16, device=device)

    vae_obj = None
    if vae_choice != "none":
        # Deploy path (``ckpt_dir`` non-None): the saved safetensors carries
        # the VAE state (it's registered as ``vae`` on the wrapper, see
        # ``pipeline_wrapper.py::__init__``). Build an empty shell so we don't
        # require ``tokenizer.pth`` to be reachable on the deploy host; the
        # architecture's ``load_checkpoint`` will populate the weights.
        # Training path (``ckpt_dir`` None) loads from ``tokenizer.pth`` as
        # before — the safetensors will then capture the loaded values on
        # the first checkpoint save.
        # Explicit ``vae_path`` always wins (it's a user override of the
        # training-bootstrap path).
        if ckpt_dir is not None and vae_path_override is None:
            logger.info("Cosmos VAE: building empty shell (deploy path; weights from unified safetensors)")
            vae_obj = _build_cosmos_predict25_vae(None, device=device, dtype=torch.bfloat16)
        else:
            vae_pth = _resolve_vae_path(model_path, vae_path_override)
            logger.info("Cosmos VAE: loading %s from %s", vae_choice, vae_pth)
            vae_obj = _build_cosmos_predict25_vae(vae_pth, device=device, dtype=torch.bfloat16)

    # Deploy path: ``reason1`` (and the empty VAE shell) live on the
    # ``meta`` device until ``arch.load_checkpoint`` materialises them. A
    # ``.to(device)`` there would recurse into the meta shell and crash
    # ("Cannot copy out of meta tensor; no data!"). So only move the DiT on the
    # TRAINING path (``ckpt_dir`` None); the deploy loader calls
    # ``set_dtype_device`` after ``load_checkpoint`` populates real weights.
    if device is not None and ckpt_dir is None:
        net = net.to(device)

    # Lightweight holder (no nn.Module wrapper). ``CosmosPredict25VideoBackbone.from_pretrained``
    # drains it into flat children (Wan holder-drain parity). Carries the five
    # geometry fields ``_probe_pipeline_geometry`` reads.
    return types.SimpleNamespace(
        net=net,
        vae=vae_obj,
        text_encoder=text_encoder_obj,
        shift_video=shift_video,
        text_dropout_p=text_dropout_p,
        text_dropout_seed=text_dropout_seed,
        **_COSMOS25_2B_GEOMETRY,
    )


__all__ = ["build_cosmos_predict25_pipeline", "import_cosmos_predict2"]
