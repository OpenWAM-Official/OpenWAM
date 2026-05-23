"""Load a SANA-Video pipeline (DiT + VAE + text encoder) for OpenWAM.

This wraps ``third_party/Sana`` upstream into a tiny ``SanaPipe`` object that
``SanaVideoBackbone`` then drives through the OpenWAM block-loop contract.
The goal here is **not** to reimplement SANA — we hold references to upstream
modules and feed them OpenWAM-shaped inputs. Pin-bumps of the submodule are
transparent as long as the upstream class signatures wrapped here don't change.

Phase 0 supports two pre-config'd variants: ``sana_video_2b_480p`` (the only
HF-published video model at time of writing) and ``mini`` (random-weight tiny
model for unit tests). Real inference / training over the published 2B weights
requires the upstream ``diffusion.model.builder`` chain — we lazy-import that
chain on first use so that ``import openwam.model.video_backbone.sana`` stays
cheap and survives a broken timm install at import time.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# SANA's own inference scripts set this to disable xformers' SDPA hijack.
# Doing it on module import keeps the upstream's `_xformers_available` guard
# at False, which forces the deterministic mask path.
os.environ.setdefault("DISABLE_XFORMERS", "1")


@dataclass
class SanaPipe:
    """Thin container of the components ``SanaVideoBackbone`` needs.

    Mirrors the role of Wan's pipe object — backbone code reaches via
    attribute access, never deep-inspects this struct's internals.
    """

    dit: nn.Module
    """The ``SanaMSVideo`` DiT (or a mini variant for tests)."""

    vae: Optional[nn.Module] = None
    """``WanVAE`` — the SANA-Video 2B 480p HF ckpt re-uses the Wan VAE."""

    text_encoder: Optional[nn.Module] = None
    """Gemma-2-2B text encoder. Not bundled in the SANA HF ckpt — load
    separately via OpenWAM's existing text-encoder utilities."""

    tokenizer: Optional[Any] = None

    scheduler: Optional[Any] = None
    """Flow-Euler scheduler with the upstream's video shift."""

    # Auxiliary metadata for the adapter
    config: dict = field(default_factory=dict)
    """Snapshot of upstream config (``model.*`` / ``vae.*`` / ``text_encoder.*``)."""


# ---------------------------------------------------------------------------
# Upstream model factory dispatch
# ---------------------------------------------------------------------------


def _resolve_upstream_factory(model_name: str) -> Any:
    """Return the upstream factory function registered under ``model_name``.

    Examples: ``SanaMSVideo_2000M_P2_D20`` is the 2B 480p model factory at
    ``third_party/Sana/diffusion/model/nets/sana_multi_scale_video.py:1054``.
    Importing the factory triggers `MODELS.register_module()` side effects
    in upstream, so the lookup must happen after the import.
    """
    # Lazy import — keeps adapter cheap to import and tolerates partial venvs.
    from diffusion.model.builder import MODELS  # type: ignore[import-not-found]
    from diffusion.model.nets import sana_multi_scale_video  # noqa: F401  # registers factories

    if model_name not in MODELS._module_dict:
        raise KeyError(
            f"SANA model factory {model_name!r} not registered. Available: "
            f"{sorted(MODELS._module_dict)[:8]}..."
        )
    return MODELS.get(model_name)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_sana_pipeline(
    cfg_or_path: Any,
    *,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    ckpt_dir: Optional[str] = None,
) -> SanaPipe:
    """Build a ``SanaPipe`` from a config or model directory.

    Accepts the same three input shapes as ``WanVideoBackbone.from_pretrained``:

    - ``omegaconf.DictConfig``: full Hydra config → reads ``video_backbone.*``
    - ``str`` directory: looks for ``checkpoints/*.pth`` + ``config.json``
    - ``dict``: full architecture params (or ``{"video_backbone": {...}}``)
      → reads ``video_backbone.*``; ``resolve_architecture_config`` returns
      plain dicts so this is the actual training-time entry point.
    """
    from omegaconf import DictConfig

    if isinstance(cfg_or_path, DictConfig):
        vb_cfg = cfg_or_path.get("video_backbone", cfg_or_path)
        spec = _spec_from_dictconfig(vb_cfg)
    elif isinstance(cfg_or_path, str):
        if not os.path.isdir(cfg_or_path):
            raise ValueError(
                f"build_sana_pipeline(str) expects a directory, got: {cfg_or_path!r}"
            )
        spec = _spec_from_model_dir(cfg_or_path)
    elif isinstance(cfg_or_path, dict):
        # Unwrap ``{video_backbone: {...}}`` if the caller passed full
        # architecture params (which is what ``resolve_architecture_config``
        # produces).
        vb_cfg = cfg_or_path.get("video_backbone", cfg_or_path)
        spec = _spec_from_dict(vb_cfg)
    else:
        raise TypeError(
            f"build_sana_pipeline: unsupported source type {type(cfg_or_path).__name__}"
        )

    return _build_pipe_from_spec(spec, device=device, dtype=dtype, ckpt_dir=ckpt_dir)


def build_mini_sana_pipeline(
    *,
    depth: int = 2,
    hidden_size: int = 128,
    num_heads: int = 4,
    linear_head_dim: int = 32,
    f: int = 4,
    h: int = 8,
    w: int = 8,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SanaPipe:
    """Mini-config factory for unit tests: random-weight ``SanaMSVideo``.

    Avoids downloading the 2B checkpoint or initializing Gemma-2-2B. The
    resulting pipe has only ``dit`` populated; VAE / text encoder are ``None``
    and tests must not call ``preprocess_input`` / ``decode_video``.
    """
    # Touch the upstream factory to trigger ``MODELS.register_module`` side effects.
    _resolve_upstream_factory("SanaMSVideo_2000M_P2_D20")
    # Direct class construction — bypass the factory wrapper so we can override
    # hidden_size + depth from defaults of the 2B model.
    from diffusion.model.nets.sana_multi_scale_video import SanaMSVideo as SanaMSVideoCls  # noqa: E402

    dit = SanaMSVideoCls(
        input_size=h,
        patch_size=(1, 2, 2),
        in_channels=16,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=2.0,
        class_dropout_prob=0.0,
        learn_sigma=False,
        pred_sigma=False,
        attn_type="LiteLAReLURope",
        ffn_type="GLUMBConvTemp",
        use_pe=True,
        pos_embed_type="wan_rope",
        qk_norm=True,
        cross_norm=True,
        y_norm=True,
        linear_head_dim=linear_head_dim,
        t_kernel_size=3,
        mlp_acts=("silu", "silu", None),
        model_max_length=8,
        caption_channels=64,
    )
    dit = dit.to(device=device, dtype=dtype).eval()

    from openwam.model.video_backbone.sana.scheduler import SanaFlowSchedulerAdapter

    return SanaPipe(
        dit=dit,
        scheduler=SanaFlowSchedulerAdapter(),
        config={"hidden_size": hidden_size, "depth": depth, "num_heads": num_heads, "fhw": (f, h, w)},
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass
class _PipeSpec:
    """Parsed inputs to ``_build_pipe_from_spec``."""

    model_factory: str = "SanaMSVideo_2000M_P2_D20"
    model_path: Optional[str] = None
    """Path to ``SANA_Video_2B_480p.pth`` (or HF URL); ``None`` ⇒ random init."""
    vae_path: Optional[str] = None
    text_encoder_name: Optional[str] = None
    model_kwargs: dict = field(default_factory=dict)
    flow_shift: float = 3.0
    """Rectified-flow shift for the scheduler. Default ``3.0`` matches
    SANA-Video upstream (``model_wrapper.py:17``)."""


def _resolve_model_path_and_kwargs(
    model_path: Optional[str],
    model_kwargs: dict,
    vae_path: Optional[str],
) -> tuple:
    """Shared auto-discovery + foot-gun guard for ``_spec_from_*``.

    Resolves ``model_path`` against two acceptable shapes:

    1. Local bundle directory (HF snapshot layout: ``config.json`` +
       ``checkpoints/*.pth`` + ``vae/Wan2.1_VAE.pth``) → returns the
       expanded ckpt path and auto-applies the published ``model_kwargs``
       preset (unless yaml already supplied explicit kwargs).
    2. Local ``.pth`` / ``.safetensors`` file path WITH explicit
       ``model_kwargs`` — caller is on the hook for the architecture spec.

    Raises ``ValueError`` on any other shape (HF URL, repo id, missing
    path, etc.) when ``model_kwargs`` is empty — i.e. anything where we
    couldn't determine the architecture. Without this guard, the factory
    would fall back to its bare defaults (``attn_type=flash``,
    ``ffn_type=mlp``, ``in_channels=4``, ``pred_sigma=True``) and
    ``load_state_dict(strict=False)`` would drop most of the published
    ckpt weights silently, leaving the DiT mostly random.

    Returns ``(model_path, model_kwargs, vae_path)``.
    """
    if isinstance(model_path, str) and os.path.isdir(model_path):
        discovered = _spec_from_model_dir(model_path)
        # Auto-discovered ckpt overrides ``model_path: <dir>`` (the dir is just
        # the bundle marker — the actual .pth lives in ``<dir>/checkpoints/``).
        model_path = discovered.model_path
        if vae_path is None:
            vae_path = discovered.vae_path
        if not model_kwargs:
            model_kwargs = discovered.model_kwargs
    elif isinstance(model_path, str) and model_path and not model_kwargs:
        raise ValueError(
            "SANA video_backbone.model_path is set but not a local bundle "
            f"directory ({model_path!r}) and no model_kwargs were provided. "
            "Either:\n"
            "  (a) Download the SANA-Video bundle locally and point "
            "model_path at the unpacked directory — _spec_from_model_dir "
            "will auto-apply the published preset; see docs/sana_vendor.md.\n"
            "  (b) Set model_kwargs explicitly in the yaml to match the "
            "checkpoint architecture (in_channels / attn_type / ffn_type / "
            "pred_sigma / etc.).\n"
            "Falling back to factory defaults here would silently load a "
            "mismatched DiT (strict=False drops most weights) and train "
            "against mostly-random init."
        )

    return model_path, model_kwargs, vae_path


def _spec_from_dictconfig(vb_cfg) -> _PipeSpec:
    """Build spec from a Hydra ``video_backbone`` config.

    Yaml-level entry. Auto-discovery + foot-gun guard are shared with
    :func:`_spec_from_dict` (the plain-dict path that
    ``resolve_architecture_config`` produces) via
    :func:`_resolve_model_path_and_kwargs` — both training-time entry
    shapes go through the same gate.
    """
    model_path, model_kwargs, vae_path = _resolve_model_path_and_kwargs(
        model_path=vb_cfg.get("model_path"),
        model_kwargs=dict(vb_cfg.get("model_kwargs", {})),
        vae_path=vb_cfg.get("vae_path"),
    )

    return _PipeSpec(
        model_factory=str(vb_cfg.get("model_factory", "SanaMSVideo_2000M_P2_D20")),
        model_path=model_path,
        vae_path=vae_path,
        text_encoder_name=vb_cfg.get("text_encoder_name", "gemma-2-2b-it"),
        model_kwargs=model_kwargs,
        flow_shift=float(vb_cfg.get("flow_shift", 3.0)),
    )


# Factory model_kwargs that must be passed to ``SanaMSVideo_2000M_P2_D20`` to
# reconstruct the architecture matching the published SANA-Video 2B 480p
# checkpoint. The factory's bare defaults are ``attn_type="flash",
# ffn_type="mlp", in_channels=4, pred_sigma=True`` — those silently load only
# a fraction of the ckpt under ``strict=False`` (random FFN, dropped temporal
# convs, mis-sized pos_embed). Source: SANA upstream training config
# ``configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml``.
_SANA_VIDEO_2B_480P_PRESET: dict = {
    "in_channels": 16,
    "input_size": 60,
    "attn_type": "LiteLAReLURope",
    "linear_head_dim": 112,
    "ffn_type": "GLUMBConvTemp",
    "qk_norm": True,
    "pred_sigma": False,
    "learn_sigma": False,
    "caption_channels": 2304,
    "model_max_length": 300,
    "mlp_ratio": 3,
    "cross_norm": True,
    "use_pe": True,
    "pos_embed_type": "wan_rope",
}


def _spec_from_model_dir(path: str) -> _PipeSpec:
    """Auto-discover from a directory containing ``checkpoints/*.pth``."""
    # The HF SANA-Video repo layout is documented in docs/sana_vendor.md.
    ckpt_candidates = []
    ckpt_dir = os.path.join(path, "checkpoints")
    if os.path.isdir(ckpt_dir):
        ckpt_candidates = [
            os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pth")
        ]
    if not ckpt_candidates:
        raise FileNotFoundError(f"No checkpoints/*.pth under {path}")
    vae_path = os.path.join(path, "vae", "Wan2.1_VAE.pth")

    # Apply preset by HF bundle name. ``config.json`` is the HF-bundled marker.
    model_kwargs: dict = {}
    config_path = os.path.join(path, "config.json")
    if os.path.isfile(config_path):
        import json

        with open(config_path) as fh:
            bundle_cfg = json.load(fh)
        model_name = str(bundle_cfg.get("model_name", ""))
        if model_name.startswith("SANA-Video-2B-480"):
            model_kwargs = dict(_SANA_VIDEO_2B_480P_PRESET)

    return _PipeSpec(
        model_path=ckpt_candidates[0],
        vae_path=vae_path if os.path.isfile(vae_path) else None,
        text_encoder_name="gemma-2-2b-it",
        model_kwargs=model_kwargs,
    )


def _spec_from_dict(d: dict) -> _PipeSpec:
    """Build spec from a plain ``video_backbone`` dict.

    Plain-dict entry; this is the **training-time** path —
    ``resolve_architecture_config`` returns plain dicts (not DictConfig),
    so the OpenWAMTrainer / deploy loader land here. Auto-discovery +
    foot-gun guard are shared with :func:`_spec_from_dictconfig` via
    :func:`_resolve_model_path_and_kwargs`.
    """
    model_path, model_kwargs, vae_path = _resolve_model_path_and_kwargs(
        model_path=d.get("model_path"),
        model_kwargs=dict(d.get("model_kwargs", {})),
        vae_path=d.get("vae_path"),
    )

    return _PipeSpec(
        model_factory=d.get("model_factory", "SanaMSVideo_2000M_P2_D20"),
        model_path=model_path,
        vae_path=vae_path,
        text_encoder_name=d.get("text_encoder_name", "gemma-2-2b-it"),
        model_kwargs=model_kwargs,
        flow_shift=float(d.get("flow_shift", 3.0)),
    )


def _build_pipe_from_spec(
    spec: _PipeSpec,
    *,
    device: Optional[str],
    dtype: torch.dtype,
    ckpt_dir: Optional[str],
) -> SanaPipe:
    factory = _resolve_upstream_factory(spec.model_factory)
    dit = factory(**spec.model_kwargs)

    if spec.model_path is not None:
        # ``find_model`` is the upstream loader for ``.pth`` files; lazy import.
        from tools.download import find_model  # type: ignore[import-not-found]

        state = find_model(spec.model_path)
        if isinstance(state, dict) and "state_dict" in state and not any(
            k.startswith("blocks.") for k in state.keys()
        ):
            state = state["state_dict"]
        # SANA's load_state_dict tolerates shape mismatches by padding (see
        # sana_multi_scale_video.py:844-1016) — keep ``strict=False`` so we
        # don't trip on optional null-embed buffers.
        result = dit.load_state_dict(state, strict=False)
        # Surface partial loads: a clean published-ckpt load against the right
        # preset has 0 missing / 0 unexpected. Default factory kwargs vs the
        # 480p ckpt previously dropped GLUMBConvTemp + temporal-conv weights
        # silently and the smoke would just produce garbage.
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                "SANA ckpt load partial: %d missing, %d unexpected keys. "
                "Sample missing=%s sample unexpected=%s",
                len(result.missing_keys),
                len(result.unexpected_keys),
                result.missing_keys[:3],
                result.unexpected_keys[:3],
            )
        logger.info("Loaded SANA DiT weights from %s", spec.model_path)

    dit = dit.to(device=device or "cuda", dtype=dtype).eval()

    vae = _load_vae(spec.vae_path, device=device, dtype=dtype) if spec.vae_path else None
    text_encoder, tokenizer = (
        _load_text_encoder(spec.text_encoder_name, device=device, dtype=dtype)
        if spec.text_encoder_name
        else (None, None)
    )

    from openwam.model.video_backbone.sana.scheduler import SanaFlowSchedulerAdapter

    scheduler = SanaFlowSchedulerAdapter(flow_shift=spec.flow_shift)

    return SanaPipe(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        config={
            "factory": spec.model_factory,
            "model_path": spec.model_path,
            "vae_path": spec.vae_path,
            "flow_shift": spec.flow_shift,
        },
    )


def _load_vae(path: Optional[str], *, device, dtype):
    """Load Wan2.1 VAE from a local ``.pth`` file.

    SANA-Video 2B 480p reuses the Wan2.1 VAE; the ckpt is bundled in the HF
    asset (``<bundle>/vae/Wan2.1_VAE.pth``). We deliberately avoid
    ``model_pool`` / HuggingFace Hub here — CI / offline machines can't
    reach the Hub and the asset already has the file on disk.

    Returns ``None`` if path is missing/unset, so smoke tests that don't
    need pixel-space encode/decode can still construct the pipeline.
    """
    if not path or not os.path.isfile(path):
        logger.warning(
            "build_sana_pipeline: VAE not loaded (path=%s missing). decode_video "
            "and preprocess_input(frames=...) will be unavailable until a path "
            "to Wan2.1_VAE.pth is provided.",
            path,
        )
        return None

    from openwam.model.video_backbone.wan.vae import WanVideoVAE

    # weights_only=True: torch>=2.6 defaults to False; pin the safe path
    # explicitly so a malformed ckpt can't smuggle arbitrary objects.
    state = torch.load(path, map_location="cpu", weights_only=True)
    vae = WanVideoVAE(z_dim=16)
    converter = getattr(WanVideoVAE, "state_dict_converter", None)
    if converter is not None:
        state = converter().from_civitai(state)
    result = vae.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        logger.warning(
            "Wan2.1 VAE load partial: %d missing, %d unexpected. "
            "Sample missing=%s sample unexpected=%s",
            len(result.missing_keys),
            len(result.unexpected_keys),
            result.missing_keys[:3],
            result.unexpected_keys[:3],
        )
    vae = vae.to(device=device or "cuda", dtype=dtype).eval()
    vae.requires_grad_(False)
    logger.info("Loaded Wan2.1 VAE from %s", path)
    return vae


def _load_text_encoder(name: Optional[str], *, device, dtype):
    """Optional Gemma-2-2B-it loader for deploy. Smoke uses ``pre_encoded_text``.

    Gemma is gated on Hugging Face; CI usually doesn't have the token. We
    fail-soft (return ``(None, None)`` + WARNING) when the model isn't
    available locally so the pipeline still constructs. Callers that hand
    in ``pre_encoded_text`` via ``preprocess_input`` never touch this path
    — see ``SanaVideoBackbone.preprocess_input``.
    """
    if not name:
        return None, None
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.warning(
            "build_sana_pipeline: transformers not installed; text encoder "
            "loading skipped (name=%s).",
            name,
        )
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name, torch_dtype=dtype)
    except Exception as e:  # gated model, no token, offline, etc.
        logger.warning(
            "build_sana_pipeline: failed to load text encoder %r (%s: %s). "
            "Pass pre_encoded_text via preprocess_input or wire a local "
            "Gemma path before deploy.",
            name,
            type(e).__name__,
            e,
        )
        return None, None

    model = model.to(device=device or "cuda").eval()
    model.requires_grad_(False)
    logger.info("Loaded text encoder %s (dtype=%s)", name, dtype)
    return model, tokenizer
