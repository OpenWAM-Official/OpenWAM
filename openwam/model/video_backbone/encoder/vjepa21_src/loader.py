"""V-JEPA 2.1 ViT + manifest loading for :class:`..vjepa21.VJEPA21VideoEncoder`.

The vendored-ViT-coupled, weight-loading concerns: manifest read/validate, the
vendored-ViT import + RoPE dtype monkey-patch, the zero-weight ViT construction,
and the pretrained-weight load. They live here as module-level functions (not
encoder methods) so ``encoder.py`` stays focused on the latent contract +
``batch_encode`` path. ``VJEPA21VideoEncoder.from_pretrained`` / ``from_skeleton``
call into these.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Geometry constants the encoder's reshape paths and spec are hard-wired
# against. The manifest can carry different ``patch`` / ``tubelet`` values
# only if a future PR also generalizes the (h = H // 16) / (Tp // 2)
# reshape and the ``spec`` block (spatial_compression=16 from ViT patch=16,
# plus a post-tubelet avg-pool stride=2 to reach temporal_compression=4).
# Today the encoder is locked to ViT-g/16 tubelet=2 — manifests that
# disagree get a fail-fast at load time instead of a confusing reshape
# error later.
_REQUIRED_MANIFEST_PATCH = 16
_REQUIRED_MANIFEST_TUBELET = 2


def read_and_validate_manifest(model_path: str) -> dict:
    manifest_path = os.path.join(model_path, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"VJEPA21 encoder requires manifest.json in {model_path}.")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    patch = int(manifest["patch"])
    tubelet = int(manifest["tubelet"])
    if patch != _REQUIRED_MANIFEST_PATCH or tubelet != _REQUIRED_MANIFEST_TUBELET:
        raise ValueError(
            f"VJEPA21 manifest patch/tubelet must be "
            f"({_REQUIRED_MANIFEST_PATCH}, {_REQUIRED_MANIFEST_TUBELET}); "
            f"got ({patch}, {tubelet}). The encoder's spec (spatial_compression=16, "
            f"temporal_compression=4 = ViT tubelet=2 × encoder pool stride=2) and "
            f"reshape logic (H//16, W//16, Tp//2) are hard-wired against these values. "
            "Use a different manifest or extend the encoder to honor the manifest geometry."
        )
    check_arch_use_rope_consistency(manifest)
    return manifest


def check_arch_use_rope_consistency(manifest: dict) -> None:
    """Manifest-internal contradiction check, isolated from the ViT import.

    Runs without importing the vendored ViT so the error stays correct in
    environments without the ViT's deps (e.g. ``timm``) installed. Called by
    ``read_and_validate_manifest`` (the ``from_pretrained`` / ``from_skeleton``
    path) and by ``load_vit`` (PR #83 V9/V10 regression tests), so all paths get
    the same fail-fast.
    """
    arch_name = manifest["arch_name"]
    manifest_use_rope = manifest.get("use_rope", True)
    if arch_name.endswith("_rope") and not manifest_use_rope:
        raise ValueError(
            f"Manifest arch_name={arch_name!r} hardcodes use_rope=True "
            "but the manifest sets use_rope=False. Pick a non-_rope "
            "arch (e.g. 'vit_giant_xformers') or set use_rope=True."
        )


def prepare_vjepa_imports_and_patch():
    """Import the vendored ViT modules + install the RoPE dtype monkey-patch.
    Idempotent. Returns the ``vision_transformer`` module.

    The ViT lives in-tree alongside this loader (MIT-licensed, adapted from
    facebookresearch/vjepa2 at commit
    ``ce64921e94f0ffdc330c00fc62618157894b74be``) — no ``third_party``
    submodule or ``sys.path`` bootstrap is needed. See
    the upstream MIT license (Copyright Meta Platforms, Inc. and affiliates).

    The RoPE dtype monkey-patch root-cause-fixes a V-JEPA / SDPA dtype mismatch
    under mixed precision: upstream ``rotate_queries_or_keys`` builds its sin/cos
    table from a fp32 mask (``1.0 * frame_ids``) and einsums it against an fp32
    ``omega``, so the rotated Q/K leave the function in fp32 even when ``x`` is
    bf16. The host backbone keeps V in bf16, and PyTorch SDPA refuses
    ``query.dtype != value.dtype``. The patch casts the output back to
    ``x.dtype`` on exit — covers all six call sites in ``AttentionRoPE.forward``
    (qd/kd, qh/kh, qw/kw) without editing the vendored source. Idempotent via the
    ``_openwam_dtype_safe`` sentinel so repeated calls (training reload, deploy
    skeleton + later weight load, EMA replicas) do not re-wrap.
    """
    from openwam.model.video_backbone.encoder.vjepa21_src import modules as vjepa_modules
    from openwam.model.video_backbone.encoder.vjepa21_src import vision_transformer as vit_encoder

    if not getattr(vjepa_modules.rotate_queries_or_keys, "_openwam_dtype_safe", False):
        _orig_rotate = vjepa_modules.rotate_queries_or_keys

        # Forward through any signature change in upstream
        # ``rotate_queries_or_keys`` (V-JEPA 2.1 added ``n_registers`` /
        # ``has_cls_first`` over V-JEPA 2; future kwargs would propagate
        # the same way). The cast-back-to-input-dtype only needs the
        # input tensor reference, so we read it from positional args
        # (or the ``x=`` kwarg as a fallback).
        def _safe_rotate(*args, **kwargs):
            out = _orig_rotate(*args, **kwargs)
            ref = args[0] if args else kwargs.get("x", None)
            if isinstance(ref, torch.Tensor) and isinstance(out, torch.Tensor):
                return out.to(ref.dtype)
            return out

        _safe_rotate._openwam_dtype_safe = True
        vjepa_modules.rotate_queries_or_keys = _safe_rotate

    return vit_encoder


def build_vit_from_manifest(vit_encoder, manifest: dict) -> nn.Module:
    """Construct a zero-weight ViT per the manifest. No weight load.

    Upstream wrappers ending in ``_rope`` (e.g.
    ``vit_giant_xformers_rope``) hardcode ``use_rope=True`` in their
    ``VisionTransformer(...)`` call and forward ``**kwargs`` to the
    same constructor — passing ``use_rope`` again from here raises
    ``TypeError: got multiple values for keyword argument 'use_rope'``.
    For non-``_rope`` arches the wrapper does not set it, so we
    forward the manifest value; we default to ``True`` (opt-out)
    because every V-JEPA 2.1 manifest we ship uses RoPE —
    ``VisionTransformer``'s own ``use_rope=False`` default is the
    wrong choice for this encoder.
    """
    arch_name = manifest["arch_name"]  # e.g. "vit_giant_xformers"
    manifest_use_rope = manifest.get("use_rope", True)
    vit_kwargs: dict[str, Any] = dict(
        patch_size=manifest["patch"],
        img_size=(manifest["img_size"], manifest["img_size"]),
        num_frames=manifest["training_num_frames"],
        tubelet_size=manifest["tubelet"],
        use_sdpa=True,
        img_temporal_dim_size=manifest.get("img_temporal_dim_size", 1),
        interpolate_rope=manifest.get("interpolate_rope", True),
    )
    if not arch_name.endswith("_rope"):
        vit_kwargs["use_rope"] = manifest_use_rope
    return vit_encoder.__dict__[arch_name](**vit_kwargs)


def load_vit_weights(vit: nn.Module, model_path: str, manifest: dict) -> None:
    """Populate a constructed ViT with pretrained weights from disk."""
    ckpt = torch.load(
        os.path.join(model_path, manifest["checkpoint_file"]),
        map_location="cpu",
    )
    state_dict = ckpt[manifest.get("checkpoint_key", "target_encoder")]
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    # ``strict=False`` is intentional but narrow: the checkpoint ships a
    # learned ``pos_embed`` for the absolute-pos-embedding variants, and
    # we always load the RoPE variants whose forward does not consume it
    # (and so the buffer/parameter does not exist on the constructed
    # ``vit`` either). Anything else missing or unexpected is a
    # manifest / checkpoint mismatch that would silently leave the frozen
    # ViT partially randomly initialized — fail fast instead. The
    # tolerated unexpected set is exactly ``{"pos_embed"}``; missing keys
    # must always be empty.
    load_result = vit.load_state_dict(state_dict, strict=False)
    unexpected = set(load_result.unexpected_keys) - {"pos_embed"}
    if unexpected or load_result.missing_keys:
        raise RuntimeError(
            "VJEPA21 checkpoint load left the ViT inconsistent with the "
            "constructed module. This usually means the manifest "
            "``arch_name`` does not match the checkpoint, or the "
            "``checkpoint_key`` extracts the wrong sub-dict. Details: "
            f"missing_keys={sorted(load_result.missing_keys)[:8]} "
            f"unexpected_keys={sorted(unexpected)[:8]}."
        )


def load_vit(model_path: str, manifest: dict) -> nn.Module:
    """Validate arch/rope, import+patch, build a zero-weight shell, load weights.

    ``check_arch_use_rope_consistency`` runs before module construction so a
    manifest mismatch raises a clear ``ValueError`` without requiring model
    weights or an external checkout.
    """
    check_arch_use_rope_consistency(manifest)
    vit_encoder = prepare_vjepa_imports_and_patch()
    vit = build_vit_from_manifest(vit_encoder, manifest)
    load_vit_weights(vit, model_path, manifest)
    return vit


# ----------------------------------------------------------------------
# Deploy / cfg helpers (lowered out of the VJEPA21VideoEncoder subclass so it
# implements only the VideoEncoder contract).
# ----------------------------------------------------------------------


def resolve_manifest_dir(ckpt_dir: str | None) -> str:
    """Return ``ckpt_dir`` when it holds a readable ``manifest.json``.

    Deploy is strictly self-contained: the manifest must live next to the
    checkpoint (written by ``save_deploy_assets``); there is no
    ``encoder.model_path`` fallback, and a missing manifest is a hard error.
    ``os.path.isfile`` (not ``exists``) rejects a directory named
    ``manifest.json`` so the failure is named here, not as a later ``json.load``.
    """
    ckpt_manifest = os.path.join(ckpt_dir, "manifest.json") if ckpt_dir else None
    if ckpt_manifest and os.path.isfile(ckpt_manifest):
        return str(ckpt_dir)
    raise FileNotFoundError(
        "VJEPA21VideoEncoder.from_skeleton: no readable manifest.json at "
        f"ckpt_dir={ckpt_manifest!r}. Re-save the checkpoint with the current "
        "code, which writes manifest.json into ckpt_dir."
    )


def cfg_has_vjepa21_forward(encoder_cfg: Any) -> bool:
    """Whether the saved encoder yaml carries the ``vjepa21_forward`` key at all
    (yaml-``null`` counts as present). ``from_skeleton`` uses this — vs.
    :func:`read_vjepa21_forward_from_cfg`, which collapses absent/null to the
    default — to fire the pre-PR-checkpoint migration warning only when the
    operator truly omitted the key.
    """
    if encoder_cfg is None:
        return False
    if isinstance(encoder_cfg, dict):
        return "vjepa21_forward" in encoder_cfg
    _MISSING = object()
    return getattr(encoder_cfg, "vjepa21_forward", _MISSING) is not _MISSING


def read_vjepa21_forward_from_cfg(encoder_cfg: Any, default: str) -> str:
    """Pick ``vjepa21_forward`` from the saved yaml; absent / yaml-null collapse
    to ``default``. The caller's ``__init__`` validates the returned value.
    """
    if encoder_cfg is None:
        return default
    if isinstance(encoder_cfg, dict):
        value = encoder_cfg.get("vjepa21_forward")
    else:
        value = getattr(encoder_cfg, "vjepa21_forward", None)
    return default if value is None else str(value)
