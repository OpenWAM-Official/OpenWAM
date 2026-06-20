"""Host-encoder wiring for the optional frozen S-VAE feature reducer.

A frozen per-token S-VAE (:mod:`openwam.model.video_backbone.encoder.svae.model`)
that compresses an encoder's raw per-token features to a smaller ``z_dim``. Any
encoder MAY opt in by holding ``self._svae = reducer.build(...)`` and routing its
``batch_encode`` / ``from_skeleton`` / ``save_deploy_assets`` through these
functions; currently only :class:`VJEPA21VideoEncoder` does. With ``svae=None``
every function is a no-op / passthrough, so a non-opting encoder's
``batch_encode`` / ``spec`` / ``state_dict`` are bit-unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from torch import Tensor

from openwam.model.video_backbone.encoder.svae.model import (
    _CHECKPOINT_FORMAT_VERSION,
    SVAE,
    build_svae,
    load_svae,
)

logger = logging.getLogger(__name__)


def build(
    svae_path: str | None,
    svae_target_dim: int | None,
    svae_config: dict | None,
) -> SVAE | None:
    """Build the optional frozen S-VAE reducer from one of three sources.

    * ``svae_path``   — training: load a standalone-trained checkpoint.
    * ``svae_config`` — deploy skeleton: rebuild a zero-weight shell from the
      sidecar config dict; the architecture's strict ``load_checkpoint`` fills
      the weights immediately after construction.
    * neither — disabled (raw passthrough; ``z_dim`` stays ``embed_dim``).

    Always returned frozen and in eval mode; :func:`reduce` calls
    :meth:`SVAE.encode_mean` (deterministic) so a recursive ``host.train()``
    cannot flip it into a stochastic path.
    """
    if svae_path is not None and svae_config is not None:
        raise ValueError("Pass only one of svae_path / svae_config, not both.")
    if svae_path is not None:
        svae = load_svae(svae_path)
    elif svae_config is not None:
        svae = build_svae(dict(svae_config))
    else:
        return None
    if svae_target_dim is not None and int(svae_target_dim) != svae.latent_dim:
        raise ValueError(
            f"svae_target_dim ({svae_target_dim}) does not match the S-VAE latent_dim ({svae.latent_dim})."
        )
    svae.eval()
    svae.requires_grad_(False)
    return svae


def effective_z_dim(svae: SVAE | None, raw_dim: int) -> int:
    """The encoder's advertised ``z_dim``: the S-VAE ``latent_dim`` when a
    reducer is attached, else ``raw_dim``."""
    return svae.latent_dim if svae is not None else int(raw_dim)


def reduce(svae: SVAE | None, z: Tensor) -> Tensor:
    """Reduce raw post-pool features with the frozen S-VAE (deterministic
    posterior mean), or pass them through unchanged when none is attached."""
    if svae is None:
        return z
    return svae.encode_mean(z)


def write_sidecar(svae: SVAE, output_dir: str, owner_name: str) -> None:
    """Write the attached S-VAE's structural config to
    ``<output_dir>/svae_config.json`` so deploy can rebuild a same-shape shell.
    Raises on IO failure — the opt-in encoder's ``save_deploy_assets`` must
    abort rather than warn-and-skip.

    Versioned with the same ``_CHECKPOINT_FORMAT_VERSION`` as the standalone
    ``svae.pt`` so a stale sidecar is rejected with a clear message on read.
    """
    dst = os.path.join(output_dir, "svae_config.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(dst, "w") as f:
        json.dump({"format_version": _CHECKPOINT_FORMAT_VERSION, "model_config": svae.config_dict()}, f)
    logger.info("%s: wrote S-VAE sidecar %s", owner_name, dst)


def read_sidecar(ckpt_dir: str | None) -> dict | None:
    """Read ``<ckpt_dir>/svae_config.json`` (written by :func:`write_sidecar`).

    Returns the structural ``model_config`` dict when the checkpoint carried an
    S-VAE reducer, else ``None``. No ``encoder.model_path`` fallback: the
    sidecar is checkpoint-local and self-contained by construction. Validates
    the sidecar ``format_version`` so a legacy unversioned / mismatched sidecar
    fails fast here rather than deeper in ``build_svae``.
    """
    if not ckpt_dir:
        return None
    path = os.path.join(ckpt_dir, "svae_config.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    fmt = payload.get("format_version") if isinstance(payload, dict) else None
    if fmt != _CHECKPOINT_FORMAT_VERSION or "model_config" not in payload:
        raise ValueError(
            f"{path!r} has unsupported S-VAE sidecar format_version={fmt!r} "
            f"(this build writes/reads version {_CHECKPOINT_FORMAT_VERSION}). "
            f"Re-export the deploy checkpoint with the current build."
        )
    return payload["model_config"]


def read_target_dim_from_cfg(encoder_cfg: Any) -> int | None:
    """Pick ``svae_target_dim`` from the saved encoder yaml if present — used
    only as a cross-check against the sidecar's ``latent_dim``. Absent / null
    collapses to ``None`` (no cross-check).
    """
    if encoder_cfg is None:
        return None
    if isinstance(encoder_cfg, dict):
        value = encoder_cfg.get("svae_target_dim")
    else:
        value = getattr(encoder_cfg, "svae_target_dim", None)
    return int(value) if value is not None else None
