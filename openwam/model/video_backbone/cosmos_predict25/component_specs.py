"""Component specs for Cosmos-Predict2.5 deploy.

Unlike Wan, CosmosPredict25 does **not** copy any tokenizer / processor directory
next to the checkpoint *for the DiT path*. The DiT (``MinimalV1LVGDiT``, ~3.9
GB) and VAE (``Wan2pt1VAEInterface``'s inner ``WanVAE_``, ~485 MB) are
registered as ``nn.Module`` children of :class:`CosmosPredict25VideoBackbone`, so
their params flow through the architecture's unified ``state_dict`` and are
saved into the same safetensors as every other dual_system / single_system
weight.

The Cosmos-Reason1-7B text encoder (~16 GB Qwen2.5-VL) **also** flows through
the unified safetensors via the ``reason1`` registration on the
wrapper (see ``pipeline_wrapper.py`` for the trick mirroring ``vae``).
The only artifacts copied alongside the checkpoint are the small JSON/
tokenizer files Qwen2.5-VL needs to bootstrap its structure at deploy time
(``config.json``, ``tokenizer.json``, etc., totalling ~10 MB) — see
:func:`copy_cosmos_predict25_artifacts`. They live under ``<ckpt_dir>/reason1/`` and
are read by :meth:`Reason1LiveTextEncoder.from_empty` to construct an empty
meta-device shell that :meth:`BaseWAMArchitecture.load_checkpoint` then
populates from the unified safetensors.

What this module does:

- :func:`generate_cosmos_predict25_component_specs` returns a non-None marker dict
  whenever the cosmos_predict25 path is in use. ``BaseWAMArchitecture.save_config``
  forwards it onto ``model.video_backbone.components`` in the saved YAML,
  which is the gate (``deploy/model_loader.py:117-122``) that makes the
  deploy loader thread ``_ckpt_dir`` into the adapter's ``from_pretrained``.
  ``build_cosmos_predict25_pipeline`` then uses ``ckpt_dir is not None`` as the
  signal to build empty DiT + VAE + Reason1 shells (skipping the
  ``model_path`` eager-load) and lets ``arch.load_checkpoint`` populate
  weights from the unified safetensors.

- :func:`copy_cosmos_predict25_artifacts` copies the Reason1 tokenizer/config JSON
  files into ``<output_dir>/reason1/``. Dispatched via
  :meth:`CosmosPredict25VideoBackbone.copy_deploy_artifacts` from the
  :class:`BaseWAMArchitecture` dispatcher.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Reason1 (Qwen2.5-VL) structural files that the meta-device shell needs.
# Weights themselves come from the unified safetensors; these are tokenizer,
# chat template, model config — collectively ~10 MB.
_REASON1_ARTIFACT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.json",
    "preprocessor_config.json",
)


def _normalize_optional_path(path: Optional[str]) -> Optional[str]:
    if path in (None, "", "None", "null"):
        return None
    return path


def _resolve_text_encoder_path(model_path_or_cfg: Any) -> Optional[str]:
    if isinstance(model_path_or_cfg, str):
        te_path: Optional[str] = model_path_or_cfg
    else:
        te_path = None
        try:
            te_path = str(model_path_or_cfg.model.video_backbone.text_encoder_path)
        except Exception:
            pass
    return _normalize_optional_path(te_path)


def _resolve_model_path(model_path_or_cfg: Any) -> Optional[str]:
    if isinstance(model_path_or_cfg, str):
        # A raw string for this module is interpreted as a Reason1 source path,
        # not the Cosmos DiT model_path.
        return None
    try:
        model_path = str(model_path_or_cfg.model.video_backbone.model_path)
    except Exception:
        return None
    return _normalize_optional_path(model_path)


def validate_reason1_artifact_source(model_path_or_cfg: Any) -> str:
    """Return a readable Reason1 artifact source or raise a clear error."""
    te_path = _resolve_text_encoder_path(model_path_or_cfg)
    if te_path and os.path.isdir(te_path):
        return te_path
    if _resolve_model_path(model_path_or_cfg):
        raise RuntimeError(
            "[component_specs] video_backbone.text_encoder_path not readable (%s); "
            "CosmosPredict25 checkpoints must copy Reason1 structural artifacts into "
            "<ckpt_dir>/reason1/ so the safetensors-saved encoder can be "
            "rebuilt at deploy time. Set model.video_backbone.text_encoder_path "
            "to a readable Cosmos-Reason1-7B bundle." % te_path
        )
    raise RuntimeError(
        "[component_specs] Reason1 artifact source not readable (%s). Pass a "
        "Cosmos-Reason1-7B directory, or a config with "
        "model.video_backbone.text_encoder_path set." % te_path
    )


def generate_cosmos_predict25_component_specs(model_path: str) -> Optional[dict]:
    """Return component specs for the deploy loader's ``_ckpt_dir`` threading.

    The cosmos_predict25 ``state_dict`` already carries DiT + VAE + Reason1 weights
    (registered as ``nn.Module`` children of :class:`CosmosPredict25VideoBackbone`
    via ``net``, ``vae``, and ``reason1`` respectively).

    Returns ``None`` only when ``model_path`` is missing or unreadable, so
    fake-pipeline tests and offline build paths bypass the deploy gate.

    Arguments:
        model_path: training-time path to the Cosmos asset bundle. Currently
            only used as a "is this a real cosmos_predict25 training run?" sanity
            check; the spec content does not capture any path-dependent state.
    """
    if not model_path or not os.path.isdir(model_path):
        return None
    # Each entry documents that the named attribute on the wrapper holds an
    # ``nn.Module`` whose state lives inside the unified safetensors under
    # the ``sub_module`` key.
    return {
        "components": [
            {"attr": "vae", "source": "state_dict", "sub_module": "vae"},
            {"attr": "text_encoder", "source": "state_dict", "sub_module": "reason1"},
        ]
    }


def copy_cosmos_predict25_artifacts(output_dir: str, model_path_or_cfg: Any) -> None:
    """Copy the small Reason1 tokenizer/config JSON files into ``<output_dir>/reason1/``.

    Reason1 model weights ride into the unified safetensors via
    ``CosmosPredict25VideoBackbone.reason1``; only the small structural
    files (~10 MB total) need to live next to the checkpoint so
    :meth:`Reason1LiveTextEncoder.from_empty` can rebuild a meta-device shell
    on a deploy host that doesn't have the original Cosmos-Reason1 bundle.

    Mirrors the contract of
    :func:`openwam.model.video_backbone.wan.component_specs.copy_video_backbone_tokenizer`.

    Arguments:
        output_dir: checkpoint output directory (where ``config.yaml`` lives).
        model_path_or_cfg: either the raw ``text_encoder_path`` string, or a
            DictConfig from which we read
            ``model.video_backbone.text_encoder_path``.
    """
    te_path = validate_reason1_artifact_source(model_path_or_cfg)

    dst_dir = os.path.join(output_dir, "reason1")
    if os.path.isdir(dst_dir) and any(os.path.isfile(os.path.join(dst_dir, name)) for name in _REASON1_ARTIFACT_FILES):
        logger.info("[component_specs] Reason1 artifacts already present, skip: %s", dst_dir)
        return

    os.makedirs(dst_dir, exist_ok=True)
    copied: list[str] = []
    for name in _REASON1_ARTIFACT_FILES:
        src = os.path.join(te_path, name)
        if not os.path.isfile(src):
            continue
        shutil.copy2(src, os.path.join(dst_dir, name))
        copied.append(name)
    if not copied:
        logger.warning(
            "[component_specs] No Reason1 structural files found under %s; "
            "expected at least config.json + tokenizer.json. Deploy may fail "
            "to construct the meta-device shell.",
            te_path,
        )
    else:
        logger.info(
            "[component_specs] Copied Reason1 artifacts (%d files):\n  src: %s\n  dst: %s",
            len(copied),
            te_path,
            dst_dir,
        )


__all__ = [
    "generate_cosmos_predict25_component_specs",
    "copy_cosmos_predict25_artifacts",
    "validate_reason1_artifact_source",
    "_resolve_text_encoder_path",
]
