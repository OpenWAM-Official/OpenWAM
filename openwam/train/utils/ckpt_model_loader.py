"""Self-contained architecture construction for finetune / resume.

When ``training.finetune_ckpt_path`` or ``training.resume_ckpt_path`` is set,
the trainer builds the architecture from that checkpoint source alone:
module skeletons from the component specs saved in its ``config.yaml``,
weights from either the requested ``checkpoint_step_*.safetensors`` file or
the latest weights in the directory (finetune), or from accelerate's
``load_state`` after prepare (resume). The original pretrained backbone
directory (``model.video_backbone.model_path``) does not need to exist on the
training host.

Train-side counterpart of the deploy loader (``openwam/deploy/model_loader.py``),
kept independent so training never imports deploy code.
"""

import logging
import os
import shutil

from omegaconf import DictConfig, OmegaConf, open_dict

from openwam.train.utils.checkpointing import find_latest_weights

logger = logging.getLogger(__name__)


def _resolve_ckpt_source(source: str) -> tuple[str, str | None]:
    """Return ``(ckpt_dir, explicit_weights)`` for a directory or safetensors file."""
    source = os.fspath(source)
    if source.endswith(".safetensors"):
        # Fail fast on a typo'd filename — otherwise safetensors only errors
        # after minutes of architecture-skeleton construction.
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Explicit checkpoint weights file does not exist: {source}")
        return os.path.dirname(os.path.abspath(source)), os.path.abspath(source)
    return source, None


def build_architecture_from_ckpt_dir(ckpt_dir: str, *, weights_required: bool):
    """Build the architecture purely from a self-contained checkpoint source.

    Skeletons come from ``<ckpt_dir>/config.yaml``'s
    ``model.video_backbone.components`` specs (tokenizer resolved against
    ``<ckpt_dir>/tokenizer/``). With ``weights_required=True`` (finetune) either
    the explicit safetensors source or latest ``checkpoint_step_*.safetensors``
    in the source dir is loaded here; with
    ``weights_required=False`` (resume) safetensors are skipped entirely —
    accelerate's ``load_state`` restores a strict superset (module weights
    incl. frozen params) after prepare, and reading the latest safetensors
    would crash on a file truncated by a mid-save kill even though the run
    is still recoverable from its accel state.

    Returns ``(resolved_arch, architecture, ckpt_cfg)``.
    """
    from openwam.model import build_architecture, resolve_architecture_config

    ckpt_dir, explicit_weights = _resolve_ckpt_source(ckpt_dir)
    tag = "finetune" if weights_required else "resume"
    ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    resolved_arch = resolve_architecture_config(ckpt_cfg.model)
    params = dict(resolved_arch.params)

    # Pin video_backbone params to a plain dict: OmegaConf would auto-promote
    # the ``_source`` assignment below into a DictConfig, and build_holder()
    # dispatches DictConfig sources to the training pipeline (which needs
    # cfg.training plus the pretrained model_path dir) instead of the
    # component-spec path.
    vb_params = params.get("video_backbone") or {}
    if not isinstance(vb_params, dict):
        vb_params = OmegaConf.to_container(vb_params, resolve=True) or {}
    params["video_backbone"] = vb_params
    vb_params["_source"] = OmegaConf.to_container(ckpt_cfg.model.video_backbone, resolve=True)
    vb_params["_ckpt_dir"] = ckpt_dir

    logger.info("[%s] building architecture from self-contained ckpt dir: %s", tag, ckpt_dir)
    architecture = build_architecture(resolved_arch.registry_name, params)

    if weights_required:
        weights = explicit_weights or find_latest_weights(ckpt_dir)
        logger.info("[%s] loading pretrained weights: %s", tag, weights)
        # Plain print so the warm-start is visible on the launch terminal even
        # when logger output is drowned out; non-main ranks have print disabled.
        print(f"[{tag}] loading pretrained weights: {weights}", flush=True)
        architecture.load_checkpoint(weights)
        print(f"[{tag}] pretrained weights loaded OK", flush=True)
    return resolved_arch, architecture, ckpt_cfg


def propagate_component_specs(ckpt_cfg: DictConfig, cfg: DictConfig) -> None:
    """Carry the ckpt's reconstruction specs into the live run cfg.

    save_video_backbone_deploy_assets() no-ops when model_path is unreadable,
    so without this the new run's config.yaml would lose the specs and its
    checkpoints would no longer be self-contained.
    """
    for key in ("components", "tokenizer"):
        val = OmegaConf.select(ckpt_cfg, f"model.video_backbone.{key}", default=None)
        if val is not None and OmegaConf.select(cfg, f"model.video_backbone.{key}", default=None) is None:
            with open_dict(cfg):
                OmegaConf.update(cfg, f"model.video_backbone.{key}", val)
            logger.info("[self-contained] propagated model.video_backbone.%s specs from ckpt config", key)


# Tokenizer artifact dirs written by save_deploy_assets, per backbone family:
# Wan uses ``tokenizer/``, cosmos3_edge uses ``text_tokenizer/``. Relaying only
# the first would produce a checkpoint that carries the ``components`` marker
# (so it claims self-containment) but no tokenizer for the deploy build to read.
_CKPT_ARTIFACT_DIRS = ("tokenizer", "text_tokenizer")


def copy_ckpt_artifacts(ckpt_dir: str, output_dir: str) -> None:
    """Relay the checkpoint's tokenizer artifact dirs into the new run dir.

    Same reason as propagate_component_specs: keeps the self-containment chain
    alive when the tokenizer's original model_path source is unreachable. Only
    the dirs that exist are copied, so this no-ops per backbone as appropriate.
    """
    ckpt_dir, _explicit_weights = _resolve_ckpt_source(ckpt_dir)
    for name in _CKPT_ARTIFACT_DIRS:
        src = os.path.join(ckpt_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.isdir(src) and not os.path.isdir(dst):
            shutil.copytree(src, dst)
            logger.info("[self-contained] relayed tokenizer dir: %s -> %s", src, dst)
