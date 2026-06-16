"""Component-spec generation for config-driven Wan checkpoint loading.

Training persists Wan component specs into ``config.yaml`` so deployment can
instantiate the video backbone architecture before checkpoint weights are
loaded. The specs are derived by hashing the source Wan model files and
matching them against OpenWAM's Wan ``MODEL_CONFIGS`` registry.
"""

from __future__ import annotations

import glob as _glob
import logging
import os
import re
import shutil
from collections import defaultdict

logger = logging.getLogger(__name__)


# model_name (from Wan MODEL_CONFIGS) -> pipe attribute name on WanVideoPipeline.
# Keep in sync with WanVideoPipeline.from_pretrained's model_pool.fetch_model calls.
_WAN_MODEL_NAME_TO_ATTR = {
    "wan_video_text_encoder": "text_encoder",
    "wan_video_dit": "dit",
    "wan_video_vae": "vae",
    "wan_video_image_encoder": "image_encoder",
    "wan_video_motion_controller": "motion_controller",
    "wan_video_vace": "vace",
    "wan_video_vap": "vap",
    "wans2v_audio_encoder": "audio_encoder",
    "wan_video_animate_adapter": "animate_adapter",
}

# Single source of truth for the Wan tokenizer layout. The same paths drive both
# the saved spec (``subdir``, read by the deploy loader) and the file copy, so
# there is no duplicated literal between spec-generation and artifact-copy.
#   source:  <model_dir>/<_WAN_TOKENIZER_REL>
#   deploy:  <ckpt_dir>/<_WAN_TOKENIZER_SUBDIR>   (the ``tokenizer/`` prefix is the
#            checkpoint-local namespace the loader strips back off as a fallback)
_WAN_TOKENIZER_REL = os.path.join("google", "umt5-xxl")
_WAN_TOKENIZER_SUBDIR = os.path.join("tokenizer", _WAN_TOKENIZER_REL)


def _list_backbone_model_paths(model_dir: str) -> list:
    """Return the model-file list used by Wan training pipeline discovery."""
    safetensors = sorted(_glob.glob(os.path.join(model_dir, "*.safetensors")))
    pth_files = sorted(_glob.glob(os.path.join(model_dir, "*.pth")))

    shard_groups: dict[str, list[str]] = defaultdict(list)
    standalone: list[str] = []
    for f in safetensors:
        basename = os.path.basename(f)
        m = re.match(r"^(.+)-\d{5}-of-\d{5}\.safetensors$", basename)
        if m:
            shard_groups[m.group(1)].append(f)
        else:
            standalone.append(f)

    paths: list = []
    for prefix in sorted(shard_groups):
        paths.append(sorted(shard_groups[prefix]))
    paths.extend(standalone)
    paths.extend(pth_files)
    return paths


def generate_video_backbone_component_specs(model_dir: str) -> dict:
    """Build config-ready component specs from a Wan backbone source directory."""
    from openwam.model.video_backbone.wan.shared.configs import MODEL_CONFIGS
    from openwam.model.video_backbone.wan.shared.core.loader.file import hash_model_file

    attr_counts: dict[str, int] = defaultdict(int)
    components: list[dict] = []

    for path in _list_backbone_model_paths(model_dir):
        h = hash_model_file(path)
        matches = [c for c in MODEL_CONFIGS if c["model_hash"] == h]
        if not matches:
            logger.info("[component_specs] No MODEL_CONFIGS match for %s (hash=%s); skipping", path, h)
            continue
        for config in matches:
            model_name = config["model_name"]
            attr_base = _WAN_MODEL_NAME_TO_ATTR.get(model_name)
            if attr_base is None:
                logger.info(
                    "[component_specs] model_name=%s has no attr mapping; skipping (path=%s)",
                    model_name,
                    path,
                )
                continue

            attr_counts[attr_base] += 1
            attr = attr_base if attr_counts[attr_base] == 1 else f"{attr_base}{attr_counts[attr_base]}"
            components.append(
                {
                    "attr": attr,
                    "model_class": config["model_class"],
                    "extra_kwargs": config.get("extra_kwargs", {}) or {},
                }
            )

    result = {"components": components}
    if os.path.isdir(os.path.join(model_dir, _WAN_TOKENIZER_REL)):
        result["tokenizer"] = {
            "class": "openwam.model.video_backbone.wan.models.text_encoder.HuggingfaceTokenizer",
            "attr": "tokenizer",
            "subdir": _WAN_TOKENIZER_SUBDIR,
            "path_kwarg": "name",
            "kwargs": {"seq_len": 512, "clean": "whitespace"},
        }
    return result


def _copy_tokenizer(model_path: str, output_dir: str) -> None:
    """Copy the Wan tokenizer dir into the self-contained checkpoint.

    Source and destination both come from the ``_WAN_TOKENIZER_*`` constants, so
    the layout is defined once and shared with the saved spec's ``subdir``.
    """
    src = os.path.join(model_path, _WAN_TOKENIZER_REL)
    dst = os.path.join(output_dir, _WAN_TOKENIZER_SUBDIR)
    if os.path.isdir(dst):
        logger.info("[component_specs] Tokenizer already present, skip: %s", dst)
    elif os.path.isdir(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst)
        logger.info("[component_specs] Copied tokenizer:\n  src: %s\n  dst: %s", src, dst)
    else:
        logger.warning(
            "[component_specs] Tokenizer source not found at %s; deploy needs tokenizer files under %s "
            "or a reachable model.video_backbone.model_path.",
            src,
            os.path.dirname(dst),
        )


def save_video_backbone_deploy_assets(output_dir: str, cfg) -> None:
    """Make the Wan video backbone's slice of the checkpoint self-contained.

    Single entry that does both halves of the self-containment, sharing one
    tokenizer-layout source so nothing is duplicated:

    1. Merge the component + tokenizer reconstruction specs into
       ``cfg.model.video_backbone`` (only when absent — never clobber an
       explicit config), so deploy rebuilds the module skeletons from
       ``config.yaml`` without the training-time ``model_path``.
    2. Copy the tokenizer files into ``output_dir``.

    No-op when ``model.video_backbone.model_path`` is unset or unreadable (e.g.
    a checkpoint already carrying ``components`` / ``tokenizer`` in its config).
    """
    from omegaconf import OmegaConf, open_dict

    model_path = OmegaConf.select(cfg, "model.video_backbone.model_path", default=None)
    if not model_path or not os.path.isdir(str(model_path)):
        logger.info(
            "[component_specs] video_backbone.model_path not readable (%s); skipping deploy-asset save.",
            model_path,
        )
        return
    model_path = str(model_path)

    specs = generate_video_backbone_component_specs(model_path)

    with open_dict(cfg):
        vb_cfg = cfg.model.video_backbone
        if "components" not in vb_cfg:
            OmegaConf.update(cfg, "model.video_backbone.components", specs["components"])
        if "tokenizer" in specs and "tokenizer" not in vb_cfg:
            OmegaConf.update(cfg, "model.video_backbone.tokenizer", specs["tokenizer"])

    if "tokenizer" in specs:
        _copy_tokenizer(model_path, output_dir)
