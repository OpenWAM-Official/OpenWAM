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
from typing import Optional

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
    tokenizer_src = os.path.join(model_dir, "google", "umt5-xxl")
    if os.path.isdir(tokenizer_src):
        result["tokenizer"] = {
            "class": "openwam.model.video_backbone.wan.text_encoder.HuggingfaceTokenizer",
            "attr": "tokenizer",
            "subdir": "tokenizer/google/umt5-xxl",
            "path_kwarg": "name",
            "kwargs": {"seq_len": 512, "clean": "whitespace"},
        }
    return result


def copy_video_backbone_tokenizer(output_dir: str, model_path_or_cfg) -> None:
    """Copy the Wan tokenizer into the checkpoint directory when available."""
    if isinstance(model_path_or_cfg, str):
        model_path = model_path_or_cfg
    else:
        model_path: Optional[str] = None
        try:
            model_path = str(model_path_or_cfg.model.video_backbone.model_path)
        except Exception:
            pass

    if not model_path or not os.path.isdir(model_path):
        logger.info(
            "[component_specs] video_backbone.model_path not readable (%s); skipping tokenizer copy.",
            model_path,
        )
        return

    tokenizer_src = os.path.join(model_path, "google", "umt5-xxl")
    tokenizer_dst = os.path.join(output_dir, "tokenizer", "google", "umt5-xxl")
    if os.path.isdir(tokenizer_dst):
        logger.info("[component_specs] Tokenizer already present, skip: %s", tokenizer_dst)
    elif os.path.isdir(tokenizer_src):
        os.makedirs(os.path.dirname(tokenizer_dst), exist_ok=True)
        shutil.copytree(tokenizer_src, tokenizer_dst)
        logger.info("[component_specs] Copied tokenizer:\n  src: %s\n  dst: %s", tokenizer_src, tokenizer_dst)
    else:
        logger.warning(
            "[component_specs] Tokenizer source not found at %s; deploy needs tokenizer files under %s "
            "or a reachable model.video_backbone.model_path.",
            tokenizer_src,
            os.path.dirname(tokenizer_dst),
        )
