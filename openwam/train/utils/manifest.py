"""Pipeline-manifest generation for self-contained checkpoints.

Writes a ``video_backbone_manifest.json`` next to the checkpoint plus a local
``tokenizer/`` copy, so deployment on any machine only needs the checkpoint
directory itself (no external Wan / video-backbone source required).

Reverse-lookup strategy: hash every *.safetensors / *.pth file under
``cfg.model.video_backbone.model_path`` and match against diffsynth's
``MODEL_CONFIGS`` registry (same mechanism ``build_training_pipeline`` uses,
but we record the matched entry instead of loading weights).

Decoupled from any specific backbone via the manifest schema (see
``build_video_backbone_from_manifest``).
"""

from __future__ import annotations

import glob as _glob
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


# model_name (from diffsynth MODEL_CONFIGS) → pipe attribute name on WanVideoPipeline.
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
    """Return the same list of file paths ``build_training_pipeline`` would feed to
    ``WanVideoPipeline.from_pretrained``.

    Sharded safetensors (``name-00001-of-00003.safetensors`` …) are grouped into
    lists; standalone ``*.safetensors`` and ``*.pth`` are left as bare strings.
    Ordering matches the training path so hashes line up.
    """
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
    for f in standalone:
        paths.append(f)
    for f in pth_files:
        paths.append(f)
    return paths


def generate_video_backbone_manifest(model_dir: str) -> dict:
    """Build a video-backbone manifest dict by hashing files under *model_dir*.

    For each file / shard-group, ``hash_model_file`` computes an MD5 over
    ``(key, shape)`` pairs in the state_dict; matching entries in
    ``MODEL_CONFIGS`` carry the ``model_class`` + ``extra_kwargs`` we need to
    re-instantiate the architecture at deploy time.

    Args:
        model_dir: Path to the backbone source directory (the same value
            ``cfg.model.video_backbone.model_path`` points to at train time).

    Returns:
        A manifest dict ready to be JSON-dumped.

    Notes:
        - Only Wan-style models are currently mapped; unmapped matches are
          logged and skipped (so e.g. an unused image_encoder file doesn't
          break manifest writing).
        - Multiple matches of the same ``model_name`` (e.g. ``dit`` + ``dit2``
          for dual-DiT Wan variants) are handled by emitting ``<attr>`` and
          ``<attr>2``.
    """
    from openwam.model.video_backbone.diffsynth.configs import MODEL_CONFIGS
    from openwam.model.video_backbone.diffsynth.core.loader.file import hash_model_file

    paths = _list_backbone_model_paths(model_dir)

    attr_counts: dict[str, int] = defaultdict(int)
    models: list[dict] = []

    for path in paths:
        h = hash_model_file(path)
        matches = [c for c in MODEL_CONFIGS if c["model_hash"] == h]
        if not matches:
            logger.info("[manifest] No MODEL_CONFIGS match for %s (hash=%s); skipping", path, h)
            continue
        for config in matches:
            model_name = config["model_name"]
            attr_base = _WAN_MODEL_NAME_TO_ATTR.get(model_name)
            if attr_base is None:
                logger.info(
                    "[manifest] model_name=%s has no attr mapping; skipping (path=%s)",
                    model_name,
                    path,
                )
                continue
            # Handle dual-DiT / dual-VACE (dit2 / vace2)
            attr_counts[attr_base] += 1
            if attr_counts[attr_base] == 1:
                attr = attr_base
            else:
                attr = f"{attr_base}{attr_counts[attr_base]}"
            models.append(
                {
                    "attr": attr,
                    "model_class": config["model_class"],
                    "extra_kwargs": config.get("extra_kwargs", {}) or {},
                }
            )

    tokenizer_block: Optional[dict] = None
    tokenizer_src = os.path.join(model_dir, "google", "umt5-xxl")
    if os.path.isdir(tokenizer_src):
        tokenizer_block = {
            "class": "openwam.model.video_backbone.diffsynth.models.wan_video_text_encoder.HuggingfaceTokenizer",
            "attr": "tokenizer",
            "subdir": "tokenizer/google/umt5-xxl",
            "path_kwarg": "name",
            "kwargs": {"seq_len": 512, "clean": "whitespace"},
        }

    manifest = {
        "pipeline": {
            "class": "openwam.model.video_backbone.diffsynth.pipelines.wan_video.WanVideoPipeline",
            "kwargs": {},
        },
        "vae_division_factor_scale": 2,
        "models": models,
    }
    if tokenizer_block is not None:
        manifest["tokenizer"] = tokenizer_block
    return manifest


def save_video_backbone_artifacts(output_dir: str, cfg) -> None:
    """Write video_backbone_manifest.json + copy tokenizer/ into *output_dir*.

    Both artifacts are written once (skipped if already present) so repeated
    calls are cheap. Deployment on a new machine then only needs *output_dir*
    — no dependency on ``cfg.model.video_backbone.model_path``.

    Args:
        output_dir: Checkpoint directory (same target as ``save_config``).
        cfg: Hydra DictConfig; reads ``cfg.model.video_backbone.model_path``.

    Silent no-op if the model_path is missing (e.g. debug / mock runs).
    """
    model_path = None
    try:
        model_path = str(cfg.model.video_backbone.model_path)
    except Exception:
        pass
    if not model_path or not os.path.isdir(model_path):
        logger.info(
            "[manifest] video_backbone.model_path not readable (%s); skipping manifest + tokenizer save.",
            model_path,
        )
        return

    manifest_path = os.path.join(output_dir, "video_backbone_manifest.json")
    tokenizer_dst = os.path.join(output_dir, "tokenizer", "google", "umt5-xxl")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Manifest (write if missing)
    if os.path.exists(manifest_path):
        logger.info("[manifest] Already present, skip: %s", manifest_path)
    else:
        manifest = generate_video_backbone_manifest(model_path)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        attrs = [m["attr"] for m in manifest["models"]]
        logger.info(
            "[manifest] Wrote %s with %d model(s): %s",
            manifest_path,
            len(manifest["models"]),
            attrs,
        )

    # 2. Tokenizer (copy if missing)
    tokenizer_src = os.path.join(model_path, "google", "umt5-xxl")
    if os.path.isdir(tokenizer_dst):
        logger.info("[manifest] Tokenizer already present, skip: %s", tokenizer_dst)
    elif os.path.isdir(tokenizer_src):
        os.makedirs(os.path.dirname(tokenizer_dst), exist_ok=True)
        shutil.copytree(tokenizer_src, tokenizer_dst)
        logger.info("[manifest] Copied tokenizer:\n  src: %s\n  dst: %s", tokenizer_src, tokenizer_dst)
    else:
        logger.warning(
            "[manifest] Tokenizer source not found at %s — deploy will fail to load tokenizer "
            "unless manifest.tokenizer is adjusted or a tokenizer dir is placed under %s",
            tokenizer_src,
            os.path.dirname(tokenizer_dst),
        )
