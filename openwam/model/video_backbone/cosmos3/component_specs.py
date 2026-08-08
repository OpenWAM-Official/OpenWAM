"""Deploy self-containment for the Cosmos3-Edge backbone.

Much simpler than predict2.5's: every weight (``dit.*`` / ``vae.*``) rides the
unified checkpoint safetensors, so the component spec is a marker only, and the
sole side artifacts are the tokenizer's structural JSON files (copied into
``<ckpt_dir>/text_tokenizer/`` — the deploy builder reads the tokenizer from
there). No ``text_encoder`` component is ever emitted, so the deploy loader's
Reason1-clearing branch never triggers for this family.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_COSMOS3_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def generate_cosmos3_component_specs(model_path, *, has_vae: bool) -> Optional[dict]:
    """Component specs for the checkpoint config, or ``None`` when ``model_path``
    is not a readable bundle (the save hook must then no-op, not raise)."""
    if model_path is None:
        return None
    root = Path(str(model_path))
    if not root.is_dir() or not (root / "text_tokenizer").is_dir():
        return None
    components = []
    if has_vae:
        components.append({"attr": "vae", "source": "state_dict", "sub_module": "vae"})
    return {"components": components}


def copy_cosmos3_artifacts(output_dir: str, model_path: str) -> None:
    """Copy the tokenizer structural files into ``<output_dir>/text_tokenizer/``.

    Per-file repair semantics: only missing files are copied, so repeated saves
    are idempotent AND a partial copy left by an interrupted save is completed
    on the next one (an any()-style "already populated" guard would freeze the
    partial state forever)."""
    src = Path(model_path) / "text_tokenizer"
    dst = Path(output_dir) / "text_tokenizer"
    if not src.is_dir():
        logger.warning("cosmos3_edge: tokenizer dir missing at %s; checkpoint will not be self-contained.", src)
        return
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    available = 0
    for name in _COSMOS3_TOKENIZER_FILES:
        if (src / name).is_file():
            available += 1
            if not (dst / name).exists():
                shutil.copy2(src / name, dst / name)
                copied += 1
    if available == 0:
        logger.warning("cosmos3_edge: no tokenizer files found under %s.", src)
