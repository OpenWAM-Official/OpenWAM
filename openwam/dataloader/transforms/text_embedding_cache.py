"""Load pre-computed text embeddings from disk and attach to dataset samples.

The Cosmos25 backbone consumes a ``pre_encoded_text`` tensor on each sample
(``(L, D)`` bf16; ``D=1024`` post-projection for Cosmos-Predict2.5 2B). Live
encoding with Cosmos-Reason1 7B during training is expensive; instead we
pre-compute once per unique caption (``python -m openwam.dataloader.reason1_
embedding_computation ...``) and read the cached ``.safetensors`` here.

Cache layout (``<cache_dir>/``):

    manifest.json           # metadata (reason1 ckpt sha, cosmos ckpt sha, dtype, count)
    empty.safetensors       # cached embedding of the empty string (CFG dropout target)
    <sha256-of-prompt>.safetensors  # one per unique caption, key="pre_encoded_text"

This transform:

* Looks up the sample's ``prompt`` string, takes the SHA-256 of its UTF-8
  bytes, loads ``<cache_dir>/<sha>.safetensors``.
* During training, with probability ``dropout_p`` replaces the load target
  with ``empty.safetensors`` (classifier-free guidance dropout).
* Attaches the result as ``sample["pre_encoded_text"]`` so the architecture
  (``BaseWAMArchitecture.prepare_inputs``) can thread it to the backbone.

A missing cache file raises ``FileNotFoundError`` with the precompute
command, since the alternative — silently dropping the field — would
defeat the whole point of opting into the cache.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

from openwam.dataloader.transforms.base import ModalityTransform

_PRECOMPUTE_HINT = (
    "Run the offline precompute first:\n"
    "  python -m openwam.dataloader.reason1_embedding_computation \\\n"
    "      --reason1-ckpt /path/to/Cosmos-Reason1-7B \\\n"
    "      --cosmos-ckpt  /path/to/Cosmos-Predict2.5-2B/base/post-trained/<uuid>_ema_bf16.pt \\\n"
    "      --dataset-config configs/dataloader/robotwin.yaml \\\n"
    "      --output-dir   <cache_dir>"
)


def sha256_for_prompt(prompt: str) -> str:
    """Stable content-addressed key for a caption.

    Always operates on UTF-8 bytes so the hash is platform-independent.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class TextEmbeddingCacheTransform(ModalityTransform):
    """Load pre-computed Reason1 text embeddings from a sha256-keyed cache.

    Args:
        cache_dir: Directory containing ``<sha>.safetensors`` files plus
            ``empty.safetensors``. Must exist at construction time.
        dropout_p: Probability of replacing the real prompt embedding with
            the empty-prompt embedding during training (CFG dropout).
            Ignored when ``training=False``. Must be in ``[0, 1]``.
        rng_seed: Optional seed for the per-instance RNG (deterministic
            dropout across workers). Default ``None`` = each instance uses
            its own ``random.Random()`` seeded from system entropy.
    """

    def __init__(
        self,
        cache_dir: str,
        *,
        dropout_p: float = 0.0,
        rng_seed: Optional[int] = None,
    ):
        super().__init__(apply_to=["prompt"], training=True)
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(
                f"text_embedding_cache_dir={cache_dir!r} is not a directory.\n{_PRECOMPUTE_HINT}"
            )
        if not 0.0 <= dropout_p <= 1.0:
            raise ValueError(f"dropout_p must be in [0, 1]; got {dropout_p}")

        self.cache_dir = cache_dir
        self.dropout_p = float(dropout_p)
        self._rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

        # Pre-resolve the empty-prompt path (always read, never optional).
        self._empty_path = os.path.join(cache_dir, "empty.safetensors")
        if dropout_p > 0.0 and not os.path.exists(self._empty_path):
            raise FileNotFoundError(
                f"dropout_p={dropout_p} but {self._empty_path} is missing.\n"
                f"The precompute script writes empty.safetensors automatically.\n{_PRECOMPUTE_HINT}"
            )

    def apply(self, data: dict) -> dict:
        prompt = data.get("prompt")
        if prompt is None:
            raise KeyError("sample['prompt'] is required for TextEmbeddingCacheTransform")

        use_empty = self.training and self.dropout_p > 0.0 and self._rng.random() < self.dropout_p
        if use_empty or prompt == "":
            path = self._empty_path
        else:
            path = os.path.join(self.cache_dir, f"{sha256_for_prompt(prompt)}.safetensors")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing cached text embedding: {path}\n"
                f"prompt={prompt!r}\n{_PRECOMPUTE_HINT}"
            )

        # Lazy import: keep CPU CI's import surface small. safetensors is a
        # cheap dep (already pulled in by transformers / accelerate), but
        # this transform should not error at import time even if it's
        # vendored away by mistake.
        from safetensors.torch import load_file

        loaded = load_file(path)
        if "pre_encoded_text" not in loaded:
            raise KeyError(
                f"Expected key 'pre_encoded_text' in {path}; got keys {sorted(loaded)}.\n"
                f"The precompute script writes a single tensor under that key."
            )
        data["pre_encoded_text"] = loaded["pre_encoded_text"]
        return data
