"""Unit tests for ``TextEmbeddingCacheTransform``.

Exercises the sha256-keyed read path, the CFG dropout substitution
against ``empty.safetensors``, and the error-on-missing-cache behavior.
No real Reason1 weights involved — caches are hand-written safetensors
files with arbitrary tensor content.
"""

from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file

from openwam.dataloader.transforms.text_embedding_cache import (
    TextEmbeddingCacheTransform,
    bucketed_cache_path_for_sha,
    sha256_for_prompt,
)


def _write_cache_file(cache_dir: str, prompt: str, tensor: torch.Tensor, *, bucketed: bool = False) -> str:
    if prompt == "":
        path = os.path.join(cache_dir, "empty.safetensors")
    elif bucketed:
        path = bucketed_cache_path_for_sha(cache_dir, sha256_for_prompt(prompt))
    else:
        path = os.path.join(cache_dir, f"{sha256_for_prompt(prompt)}.safetensors")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({"pre_encoded_text": tensor.contiguous()}, path, metadata={"prompt": prompt})
    return path


@pytest.fixture
def populated_cache(tmp_path):
    cache_dir = str(tmp_path / "text_cache")
    os.makedirs(cache_dir)
    real = torch.randn(8, 1024)
    empty = torch.full((8, 1024), -1.0)  # distinguishable from real
    _write_cache_file(cache_dir, "pick up the block", real)
    _write_cache_file(cache_dir, "", empty)
    return cache_dir, real, empty


def test_sha256_stability():
    # UTF-8 hash; a fixed prompt → a fixed sha. Hard-coded value pins the
    # hash for cross-version stability (collisions across precompute runs).
    assert (
        sha256_for_prompt("pick up the block")
        == "98fffccbfa71a725c80a9f6370854ba0345582b4e1c6b0459dc306ea63b93f3b"
    )


def test_load_real_prompt(populated_cache):
    cache_dir, real, _ = populated_cache
    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=0.0)
    out = tx.apply({"prompt": "pick up the block"})
    assert "pre_encoded_text" in out
    assert torch.equal(out["pre_encoded_text"], real)


def test_load_bucketed_prompt(tmp_path):
    cache_dir = str(tmp_path / "text_cache")
    os.makedirs(cache_dir)
    real = torch.randn(8, 1024)
    empty = torch.full((8, 1024), -1.0)
    bucketed_path = _write_cache_file(cache_dir, "pick up the block", real, bucketed=True)
    _write_cache_file(cache_dir, "", empty)

    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=0.0)
    out = tx.apply({"prompt": "pick up the block"})

    assert bucketed_path.endswith(
        "98/98fffccbfa71a725c80a9f6370854ba0345582b4e1c6b0459dc306ea63b93f3b.safetensors"
    )
    assert torch.equal(out["pre_encoded_text"], real)


def test_dropout_zero_never_uses_empty(populated_cache):
    cache_dir, real, _ = populated_cache
    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=0.0, rng_seed=0)
    for _ in range(50):
        out = tx.apply({"prompt": "pick up the block"})
        assert torch.equal(out["pre_encoded_text"], real)


def test_dropout_one_always_uses_empty(populated_cache):
    cache_dir, _, empty = populated_cache
    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=1.0, rng_seed=0)
    for _ in range(50):
        out = tx.apply({"prompt": "pick up the block"})
        assert torch.equal(out["pre_encoded_text"], empty)


def test_dropout_half_split(populated_cache):
    cache_dir, real, empty = populated_cache
    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=0.5, rng_seed=42)
    real_count = 0
    n = 1000
    for _ in range(n):
        out = tx.apply({"prompt": "pick up the block"})
        if torch.equal(out["pre_encoded_text"], real):
            real_count += 1
    # Binomial sample within tolerance: 95% CI ≈ ±3.1%.
    assert 0.42 * n < real_count < 0.58 * n, f"got {real_count}/{n}"


def test_dropout_zero_with_empty_prompt_still_uses_empty(populated_cache):
    """Even with dropout disabled, an empty prompt string maps to empty.safetensors."""
    cache_dir, _, empty = populated_cache
    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=0.0)
    out = tx.apply({"prompt": ""})
    assert torch.equal(out["pre_encoded_text"], empty)


def test_eval_mode_disables_dropout(populated_cache):
    cache_dir, real, empty = populated_cache
    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=1.0, rng_seed=0)
    tx.eval()
    for _ in range(20):
        out = tx.apply({"prompt": "pick up the block"})
        # eval disables training-mode dropout, so we get the real embedding.
        assert torch.equal(out["pre_encoded_text"], real)
    # Empty string still maps to empty embedding (not dropout — semantics).
    out_empty = tx.apply({"prompt": ""})
    assert torch.equal(out_empty["pre_encoded_text"], empty)


def test_missing_cache_file_raises_with_precompute_hint(populated_cache):
    cache_dir, _, _ = populated_cache
    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=0.0)
    with pytest.raises(FileNotFoundError) as exc_info:
        tx.apply({"prompt": "an unseen caption that was never cached"})
    assert "reason1_embedding_computation" in str(exc_info.value)
    assert "an unseen caption that was never cached" in str(exc_info.value)


def test_missing_cache_dir_at_construct_raises(tmp_path):
    bad = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="not a directory"):
        TextEmbeddingCacheTransform(cache_dir=bad)


def test_dropout_requires_empty_safetensors(tmp_path):
    cache_dir = str(tmp_path / "no_empty")
    os.makedirs(cache_dir)
    _write_cache_file(cache_dir, "x", torch.randn(4, 1024))
    # dropout_p>0 + missing empty.safetensors → fail fast.
    with pytest.raises(FileNotFoundError, match="empty.safetensors"):
        TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=0.1)
    # dropout_p=0 + no empty.safetensors is fine (only used when prompt == "").
    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=0.0)
    out = tx.apply({"prompt": "x"})
    assert "pre_encoded_text" in out


def test_dropout_p_out_of_range_rejected(populated_cache):
    cache_dir, _, _ = populated_cache
    with pytest.raises(ValueError, match=r"dropout_p"):
        TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=-0.1)
    with pytest.raises(ValueError, match=r"dropout_p"):
        TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=1.5)


def test_missing_prompt_field_raises(populated_cache):
    cache_dir, _, _ = populated_cache
    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=0.0)
    with pytest.raises(KeyError, match="prompt"):
        tx.apply({"not_a_prompt": "x"})


def test_wrong_safetensors_key_raises(populated_cache):
    cache_dir, _, _ = populated_cache
    # Write a cache file with a wrong key — should raise on read, not silently
    # accept whatever key happens to be in the file.
    path = os.path.join(cache_dir, f"{sha256_for_prompt('weird')}.safetensors")
    save_file({"wrong_key": torch.zeros(4, 1024)}, path)
    tx = TextEmbeddingCacheTransform(cache_dir=cache_dir, dropout_p=0.0)
    with pytest.raises(KeyError, match="pre_encoded_text"):
        tx.apply({"prompt": "weird"})
