"""CPU smoke for Phase 4 VAE config plumbing — no ``cosmos_predict2`` needed.

Covers the path-resolution + config-validation seam between the user-facing
``video_backbone.vae`` / ``video_backbone.vae_path`` knobs and the actual
upstream tokenizer load. The real ``Wan2pt1VAEInterface`` construction is
gated by the lazy upstream import and lives behind ``_build_cosmos_predict25_vae``;
that path is exercised on GPU by ``tests/test_cosmos_predict25_real_load.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openwam.model.video_backbone.cosmos_predict25 import pipeline_builder
from openwam.model.video_backbone.cosmos_predict25.pipeline_builder import (
    _COSMOS25_VAE_FILENAME,
    _VAE_CHOICES,
    _resolve_vae_path,
)


def test_vae_choices_constant_includes_none_and_wan2pt1():
    assert _VAE_CHOICES == {"none", "wan2pt1"}


def test_resolve_vae_path_defaults_to_tokenizer_pth_under_model_path(tmp_path: Path):
    """``vae_path=None`` ⇒ ``<model_path>/tokenizer.pth``."""
    tok = tmp_path / _COSMOS25_VAE_FILENAME
    tok.write_bytes(b"fake-tokenizer")

    resolved = _resolve_vae_path(tmp_path, None)
    assert resolved == tok


def test_resolve_vae_path_honors_explicit_override(tmp_path: Path):
    """``vae_path=...`` is used verbatim even when ``<model_path>/tokenizer.pth`` exists."""
    elsewhere = tmp_path / "subdir" / "my_tokenizer.pth"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_bytes(b"fake")
    # Distractor: default location also exists but must NOT be picked.
    (tmp_path / _COSMOS25_VAE_FILENAME).write_bytes(b"wrong")

    resolved = _resolve_vae_path(tmp_path, str(elsewhere))
    assert resolved == elsewhere


def test_resolve_vae_path_missing_file_raises_clear_error(tmp_path: Path):
    """The error must mention all three escape hatches so users aren't stuck."""
    with pytest.raises(FileNotFoundError) as exc_info:
        _resolve_vae_path(tmp_path, None)
    message = str(exc_info.value)
    assert _COSMOS25_VAE_FILENAME in message
    # Each of the two exits must be discoverable from the error text.
    assert "vae_path" in message
    assert "vae: none" in message


def test_build_cosmos_predict25_vae_is_a_module_attribute():
    """Smoke: ``_build_cosmos_predict25_vae`` is importable without triggering the
    upstream ``cosmos_predict2`` import (the import happens inside the
    function body, not at module-load time)."""
    assert callable(pipeline_builder._build_cosmos_predict25_vae)
