"""Cosmos-Predict2.5 video backbone registry smoke.

Confirms the new backbone names land in the global registry without
requiring the ``cosmos_predict2`` runtime dependency. Importing the
backbone package must stay side-effect-free on CPU-only hosts.
"""

from __future__ import annotations

import pytest

from openwam.model.video_backbone import (
    _VIDEO_BACKBONE_REGISTRY,
    CosmosPredict25VideoBackbone,
    build_video_backbone,
    register_video_backbone,
)


def test_cosmos_predict25_name_registered():
    assert _VIDEO_BACKBONE_REGISTRY["cosmos_predict25_2b"] is CosmosPredict25VideoBackbone


def test_cosmos_predict25_build_with_bad_path_raises_clear_error():
    """`build_video_backbone` should fail with a clear cosmos / path error rather
    than an AttributeError deep inside the wrapper.

    Accepts three failure modes:
    - ``ImportError`` when ``cosmos_predict2`` is not installed at all,
    - ``NotImplementedError`` when the upstream extra is partially wired,
    - ``FileNotFoundError`` when the package is installed but
      ``model_path`` points nowhere.
    """
    cfg = {
        "video_backbone": {
            "name": "cosmos_predict25_2b",
            "model_path": "/nonexistent",
            "text_encoder_path": "/nonexistent",
        }
    }
    with pytest.raises((ImportError, NotImplementedError, FileNotFoundError)) as exc_info:
        build_video_backbone("cosmos_predict25_2b", cfg)
    message = str(exc_info.value).lower()
    assert any(token in message for token in ("cosmos", "nonexistent", "variant"))


def test_cosmos_predict25_register_idempotency_guard():
    """The registry must reject double-registration of the same name."""
    with pytest.raises(ValueError, match="already registered"):
        register_video_backbone("cosmos_predict25_2b")(CosmosPredict25VideoBackbone)


def test_cosmos_predict25_invalid_sac_mode_rejected():
    """`sac_mode=foo` must fail-fast with a clear ValueError before any
    `cosmos_predict2` import so the error is testable on CPU CI."""
    cfg = {
        "video_backbone": {
            "name": "cosmos_predict25_2b",
            "model_path": "/nonexistent",
            "sac_mode": "totally_bogus",
        }
    }
    with pytest.raises(ValueError) as exc_info:
        build_video_backbone("cosmos_predict25_2b", cfg)
    message = str(exc_info.value)
    assert "sac_mode" in message
    assert "totally_bogus" in message
    for valid in ("none", "mm_only", "block_wise"):
        assert valid in message


def test_cosmos_predict25_invalid_vae_choice_rejected():
    """`vae=foo` must fail-fast with a clear ValueError before any
    `cosmos_predict2` import so the error is testable on CPU CI."""
    cfg = {
        "video_backbone": {
            "name": "cosmos_predict25_2b",
            "model_path": "/nonexistent",
            "vae": "garbage_vae",
        }
    }
    with pytest.raises(ValueError) as exc_info:
        build_video_backbone("cosmos_predict25_2b", cfg)
    message = str(exc_info.value)
    assert "vae" in message
    assert "garbage_vae" in message
    for valid in ("none", "wan2pt1"):
        assert valid in message


def test_cosmos_predict25_missing_text_encoder_path_rejected():
    """A training-path build without `text_encoder_path` must raise a clear
    ValueError before any heavy import — Reason1 live encoding is the only
    text pathway."""
    cfg = {
        "video_backbone": {
            "name": "cosmos_predict25_2b",
            "model_path": "/nonexistent",
            # `text_encoder_path` intentionally omitted.
        }
    }
    with pytest.raises(ValueError, match="text_encoder_path"):
        build_video_backbone("cosmos_predict25_2b", cfg)


def test_cosmos_predict25_text_encoder_dropout_out_of_range_rejected():
    """`text_encoder_dropout` outside [0, 1] must fail-fast at build time
    (CPU CI testable). Mirrors the backbone-side ValueError; catching it at
    the builder avoids any upstream import / 5 GB DiT load for a config typo."""
    cfg = {
        "video_backbone": {
            "name": "cosmos_predict25_2b",
            "model_path": "/nonexistent",
            "text_encoder_path": "/nonexistent",
            "text_encoder_dropout": 1.5,
        }
    }
    with pytest.raises(ValueError) as exc_info:
        build_video_backbone("cosmos_predict25_2b", cfg)
    message = str(exc_info.value)
    assert "text_encoder_dropout" in message
    assert "1.5" in message
