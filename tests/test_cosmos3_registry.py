"""Registry + fail-fast config validation for the cosmos3_edge backbone.

Every bad-config case must raise from the pure-Python validation phase (before
any heavy import), so these tests run identically on CPU CI without diffusers.
"""

import pytest

from openwam.model.video_backbone import (
    _VIDEO_BACKBONE_REGISTRY,
    Cosmos3EdgeVideoBackbone,
    build_video_backbone,
    register_video_backbone,
)


def test_cosmos3_edge_registered():
    assert _VIDEO_BACKBONE_REGISTRY["cosmos3_edge"] is Cosmos3EdgeVideoBackbone


def test_double_registration_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_video_backbone("cosmos3_edge")(Cosmos3EdgeVideoBackbone)


def _cfg(**vb):
    return {"model": {"video_backbone": vb}}


def test_unknown_name_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="cosmos3_edge"):
        build_video_backbone("cosmos3_edge", _cfg(name="cosmos3_nano", model_path="/tmp/x"))


def test_missing_model_path_raises():
    with pytest.raises(ValueError, match="model_path"):
        build_video_backbone("cosmos3_edge", _cfg(name="cosmos3_edge"))


def test_bad_text_dropout_raises():
    with pytest.raises(ValueError, match="text_encoder_dropout"):
        build_video_backbone(
            "cosmos3_edge",
            _cfg(name="cosmos3_edge", model_path="/tmp/x", text_encoder_dropout=1.5),
        )


def test_bad_max_text_tokens_raises():
    with pytest.raises(ValueError, match="max_text_tokens"):
        build_video_backbone(
            "cosmos3_edge",
            _cfg(name="cosmos3_edge", model_path="/tmp/x", max_text_tokens=2),
        )


def test_freeze_und_false_rejected():
    with pytest.raises(NotImplementedError, match="freeze_und"):
        build_video_backbone(
            "cosmos3_edge",
            _cfg(name="cosmos3_edge", model_path="/tmp/x", freeze_und=False),
        )


def test_nonexistent_model_path_fails_fast(tmp_path):
    # With diffusers installed this is a FileNotFoundError on the missing
    # transformer/ subfolder; without diffusers it is the install-hint
    # ImportError. Both are acceptable fail-fast outcomes on CPU.
    with pytest.raises((FileNotFoundError, ImportError)):
        build_video_backbone(
            "cosmos3_edge",
            _cfg(name="cosmos3_edge", model_path=str(tmp_path / "nonexistent")),
        )
