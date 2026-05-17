"""Wan video backbone registry smoke tests.

Verifies that the three Wan backbones plan §1 promises are wired up
without touching GPU or real weights.
"""

from __future__ import annotations

import pytest


def test_registry_keys_present():
    from openwam.model.video_backbone import _VIDEO_BACKBONE_REGISTRY

    assert {"wan22_ti2v_5b", "wan21_vace_1_3b", "wan21_i2v_14b_480p"} <= set(_VIDEO_BACKBONE_REGISTRY)


def test_unknown_key_raises():
    from openwam.model.video_backbone import build_video_backbone

    with pytest.raises(KeyError, match="Unknown video backbone"):
        build_video_backbone("nonexistent_backbone", {})
