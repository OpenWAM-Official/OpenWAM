"""API-hygiene guard for the Cosmos25 backbone.

The user constraint for the port: ``Cosmos25VideoBackbone`` may expose publicly
*only* methods/properties that already exist on the ``VideoBackbone`` base
contract (``base.py``). Every Cosmos-specific helper must be private/auxiliary.
This test fails loudly if a new public method/property leaks onto the backbone.
"""

from __future__ import annotations

from openwam.model.video_backbone import _VIDEO_BACKBONE_REGISTRY
from openwam.model.video_backbone.base import VideoBackbone
from openwam.model.video_backbone.cosmos25_backbone import Cosmos25VideoBackbone


def _public_members(cls) -> set[str]:
    return {name for name in dir(cls) if not name.startswith("_")}


def test_no_public_surface_beyond_base_contract():
    extra = _public_members(Cosmos25VideoBackbone) - _public_members(VideoBackbone)
    assert extra == set(), (
        "Cosmos25VideoBackbone exposes public members absent from the VideoBackbone "
        f"base contract (must be private helpers): {sorted(extra)}"
    )


def test_subclasses_base_contract():
    assert issubclass(Cosmos25VideoBackbone, VideoBackbone)


def test_registered_under_expected_names():
    for name in ("cosmos25_predict_2b", "cosmos25_predict_14b"):
        assert _VIDEO_BACKBONE_REGISTRY[name] is Cosmos25VideoBackbone


def test_inference_and_deploy_are_wired_not_stubs():
    """Inference/deploy hooks are implemented — no longer NotImplementedError stubs.

    Detailed behavior lives in test_cosmos25_cfg_inference.py /
    test_cosmos25_deploy_artifacts.py; here we only assert the stubs are gone.
    """
    import pytest

    bb = Cosmos25VideoBackbone.__new__(Cosmos25VideoBackbone)
    bb.text_encoder = None  # no live source → inference gate raises

    # Inference now reaches the real no-prompt-source gate (ValueError), not the
    # old NotImplementedError stub.
    with pytest.raises(ValueError, match="no prompt source"):
        bb.preprocess_input_for_inference(prompt="x", cfg_scale=1.0)

    # save_deploy_assets is a safe no-op when model_path is unreadable (it must
    # not raise NotImplementedError anymore).
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"model": {"video_backbone": {"model_path": "/no/such/dir"}}})
    bb.save_deploy_assets("/tmp/does-not-matter", cfg=cfg)
    assert "components" not in cfg.model.video_backbone  # unreadable path → no mutation
