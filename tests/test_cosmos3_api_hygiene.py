"""API hygiene for the cosmos3_edge backbone (predict2.5 parity rules).

The backbone class may expose zero public members beyond the VideoBackbone ABC;
deploy hooks must be wired (real validation errors, not NotImplementedError);
save_deploy_assets must no-op without mutating cfg when model_path is unreadable.
"""

import pytest
from omegaconf import OmegaConf

from openwam.model.video_backbone import Cosmos3EdgeVideoBackbone, VideoBackbone


def _public_members(cls):
    return {n for n in dir(cls) if not n.startswith("_")}


def test_no_public_surface_beyond_base_contract():
    extra = _public_members(Cosmos3EdgeVideoBackbone) - _public_members(VideoBackbone)
    assert extra == set(), f"cosmos3_edge leaks public members beyond the ABC: {sorted(extra)}"


def test_is_video_backbone_subclass():
    assert issubclass(Cosmos3EdgeVideoBackbone, VideoBackbone)


def _bare_instance():
    return Cosmos3EdgeVideoBackbone.__new__(Cosmos3EdgeVideoBackbone)


def test_inference_hook_is_wired_not_stub():
    vb = _bare_instance()
    try:
        vb.preprocess_input_for_inference()
    except ValueError:
        pass  # the real gate: missing `prompt`
    except NotImplementedError as exc:  # pragma: no cover
        raise AssertionError("preprocess_input_for_inference is an unwired stub") from exc


def test_decode_hook_is_wired_not_stub():
    vb = _bare_instance()
    try:
        vb.decode_video(None)  # type: ignore[arg-type]
    except ValueError:
        pass  # the real gate: no VAE attached
    except NotImplementedError as exc:  # pragma: no cover
        raise AssertionError("decode_video is an unwired stub") from exc


def _validating_instance():
    vb = _bare_instance()
    object.__setattr__(vb, "_temporal_compression", 4)
    return vb


def test_deploy_cfg_gt1_rejected_loudly():
    vb = _validating_instance()
    with pytest.raises(NotImplementedError, match="cfg_scale"):
        vb.preprocess_input_for_inference(prompt="x", first_frame_image=object(), cfg_scale=2.0)


def test_deploy_num_frames_must_be_4k_plus_1():
    vb = _validating_instance()
    with pytest.raises(ValueError, match="4k\\+1"):
        vb.preprocess_input_for_inference(prompt="x", first_frame_image=object(), num_frames=32)


def test_deploy_multi_image_list_rejected():
    vb = _validating_instance()
    with pytest.raises(ValueError, match="exactly one"):
        vb.preprocess_input_for_inference(prompt="x", first_frame_image=[object(), object()])


def test_save_deploy_assets_noop_on_unreadable_model_path(tmp_path):
    vb = _bare_instance()
    cfg = OmegaConf.create({"model": {"video_backbone": {"model_path": str(tmp_path / "missing")}}})
    vb.save_deploy_assets(str(tmp_path), cfg)
    assert OmegaConf.select(cfg, "model.video_backbone.components") is None
    assert not (tmp_path / "text_tokenizer").exists()
