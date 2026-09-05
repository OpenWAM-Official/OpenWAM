"""Unit tests for ObsPreprocessor — obs payload validation + preprocessing.

Engine-free / GPU-free: ObsPreprocessor is constructed from an explicit view config,
so every obs-contract branch (single/multi view, missing cameras, bad base64,
proprio passthrough) is covered without a checkpoint or a live server.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from openwam.dataloader.transforms.multiview import (
    DEFAULT_MULTIVIEW_CAMERA_LAYOUT,
    format_prompt_for_inference,
)
from openwam.deploy.obs_preprocess import ObsPreprocessor, ObsValidationError


def _jpeg_b64(h: int = 48, w: int = 64, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _single_view(*, requires_proprio=False) -> ObsPreprocessor:
    return ObsPreprocessor(
        multiview=False,
        camera_layout=["head_camera"],
        img_height=32,
        img_width=32,
        requires_proprio=requires_proprio,
    )


def _multi_view(*, requires_proprio=False) -> ObsPreprocessor:
    return ObsPreprocessor(
        multiview=True,
        camera_layout=list(DEFAULT_MULTIVIEW_CAMERA_LAYOUT),
        img_height=32,
        img_width=32,
        requires_proprio=requires_proprio,
    )


def test_single_view_head_only_resizes_and_wraps_prompt():
    obs = _single_view().preprocess({"images": {"head_camera": _jpeg_b64()}, "prompt": "pick up the bottle"})
    assert obs["image"].size == (32, 32)
    assert isinstance(obs["prompt"], str) and obs["prompt"]  # wrapped, non-empty


def test_single_view_ignores_wrist_inputs():
    obs = _single_view().preprocess(
        {"images": {"head_camera": _jpeg_b64(), "left_wrist_camera": _jpeg_b64(seed=1)}, "prompt": "x"}
    )
    assert obs["image"].size == (32, 32)


def test_multiview_black_fills_missing_wrists():
    obs = _multi_view().preprocess(
        {"images": {"head_camera": _jpeg_b64(), "left_wrist_camera": None, "right_wrist_camera": None}, "prompt": "x"}
    )
    assert obs["image"].size == (32, 32)


def test_missing_images_dict_raises():
    with pytest.raises(ObsValidationError, match="images"):
        _single_view().preprocess({"prompt": "x"})


def test_missing_head_camera_raises():
    with pytest.raises(ObsValidationError, match="head_camera"):
        _single_view().preprocess({"images": {"head_camera": None}, "prompt": "x"})


def test_bad_base64_raises():
    with pytest.raises(ObsValidationError, match="head_camera"):
        _single_view().preprocess({"images": {"head_camera": "!!!not-a-jpeg!!!"}, "prompt": "x"})


def test_multiview_requires_three_camera_layout():
    dec = ObsPreprocessor(multiview=True, camera_layout=["only_one"], img_height=32, img_width=32)
    with pytest.raises(ObsValidationError, match="camera_layout"):
        dec.preprocess({"images": {"head_camera": _jpeg_b64()}, "prompt": "x"})


def test_state_any_dim_passes_through():
    # State-dim validation was removed: a state of any width is accepted and flattened.
    obs = _single_view().preprocess({"images": {"head_camera": _jpeg_b64()}, "prompt": "x", "state": list(range(14))})
    assert isinstance(obs["state"], np.ndarray) and obs["state"].shape == (14,)


def test_state_passthrough():
    obs = _single_view().preprocess({"images": {"head_camera": _jpeg_b64()}, "prompt": "x", "state": list(range(20))})
    assert isinstance(obs["state"], np.ndarray) and obs["state"].shape == (20,)


def test_requires_proprio_but_no_state_raises():
    with pytest.raises(ObsValidationError, match="requires obs"):
        _single_view(requires_proprio=True).preprocess({"images": {"head_camera": _jpeg_b64()}, "prompt": "x"})


def test_from_cfg_resolves_view_config():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "dataloader": {"multiview": True, "camera_layout": ["a", "b", "c"]},
            "inference": {"height": 384, "width": 320},
            "model": {"architecture": {"use_proprioception": True, "state_dim": 20}},
        }
    )
    dec = ObsPreprocessor.from_cfg(cfg, engine=None)
    assert dec.multiview is True
    assert dec.camera_layout == ["a", "b", "c"]
    assert (dec.img_height, dec.img_width) == (384, 320)
    assert dec.requires_proprio is True


# --- Pixel-level layout checks (migrated from test_policy_server_obs.py) ---


def _solid_b64(color=(240, 240, 240), w: int = 640, h: int = 480) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _multi_view_canvas(h: int = 384, w: int = 320) -> ObsPreprocessor:
    return ObsPreprocessor(
        multiview=True,
        camera_layout=list(DEFAULT_MULTIVIEW_CAMERA_LAYOUT),
        img_height=h,
        img_width=w,
    )


def test_multiview_all_three_bright_canvas():
    obs = _multi_view_canvas().preprocess(
        {
            "images": {
                "head_camera": _solid_b64(),
                "left_wrist_camera": _solid_b64(),
                "right_wrist_camera": _solid_b64(),
            },
            "prompt": "go",
        }
    )
    assert obs["image"].size == (320, 384)  # PIL.size is (width, height)
    assert np.asarray(obs["image"]).mean() > 200


def test_multiview_black_fill_pixels_when_wrists_none():
    obs = _multi_view_canvas().preprocess(
        {
            "images": {"head_camera": _solid_b64(), "left_wrist_camera": None, "right_wrist_camera": None},
            "prompt": "go",
        }
    )
    arr = np.asarray(obs["image"])
    # Layout: top 2/3 (rows 0-255) is head, bottom 1/3 is the two wrist slots.
    assert arr[:256].mean() > 200
    assert arr[256:].max() < 5


def test_multiview_black_fill_one_wrist_missing():
    obs = _multi_view_canvas().preprocess(
        {
            "images": {"head_camera": _solid_b64(), "left_wrist_camera": _solid_b64()},
            "prompt": "go",
        }
    )
    bottom = np.asarray(obs["image"])[256:]
    assert bottom[:, : 320 // 2].mean() > 200
    assert bottom[:, 320 // 2 :].max() < 5


# --- Prompt template contract (migrated from test_policy_server_obs.py) ---


def test_prompt_wrap_plain_instruction():
    assert format_prompt_for_inference("pick up the bottle") == (
        "A video recorded from a robot's point of view executing the following instruction: pick up the bottle"
    )


def test_prompt_wrap_keeps_trailing_period():
    assert format_prompt_for_inference("pick up the bottle.") == (
        "A video recorded from a robot's point of view executing the following instruction: pick up the bottle."
    )


def test_prompt_wrap_empty_instruction():
    assert format_prompt_for_inference("") == (
        "A video recorded from a robot's point of view executing the following instruction: "
    )


def test_prompt_wrap_matches_dataset_training_output():
    """Regression guard: deploy-side wrapping must equal the dataset's
    training-time prompt byte-for-byte."""
    from openwam.dataloader.robotwin import _resolve_prompt

    base = "pick up the red bottle"
    training_output = _resolve_prompt(
        instructions={"episode0.json": {"seen": [base]}},
        ep_file="episode0.hdf5",
        split="val",  # deterministic: picks pool[0]
        task_name="dummy",
    )
    assert training_output == format_prompt_for_inference(base)


def test_decode_passes_prompt_through():
    # The server is prompt-agnostic: it forwards the client's prompt verbatim.
    # Prompt wrapping now lives in each benchmark's client, not the server.
    obs = _single_view().preprocess({"images": {"head_camera": _jpeg_b64()}, "prompt": "pick up the bottle"})
    assert obs["prompt"] == "pick up the bottle"


def test_decode_missing_prompt_normalized_to_empty():
    obs = _single_view().preprocess({"images": {"head_camera": _jpeg_b64()}})
    assert obs["prompt"] == ""


def test_state_accepts_nested_list_and_flattens():
    obs = _single_view().preprocess(
        {
            "images": {"head_camera": _jpeg_b64()},
            "prompt": "x",
            "state": np.arange(20, dtype=np.float32).reshape(1, 20).tolist(),
        }
    )
    assert obs["state"].shape == (20,)
    assert obs["state"].dtype == np.float32
