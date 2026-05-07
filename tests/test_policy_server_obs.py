"""Unit tests for PolicyServer._decode_obs and format_prompt_for_inference.

Tests are pure CPU — no GPU, no network, no model loading. Each case
constructs a minimal OmegaConf cfg and a bare PolicyServer, sets the view
config attributes directly (bypassing _init_policy which would instantiate
a WAMPolicy with a real engine), then exercises _decode_obs / prompt
wrapping with synthetic PIL images.
"""

import base64
import io

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.transforms.multiview import format_prompt_for_inference
from openwam.deploy.policy_server import ObsValidationError, PolicyServer

# --- Test fixtures / helpers ---


def _make_image_b64(w: int = 640, h: int = 480, color: tuple = (128, 128, 128)) -> str:
    """Encode a solid-color PIL image as base64 JPEG."""
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_bright_b64(w: int = 640, h: int = 480) -> str:
    """Encode a bright-white image so we can distinguish it from black."""
    return _make_image_b64(w, h, color=(240, 240, 240))


def _bare_server(multiview: bool, camera_layout=None, out_h: int = 384, out_w: int = 320) -> PolicyServer:
    """Build a PolicyServer with view config set, bypassing _init_policy."""
    cfg = OmegaConf.create({})  # unused by _decode_obs once attrs are set
    server = PolicyServer(engine=None, cfg=cfg)
    server._policy = object()  # sentinel so _init_policy short-circuits
    server._multiview = multiview
    server._camera_layout = list(camera_layout or ["head_camera", "left_camera", "right_camera"])
    server._target_camera = "head_camera"
    server._img_height = out_h
    server._img_width = out_w
    return server


# --- Image-branch tests (cases 1-9 from the plan) ---


def test_multiview_false_head_only():
    server = _bare_server(multiview=False)
    obs = {"images": {"head_camera": _make_bright_b64()}, "prompt": "go"}
    out = server._decode_obs(obs)
    assert isinstance(out["image"], Image.Image)
    assert out["image"].size == (320, 384)  # PIL.size is (width, height)
    assert out["image"].mode == "RGB"


def test_multiview_false_ignores_wrist():
    server = _bare_server(multiview=False)
    obs_head_only = {"images": {"head_camera": _make_bright_b64()}, "prompt": "go"}
    obs_with_wrist = {
        "images": {
            "head_camera": _make_bright_b64(),
            "left_wrist_camera": _make_bright_b64(),
            "right_wrist_camera": _make_bright_b64(),
        },
        "prompt": "go",
    }
    out_a = server._decode_obs(dict(obs_head_only))
    out_b = server._decode_obs(dict(obs_with_wrist))
    assert np.array_equal(np.asarray(out_a["image"]), np.asarray(out_b["image"]))


def test_multiview_true_all_three():
    server = _bare_server(multiview=True)
    obs = {
        "images": {
            "head_camera": _make_bright_b64(),
            "left_wrist_camera": _make_bright_b64(),
            "right_wrist_camera": _make_bright_b64(),
        },
        "prompt": "go",
    }
    out = server._decode_obs(obs)
    assert out["image"].size == (320, 384)
    # With 3 bright cameras, the composed canvas should be mostly bright.
    arr = np.asarray(out["image"])
    assert arr.mean() > 200  # bright everywhere


def test_multiview_true_black_fill_wrists_none():
    server = _bare_server(multiview=True, out_h=384, out_w=320)
    obs = {
        "images": {
            "head_camera": _make_bright_b64(),
            "left_wrist_camera": None,
            "right_wrist_camera": None,
        },
        "prompt": "go",
    }
    out = server._decode_obs(obs)
    arr = np.asarray(out["image"])
    # Layout: top 2/3 is head (bright), bottom 1/3 is wrists (black).
    # top_h = round(384 * 2/3) = 256
    top = arr[:256]
    bottom = arr[256:]
    assert top.mean() > 200  # head is bright
    assert bottom.max() < 5  # wrist slots are black


def test_multiview_true_black_fill_one_wrist_missing():
    server = _bare_server(multiview=True, out_h=384, out_w=320)
    obs = {
        "images": {
            "head_camera": _make_bright_b64(),
            "left_wrist_camera": _make_bright_b64(),
            # right_wrist_camera intentionally omitted from the dict
        },
        "prompt": "go",
    }
    out = server._decode_obs(obs)
    arr = np.asarray(out["image"])
    # Bottom row: left half bright, right half black.
    bottom = arr[256:]
    half_w = 320 // 2
    left_bot = bottom[:, :half_w]
    right_bot = bottom[:, half_w:]
    assert left_bot.mean() > 200
    assert right_bot.max() < 5


def test_reject_missing_head_camera():
    server = _bare_server(multiview=True)
    with pytest.raises(ObsValidationError, match="head_camera"):
        server._decode_obs({"images": {"head_camera": None}, "prompt": "go"})
    with pytest.raises(ObsValidationError, match="head_camera"):
        server._decode_obs({"images": {}, "prompt": "go"})


def test_reject_legacy_image_field():
    server = _bare_server(multiview=False)
    with pytest.raises(ObsValidationError, match="images"):
        server._decode_obs({"image": _make_bright_b64(), "prompt": "go"})


def test_reject_bad_base64():
    server = _bare_server(multiview=False)
    with pytest.raises(ObsValidationError, match="failed to decode"):
        server._decode_obs({"images": {"head_camera": "!!!not-valid-base64!!!"}, "prompt": "go"})


def test_multiview_true_short_camera_layout_errors():
    server = _bare_server(multiview=True, camera_layout=["head_camera", "left_camera"])  # only 2
    with pytest.raises(ObsValidationError, match="camera_layout"):
        server._decode_obs({"images": {"head_camera": _make_bright_b64()}, "prompt": "go"})


# --- Prompt-branch tests (cases 10-14 from the plan) ---


def test_prompt_passthrough_single_view():
    assert format_prompt_for_inference("pick up the bottle") == (
        "A video recorded from a robot's point of view executing the following instruction: pick up the bottle"
    )


def test_prompt_wrap_multiview_with_period():
    out = format_prompt_for_inference("pick up the bottle.")
    assert out == (
        "A video recorded from a robot's point of view executing the following instruction: pick up the bottle."
    )


def test_prompt_wrap_multiview_auto_period():
    out = format_prompt_for_inference("pick up the bottle")
    assert out == (
        "A video recorded from a robot's point of view executing the following instruction: pick up the bottle"
    )


def test_prompt_wrap_matches_dataset_training_output():
    """Regression guard: the server-side helper must produce byte-for-byte
    identical prompts to what the dataset produces at training time.

    Calls the pure ``_resolve_prompt`` helper directly (what
    ``RoboTwinDataset._get_prompt`` delegates to) so the test no longer
    depends on poking private attributes via ``__new__``.
    """
    from openwam.dataloader.robotwin_dataset import _resolve_prompt

    base = "pick up the red bottle"

    training_output = _resolve_prompt(
        instructions={"episode0.json": {"seen": [base]}},
        ep_file="episode0.hdf5",
        split="val",  # deterministic: picks pool[0]
        task_name="dummy",
    )
    helper_output = format_prompt_for_inference(base)

    assert training_output == helper_output


def test_prompt_empty_multiview_fallback():
    out = format_prompt_for_inference("")
    assert out == "A video recorded from a robot's point of view executing the following instruction: "


# --- End-to-end prompt-wrapping through _decode_obs ---


def test_decode_obs_wraps_prompt_for_multiview():
    server = _bare_server(multiview=True)
    obs = {
        "images": {
            "head_camera": _make_bright_b64(),
            "left_wrist_camera": None,
            "right_wrist_camera": None,
        },
        "prompt": "pick up the bottle",
    }
    out = server._decode_obs(obs)
    assert out["prompt"] == (
        "A video recorded from a robot's point of view executing the following instruction: pick up the bottle"
    )


def test_decode_obs_passes_prompt_through_single_view():
    server = _bare_server(multiview=False)
    obs = {"images": {"head_camera": _make_bright_b64()}, "prompt": "pick up the bottle"}
    out = server._decode_obs(obs)
    assert out["prompt"] == (
        "A video recorded from a robot's point of view executing the following instruction: pick up the bottle"
    )
