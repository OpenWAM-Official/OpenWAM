"""Pixel-level contracts for benchmark eval image transport."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from benchmarks.utils import encode_numpy_b64, resize_for_lshape_slot
from openwam.deploy.obs_preprocess import ObsPreprocessor


def _random_rgb(height: int, width: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _decode(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_numpy_transport_is_lossless_png():
    source = _random_rgb(47, 61, seed=0)
    decoded = _decode(encode_numpy_b64(source))
    assert decoded.format == "PNG"
    np.testing.assert_array_equal(np.asarray(decoded.convert("RGB")), source)


@pytest.mark.parametrize(
    ("slot", "expected_size"),
    [
        ("head_camera", (320, 256)),
        ("left_wrist_camera", (160, 128)),
        ("right_wrist_camera", (160, 128)),
    ],
)
def test_eval_slot_resize_is_exact_pillow_lanczos(slot, expected_size):
    source = _random_rgb(256, 256, seed=len(slot))
    actual = resize_for_lshape_slot(source, slot)
    expected = np.asarray(Image.fromarray(source).resize(expected_size, Image.Resampling.LANCZOS))
    assert actual.shape == (expected_size[1], expected_size[0], 3)
    np.testing.assert_array_equal(actual, expected)


def test_lanczos_png_server_composition_is_pixel_exact_for_two_views():
    head = resize_for_lshape_slot(_random_rgb(256, 256, seed=1), "head_camera")
    wrist = resize_for_lshape_slot(_random_rgb(256, 256, seed=2), "left_wrist_camera")
    preprocessor = ObsPreprocessor(
        multiview=True,
        camera_layout=["head", "left", "right"],
        img_height=384,
        img_width=320,
    )
    result = preprocessor.preprocess(
        {
            "images": {
                "head_camera": encode_numpy_b64(head),
                "left_wrist_camera": encode_numpy_b64(wrist),
                "right_wrist_camera": None,
            },
            "prompt": "native prompt",
        }
    )

    expected = np.zeros((384, 320, 3), dtype=np.uint8)
    expected[:256, :] = head
    expected[256:, :160] = wrist
    np.testing.assert_array_equal(np.asarray(result["image"]), expected)


def test_slot_resize_rejects_unknown_slot():
    with pytest.raises(ValueError, match="unknown L-shape image slot"):
        resize_for_lshape_slot(np.zeros((8, 8, 3), dtype=np.uint8), "camera4")


def test_vlabench_adapter_resizes_each_eval_slot(monkeypatch):
    from benchmarks.vlabench import openwam2vlabench_interface as interface

    captured = []
    monkeypatch.setattr(
        interface.client,
        "encode_numpy_b64",
        lambda image: captured.append(np.asarray(image).shape) or "png",
    )
    policy = interface.OpenWAMVLABenchPolicy.__new__(interface.OpenWAMVLABenchPolicy)
    obs = {"rgb": np.zeros((4, 32, 48, 3), dtype=np.uint8)}
    assert policy._maybe_encode(obs, 3, "left_wrist", "left_wrist_camera") == "png"
    assert policy._maybe_encode(obs, 0, "right_wrist", "right_wrist_camera") == "png"
    assert captured == [(128, 160, 3), (128, 160, 3)]


def test_robocasa_gr1_adapter_resizes_eval_head_slot(monkeypatch):
    from benchmarks.robocasa_gr1 import openwam2robocasa_gr1_interface as interface

    captured = []
    monkeypatch.setattr(
        interface.client,
        "encode_numpy_b64",
        lambda image: captured.append(np.asarray(image).shape) or "png",
    )
    policy = interface.OpenWAMRoboCasaGR1Policy.__new__(interface.OpenWAMRoboCasaGR1Policy)
    obs = {"head": np.zeros((256, 256, 3), dtype=np.uint8)}
    assert policy._maybe_encode(obs, "head", "head_camera") == "png"
    assert captured == [(256, 320, 3)]
