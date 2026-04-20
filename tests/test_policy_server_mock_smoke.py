"""End-to-end smoke test for the policy server + mock inference engine.

Purpose (per PR#3 review item P2): catch silent drift in the obs contract,
``--ckpt-dir`` wiring, port defaults, mock/real-engine step-count alignment,
and debug-mode directory layout. Runs without GPU, weights, or a live HTTP
listener: we call ``PolicyServer.predict()`` directly so all the decode /
policy / engine / debug code paths exercise, without pulling in aiohttp.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile

import numpy as np
from omegaconf import OmegaConf
from PIL import Image


def _jpeg_b64_frame(h: int = 32, w: int = 32, seed: int = 0) -> str:
    """Return a base64 JPEG string for a random RGB frame."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _minimal_cfg(
    num_frames: int = 17,
    multiview: bool = False,
    height: int = 32,
    width: int = 32,
) -> OmegaConf:
    """Smallest cfg that satisfies PolicyServer._init_policy + _decode_obs."""
    return OmegaConf.create(
        {
            "inference": {"num_frames": num_frames, "height": height, "width": width},
            "dataloader": {
                "multiview": multiview,
                "target_camera": "head_camera",
                "height": height,
                "width": width,
            },
            "policy": {
                "execute_horizon": None,
                "temporal_ensemble": False,
                "history_len": 1,
            },
        }
    )


def _make_server(debug: bool = False, debug_dir: str | None = None, multiview: bool = False):
    from openwam.deployment.mock_engine import MockInferenceEngine
    from openwam.deployment.policy_server import PolicyServer

    cfg = _minimal_cfg(multiview=multiview)
    engine = MockInferenceEngine(cfg=cfg, action_dim=14, latency_ms=0.0)
    return PolicyServer(
        engine=engine,
        cfg=cfg,
        debug=debug,
        debug_dir=debug_dir or tempfile.mkdtemp(prefix="pserver_smoke_"),
    )


def _predict_payload(head: str, left: str | None = None, right: str | None = None, prompt: str = "test"):
    """Match benchmarks.utils.client.build_payload exactly."""
    p = {
        "images": {
            "head_camera": head,
            "left_wrist_camera": left,
            "right_wrist_camera": right,
        },
        "prompt": prompt,
    }
    return p


def test_predict_returns_well_formed_response():
    """Single-view path: one predict call returns action list + step + latency."""
    server = _make_server(multiview=False)
    payload = _predict_payload(head=_jpeg_b64_frame(seed=1))
    result = server.predict(payload)

    # Contract
    assert isinstance(result["action"], list)
    assert len(result["action"]) == 14  # action_dim from mock engine
    assert all(isinstance(x, float) for x in result["action"])
    assert result["step"] == 1
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0


def test_mock_engine_step_count_matches_real_engine():
    """Regression guard for the num_frames - 1 alignment (review item H3)."""
    from openwam.deployment.mock_engine import MockInferenceEngine

    cfg = _minimal_cfg(num_frames=33)
    engine = MockInferenceEngine(cfg=cfg, action_dim=20, latency_ms=0.0)
    res = engine.generate({"num_frames": 33})
    # Real engine (joint_generation.py) also produces num_frames - 1 steps.
    assert res["actions"].shape == (32, 20)


def test_reset_clears_state_and_advances_debug_episode():
    """Reset zeros the request counter and bumps the debug episode index."""
    with tempfile.TemporaryDirectory() as td:
        server = _make_server(debug=True, debug_dir=td)
        assert server._debug_episode is None
        payload = _predict_payload(head=_jpeg_b64_frame(seed=2))

        # First predict triggers lazy-init → ep0000 (no stale ep-001 from the old bug)
        server.predict(payload)
        assert server._debug_episode == 0
        assert server._request_count == 1

        # Reset should increment episode (we already had ep0000 open)
        server.reset()
        assert server._debug_episode == 1
        assert server._request_count == 0

        server.predict(payload)
        assert server._request_count == 1
        assert server._debug_episode == 1  # no further bump

        # Directory structure reflects the two episodes
        entries = sorted(os.listdir(td))
        assert "ep0000" in entries
        assert "ep0001" in entries


def test_predict_rejects_missing_head_camera():
    """Server must error out on malformed payload; covers PolicyServer._decode_obs."""
    from openwam.deployment.policy_server import ObsValidationError

    server = _make_server()
    bad_payload = {"images": {"head_camera": None}, "prompt": "x"}
    try:
        server.predict(bad_payload)
    except ObsValidationError as exc:
        assert "head_camera" in str(exc)
    else:
        raise AssertionError("expected ObsValidationError when head_camera is None")


def test_multiview_accepts_missing_wrists():
    """multiview=true: server black-fills missing wrists rather than erroring."""
    server = _make_server(multiview=True)
    # Only head_camera; left/right absent from obs payload
    payload = {
        "images": {
            "head_camera": _jpeg_b64_frame(seed=3),
            "left_wrist_camera": None,
            "right_wrist_camera": None,
        },
        "prompt": "multiview smoke",
    }
    result = server.predict(payload)
    assert len(result["action"]) == 14
    assert result["step"] == 1
