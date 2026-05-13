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
import pytest
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
    async_mode: str = "none",
    temporal_ensemble: bool = False,
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
                "temporal_ensemble": temporal_ensemble,
                "history_len": 1,
            },
            "optimization": {
                "async_inference": {
                    "mode": async_mode,
                    "vanilla": {
                        "execution_horizon": 4,
                        "inference_delay_steps": 1,
                    },
                }
            },
        }
    )


def _make_server(debug: bool = False, debug_dir: str | None = None, multiview: bool = False):
    from openwam.deploy.mock_engine import MockInferenceEngine
    from openwam.deploy.policy_server import PolicyServer

    cfg = _minimal_cfg(multiview=multiview)
    engine = MockInferenceEngine(cfg=cfg, action_dim=14, latency_ms=0.0)
    return PolicyServer(
        engine=engine,
        cfg=cfg,
        debug=debug,
        debug_dir=debug_dir or tempfile.mkdtemp(prefix="pserver_smoke_"),
    )


def _make_async_server():
    from openwam.deploy.mock_engine import MockInferenceEngine
    from openwam.deploy.policy_server import PolicyServer

    cfg = _minimal_cfg(async_mode="vanilla", temporal_ensemble=True)
    engine = MockInferenceEngine(cfg=cfg, action_dim=14, latency_ms=0.0)
    return PolicyServer(
        engine=engine,
        cfg=cfg,
        debug=False,
        debug_dir=tempfile.mkdtemp(prefix="pserver_smoke_"),
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


def test_info_reports_async_inference_state():
    """Server info should expose async mode and executor stats."""
    server = _make_async_server()
    before_predict = server.get_info()["async_inference"]
    assert before_predict["enabled"] is True
    assert before_predict["mode"] == "vanilla"
    assert before_predict["effective_temporal_ensemble"] is False
    assert before_predict["num_inferences"] == 0
    assert before_predict["pending"] is False

    payload = _predict_payload(head=_jpeg_b64_frame(seed=4))
    server.predict(payload)

    info = server.get_info()
    assert set(info["async_inference"]) == set(before_predict)
    assert info["async_inference"]["enabled"] is True
    assert info["async_inference"]["mode"] == "vanilla"
    assert info["policy_config"]["temporal_ensemble"] is True
    assert info["async_inference"]["effective_temporal_ensemble"] is False
    assert info["async_inference"]["execution_horizon"] == 4
    assert info["async_inference"]["resolved_inference_delay_steps"] == 1
    assert info["async_inference"]["num_sync_inferences"] == 1


def test_info_reports_auto_async_delay_before_first_predict():
    """Auto delay should be visible before the executor sees the first action chunk."""
    from openwam.deploy.mock_engine import MockInferenceEngine
    from openwam.deploy.policy_server import PolicyServer

    cfg = _minimal_cfg(async_mode="vanilla")
    OmegaConf.update(cfg, "optimization.async_inference.vanilla.inference_delay_steps", None, merge=False)
    engine = MockInferenceEngine(cfg=cfg, action_dim=14, latency_ms=0.0)
    server = PolicyServer(engine=engine, cfg=cfg)

    info = server.get_info()["async_inference"]
    assert info["execution_horizon"] == 4
    assert info["inference_delay_steps"] is None
    assert info["resolved_inference_delay_steps"] == 2
    assert info["lead_time_steps"] == 2


def test_policy_server_cli_async_numeric_overrides():
    """Direct policy_server CLI should expose the same async sweep knobs."""
    from openwam.deploy.policy_server import _apply_async_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(
        [
            "--mock",
            "--async-mode",
            "vanilla",
            "--async-execution-horizon",
            "24",
            "--async-inference-delay-steps",
            "6",
        ]
    )
    cfg = _apply_async_cli_overrides(OmegaConf.create({}), args)

    assert OmegaConf.select(cfg, "optimization.async_inference.mode") == "vanilla"
    assert OmegaConf.select(cfg, "optimization.async_inference.vanilla.execution_horizon") == 24
    assert OmegaConf.select(cfg, "optimization.async_inference.vanilla.inference_delay_steps") == 6


def test_policy_server_cli_async_numeric_overrides_require_vanilla():
    from openwam.deploy.policy_server import _apply_async_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--mock", "--async-execution-horizon", "24"])

    with pytest.raises(ValueError, match="--async-mode vanilla"):
        _apply_async_cli_overrides(OmegaConf.create({}), args)

    legacy_cfg = OmegaConf.create({"optimization": {"async_inference": {"enabled": True}}})
    cfg = _apply_async_cli_overrides(legacy_cfg, args)
    assert OmegaConf.select(cfg, "optimization.async_inference.vanilla.execution_horizon") == 24


def test_policy_server_cli_async_numeric_overrides_fail_fast_on_invalid_ranges():
    from openwam.deploy.policy_server import _apply_async_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--mock", "--async-mode", "vanilla", "--async-execution-horizon", "4"])
    args.async_inference_delay_steps = 4

    with pytest.raises(ValueError, match="inference_delay_steps must be < execution_horizon"):
        _apply_async_cli_overrides(OmegaConf.create({}), args)


def test_mock_engine_step_count_matches_real_engine():
    """Regression guard for the num_frames - 1 alignment (review item H3)."""
    from openwam.deploy.mock_engine import MockInferenceEngine

    cfg = _minimal_cfg(num_frames=33)
    engine = MockInferenceEngine(cfg=cfg, action_dim=20, latency_ms=0.0)
    res = engine.generate({"num_frames": 33})
    # Real engine also produces num_frames - 1 steps.
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
    from openwam.deploy.policy_server import ObsValidationError

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
