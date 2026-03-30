"""Tests for the deployment policy server."""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace
from PIL import Image

from open_wam.serving.policy_server import PolicyServer, _build_argparser


class MockEngine:
    """Mock inference engine."""
    def __init__(self, action_dim=7):
        self.action_dim = action_dim
        self.call_count = 0

    def generate(self, conditions):
        self.call_count += 1
        return {
            "actions": np.ones((10, self.action_dim), dtype=np.float32) * self.call_count,
            "video": None,
        }


def make_cfg(**kwargs):
    defaults = {"history_len": 3, "execute_horizon": None, "policy": SimpleNamespace(history_len=3, execute_horizon=None)}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_base64_image(width=64, height=64):
    """Create a small base64-encoded JPEG image."""
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_server_import():
    """PolicyServer should be importable."""
    from open_wam.serving import PolicyServer
    assert callable(PolicyServer)


def test_server_predict_basic():
    """Basic predict should return action dict."""
    server = PolicyServer(MockEngine(), make_cfg())
    obs = {"image": Image.new("RGB", (64, 64))}
    result = server.predict(obs)

    assert "action" in result
    assert "step" in result
    assert "latency_ms" in result
    assert isinstance(result["action"], list)
    assert len(result["action"]) == 7
    assert result["step"] == 1


def test_server_predict_base64_image():
    """Server should decode base64 images."""
    server = PolicyServer(MockEngine(), make_cfg())
    b64 = make_base64_image()
    result = server.predict({"image": b64})

    assert "action" in result
    assert len(result["action"]) == 7


def test_server_predict_with_state():
    """Server should handle state arrays."""
    server = PolicyServer(MockEngine(), make_cfg())
    result = server.predict({
        "image": Image.new("RGB", (64, 64)),
        "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    })

    assert "action" in result


def test_server_reset():
    """Reset should clear state."""
    server = PolicyServer(MockEngine(), make_cfg())
    server.predict({"image": Image.new("RGB", (64, 64))})
    assert server._request_count == 1

    server.reset()
    assert server._request_count == 0


def test_server_info():
    """Info should return server stats."""
    server = PolicyServer(MockEngine(), make_cfg())
    server.predict({"image": Image.new("RGB", (64, 64))})
    server.predict({"image": Image.new("RGB", (64, 64))})

    info = server.get_info()
    assert info["model"] == "OpenWAM"
    assert info["total_requests"] == 2
    assert info["avg_latency_ms"] > 0


def test_server_with_embodiment():
    """Server should convert actions with embodiment adapter."""
    server = PolicyServer(MockEngine(action_dim=14), make_cfg(), embodiment="arx-x5")
    result = server.predict({"image": Image.new("RGB", (64, 64))})

    assert "action" in result
    assert len(result["action"]) == 14


def test_server_multiple_predictions():
    """Multiple predictions should increment step counter."""
    server = PolicyServer(MockEngine(), make_cfg())
    for i in range(5):
        result = server.predict({"image": Image.new("RGB", (64, 64))})
        assert result["step"] == i + 1


def test_server_latency_tracking():
    """Latency should be tracked per request."""
    server = PolicyServer(MockEngine(), make_cfg())
    result = server.predict({"image": Image.new("RGB", (64, 64))})

    assert result["latency_ms"] >= 0
    info = server.get_info()
    assert info["avg_latency_ms"] >= 0


def test_server_cli_parser():
    """CLI parser should expose the serving startup surface."""
    parser = _build_argparser()
    args = parser.parse_args(["--ckpt-path", "model.safetensors", "--ws-port", "9000"])

    assert args.ckpt_path == "model.safetensors"
    assert args.ws_port == 9000
    assert args.http_port is None


def test_policy_client_script_exists():
    """Minimal deployment client script should ship with the repo."""
    assert Path("scripts/policy_client.py").exists()
