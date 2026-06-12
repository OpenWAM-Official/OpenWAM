"""CLI tests for the OpenWAM policy server — async inference override wiring.

Exercises ``_apply_async_cli_overrides`` / ``_build_argparser`` without an
engine, GPU, or weights: the async sweep knobs are pure config logic.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf


def test_policy_server_cli_async_numeric_overrides():
    """Direct policy_server CLI should expose the async sweep knobs."""
    from openwam.deploy.server import _apply_async_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(
        [
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
    from openwam.deploy.server import _apply_async_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--async-execution-horizon", "24"])

    with pytest.raises(ValueError, match="--async-mode vanilla"):
        _apply_async_cli_overrides(OmegaConf.create({}), args)

    legacy_cfg = OmegaConf.create({"optimization": {"async_inference": {"enabled": True}}})
    cfg = _apply_async_cli_overrides(legacy_cfg, args)
    assert OmegaConf.select(cfg, "optimization.async_inference.vanilla.execution_horizon") == 24


def test_policy_server_cli_async_numeric_overrides_fail_fast_on_invalid_ranges():
    from openwam.deploy.server import _apply_async_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--async-mode", "vanilla", "--async-execution-horizon", "4"])
    args.async_inference_delay_steps = 4

    with pytest.raises(ValueError, match="inference_delay_steps must be < execution_horizon"):
        _apply_async_cli_overrides(OmegaConf.create({}), args)
