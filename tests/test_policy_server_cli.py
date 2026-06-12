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


# --- Unified CLI: the package entrypoint is a strict superset of scripts/deploy.py ---


def test_cli_exposes_ckpt_name_and_inference_overrides():
    """Flags absorbed from scripts/deploy.py parse with the documented defaults."""
    from openwam.deploy.server import _build_argparser

    args = _build_argparser().parse_args([])
    assert args.ckpt_name is None
    assert args.denoise_steps is None
    assert args.schedule_type is None
    assert args.shift is None
    assert args.device is None  # fallback chain resolves later: CLI > yaml > cuda

    args = _build_argparser().parse_args(
        ["--ckpt-name", "checkpoint_step_42.safetensors", "--denoise-steps", "7", "--shift", "3.5"]
    )
    assert args.ckpt_name == "checkpoint_step_42.safetensors"
    assert args.denoise_steps == 7
    assert args.shift == 3.5


def test_cli_schedule_type_only_accepts_sync():
    from openwam.deploy.server import _build_argparser

    parser = _build_argparser()
    assert parser.parse_args(["--schedule-type", "sync"]).schedule_type == "sync"
    with pytest.raises(SystemExit):
        parser.parse_args(["--schedule-type", "cascade"])


def test_cli_dotlist_overrides_coexist_with_value_flags():
    """Positional dotlist overrides must not swallow values of the new flags."""
    from openwam.deploy.server import _build_argparser

    args = _build_argparser().parse_args(["--denoise-steps", "7", "foo.bar=1", "inference.shift=9.0"])
    assert args.denoise_steps == 7
    assert args.overrides == ["foo.bar=1", "inference.shift=9.0"]
