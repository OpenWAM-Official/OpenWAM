"""CLI tests for the OpenWAM policy server — async inference override wiring.

Exercises ``_apply_execution_cli_overrides`` / ``_build_argparser`` without an
engine, GPU, or weights: the async sweep knobs are pure config logic.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf


def test_cli_execution_overrides_write_inference_section():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--execution-mode", "async", "--execution-horizon", "24", "--inference-delay-steps", "6"])
    cfg = _apply_execution_cli_overrides(OmegaConf.create({}), args)

    assert OmegaConf.select(cfg, "inference.execution_mode") == "async"
    assert OmegaConf.select(cfg, "inference.execution_horizon") == 24
    assert OmegaConf.select(cfg, "inference.inference_delay_steps") == 6


def test_cli_execution_timing_flags_require_async():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--execution-horizon", "24"])

    with pytest.raises(ValueError, match="--execution-mode async"):
        _apply_execution_cli_overrides(OmegaConf.create({}), args)


def test_cli_execution_overrides_fail_fast_on_invalid_ranges():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--execution-mode", "async", "--execution-horizon", "4"])
    args.inference_delay_steps = 4

    with pytest.raises(ValueError, match="inference_delay_steps must be < execution_horizon"):
        _apply_execution_cli_overrides(OmegaConf.create({}), args)


def test_legacy_async_inference_section_rejected():
    from openwam.deploy.executors import resolve_execution_config

    legacy = OmegaConf.create({"optimization": {"async_inference": {"mode": "vanilla"}}})
    with pytest.raises(ValueError, match="has been removed"):
        resolve_execution_config(legacy)


@pytest.mark.parametrize("legacy_value", ["none", "vanilla"])
def test_cli_execution_mode_rejects_legacy_values(legacy_value):
    from openwam.deploy.server import _build_argparser

    with pytest.raises(SystemExit):
        _build_argparser().parse_args(["--execution-mode", legacy_value])


# --- Unified CLI: the package entrypoint is a strict superset of scripts/deploy.py ---


def test_cli_exposes_ckpt_name_and_inference_overrides():
    """Flags absorbed from scripts/deploy.py parse with the documented defaults."""
    from openwam.deploy.server import _build_argparser

    args = _build_argparser().parse_args([])
    assert args.ckpt_name is None
    assert args.denoise_steps is None
    assert args.schedule_type is None
    assert args.device is None  # fallback chain resolves later: CLI > yaml > cuda

    args = _build_argparser().parse_args(["--ckpt-name", "checkpoint_step_42.safetensors", "--denoise-steps", "7"])
    assert args.ckpt_name == "checkpoint_step_42.safetensors"
    assert args.denoise_steps == 7


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
