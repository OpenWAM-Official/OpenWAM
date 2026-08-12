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
    args = parser.parse_args(["--inference-mode", "async", "--inference-horizon", "24", "--inference-delay-steps", "6"])
    cfg = _apply_execution_cli_overrides(OmegaConf.create({}), args)

    assert OmegaConf.select(cfg, "inference.inference_mode") == "async"
    assert OmegaConf.select(cfg, "inference.inference_horizon") == 24
    assert OmegaConf.select(cfg, "inference.inference_delay_steps") == 6


def test_cli_execution_timing_flags_require_async():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--inference-horizon", "24"])

    with pytest.raises(ValueError, match="--inference-mode async"):
        _apply_execution_cli_overrides(OmegaConf.create({}), args)


def test_cli_execution_overrides_fail_fast_on_invalid_ranges():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--inference-mode", "async", "--inference-horizon", "4"])
    args.inference_delay_steps = 4

    with pytest.raises(ValueError, match="inference_delay_steps must be < inference_horizon"):
        _apply_execution_cli_overrides(OmegaConf.create({}), args)


def test_cli_inference_mode_rejects_invalid_value():
    from openwam.deploy.server import _build_argparser

    with pytest.raises(SystemExit):
        _build_argparser().parse_args(["--inference-mode", "unsupported"])


# --- Unified CLI: the package entrypoint is a strict superset of scripts/deploy.py ---


def test_cli_exposes_ckpt_name_and_inference_overrides():
    """Flags absorbed from scripts/deploy.py parse with the documented defaults."""
    from openwam.deploy.server import _build_argparser

    args = _build_argparser().parse_args([])
    assert args.ckpt_name is None
    assert args.denoise_steps is None
    assert args.denoise_mode is None
    assert args.device is None  # fallback chain resolves later: CLI > yaml > cuda

    args = _build_argparser().parse_args(["--ckpt-name", "checkpoint_step_42.safetensors", "--denoise-steps", "7"])
    assert args.ckpt_name == "checkpoint_step_42.safetensors"
    assert args.denoise_steps == 7


def test_cli_denoise_mode_accepts_sync_and_async():
    from openwam.deploy.server import _build_argparser

    parser = _build_argparser()
    assert parser.parse_args(["--denoise-mode", "sync"]).denoise_mode == "sync"
    assert parser.parse_args(["--denoise-mode", "async"]).denoise_mode == "async"
    with pytest.raises(SystemExit):
        parser.parse_args(["--denoise-mode", "unsupported"])


def test_cli_async_denoising_overrides_parse():
    from openwam.deploy.server import _build_argparser

    args = _build_argparser().parse_args(
        [
            "--lead-modality",
            "video",
            "--variance-shift-alpha",
            "9",
            "--linear-offset",
            "0.2",
        ]
    )
    assert args.lead_modality == "video"
    assert args.variance_shift_alpha == 9.0
    assert args.linear_offset == 0.2


def test_cli_async_denoising_overrides_require_async_mode():
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(["--variance-shift-alpha", "9"])
    with pytest.raises(ValueError, match="--denoise-mode async"):
        _apply_inference_overrides(OmegaConf.create({}), args)


@pytest.mark.parametrize(
    ("flag", "value", "match"),
    [
        ("--variance-shift-alpha", "0", "variance_shift_alpha must be >= 1"),
        ("--linear-offset", "-0.1", "linear_offset must satisfy"),
        ("--linear-offset", "1", "linear_offset must satisfy"),
    ],
)
def test_cli_async_denoising_overrides_validate_ranges(flag, value, match):
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(["--denoise-mode", "async", flag, value])
    with pytest.raises(ValueError, match=match):
        _apply_inference_overrides(OmegaConf.create({}), args)


@pytest.mark.parametrize(
    ("inference", "match"),
    [
        ({"denoise_mode": "unsupported"}, "Unsupported denoise mode"),
        ({"denoise_mode": "sync", "variance_shift_alpha": 9.0}, "require denoise_mode='async'"),
        ({"inference_mode": "sync", "inference_horizon": 8}, "require inference_mode='async'"),
    ],
)
def test_deploy_inference_config_is_validated_at_startup(inference, match):
    from openwam.deploy.server import _validate_inference_config

    with pytest.raises(ValueError, match=match):
        _validate_inference_config(OmegaConf.create({"inference": inference}))


def test_cli_dotlist_overrides_coexist_with_value_flags():
    """Positional dotlist overrides must not swallow values of the new flags."""
    from openwam.deploy.server import _build_argparser

    args = _build_argparser().parse_args(["--denoise-steps", "7", "foo.bar=1", "inference.shift=9.0"])
    assert args.denoise_steps == 7
    assert args.overrides == ["foo.bar=1", "inference.shift=9.0"]
