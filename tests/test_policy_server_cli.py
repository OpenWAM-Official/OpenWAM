"""CLI tests for the OpenWAM policy server execution override wiring.

Exercises ``_apply_execution_cli_overrides`` / ``_build_argparser`` without an
engine, GPU, or weights: these controls are pure config logic.
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


def test_cli_inference_horizon_is_valid_in_sync_mode():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    parser = _build_argparser()
    args = parser.parse_args(["--inference-mode", "sync", "--inference-horizon", "24"])
    cfg = _apply_execution_cli_overrides(OmegaConf.create({}), args)

    assert OmegaConf.select(cfg, "inference.inference_mode") == "sync"
    assert OmegaConf.select(cfg, "inference.inference_horizon") == 24
    assert OmegaConf.select(cfg, "inference.inference_delay_steps") is None


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


def _async_denoise_cfg():
    return OmegaConf.create(
        {
            "inference": {
                "denoise_mode": "async",
                "lead_modality": "action",
                "variance_shift_alpha": 3.0,
                "linear_offset": 0.1,
            }
        }
    )


def test_cli_denoise_mode_sync_resets_async_controls():
    """Running the sync baseline against an async yaml needs no field-by-field unset."""
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(["--denoise-mode", "sync"])
    cfg = _apply_inference_overrides(_async_denoise_cfg(), args)

    assert OmegaConf.select(cfg, "inference.denoise_mode") == "sync"
    assert OmegaConf.select(cfg, "inference.lead_modality") == "video"
    assert OmegaConf.select(cfg, "inference.variance_shift_alpha") == 1.0
    assert OmegaConf.select(cfg, "inference.linear_offset") == 0.0


def test_cli_denoise_mode_sync_still_rejects_contradictory_async_flags():
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(["--denoise-mode", "sync", "--variance-shift-alpha", "3"])
    with pytest.raises(ValueError, match="--variance-shift-alpha"):
        _apply_inference_overrides(_async_denoise_cfg(), args)


def test_cli_async_flags_at_their_defaults_are_accepted_under_sync():
    """The CLI must be no stricter than the same value written in the yaml."""
    from openwam.deploy.server import _apply_inference_overrides, _build_argparser

    args = _build_argparser().parse_args(
        ["--denoise-mode", "sync", "--lead-modality", "video", "--variance-shift-alpha", "1.0", "--linear-offset", "0"]
    )
    cfg = _apply_inference_overrides(_async_denoise_cfg(), args)
    assert OmegaConf.select(cfg, "inference.denoise_mode") == "sync"
    assert OmegaConf.select(cfg, "inference.variance_shift_alpha") == 1.0


def test_cli_inference_mode_sync_keeps_horizon_and_resets_async_delay():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    cfg = OmegaConf.create(
        {"inference": {"inference_mode": "async", "inference_horizon": 8, "inference_delay_steps": 2}}
    )
    args = _build_argparser().parse_args(["--inference-mode", "sync"])
    cfg = _apply_execution_cli_overrides(cfg, args)

    assert OmegaConf.select(cfg, "inference.inference_mode") == "sync"
    assert OmegaConf.select(cfg, "inference.inference_horizon") == 8
    assert OmegaConf.select(cfg, "inference.inference_delay_steps") is None


def test_cli_inference_mode_sync_rejects_async_delay_flag():
    from openwam.deploy.server import _apply_execution_cli_overrides, _build_argparser

    cfg = OmegaConf.create({"inference": {"inference_mode": "async", "inference_horizon": 8}})
    args = _build_argparser().parse_args(["--inference-mode", "sync", "--inference-delay-steps", "2"])
    with pytest.raises(ValueError, match="--inference-mode async"):
        _apply_execution_cli_overrides(cfg, args)


def test_noop_async_denoising_warns_at_startup(caplog):
    from openwam.deploy.server import _validate_inference_config

    cfg = OmegaConf.create({"inference": {"denoise_mode": "async"}})
    with caplog.at_level("WARNING", logger="openwam.deploy.server"):
        _validate_inference_config(cfg)
    assert any("reproduces the sync trajectory" in record.getMessage() for record in caplog.records)


def test_shifted_async_denoising_does_not_warn(caplog):
    from openwam.deploy.server import _validate_inference_config

    cfg = OmegaConf.create({"inference": {"denoise_mode": "async", "variance_shift_alpha": 3.0}})
    with caplog.at_level("WARNING", logger="openwam.deploy.server"):
        _validate_inference_config(cfg)
    assert not any("reproduces the sync trajectory" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    ("inference", "match"),
    [
        ({"denoise_mode": "unsupported"}, "Unsupported denoise mode"),
        ({"denoise_mode": "sync", "variance_shift_alpha": 9.0}, "denoise_mode='async'"),
        ({"inference_mode": "sync", "inference_delay_steps": 2}, "inference_mode='async'"),
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
