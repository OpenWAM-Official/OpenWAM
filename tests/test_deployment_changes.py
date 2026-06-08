"""Tests for deployment-related changes.

Covers mixed-precision save/load behavior, deploy.yaml inference/optimization
sections, deploy.py config loading, CLI override logic, attention backend
logging, and joint_engine compile flags.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_module(dtype=torch.float32) -> nn.Module:
    m = nn.Linear(4, 4)
    m.to(dtype=dtype)
    return m


def _tiny_pipe(dtype=torch.float32):
    """Minimal duck-typed pipeline with named_parameters/named_buffers."""
    pipe = MagicMock()
    linear = nn.Linear(4, 4).to(dtype=dtype)
    pipe.named_parameters.return_value = linear.named_parameters()
    pipe.named_buffers.return_value = linear.named_buffers()
    return pipe


# ---------------------------------------------------------------------------
# 1. mixed_precision in accelerate yaml (single source of truth)
# ---------------------------------------------------------------------------


class TestAccelerateYamlMixedPrecision:
    """``cfg.accelerate.mixed_precision`` is the sole source of truth.

    ``cfg.training.mixed_precision`` was removed; both training and deploy
    must read from the accelerate yaml that ``configs/train.yaml`` composes
    from (default: ``accelerate/deepspeed_zero2.yaml``).
    """

    def test_training_field_removed(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "train.yaml")
        assert OmegaConf.select(cfg, "training.mixed_precision") is None, (
            "training.mixed_precision should be removed; accelerate.mixed_precision is now the only source"
        )

    @pytest.mark.parametrize("stage", ["deepspeed_zero1", "deepspeed_zero2", "deepspeed_zero3"])
    def test_accelerate_field_exists(self, stage):
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "accelerate" / f"{stage}.yaml")
        mp = OmegaConf.select(cfg, "mixed_precision")
        assert mp is not None, f"mixed_precision missing from {stage}.yaml"
        assert mp == "bf16", f"Expected 'bf16', got {mp!r}"


# ---------------------------------------------------------------------------
# 2. checkpointing — save dtype enforcement
# ---------------------------------------------------------------------------


class TestSaveTrainableCheckpoint:
    def test_saves_bf16(self, tmp_path):
        from safetensors.torch import load_file

        from openwam.train.utils.checkpointing import save_trainable_checkpoint

        action_dit = _tiny_module(dtype=torch.float32)
        pipe = _tiny_pipe(dtype=torch.float32)

        path = str(tmp_path / "ckpt.safetensors")
        save_trainable_checkpoint(path, action_dit, pipe, lambda_action=1.0, mixed_precision="bf16")

        sd = load_file(path)
        for k, v in sd.items():
            assert v.dtype == torch.bfloat16, f"{k}: expected bf16, got {v.dtype}"

    def test_saves_fp16(self, tmp_path):
        from safetensors.torch import load_file

        from openwam.train.utils.checkpointing import save_trainable_checkpoint

        action_dit = _tiny_module(dtype=torch.float32)
        pipe = _tiny_pipe(dtype=torch.float32)

        path = str(tmp_path / "ckpt_fp16.safetensors")
        save_trainable_checkpoint(path, action_dit, pipe, lambda_action=1.0, mixed_precision="fp16")

        sd = load_file(path)
        for k, v in sd.items():
            assert v.dtype == torch.float16, f"{k}: expected fp16, got {v.dtype}"

    def test_saves_fp32_when_no(self, tmp_path):
        from safetensors.torch import load_file

        from openwam.train.utils.checkpointing import save_trainable_checkpoint

        action_dit = _tiny_module(dtype=torch.float32)
        pipe = _tiny_pipe(dtype=torch.float32)

        path = str(tmp_path / "ckpt_fp32.safetensors")
        save_trainable_checkpoint(path, action_dit, pipe, lambda_action=1.0, mixed_precision="no")

        sd = load_file(path)
        for k, v in sd.items():
            assert v.dtype == torch.float32, f"{k}: expected fp32, got {v.dtype}"

    def test_invalid_precision_raises(self, tmp_path):
        from openwam.train.utils.checkpointing import save_trainable_checkpoint

        action_dit = _tiny_module()
        pipe = _tiny_pipe()
        with pytest.raises(ValueError, match="Unknown mixed_precision"):
            save_trainable_checkpoint(str(tmp_path / "x.safetensors"), action_dit, pipe, 1.0, mixed_precision="int8")

    def test_integer_buffers_not_cast(self, tmp_path):
        """Integer / bool buffers must survive unchanged through the cast."""
        from safetensors.torch import load_file

        from openwam.train.utils.checkpointing import save_trainable_checkpoint

        class ModWithIntBuf(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(4, 4))
                self.register_buffer("step", torch.tensor(42, dtype=torch.int64))

            def forward(self, x):
                return x

        m = ModWithIntBuf()
        pipe = _tiny_pipe(dtype=torch.float32)

        path = str(tmp_path / "ckpt_int.safetensors")
        save_trainable_checkpoint(path, m, pipe, lambda_action=1.0, mixed_precision="bf16")

        sd = load_file(path)
        assert sd["action_backbone.step"].dtype == torch.int64, "int64 buffer should not be cast"
        assert sd["action_backbone.w"].dtype == torch.bfloat16, "float param should be cast to bf16"

    def test_default_precision_is_bf16(self, tmp_path):
        """Calling without mixed_precision arg should default to bf16."""
        from safetensors.torch import load_file

        from openwam.train.utils.checkpointing import save_trainable_checkpoint

        action_dit = _tiny_module(dtype=torch.float32)
        pipe = _tiny_pipe(dtype=torch.float32)

        path = str(tmp_path / "ckpt_default.safetensors")
        save_trainable_checkpoint(path, action_dit, pipe, lambda_action=1.0)  # no mixed_precision kwarg

        sd = load_file(path)
        for k, v in sd.items():
            assert v.dtype == torch.bfloat16, f"{k}: default should be bf16, got {v.dtype}"


# ---------------------------------------------------------------------------
# 3. deployment.yaml — inference + deploy sections
# ---------------------------------------------------------------------------


class TestDeploymentYaml:
    def _load(self):
        from omegaconf import OmegaConf

        return OmegaConf.load(PROJECT_ROOT / "configs" / "deploy.yaml")

    def test_inference_section_exists(self):
        cfg = self._load()
        from omegaconf import OmegaConf

        assert OmegaConf.select(cfg, "inference") is not None

    def test_inference_denoise_steps(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "inference.denoise_steps") is not None

    def test_inference_cfg_fields_default_noop(self):
        """§15 — `cfg_scale` is back in deploy.yaml as a Cosmos25 CFG knob.

        The default MUST be the no-op (`cfg_scale=1.0` / `cfg_merge=false` /
        `text_embedding_cache_dir=null`) so that existing Wan deployments and
        Cosmos25 synthetic smoke paths keep their behavior unchanged. CFG only
        activates when a user explicitly overrides ``inference.cfg_scale`` to
        a value greater than 1.0.
        """
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "inference.cfg_scale") == 1.0
        assert OmegaConf.select(cfg, "inference.cfg_merge") is False
        assert OmegaConf.select(cfg, "inference.text_embedding_cache_dir") is None

    def test_inference_schedule_type(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "inference.schedule_type") is not None

    def test_optimization_section_exists(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "optimization") is not None

    def test_optimization_compile_section_exists(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "optimization.compile") is not None
        assert OmegaConf.select(cfg, "optimization.compile.mode") == "auto"
        assert OmegaConf.select(cfg, "optimization.compile.self_attn.torch_mode") == "default"
        assert OmegaConf.select(cfg, "optimization.compile.self_attn.dynamic") is False
        assert OmegaConf.select(cfg, "optimization.compile.cross_attn.torch_mode") == "default"
        assert OmegaConf.select(cfg, "optimization.compile.cross_attn.dynamic") is False

    def test_optimization_dit_cache_defaults_off(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert not OmegaConf.select(cfg, "optimization.dit_cache.enabled")

    def test_optimization_async_inference_defaults_off(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "optimization.async_inference.mode") == "none"
        assert OmegaConf.select(cfg, "optimization.async_inference.vanilla.inference_delay_steps") is None

    def test_optimization_schedule_action_steps(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "optimization.schedule.action_steps") is not None

    def test_server_defaults_present(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "server.ws_port") is not None
        assert OmegaConf.select(cfg, "server.http_port") is not None


# ---------------------------------------------------------------------------
# 4. deploy.py — config loading and CLI override logic
# ---------------------------------------------------------------------------


class TestDeployConfigLoading:
    """Test _load_deploy_config and _apply_cli_overrides without running a server."""

    def _import(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("deploy", PROJECT_ROOT / "scripts" / "deploy.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _compile_options(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "compile_options", PROJECT_ROOT / "openwam" / "model" / "compile_options.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _policy_server(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "policy_server", PROJECT_ROOT / "openwam" / "deploy" / "policy_server.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _blank_args(self):
        args = MagicMock()
        for attr in (
            "device",
            "host",
            "ws_port",
            "http_port",
            "denoise_steps",
            "schedule_type",
            "shift",
            "compile_mode",
            "async_mode",
            "async_execution_horizon",
            "async_inference_delay_steps",
        ):
            setattr(args, attr, None)
        return args

    def test_load_deploy_config_returns_omegaconf(self):
        deploy = self._import()
        cfg = deploy._load_deploy_config()
        from omegaconf import DictConfig

        assert isinstance(cfg, DictConfig)

    def test_load_deploy_config_has_inference(self):
        from omegaconf import OmegaConf

        deploy = self._import()
        cfg = deploy._load_deploy_config()
        assert OmegaConf.select(cfg, "inference.denoise_steps") is not None

    def test_cli_device_override(self):
        from omegaconf import OmegaConf

        deploy = self._import()

        cfg = deploy._load_deploy_config()
        args = self._blank_args()
        args.device = "cuda:3"

        cfg = deploy._apply_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "device") == "cuda:3"

    def test_cli_denoise_steps_override(self):
        from omegaconf import OmegaConf

        deploy = self._import()

        cfg = deploy._load_deploy_config()
        args = self._blank_args()
        args.denoise_steps = 20

        cfg = deploy._apply_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.denoise_steps") == 20

    def test_cli_compile_mode_none_disables_compile(self):
        from omegaconf import OmegaConf

        deploy = self._import()
        compile_options = self._compile_options()

        cfg = deploy._load_deploy_config()
        args = self._blank_args()
        args.compile_mode = "none"

        cfg = deploy._apply_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "optimization.compile.mode") == "none"
        compile_cfg = OmegaConf.select(cfg, "optimization.compile")
        assert compile_options.compile_mode(compile_cfg, strict=True) == "none"

    def test_cli_compile_mode_auto_keeps_architecture_selection(self):
        from omegaconf import OmegaConf

        deploy = self._import()
        compile_options = self._compile_options()

        cfg = deploy._load_deploy_config()
        args = self._blank_args()
        args.compile_mode = "auto"

        cfg = deploy._apply_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "optimization.compile.mode") == "auto"
        compile_cfg = OmegaConf.select(cfg, "optimization.compile")
        assert compile_options.compile_mode(compile_cfg, strict=True) == "auto"

    def test_cli_compile_mode_accepts_only_auto_and_none(self):
        import argparse

        deploy = self._import()

        parser = argparse.ArgumentParser()
        parser.add_argument("--compile-mode", type=deploy._normalize_compile_mode_arg, choices=deploy._COMPILE_MODES)

        assert parser.parse_args(["--compile-mode", "auto"]).compile_mode == "auto"
        assert parser.parse_args(["--compile-mode", "none"]).compile_mode == "none"
        with pytest.raises(SystemExit):
            parser.parse_args(["--compile-mode", "self-attn"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--compile-mode", "cross-attn"])

    def test_mode_only_compile_sections_use_fast_path_defaults(self):
        compile_options = self._compile_options()

        self_cfg = {"self_attn": {}}
        self_section = compile_options.self_attn_compile_cfg(self_cfg)
        assert compile_options.section_enabled(self_section, default=False) is True
        assert compile_options.torch_compile_kwargs(self_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }

        cross_cfg = {"cross_attn": {}}
        cross_section = compile_options.cross_attn_compile_cfg(cross_cfg)
        assert compile_options.section_enabled(cross_section, default=False) is True
        assert compile_options.torch_compile_kwargs(cross_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }

        idm_cfg = {"idm": {}}
        idm_section = compile_options.idm_compile_cfg(idm_cfg)
        assert compile_options.section_enabled(idm_section, default=False) is True
        assert compile_options.torch_compile_kwargs(idm_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }
        assert compile_options.section_enabled(idm_section.video_loop, default=False) is True
        assert compile_options.section_enabled(idm_section.action_cache, default=False) is True

        tri_cfg = {"tri_system": {}}
        tri_section = compile_options.tri_system_compile_cfg(tri_cfg)
        assert compile_options.section_enabled(tri_section, default=False) is True
        assert compile_options.torch_compile_kwargs(tri_section) == {
            "dynamic": False,
            "mode": "reduce-overhead",
        }

    def test_compile_section_enabled_false_is_respected(self):
        compile_options = self._compile_options()

        self_cfg = {"self_attn": {"enabled": False}}
        self_section = compile_options.self_attn_compile_cfg(self_cfg)
        assert compile_options.section_enabled(self_section, default=True) is False

        cross_cfg = {"cross_attn": {"enabled": False}}
        cross_section = compile_options.cross_attn_compile_cfg(cross_cfg)
        assert compile_options.section_enabled(cross_section, default=True) is False

        idm_cfg = {"idm": {"enabled": False}}
        idm_section = compile_options.idm_compile_cfg(idm_cfg)
        assert compile_options.section_enabled(idm_section, default=True) is False
        assert compile_options.section_enabled(idm_section.video_loop, default=True) is False
        assert compile_options.section_enabled(idm_section.action_cache, default=True) is False

        tri_cfg = {"tri_system": {"enabled": False}}
        tri_section = compile_options.tri_system_compile_cfg(tri_cfg)
        assert compile_options.section_enabled(tri_section, default=True) is False

    def test_idm_compile_subsections_can_be_disabled_independently(self):
        compile_options = self._compile_options()

        idm_section = compile_options.idm_compile_cfg(
            {
                "idm": {
                    "video_loop": {"enabled": False},
                    "action_cache": {"enabled": True},
                }
            }
        )

        assert compile_options.section_enabled(idm_section, default=True) is True
        assert compile_options.section_enabled(idm_section.video_loop, default=True) is False
        assert compile_options.section_enabled(idm_section.action_cache, default=False) is True

    def test_removed_compile_modes_are_rejected(self):
        compile_options = self._compile_options()

        assert compile_options.normalize_compile_mode("auto") == "auto"
        assert compile_options.normalize_compile_mode("none") == "none"
        with pytest.raises(ValueError, match="Unknown compile mode"):
            compile_options.normalize_compile_mode("default")
        with pytest.raises(ValueError, match="Unknown compile mode"):
            compile_options.compile_mode({"mode": "default"}, strict=True)
        with pytest.raises(ValueError, match="Unknown compile mode"):
            compile_options.normalize_compile_mode("self_attn")
        with pytest.raises(ValueError, match="Unknown compile mode"):
            compile_options.normalize_compile_mode("cross-attn")

    def test_policy_server_entrypoint_validates_compile_mode(self):
        from omegaconf import OmegaConf

        policy_server = self._policy_server()

        cfg = OmegaConf.create({"optimization": {"compile": {"mode": "none"}}})
        policy_server._normalize_compile_mode_in_cfg(cfg)
        assert OmegaConf.select(cfg, "optimization.compile.mode") == "none"

        auto_cfg = OmegaConf.create({"optimization": {"compile": {"mode": "auto"}}})
        policy_server._normalize_compile_mode_in_cfg(auto_cfg)
        assert OmegaConf.select(auto_cfg, "optimization.compile.mode") == "auto"

        bad_cfg = OmegaConf.create({"optimization": {"compile": {"mode": "default"}}})
        with pytest.raises(ValueError, match="Unknown compile mode"):
            policy_server._normalize_compile_mode_in_cfg(bad_cfg)

    def test_policy_server_compile_mode_cli_override(self):
        from omegaconf import OmegaConf

        policy_server = self._policy_server()

        args = policy_server._build_argparser().parse_args(["--mock", "--compile-mode", "none"])
        assert args.compile_mode == "none"

        cfg = OmegaConf.create({})
        policy_server._apply_compile_mode_override(cfg, args.compile_mode)
        assert OmegaConf.select(cfg, "optimization.compile.mode") == "none"

        with pytest.raises(SystemExit):
            policy_server._build_argparser().parse_args(["--mock", "--compile-mode", "self-attn"])

    def test_cli_async_mode_override(self):
        from omegaconf import OmegaConf

        deploy = self._import()

        cfg = deploy._load_deploy_config()
        args = self._blank_args()
        args.async_mode = "vanilla"

        cfg = deploy._apply_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "optimization.async_inference.mode") == "vanilla"

    def test_cli_async_numeric_overrides(self):
        from omegaconf import OmegaConf

        deploy = self._import()

        cfg = deploy._load_deploy_config()
        args = self._blank_args()
        args.async_mode = "vanilla"
        args.async_execution_horizon = 24
        args.async_inference_delay_steps = 6

        cfg = deploy._apply_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "optimization.async_inference.mode") == "vanilla"
        assert OmegaConf.select(cfg, "optimization.async_inference.vanilla.execution_horizon") == 24
        assert OmegaConf.select(cfg, "optimization.async_inference.vanilla.inference_delay_steps") == 6

        legacy_cfg = deploy._load_deploy_config()
        OmegaConf.update(legacy_cfg, "optimization.async_inference.mode", None, merge=False)
        OmegaConf.update(legacy_cfg, "optimization.async_inference.enabled", True, merge=False)
        legacy_args = self._blank_args()
        legacy_args.async_execution_horizon = 16
        legacy_cfg = deploy._apply_cli_overrides(legacy_cfg, legacy_args)
        assert OmegaConf.select(legacy_cfg, "optimization.async_inference.vanilla.execution_horizon") == 16

    def test_cli_async_numeric_overrides_require_vanilla(self):
        deploy = self._import()

        cfg = deploy._load_deploy_config()
        args = self._blank_args()
        args.async_execution_horizon = 24

        with pytest.raises(ValueError, match="--async-mode vanilla"):
            deploy._apply_cli_overrides(cfg, args)

        args.async_mode = "none"
        with pytest.raises(ValueError, match="--async-mode vanilla"):
            deploy._apply_cli_overrides(cfg, args)

    def test_cli_async_numeric_overrides_fail_fast_on_invalid_ranges(self):
        deploy = self._import()

        cfg = deploy._load_deploy_config()
        args = self._blank_args()
        args.async_mode = "vanilla"
        args.async_execution_horizon = 0
        with pytest.raises(ValueError, match="execution_horizon must be positive"):
            deploy._apply_cli_overrides(cfg, args)

        cfg = deploy._load_deploy_config()
        args = self._blank_args()
        args.async_mode = "vanilla"
        args.async_execution_horizon = 4
        args.async_inference_delay_steps = 4
        with pytest.raises(ValueError, match="inference_delay_steps must be < execution_horizon"):
            deploy._apply_cli_overrides(cfg, args)

    def test_none_args_do_not_override(self):
        from omegaconf import OmegaConf

        deploy = self._import()

        cfg = deploy._load_deploy_config()
        original_steps = OmegaConf.select(cfg, "inference.denoise_steps")

        args = self._blank_args()
        # All None — nothing should change
        for attr in ("device", "host", "ws_port", "http_port", "denoise_steps", "schedule_type", "shift", "async_mode"):
            setattr(args, attr, None)

        cfg = deploy._apply_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.denoise_steps") == original_steps

    def test_merge_with_training_cfg_uses_dataloader_dims(self):
        from omegaconf import OmegaConf

        deploy = self._import()

        training_cfg = OmegaConf.create({"dataloader": {"num_frames": 49, "height": 720, "width": 1280}})
        deploy_cfg = deploy._load_deploy_config()

        # Remove inference dims so they should be filled from dataloader
        OmegaConf.update(deploy_cfg, "inference.num_frames", None, merge=False)
        OmegaConf.update(deploy_cfg, "inference.height", None, merge=False)
        OmegaConf.update(deploy_cfg, "inference.width", None, merge=False)

        merged = deploy._merge_with_training_cfg(training_cfg, deploy_cfg)
        assert OmegaConf.select(merged, "inference.num_frames") == 49
        assert OmegaConf.select(merged, "inference.height") == 720
        assert OmegaConf.select(merged, "inference.width") == 1280

    def test_deploy_cfg_wins_over_training_cfg_on_overlap(self):
        from omegaconf import OmegaConf

        deploy = self._import()

        training_cfg = OmegaConf.create({"inference": {"denoise_steps": 99}})
        deploy_cfg = deploy._load_deploy_config()
        OmegaConf.update(deploy_cfg, "inference.denoise_steps", 10, merge=False)

        merged = deploy._merge_with_training_cfg(training_cfg, deploy_cfg)
        assert OmegaConf.select(merged, "inference.denoise_steps") == 10


# ---------------------------------------------------------------------------
# 5. joint_engine.py — compile flags parsed from cfg.optimization.compile
# ---------------------------------------------------------------------------


class TestJointEngineCompileFlags:
    """Verify compile mode routing without broad default compile side effects."""

    def _make_filter_engine(self, architecture):
        from openwam.deploy.joint_engine import JointInferenceEngine

        engine = JointInferenceEngine.__new__(JointInferenceEngine)
        engine.architecture = architecture
        engine._architecture_generate_accepts_extra_kwargs = None
        engine._architecture_generate_kwarg_names = None
        engine._architecture_generate_warned_dropped_kwargs = set()
        return engine

    def _make_engine(self, compile_mode="none", return_arch=False):
        from omegaconf import OmegaConf

        from openwam.deploy.joint_engine import JointInferenceEngine

        cfg = OmegaConf.create(
            {
                "inference": {
                    "denoise_steps": 10,
                    "schedule_type": "sync",
                    "shift": 5.0,
                    "num_frames": 33,
                    "height": 384,
                    "width": 320,
                },
                "optimization": {
                    "decode_video": True,
                    "compile": {
                        "mode": compile_mode,
                        "self_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
                        "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
                        "tri_system": {"torch_mode": "reduce-overhead", "dynamic": False},
                    },
                    "dit_cache": {"enabled": False},
                    "schedule": {"type": None, "action_steps": 4},
                },
            }
        )

        arch = MagicMock()
        with patch("torch.compile") as mock_compile:
            engine = JointInferenceEngine.__new__(JointInferenceEngine)
            engine.cfg = cfg
            engine.architecture = arch
            engine.action_dit = None
            engine.action_repr = None
            engine._init_optimizations()
        if return_arch:
            return engine, mock_compile, arch
        return engine, mock_compile

    def test_compile_mode_none_does_not_broad_compile(self):
        _engine, mock_compile, arch = self._make_engine(return_arch=True)
        mock_compile.assert_not_called()
        arch.apply_compile_optimizations.assert_called_once()

    def test_auto_mode_is_passed_to_architecture(self):
        from omegaconf import OmegaConf

        _engine, mock_compile, arch = self._make_engine("auto", return_arch=True)
        mock_compile.assert_not_called()
        compile_cfg = arch.apply_compile_optimizations.call_args.args[0]
        assert OmegaConf.select(compile_cfg, "mode") == "auto"

    def test_generate_kwarg_filter_warns_once_for_meaningful_drops(self, caplog):
        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())

        caplog.set_level("WARNING", logger="openwam.deploy.joint_engine")
        kwargs = {
            "schedule": object(),
            "prompt": "pick up the cube",
            "cfg_scale": 1.5,
            "cfg_merge": False,
            "pre_encoded_text": None,
            "prompt_embed_cache": object(),
        }

        filtered = engine._filter_architecture_generate_kwargs(kwargs)
        engine._filter_architecture_generate_kwargs(kwargs)

        assert set(filtered) == {"schedule", "prompt"}
        warning_messages = [
            record.getMessage() for record in caplog.records if "does not accept deploy kwarg" in record.getMessage()
        ]
        assert len(warning_messages) == 1
        assert "cfg_scale" in warning_messages[0]
        assert "prompt_embed_cache" in warning_messages[0]

    def test_generate_kwarg_filter_keeps_default_noop_drops_quiet(self, caplog):
        from openwam.deploy.joint_engine import _BoundedPromptEmbedCache

        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())

        caplog.set_level("WARNING", logger="openwam.deploy.joint_engine")
        kwargs = {
            "schedule": object(),
            "prompt": "pick up the cube",
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "pre_encoded_text": None,
            "prompt_embed_cache": _BoundedPromptEmbedCache(),
        }

        filtered = engine._filter_architecture_generate_kwargs(kwargs)

        assert set(filtered) == {"schedule", "prompt"}
        assert not [record for record in caplog.records if "does not accept deploy kwarg" in record.getMessage()]

    def test_generate_kwarg_filter_warns_for_configured_prompt_cache_drop(self, caplog):
        from openwam.deploy.joint_engine import _BoundedPromptEmbedCache

        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())

        caplog.set_level("WARNING", logger="openwam.deploy.joint_engine")
        filtered = engine._filter_architecture_generate_kwargs(
            {
                "schedule": object(),
                "prompt": "pick up the cube",
                "cfg_scale": 1.0,
                "prompt_embed_cache": _BoundedPromptEmbedCache(maxsize=64),
            }
        )

        assert set(filtered) == {"schedule", "prompt"}
        warning_messages = [
            record.getMessage() for record in caplog.records if "does not accept deploy kwarg" in record.getMessage()
        ]
        assert len(warning_messages) == 1
        assert "prompt_embed_cache" in warning_messages[0]

    def test_base_architecture_does_not_broad_compile_backbones(self):
        from omegaconf import OmegaConf

        from tests.test_openwam_trainer import _make_tiny_arch

        arch = _make_tiny_arch()
        cfg = OmegaConf.create(
            {
                "mode": "none",
                "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )

        with patch("torch.compile") as mock_compile:
            arch.apply_compile_optimizations(cfg)

        mock_compile.assert_not_called()

    def test_base_architecture_auto_compile_is_eager(self):
        from omegaconf import OmegaConf

        from tests.test_openwam_trainer import _make_tiny_arch

        arch = _make_tiny_arch()
        cfg = OmegaConf.create({"mode": "auto"})

        with patch("torch.compile") as mock_compile:
            arch.apply_compile_optimizations(cfg)

        mock_compile.assert_not_called()


# ---------------------------------------------------------------------------
# 6. model_loader.py — dtype applied to all pipeline modules
# ---------------------------------------------------------------------------


class TestModelLoaderDtype:
    """Unit-test the dtype-selection logic in load_from_checkpoint_dir."""

    def test_dtype_map_bf16(self):
        # Simulate what model_loader does when mixed_precision = bf16
        _mp = "bf16"
        _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
        dtype = _DTYPE_MAP.get(str(_mp).strip().lower(), torch.bfloat16)
        assert dtype == torch.bfloat16

    def test_dtype_map_fp16(self):
        _mp = "fp16"
        _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
        dtype = _DTYPE_MAP.get(str(_mp).strip().lower(), torch.bfloat16)
        assert dtype == torch.float16

    def test_dtype_map_no(self):
        _mp = "no"
        _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
        dtype = _DTYPE_MAP.get(str(_mp).strip().lower(), torch.bfloat16)
        assert dtype == torch.float32

    def test_dtype_fallback_to_bf16_on_unknown(self):
        _mp = "unknown_value"
        _DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
        dtype = _DTYPE_MAP.get(str(_mp).strip().lower(), torch.bfloat16)
        assert dtype == torch.bfloat16

    def test_dtype_read_from_accelerate_cfg(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"accelerate": {"mixed_precision": "fp16"}})
        _mp = OmegaConf.select(cfg, "accelerate.mixed_precision", default="bf16")
        assert _mp == "fp16"

    def test_dtype_defaults_to_bf16_when_missing(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({})  # no accelerate.mixed_precision
        _mp = OmegaConf.select(cfg, "accelerate.mixed_precision", default="bf16")
        assert _mp == "bf16"


# ---------------------------------------------------------------------------
# 7. deploy.py — _log_attention_backends output format
# ---------------------------------------------------------------------------


class TestLogAttentionBackends:
    """Verify that attention backend diagnostics no longer emit ✗/SLOW markers."""

    def _import_deploy(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("deploy", PROJECT_ROOT / "scripts" / "deploy.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_no_slow_marker(self, caplog):
        import logging

        deploy = self._import_deploy()
        with caplog.at_level(logging.INFO):
            deploy._log_attention_backends(logging.getLogger("test_attn"))
        assert "✗" not in caplog.text, "✗ marker should have been removed"
        assert "SLOW" not in caplog.text, "SLOW annotation should have been removed"

    def test_no_install_hint(self, caplog):
        import logging

        deploy = self._import_deploy()
        with caplog.at_level(logging.INFO):
            deploy._log_attention_backends(logging.getLogger("test_attn"))
        assert "install flash-attn" not in caplog.text
        assert "pip install" not in caplog.text

    def test_three_subsystems_reported(self, caplog):
        import logging

        deploy = self._import_deploy()
        with caplog.at_level(logging.INFO):
            deploy._log_attention_backends(logging.getLogger("test_attn"))
        assert "ActionDiT" in caplog.text
        assert "Video DiT" in caplog.text
        assert "Wan shared core" in caplog.text

    def test_check_mark_absent_without_flash_attn(self, caplog):
        """Without flash-attn, torch_sdpa is the backend — no ✓ expected either."""
        import logging

        deploy = self._import_deploy()
        with caplog.at_level(logging.INFO):
            deploy._log_attention_backends(logging.getLogger("test_attn"))
        # Output is purely informational: just a name, no judgement symbols
        assert "✓" not in caplog.text


# ---------------------------------------------------------------------------
# 8. architecture.generate() — cudagraph_mark_step_begin placement
# ---------------------------------------------------------------------------


class TestCudagraphMarkStepBegin:
    """Structural check: every architecture forward dispatch in generate()
    must be preceded by ``torch.compiler.cudagraph_mark_step_begin()`` to
    prevent CUDA Graph tree from raising 'tensor output overwritten by
    subsequent run' when fixed-shape compile paths are active.

    After the architecture refactor the dispatch site is in
    ``BaseWAMArchitecture.generate()`` in ``base.py``.
    """

    _DISPATCH_SUBSTRINGS = (
        "noise_pred, action_noise_pred = self.forward(",
        "noise_pred = vb.finalize(state)",
        # §15 — CFG forward dispatch sites inside _forward_with_cfg
        "merged_noise, merged_action = self.forward(",
        "cond_noise, cond_action = self.forward(",
        "uncond_noise, uncond_action = self.forward(",
    )

    def _source(self):
        # Only check generate() method, not compute_loss()
        src = (PROJECT_ROOT / "openwam" / "model" / "base.py").read_text()
        marker = "def generate("
        idx = src.index(marker)
        return src[idx:]

    def test_mark_count_equals_dispatch_sites(self):
        src = self._source()
        mark_count = src.count("torch.compiler.cudagraph_mark_step_begin()")
        expected = sum(src.count(sub) for sub in self._DISPATCH_SUBSTRINGS)
        assert mark_count == expected, (
            f"Expected {expected} cudagraph_mark_step_begin() call(s), found {mark_count}. "
            "Add torch.compiler.cudagraph_mark_step_begin() before any new "
            "architecture forward dispatch."
        )

    def test_mark_appears_before_not_after(self):
        """The mark must appear on the line immediately before each dispatch site."""
        src = self._source()
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if any(sub in line for sub in self._DISPATCH_SUBSTRINGS):
                prev = i - 1
                while prev >= 0 and (lines[prev].strip() == "" or lines[prev].lstrip().startswith("#")):
                    prev -= 1
                assert "cudagraph_mark_step_begin" in lines[prev], (
                    f"Line {i + 1}: dispatch not preceded by "
                    f"cudagraph_mark_step_begin(). Found instead: {lines[prev]!r}"
                )
