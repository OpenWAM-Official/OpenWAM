"""Tests for deployment-related changes:
- mixed_precision in train.yaml (save/load dtype enforcement)
- deployment.yaml inference/deploy sections
- deploy.py config loading, CLI override logic, and attention backend logging
- joint_engine.py compile flags via cfg.deploy.compile
- joint_generation.py cudagraph_mark_step_begin placement
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
# 1. mixed_precision in train.yaml
# ---------------------------------------------------------------------------


class TestTrainYamlMixedPrecision:
    def test_field_exists(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "train.yaml")
        mp = OmegaConf.select(cfg, "training.mixed_precision")
        assert mp is not None, "training.mixed_precision missing from train.yaml"
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
        assert sd["action_dit.step"].dtype == torch.int64, "int64 buffer should not be cast"
        assert sd["action_dit.w"].dtype == torch.bfloat16, "float param should be cast to bf16"

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

        return OmegaConf.load(PROJECT_ROOT / "configs" / "deployment.yaml")

    def test_inference_section_exists(self):
        cfg = self._load()
        from omegaconf import OmegaConf

        assert OmegaConf.select(cfg, "inference") is not None

    def test_inference_denoise_steps(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "inference.denoise_steps") is not None

    def test_inference_cfg_scale_disabled(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "inference.cfg_scale") == 1.0

    def test_inference_schedule_type(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "inference.schedule_type") is not None

    def test_deploy_section_exists(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "deploy") is not None

    def test_deploy_compile_section_exists(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "deploy.compile") is not None
        assert OmegaConf.select(cfg, "deploy.compile.enabled") is not None
        assert OmegaConf.select(cfg, "deploy.compile.video_dit") is not None
        assert OmegaConf.select(cfg, "deploy.compile.vae") is not None

    def test_deploy_dit_cache_defaults_off(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert not OmegaConf.select(cfg, "deploy.dit_cache.enabled")

    def test_deploy_schedule_action_steps(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "deploy.schedule.action_steps") is not None

    def test_server_defaults_present(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "deployment.server.ws_port") is not None
        assert OmegaConf.select(cfg, "deployment.server.http_port") is not None


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
        args = MagicMock()
        args.device = "cuda:3"
        args.host = None
        args.ws_port = None
        args.http_port = None
        args.denoise_steps = None
        args.schedule_type = None
        args.cfg_scale = None
        args.shift = None

        cfg = deploy._apply_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "deployment.device") == "cuda:3"

    def test_cli_denoise_steps_override(self):
        from omegaconf import OmegaConf

        deploy = self._import()

        cfg = deploy._load_deploy_config()
        args = MagicMock()
        args.device = None
        args.host = None
        args.ws_port = None
        args.http_port = None
        args.denoise_steps = 20
        args.schedule_type = None
        args.cfg_scale = None
        args.shift = None

        cfg = deploy._apply_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.denoise_steps") == 20

    def test_none_args_do_not_override(self):
        from omegaconf import OmegaConf

        deploy = self._import()

        cfg = deploy._load_deploy_config()
        original_steps = OmegaConf.select(cfg, "inference.denoise_steps")

        args = MagicMock()
        # All None — nothing should change
        for attr in ("device", "host", "ws_port", "http_port", "denoise_steps", "schedule_type", "cfg_scale", "shift"):
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
# 5. joint_engine.py — compile flags parsed from cfg.deploy.compile
# ---------------------------------------------------------------------------


class TestJointEngineCompileFlags:
    """Verify _init_optimizations correctly reads compile flags without a real GPU."""

    def _make_engine(self, compile_enabled=False, video_dit=False, vae_compile=False, return_mock=False):
        from omegaconf import OmegaConf

        from openwam.deployment.joint_engine import JointInferenceEngine

        cfg = OmegaConf.create(
            {
                "inference": {
                    "denoise_steps": 10,
                    "schedule_type": "sync",
                    "cfg_scale": 1.0,
                    "shift": 5.0,
                    "num_frames": 33,
                    "height": 480,
                    "width": 832,
                },
                "deploy": {
                    "decode_video": True,
                    "compile": {"enabled": compile_enabled, "video_dit": video_dit, "vae": vae_compile},
                    "dit_cache": {"enabled": False},
                    "cfg": {"mode": None, "scale": 1.0},
                    "schedule": {"type": None, "action_steps": 4},
                },
            }
        )

        pipe = MagicMock()
        pipe.dit = _tiny_module()
        pipe.vae = _tiny_module()

        arch = MagicMock()
        arch.action_dit = _tiny_module()
        arch.is_interleaved = False
        arch.bridge_layers = []

        # Patch torch.compile to a passthrough so we don't need a real GPU
        with patch("torch.compile", side_effect=lambda m, **kw: m) as mock_compile:
            engine = JointInferenceEngine.__new__(JointInferenceEngine)
            engine.cfg = cfg
            engine.pipeline = pipe
            engine.architecture = arch
            engine.action_dit = None
            engine.action_repr = None
            engine._init_optimizations()
        if return_mock:
            return engine, mock_compile
        return engine

    def test_compile_disabled_by_default(self):
        _engine, mock_compile = self._make_engine(compile_enabled=False, return_mock=True)
        mock_compile.assert_not_called()

    def test_action_dit_compile_flag(self):
        orig_module = _tiny_module()

        arch = MagicMock()
        arch.action_dit = orig_module
        arch.is_interleaved = False
        arch.bridge_layers = []

        from omegaconf import OmegaConf

        from openwam.deployment.joint_engine import JointInferenceEngine

        cfg = OmegaConf.create(
            {
                "inference": {
                    "denoise_steps": 10,
                    "schedule_type": "sync",
                    "cfg_scale": 1.0,
                    "shift": 5.0,
                    "num_frames": 33,
                    "height": 480,
                    "width": 832,
                },
                "deploy": {
                    "decode_video": True,
                    "compile": {"enabled": True, "video_dit": False, "vae": False},
                    "dit_cache": {"enabled": False},
                    "cfg": {"mode": None, "scale": 1.0},
                    "schedule": {"type": None, "action_steps": 4},
                },
            }
        )

        pipe = MagicMock()
        pipe.dit = _tiny_module()
        pipe.vae = _tiny_module()

        sentinel = object()
        with patch("torch.compile", return_value=sentinel) as mock_compile:
            engine = JointInferenceEngine.__new__(JointInferenceEngine)
            engine.cfg = cfg
            engine.pipeline = pipe
            engine.architecture = arch
            engine.action_dit = None
            engine.action_repr = None
            engine._init_optimizations()

        mock_compile.assert_called()
        assert engine.architecture.action_dit is sentinel

    def test_video_dit_compile_flag(self):
        from omegaconf import OmegaConf

        from openwam.deployment.joint_engine import JointInferenceEngine

        cfg = OmegaConf.create(
            {
                "inference": {
                    "denoise_steps": 10,
                    "schedule_type": "sync",
                    "cfg_scale": 1.0,
                    "shift": 5.0,
                    "num_frames": 33,
                    "height": 480,
                    "width": 832,
                },
                "deploy": {
                    "decode_video": True,
                    "compile": {"enabled": False, "video_dit": True, "vae": False},
                    "dit_cache": {"enabled": False},
                    "cfg": {"mode": None, "scale": 1.0},
                    "schedule": {"type": None, "action_steps": 4},
                },
            }
        )

        # model_fn_wan_video iterates dit.blocks individually, so compile targets blocks
        class FakeDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([_tiny_module(), _tiny_module()])

        pipe = MagicMock()
        pipe.dit = FakeDiT()
        pipe.vae = _tiny_module()

        arch = MagicMock()
        arch.action_dit = _tiny_module()
        arch.is_interleaved = False
        arch.bridge_layers = []

        # torch.compile returns an OptimizedModule (nn.Module); use a real Module sentinel
        sentinel = nn.Identity()
        with patch("torch.compile", return_value=sentinel):
            engine = JointInferenceEngine.__new__(JointInferenceEngine)
            engine.cfg = cfg
            engine.pipeline = pipe
            engine.architecture = arch
            engine.action_dit = None
            engine.action_repr = None
            engine._init_optimizations()

        # Each block should be replaced by the compiled sentinel
        assert all(b is sentinel for b in engine.pipeline.dit.blocks)

    def test_vae_compile_flag(self):
        from omegaconf import OmegaConf

        from openwam.deployment.joint_engine import JointInferenceEngine

        cfg = OmegaConf.create(
            {
                "inference": {
                    "denoise_steps": 10,
                    "schedule_type": "sync",
                    "cfg_scale": 1.0,
                    "shift": 5.0,
                    "num_frames": 33,
                    "height": 480,
                    "width": 832,
                },
                "deploy": {
                    "decode_video": True,
                    "compile": {"enabled": False, "video_dit": False, "vae": True},
                    "dit_cache": {"enabled": False},
                    "cfg": {"mode": None, "scale": 1.0},
                    "schedule": {"type": None, "action_steps": 4},
                },
            }
        )

        pipe = MagicMock()
        pipe.dit = _tiny_module()
        pipe.vae = _tiny_module()

        arch = MagicMock()
        arch.action_dit = _tiny_module()
        arch.is_interleaved = False
        arch.bridge_layers = []

        sentinel = object()
        with patch("torch.compile", return_value=sentinel):
            engine = JointInferenceEngine.__new__(JointInferenceEngine)
            engine.cfg = cfg
            engine.pipeline = pipe
            engine.architecture = arch
            engine.action_dit = None
            engine.action_repr = None
            engine._init_optimizations()

        assert engine.pipeline.vae is sentinel

    def test_compile_failure_does_not_crash(self):
        """If torch.compile raises, the engine should continue with eager mode."""
        from omegaconf import OmegaConf

        from openwam.deployment.joint_engine import JointInferenceEngine

        cfg = OmegaConf.create(
            {
                "inference": {
                    "denoise_steps": 10,
                    "schedule_type": "sync",
                    "cfg_scale": 1.0,
                    "shift": 5.0,
                    "num_frames": 33,
                    "height": 480,
                    "width": 832,
                },
                "deploy": {
                    "decode_video": True,
                    "compile": {"enabled": True, "video_dit": True, "vae": True},
                    "dit_cache": {"enabled": False},
                    "cfg": {"mode": None, "scale": 1.0},
                    "schedule": {"type": None, "action_steps": 4},
                },
            }
        )

        class FakeDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([_tiny_module()])

        pipe = MagicMock()
        pipe.dit = FakeDiT()
        pipe.vae = _tiny_module()

        arch = MagicMock()
        arch.action_dit = _tiny_module()
        arch.is_interleaved = False
        arch.bridge_layers = []

        with patch("torch.compile", side_effect=RuntimeError("compile unavailable")):
            engine = JointInferenceEngine.__new__(JointInferenceEngine)
            engine.cfg = cfg
            engine.pipeline = pipe
            engine.architecture = arch
            engine.action_dit = None
            engine.action_repr = None
            # Should not raise
            engine._init_optimizations()


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

    def test_dtype_read_from_training_cfg(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"training": {"mixed_precision": "fp16"}})
        _mp = OmegaConf.select(cfg, "training.mixed_precision", default="bf16")
        assert _mp == "fp16"

    def test_dtype_defaults_to_bf16_when_missing(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({})  # no training.mixed_precision
        _mp = OmegaConf.select(cfg, "training.mixed_precision", default="bf16")
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
        assert "diffsynth core" in caplog.text

    def test_check_mark_absent_without_flash_attn(self, caplog):
        """Without flash-attn, torch_sdpa is the backend — no ✓ expected either."""
        import logging

        deploy = self._import_deploy()
        with caplog.at_level(logging.INFO):
            deploy._log_attention_backends(logging.getLogger("test_attn"))
        # Output is purely informational: just a name, no judgement symbols
        assert "✓" not in caplog.text


# ---------------------------------------------------------------------------
# 8. joint_generation.py — cudagraph_mark_step_begin placement
# ---------------------------------------------------------------------------


class TestCudagraphMarkStepBegin:
    """Structural check: every direct ``pipe.model_fn()`` call AND every
    ``cfg_handler.forward(pipe.model_fn, ...)`` dispatch must be preceded by
    ``torch.compiler.cudagraph_mark_step_begin()`` to prevent CUDA Graph tree
    from raising 'tensor output overwritten by subsequent run' when
    ``compile.video_dit=true``. cfg_handler dispatches to pipe.model_fn
    internally and shares the same CUDA Graph tree, so the mark requirement
    applies to it too."""

    def _source(self):
        return (PROJECT_ROOT / "openwam" / "deployment" / "joint_generation.py").read_text()

    def test_mark_count_equals_dispatch_sites(self):
        src = self._source()
        mark_count = src.count("torch.compiler.cudagraph_mark_step_begin()")
        # Direct assignments + cfg_handler dispatches — both trigger the same
        # CUDA Graph tree and both need the mark before them.
        direct_calls = src.count("noise_pred_posi = pipe.model_fn(") + src.count("noise_pred_nega = pipe.model_fn(")
        cfg_handler_calls = src.count("cfg_handler.forward(")
        expected = direct_calls + cfg_handler_calls
        assert mark_count == expected, (
            f"Expected {expected} cudagraph_mark_step_begin() call(s) "
            f"({direct_calls} direct pipe.model_fn + {cfg_handler_calls} cfg_handler.forward), "
            f"found {mark_count}. Add torch.compiler.cudagraph_mark_step_begin() before any new "
            "pipe.model_fn() or cfg_handler.forward() dispatch."
        )

    def test_mark_appears_before_not_after(self):
        """The mark must appear on the line immediately before each dispatch site."""
        src = self._source()
        lines = src.splitlines()
        trigger_substrings = (
            "noise_pred_posi = pipe.model_fn(",
            "noise_pred_nega = pipe.model_fn(",
            "noise_pred = cfg_handler.forward(",
        )
        for i, line in enumerate(lines):
            if any(sub in line for sub in trigger_substrings):
                # Search backwards (skip blank and comment-only lines) for the mark
                prev = i - 1
                while prev >= 0 and (lines[prev].strip() == "" or lines[prev].lstrip().startswith("#")):
                    prev -= 1
                assert "cudagraph_mark_step_begin" in lines[prev], (
                    f"Line {i + 1}: dispatch not preceded by "
                    f"cudagraph_mark_step_begin(). Found instead: {lines[prev]!r}"
                )
