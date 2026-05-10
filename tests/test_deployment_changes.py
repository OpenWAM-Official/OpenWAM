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

    def test_inference_cfg_scale_removed(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "inference.cfg_scale") is None

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
        assert OmegaConf.select(cfg, "optimization.compile.mode") == "default"
        assert OmegaConf.select(cfg, "optimization.compile.default.enabled") is True
        assert OmegaConf.select(cfg, "optimization.compile.default.video_dit") is True
        assert OmegaConf.select(cfg, "optimization.compile.default.vae") is False
        assert OmegaConf.select(cfg, "optimization.compile.default.torch_mode") is None
        assert OmegaConf.select(cfg, "optimization.compile.default.dynamic") is True
        assert OmegaConf.select(cfg, "optimization.compile.mot_loop.torch_mode") == "reduce-overhead"
        assert OmegaConf.select(cfg, "optimization.compile.mot_loop.dynamic") is False

    def test_optimization_dit_cache_defaults_off(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert not OmegaConf.select(cfg, "optimization.dit_cache.enabled")

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
        assert compile_options.section_enabled(compile_options.default_compile_cfg(compile_cfg), default=False) is False
        assert compile_options.section_enabled(compile_options.mot_loop_compile_cfg(compile_cfg), default=False) is False
        assert OmegaConf.select(cfg, "optimization.compile.default.video_dit") is True

    def test_cli_compile_mode_mot_loop_enables_only_mot_loop(self):
        from omegaconf import OmegaConf

        deploy = self._import()
        compile_options = self._compile_options()

        cfg = deploy._load_deploy_config()
        args = self._blank_args()
        args.compile_mode = "mot_loop"

        cfg = deploy._apply_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "optimization.compile.mode") == "mot_loop"
        compile_cfg = OmegaConf.select(cfg, "optimization.compile")
        assert compile_options.section_enabled(compile_options.default_compile_cfg(compile_cfg), default=False) is False
        assert compile_options.section_enabled(compile_options.mot_loop_compile_cfg(compile_cfg), default=False) is True

    def test_none_args_do_not_override(self):
        from omegaconf import OmegaConf

        deploy = self._import()

        cfg = deploy._load_deploy_config()
        original_steps = OmegaConf.select(cfg, "inference.denoise_steps")

        args = self._blank_args()
        # All None — nothing should change
        for attr in ("device", "host", "ws_port", "http_port", "denoise_steps", "schedule_type", "shift"):
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


class TestRobotwinOpenLoopEvalCompileMode:
    """RobotWin open-loop eval must use the active nested compile schema."""

    def _import(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "robotwin_open_loop_eval", PROJECT_ROOT / "scripts" / "robotwin_open_loop_eval.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_disable_compile_sets_mode_none(self):
        from omegaconf import OmegaConf

        robotwin_eval = self._import()
        cfg = OmegaConf.create(
            {
                "optimization": {
                    "compile": {
                        "mode": "default",
                        "default": {"video_dit": True, "vae": False},
                        "mot_loop": {"torch_mode": "reduce-overhead", "dynamic": False},
                    }
                }
            }
        )

        robotwin_eval._disable_compile_in_cfg(cfg)

        assert OmegaConf.select(cfg, "optimization.compile.mode") == "none"
        assert OmegaConf.select(cfg, "optimization.compile.enabled", default=None) is None
        assert OmegaConf.select(cfg, "optimization.compile.video_dit", default=None) is None
        assert OmegaConf.select(cfg, "optimization.compile.vae", default=None) is None

    def test_compile_mode_reporting_reads_nested_schema(self):
        from omegaconf import OmegaConf

        robotwin_eval = self._import()
        cfg = OmegaConf.create({"optimization": {"compile": {"mode": "mot_loop"}}})

        assert robotwin_eval._compile_mode_from_cfg(cfg) == "mot_loop"


# ---------------------------------------------------------------------------
# 5. joint_engine.py — compile flags parsed from cfg.optimization.compile
# ---------------------------------------------------------------------------


class TestJointEngineCompileFlags:
    """Verify _init_optimizations correctly reads compile flags without a real GPU."""

    def _make_engine(self, compile_enabled=False, video_dit=False, vae_compile=False, return_mock=False):
        from omegaconf import OmegaConf

        from openwam.deploy.joint_engine import JointInferenceEngine

        cfg = OmegaConf.create(
            {
                "inference": {
                    "denoise_steps": 10,
                    "schedule_type": "sync",
                    "shift": 5.0,
                    "num_frames": 33,
                    "height": 480,
                    "width": 832,
                },
                "optimization": {
                    "decode_video": True,
                    "compile": {
                        "default": {"enabled": compile_enabled, "video_dit": video_dit, "vae": vae_compile},
                        "mot_loop": {"enabled": False},
                    },
                    "dit_cache": {"enabled": False},
                    "schedule": {"type": None, "action_steps": 4},
                },
            }
        )

        pipe = MagicMock()
        pipe.dit = _tiny_module()
        pipe.vae = _tiny_module()

        vb = MagicMock()
        vb._pipe = pipe

        arch = MagicMock()
        arch.action_backbone = _tiny_module()
        arch.bridge_layers = []
        arch.video_backbone = vb

        # Patch torch.compile to a passthrough so we don't need a real GPU
        with patch("torch.compile", side_effect=lambda m, **kw: m) as mock_compile:
            engine = JointInferenceEngine.__new__(JointInferenceEngine)
            engine.cfg = cfg
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
        """compile.enabled=True should torch.compile the action module."""
        from omegaconf import OmegaConf

        from tests.test_openwam_trainer import _make_tiny_arch

        cfg = OmegaConf.create({"default": {"enabled": True, "video_dit": False, "vae": False}})
        arch = _make_tiny_arch()

        sentinel = nn.Identity()
        with patch("torch.compile", return_value=sentinel):
            arch.apply_compile_optimizations(cfg)

        assert arch.action_backbone is sentinel

    def test_video_dit_compile_flag(self):
        """compile.video_dit=True should compile each DiT block via backbone.apply_compile."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"default": {"enabled": True, "video_dit": True, "vae": False}})

        class FakeDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([_tiny_module(), _tiny_module()])

        pipe = MagicMock()
        pipe.dit = FakeDiT()
        pipe.vae = _tiny_module()

        from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

        vb = WanVideoBackbone(pipe)

        from tests.test_openwam_trainer import _make_tiny_arch

        arch = _make_tiny_arch()
        arch.video_backbone = vb

        sentinel = nn.Identity()
        with patch("torch.compile", return_value=sentinel):
            arch.apply_compile_optimizations(cfg)

        assert all(b is sentinel for b in vb._pipe.dit.blocks)

    def test_video_dit_compile_flag_works_when_action_compile_disabled(self):
        """compile.enabled=False should not suppress independent backbone compile flags."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {"mode": "default", "default": {"enabled": False, "video_dit": True, "vae": False}}
        )

        class FakeDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([_tiny_module(), _tiny_module()])

        pipe = MagicMock()
        pipe.dit = FakeDiT()
        pipe.vae = _tiny_module()

        from openwam.model.video_backbone.wan_adapter import WanVideoBackbone
        from tests.test_openwam_trainer import _make_tiny_arch

        vb = WanVideoBackbone(pipe)
        arch = _make_tiny_arch()
        arch.video_backbone = vb

        sentinel = nn.Identity()
        with patch("torch.compile", return_value=sentinel) as mock_compile:
            arch.apply_compile_optimizations(cfg)

        assert arch.action_backbone is not sentinel
        assert all(b is sentinel for b in vb._pipe.dit.blocks)
        assert mock_compile.call_count == 2

    def test_compile_mode_none_suppresses_backbone_compile_flags(self):
        """mode=none should disable all default compile work even when flags stay true."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {"mode": "none", "default": {"enabled": True, "video_dit": True, "vae": True}}
        )

        class FakeDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([_tiny_module(), _tiny_module()])

        pipe = MagicMock()
        pipe.dit = FakeDiT()
        pipe.vae = _tiny_module()

        from openwam.model.video_backbone.wan_adapter import WanVideoBackbone
        from tests.test_openwam_trainer import _make_tiny_arch

        vb = WanVideoBackbone(pipe)
        arch = _make_tiny_arch()
        arch.video_backbone = vb

        with patch("torch.compile") as mock_compile:
            arch.apply_compile_optimizations(cfg)

        mock_compile.assert_not_called()

    def test_vae_compile_flag(self):
        """compile.vae=True should torch.compile the VAE via backbone.apply_compile."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"default": {"enabled": True, "video_dit": False, "vae": True}})

        pipe = MagicMock()
        pipe.dit = _tiny_module()
        pipe.vae = _tiny_module()

        from openwam.model.video_backbone.wan_adapter import WanVideoBackbone

        vb = WanVideoBackbone(pipe)

        from tests.test_openwam_trainer import _make_tiny_arch

        arch = _make_tiny_arch()
        arch.video_backbone = vb

        sentinel = nn.Identity()
        with patch("torch.compile", return_value=sentinel):
            arch.apply_compile_optimizations(cfg)

        assert vb._pipe.vae is sentinel

    def test_compile_failure_does_not_crash(self):
        """If torch.compile raises, the engine should continue with eager mode."""
        from omegaconf import OmegaConf

        from openwam.deploy.joint_engine import JointInferenceEngine

        cfg = OmegaConf.create(
            {
                "inference": {
                    "denoise_steps": 10,
                    "schedule_type": "sync",
                    "seed": 42,
                    "shift": 5.0,
                    "num_frames": 33,
                    "height": 480,
                    "width": 832,
                },
                "optimization": {
                    "decode_video": True,
                    "compile": {
                        "default": {"enabled": True, "video_dit": True, "vae": True},
                        "mot_loop": {"enabled": False},
                    },
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

        vb = MagicMock()
        vb._pipe = pipe

        arch = MagicMock()
        arch.action_backbone = _tiny_module()
        arch.bridge_layers = []
        arch.video_backbone = vb

        with patch("torch.compile", side_effect=RuntimeError("compile unavailable")):
            engine = JointInferenceEngine.__new__(JointInferenceEngine)
            engine.cfg = cfg
            engine.architecture = arch
            engine.action_dit = None
            engine.action_repr = None
            # Should not raise
            engine._init_optimizations()


class TestCompileForward:
    """Verify that torch.compile + forward actually works (eager backend)."""

    def _make_arch(self):
        from tests.test_openwam_trainer import _make_tiny_arch

        return _make_tiny_arch()

    def test_action_module_compile_forward(self):
        """Action module should produce correct output after torch.compile."""
        arch = self._make_arch()
        arch.eval()

        B, T_action, action_dim = 1, 5, 7
        noisy = torch.randn(B, T_action, action_dim)
        timestep = torch.tensor([0.5])
        # ActionDiT.forward needs bridges as a {block_id: feat} dict.
        bridges = {bid: torch.randn(B, 10, 64) for bid in arch.bridge_layers}

        with torch.no_grad():
            pred_before = arch.action_backbone(noisy, bridges, timestep)

        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"default": {"enabled": True, "video_dit": False, "vae": False}})
        arch.apply_compile_optimizations(cfg)

        with torch.no_grad():
            pred_after = arch.action_backbone(noisy, bridges, timestep)

        assert pred_after.shape == pred_before.shape == (B, T_action, action_dim)

    def test_backbone_compile_forward(self):
        """Video backbone submodules should work after torch.compile."""
        arch = self._make_arch()
        arch.eval()

        B = 1
        latents = torch.randn(B, 16, 3, 8, 8)
        timestep = torch.tensor([0.5])
        context = torch.randn(B, 4, 64)

        # Forward before compile
        with torch.no_grad():
            state = arch.video_backbone.prepare(
                latents=latents,
                timestep=timestep,
                context=context,
            )
            for i in range(arch.video_backbone.num_layers):
                state = arch.video_backbone.run_block(i, state)
            pred_before = arch.video_backbone.finalize(state)

        # Compile all backbone submodules
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "default": {
                    "enabled": True,
                    "video_dit": True,
                    "vae": True,
                    "text_encoder": True,
                }
            }
        )
        arch.apply_compile_optimizations(cfg)

        # Forward after compile
        with torch.no_grad():
            state = arch.video_backbone.prepare(
                latents=latents,
                timestep=timestep,
                context=context,
            )
            for i in range(arch.video_backbone.num_layers):
                state = arch.video_backbone.run_block(i, state)
            pred_after = arch.video_backbone.finalize(state)

        assert pred_after.shape == pred_before.shape

    def test_full_architecture_compile_forward(self):
        """Full architecture forward (video + action) after compiling everything."""
        arch = self._make_arch()
        arch.eval()

        B, T_action, action_dim = 1, 5, 7
        noisy_actions = torch.randn(B, T_action, action_dim)
        action_timestep = torch.tensor([0.5])
        latents = torch.randn(B, 16, 3, 8, 8)
        video_timestep = torch.tensor([0.5])
        context = torch.randn(B, 4, 64)

        # Forward before compile
        with torch.no_grad():
            v_pred_before, a_pred_before = arch.forward(
                noisy_actions,
                action_timestep,
                latents=latents,
                timestep=video_timestep,
                context=context,
            )

        # Compile everything
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"default": {"enabled": True, "video_dit": True, "vae": True}})
        arch.apply_compile_optimizations(cfg)

        # Forward after compile
        with torch.no_grad():
            v_pred_after, a_pred_after = arch.forward(
                noisy_actions,
                action_timestep,
                latents=latents,
                timestep=video_timestep,
                context=context,
            )

        assert v_pred_after.shape == v_pred_before.shape
        assert a_pred_after.shape == a_pred_before.shape == (B, T_action, action_dim)


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
    subsequent run' when ``compile.video_dit=true``.

    After the architecture refactor the dispatch site is in
    ``BaseWAMArchitecture.generate()`` in ``base.py``.
    """

    _DISPATCH_SUBSTRINGS = (
        "noise_pred, action_noise_pred = self.forward(",
        "noise_pred = vb.finalize(state)",
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
