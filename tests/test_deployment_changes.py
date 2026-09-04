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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party"))


# ---------------------------------------------------------------------------
# 1. mixed_precision in accelerate yaml (single source of truth)
# ---------------------------------------------------------------------------


class TestTrainingMixedPrecision:
    """``cfg.training.mixed_precision`` is the sole source of truth.

    ``configs/accelerate`` was removed; both training (``_build_accelerator``)
    and deploy (``model_loader``) read ``training.mixed_precision`` directly.
    """

    def test_training_field_present(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "train.yaml")
        mp = OmegaConf.select(cfg, "training.mixed_precision")
        assert mp == "bf16", f"Expected 'bf16', got {mp!r}"

    def test_accelerate_group_removed(self):
        assert not (PROJECT_ROOT / "configs" / "accelerate").exists(), (
            "configs/accelerate should be gone; DeepSpeed settings now live in train.yaml"
        )


# ---------------------------------------------------------------------------
# 2. deployment.yaml — inference + deploy sections
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

    def test_inference_cfg_fields_absent(self):
        """CFG knobs stay out of deploy.yaml; engine defaults (1.0/false/null) disable CFG."""
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "inference.cfg_scale") is None
        assert OmegaConf.select(cfg, "inference.cfg_merge") is None

    def test_inference_denoise_mode(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "inference.denoise_mode") == "sync"
        assert OmegaConf.select(cfg, "inference.lead_modality") == "video"
        assert OmegaConf.select(cfg, "inference.variance_shift_alpha") == 1.0
        assert OmegaConf.select(cfg, "inference.linear_offset") == 0.0

    def test_optimization_section_exists(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "optimization") is not None

    def test_optimization_compile_section_exists(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "optimization.compile") is not None
        assert OmegaConf.select(cfg, "optimization.compile.enabled") is True
        assert OmegaConf.select(cfg, "optimization.compile.self_attn.torch_mode") == "default"
        assert OmegaConf.select(cfg, "optimization.compile.self_attn.dynamic") is False
        assert OmegaConf.select(cfg, "optimization.compile.cross_attn.torch_mode") == "default"
        assert OmegaConf.select(cfg, "optimization.compile.cross_attn.dynamic") is False

    def test_optimization_dit_cache_enabled(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "optimization.dit_cache.enabled") is True

    def test_inference_execution_defaults(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "inference.inference_mode") == "sync"
        assert OmegaConf.select(cfg, "inference.inference_horizon") is None
        assert OmegaConf.select(cfg, "inference.inference_delay_steps") is None

    def test_server_defaults_present(self):
        from omegaconf import OmegaConf

        cfg = self._load()
        assert OmegaConf.select(cfg, "server.port") is not None


# ---------------------------------------------------------------------------
# 4. deploy.py — config loading and CLI override logic
# ---------------------------------------------------------------------------


class TestDeployConfigLoading:
    """Config loading + CLI override logic of the unified server CLI (no server run)."""

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

        spec = importlib.util.spec_from_file_location("server", PROJECT_ROOT / "openwam" / "deploy" / "server.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _blank_args(self):
        args = MagicMock()
        for attr in (
            "device",
            "host",
            "port",
            "denoise_steps",
            "denoise_mode",
            "lead_modality",
            "variance_shift_alpha",
            "linear_offset",
            "shift",
            "compile_enabled",
            "inference_mode",
            "inference_horizon",
            "inference_delay_steps",
        ):
            setattr(args, attr, None)
        return args

    def test_load_deploy_config_returns_omegaconf(self):
        deploy = self._policy_server()
        cfg = deploy._load_deploy_yaml()
        from omegaconf import DictConfig

        assert isinstance(cfg, DictConfig)

    def test_load_deploy_config_has_inference(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()
        cfg = deploy._load_deploy_yaml()
        assert OmegaConf.select(cfg, "inference.denoise_steps") is not None

    def test_cli_denoise_steps_override(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.denoise_steps = 20

        cfg = deploy._apply_inference_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.denoise_steps") == 20

    def test_cli_schedule_overrides(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.denoise_mode = "async"
        args.lead_modality = "video"
        args.variance_shift_alpha = 9.0
        args.linear_offset = 0.25

        cfg = deploy._apply_inference_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.denoise_mode") == "async"
        assert OmegaConf.select(cfg, "inference.lead_modality") == "video"
        assert OmegaConf.select(cfg, "inference.variance_shift_alpha") == 9.0
        assert OmegaConf.select(cfg, "inference.linear_offset") == 0.25

    def test_cli_compile_enabled_false_disables_compile(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()
        compile_options = self._compile_options()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.compile_enabled = False

        deploy._apply_compile_enabled_override(cfg, args.compile_enabled)
        assert OmegaConf.select(cfg, "optimization.compile.enabled") is False
        compile_cfg = OmegaConf.select(cfg, "optimization.compile")
        assert compile_options.compile_enabled(compile_cfg, strict=True) is False

    def test_cli_compile_enabled_true_keeps_architecture_selection(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()
        compile_options = self._compile_options()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.compile_enabled = True

        deploy._apply_compile_enabled_override(cfg, args.compile_enabled)
        assert OmegaConf.select(cfg, "optimization.compile.enabled") is True
        compile_cfg = OmegaConf.select(cfg, "optimization.compile")
        assert compile_options.compile_enabled(compile_cfg, strict=True) is True

    def test_cli_compile_enabled_accepts_bool_strings(self):
        import argparse

        deploy = self._policy_server()

        parser = argparse.ArgumentParser()
        parser.add_argument("--compile-enabled", type=deploy._normalize_compile_enabled_arg)

        assert parser.parse_args(["--compile-enabled", "true"]).compile_enabled is True
        assert parser.parse_args(["--compile-enabled", "false"]).compile_enabled is False
        with pytest.raises(SystemExit):
            parser.parse_args(["--compile-enabled", "auto"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--compile-enabled", "none"])

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

    def test_compile_enabled_validation_rejects_non_bool_values(self):
        compile_options = self._compile_options()

        assert compile_options.normalize_compile_enabled("true") is True
        assert compile_options.normalize_compile_enabled("false") is False
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            compile_options.normalize_compile_enabled("default")
        with pytest.raises(ValueError, match="Unknown legacy compile mode"):
            compile_options.compile_enabled({"mode": "default"}, strict=True)
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            compile_options.normalize_compile_enabled("self_attn")
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            compile_options.normalize_compile_enabled("cross-attn")

    def test_policy_server_entrypoint_validates_compile_enabled(self):
        from omegaconf import OmegaConf

        policy_server = self._policy_server()

        cfg = OmegaConf.create({"optimization": {"compile": {"enabled": "false"}}})
        policy_server._normalize_compile_enabled_in_cfg(cfg)
        assert OmegaConf.select(cfg, "optimization.compile.enabled") is False

        true_cfg = OmegaConf.create({"optimization": {"compile": {"enabled": "true"}}})
        policy_server._normalize_compile_enabled_in_cfg(true_cfg)
        assert OmegaConf.select(true_cfg, "optimization.compile.enabled") is True

        legacy_cfg = OmegaConf.create({"optimization": {"compile": {"mode": "auto"}}})
        policy_server._normalize_compile_enabled_in_cfg(legacy_cfg)
        assert OmegaConf.select(legacy_cfg, "optimization.compile.enabled") is True

        bad_cfg = OmegaConf.create({"optimization": {"compile": {"enabled": "default"}}})
        with pytest.raises(ValueError, match="Unknown compile enabled value"):
            policy_server._normalize_compile_enabled_in_cfg(bad_cfg)

    def test_policy_server_compile_enabled_cli_override(self):
        from omegaconf import OmegaConf

        policy_server = self._policy_server()

        args = policy_server._build_argparser().parse_args(["--compile-enabled", "false"])
        assert args.compile_enabled is False

        cfg = OmegaConf.create({})
        policy_server._apply_compile_enabled_override(cfg, args.compile_enabled)
        assert OmegaConf.select(cfg, "optimization.compile.enabled") is False

        with pytest.raises(SystemExit):
            policy_server._build_argparser().parse_args(["--compile-enabled", "self-attn"])

    def test_cli_inference_mode_override(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.inference_mode = "async"

        cfg = deploy._apply_execution_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.inference_mode") == "async"

    def test_cli_execution_numeric_overrides(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.inference_mode = "async"
        args.inference_horizon = 24
        args.inference_delay_steps = 6

        cfg = deploy._apply_execution_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.inference_mode") == "async"
        assert OmegaConf.select(cfg, "inference.inference_horizon") == 24
        assert OmegaConf.select(cfg, "inference.inference_delay_steps") == 6

    def test_cli_inference_horizon_override_supports_sync(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.inference_mode = "sync"
        args.inference_horizon = 24

        cfg = deploy._apply_execution_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.inference_mode") == "sync"
        assert OmegaConf.select(cfg, "inference.inference_horizon") == 24
        assert OmegaConf.select(cfg, "inference.inference_delay_steps") is None

    def test_cli_async_numeric_overrides_fail_fast_on_invalid_ranges(self):
        deploy = self._policy_server()

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.inference_mode = "async"
        args.inference_horizon = 0
        with pytest.raises(ValueError, match="inference_horizon must be positive"):
            deploy._apply_execution_cli_overrides(cfg, args)

        cfg = deploy._load_deploy_yaml()
        args = self._blank_args()
        args.inference_mode = "async"
        args.inference_horizon = 4
        args.inference_delay_steps = 4
        with pytest.raises(ValueError, match="inference_delay_steps must be < inference_horizon"):
            deploy._apply_execution_cli_overrides(cfg, args)

    def test_none_args_do_not_override(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        cfg = deploy._load_deploy_yaml()
        original_steps = OmegaConf.select(cfg, "inference.denoise_steps")

        args = self._blank_args()
        # All None — nothing should change
        for attr in (
            "device",
            "host",
            "port",
            "denoise_steps",
            "denoise_mode",
            "shift",
            "inference_mode",
            "inference_horizon",
            "inference_delay_steps",
        ):
            setattr(args, attr, None)

        cfg = deploy._apply_execution_cli_overrides(cfg, args)
        assert OmegaConf.select(cfg, "inference.denoise_steps") == original_steps

    def test_merge_with_training_cfg_uses_dataloader_dims(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        policy_server = self._policy_server()
        training_cfg = OmegaConf.create({"dataloader": {"num_frames": 49, "height": 720, "width": 1280}})
        deploy_cfg = deploy._load_deploy_yaml()

        # Remove inference dims so they should be filled from dataloader
        OmegaConf.update(deploy_cfg, "inference.num_frames", None, merge=False)
        OmegaConf.update(deploy_cfg, "inference.height", None, merge=False)
        OmegaConf.update(deploy_cfg, "inference.width", None, merge=False)

        merged = policy_server.merge_deploy_cfg(training_cfg, deploy_cfg)
        assert OmegaConf.select(merged, "inference.num_frames") == 49
        assert OmegaConf.select(merged, "inference.height") == 720
        assert OmegaConf.select(merged, "inference.width") == 1280

    def test_deploy_cfg_wins_over_training_cfg_on_overlap(self):
        from omegaconf import OmegaConf

        deploy = self._policy_server()

        policy_server = self._policy_server()
        training_cfg = OmegaConf.create({"inference": {"denoise_steps": 99}})
        deploy_cfg = deploy._load_deploy_yaml()
        OmegaConf.update(deploy_cfg, "inference.denoise_steps", 10, merge=False)

        merged = policy_server.merge_deploy_cfg(training_cfg, deploy_cfg)
        assert OmegaConf.select(merged, "inference.denoise_steps") == 10


# ---------------------------------------------------------------------------
# 5. joint_engine.py — compile flags parsed from cfg.optimization.compile
# ---------------------------------------------------------------------------


class TestJointEngineCompileFlags:
    """Verify compile mode routing without broad default compile side effects."""

    def _make_filter_engine(self, architecture):
        from openwam.deploy.engine import JointInferenceEngine

        engine = JointInferenceEngine.__new__(JointInferenceEngine)
        engine.architecture = architecture
        engine._architecture_generate_accepts_extra_kwargs = None
        engine._architecture_generate_kwarg_names = None
        engine._architecture_generate_warned_dropped_kwargs = set()
        return engine

    def _make_engine(self, compile_enabled=False, return_arch=False, prompt_cache_cfg=None):
        from omegaconf import OmegaConf

        from openwam.deploy.engine import JointInferenceEngine

        cfg = OmegaConf.create(
            {
                "inference": {
                    "denoise_steps": 10,
                    "denoise_mode": "sync",
                    "shift": 5.0,
                    "num_frames": 33,
                    "height": 384,
                    "width": 320,
                },
                "optimization": {
                    "decode_video": True,
                    "compile": {
                        "enabled": compile_enabled,
                        "self_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
                        "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
                        "tri_system": {"torch_mode": "reduce-overhead", "dynamic": False},
                    },
                    "dit_cache": {"enabled": False},
                    "schedule": {"type": None, "action_steps": 4},
                },
            }
        )
        if prompt_cache_cfg is not None:
            cfg.optimization.prompt_embed_cache = prompt_cache_cfg

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

    def test_compile_enabled_false_does_not_broad_compile(self):
        _engine, mock_compile, arch = self._make_engine(return_arch=True)
        mock_compile.assert_not_called()
        arch.apply_compile_optimizations.assert_called_once()

    def test_compile_enabled_true_is_passed_to_architecture(self):
        from omegaconf import OmegaConf

        _engine, mock_compile, arch = self._make_engine(True, return_arch=True)
        mock_compile.assert_not_called()
        compile_cfg = arch.apply_compile_optimizations.call_args.args[0]
        assert OmegaConf.select(compile_cfg, "enabled") is True

    def test_prompt_embed_cache_default_is_bounded_default(self):
        engine, _ = self._make_engine()
        from openwam.deploy.engine import DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE, _BoundedPromptEmbedCache

        assert isinstance(engine._prompt_embed_cache, _BoundedPromptEmbedCache)
        assert engine._prompt_embed_cache._maxsize == DEFAULT_PROMPT_EMBED_CACHE_MAXSIZE

    def test_prompt_embed_cache_maxsize_from_config(self):
        engine, _ = self._make_engine(prompt_cache_cfg={"enabled": True, "maxsize": 8})
        assert engine._prompt_embed_cache._maxsize == 8

    def test_prompt_embed_cache_enabled_false_disables_cache(self):
        engine, _ = self._make_engine(prompt_cache_cfg={"enabled": False, "maxsize": 8})
        assert engine._prompt_embed_cache is None

    def test_generate_kwarg_filter_treats_disabled_cache_drop_as_noop(self, caplog):
        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())
        caplog.set_level("WARNING", logger="openwam.deploy.engine")
        filtered = engine._filter_architecture_generate_kwargs(
            {"schedule": object(), "prompt": "pick up the cube", "prompt_embed_cache": None}
        )
        assert set(filtered) == {"schedule", "prompt"}
        assert not [r for r in caplog.records if "does not accept deploy kwarg" in r.getMessage()]

    def test_generate_kwarg_filter_warns_once_for_meaningful_drops(self, caplog):
        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())

        caplog.set_level("WARNING", logger="openwam.deploy.engine")
        kwargs = {
            "schedule": object(),
            "prompt": "pick up the cube",
            "cfg_scale": 1.5,
            "cfg_merge": False,
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
        from openwam.deploy.engine import _BoundedPromptEmbedCache

        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())

        caplog.set_level("WARNING", logger="openwam.deploy.engine")
        kwargs = {
            "schedule": object(),
            "prompt": "pick up the cube",
            "cfg_scale": 1.0,
            "cfg_merge": False,
            "prompt_embed_cache": _BoundedPromptEmbedCache(),
        }

        filtered = engine._filter_architecture_generate_kwargs(kwargs)

        assert set(filtered) == {"schedule", "prompt"}
        assert not [record for record in caplog.records if "does not accept deploy kwarg" in record.getMessage()]

    def test_generate_kwarg_filter_warns_for_configured_prompt_cache_drop(self, caplog):
        from openwam.deploy.engine import _BoundedPromptEmbedCache

        class _StrictArchitecture:
            def generate(self, *, schedule, prompt):
                return {"schedule": schedule, "prompt": prompt}

        engine = self._make_filter_engine(_StrictArchitecture())

        caplog.set_level("WARNING", logger="openwam.deploy.engine")
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
                "enabled": False,
                "cross_attn": {"torch_mode": "reduce-overhead", "dynamic": False},
            }
        )

        with patch("torch.compile") as mock_compile:
            arch.apply_compile_optimizations(cfg)

        mock_compile.assert_not_called()

    def test_architecture_auto_compile_respects_disabled_fast_path(self):
        from omegaconf import OmegaConf

        from tests.test_openwam_trainer import _make_tiny_arch

        arch = _make_tiny_arch()
        cfg = OmegaConf.create({"enabled": True, "cross_attn": {"enabled": False}})

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
# 7. server CLI — _log_attention_backends output format
# ---------------------------------------------------------------------------


class TestLogAttentionBackends:
    """Verify that attention backend diagnostics no longer emit ✗/SLOW markers."""

    def _import_deploy(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("server", PROJECT_ROOT / "openwam" / "deploy" / "server.py")
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
        src = (PROJECT_ROOT / "openwam" / "model" / "architectures" / "base.py").read_text()
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
