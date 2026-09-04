"""Config-parsing tests for benchmarks/robocasa365/single_eval.py.

Only the pure config helpers are tested here; ``run_eval`` needs the robosuite
sim (separate env) and is exercised by the user-run smoke. ``single_eval.py`` is
loaded under a unique module name so it does not collide with the identically
named ``benchmarks/libero/single_eval.py`` in the same pytest session.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "robocasa365" / "single_eval.py"
# single_eval.py does `from openwam2robocasa365_interface import ...`, so its dir
# must be importable.
if str(_SE_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_SE_PATH.parent))
_spec = importlib.util.spec_from_file_location("robocasa365_single_eval", _SE_PATH)
single_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(single_eval)


def test_parse_bool_variants():
    assert single_eval._parse_bool(True, "x") is True
    assert single_eval._parse_bool("yes", "x") is True
    assert single_eval._parse_bool("On", "x") is True
    assert single_eval._parse_bool("off", "x") is False
    assert single_eval._parse_bool("", "x") is False
    with pytest.raises(ValueError):
        single_eval._parse_bool("maybe", "x")


def test_normalize_optional():
    assert single_eval._normalize_optional("none") is None
    assert single_eval._normalize_optional("null") is None
    assert single_eval._normalize_optional("") is None
    assert single_eval._normalize_optional("video.robot0_agentview_left") == "video.robot0_agentview_left"
    assert single_eval._normalize_optional(None) is None


def test_parse_optional_int():
    assert single_eval._parse_optional_int("16", "x") == 16
    assert single_eval._parse_optional_int(16, "x") == 16
    assert single_eval._parse_optional_int("none", "x") is None
    with pytest.raises(ValueError):
        single_eval._parse_optional_int("0", "x")
    with pytest.raises(ValueError):
        single_eval._parse_optional_int("-3", "x")


def test_load_config_roundtrip(tmp_path):
    p = tmp_path / "c.yml"
    p.write_text("task: OpenDrawer\nport: 8848\nstate_dim: 16\n", encoding="utf-8")
    cfg = single_eval._load_config(p)
    assert cfg["task"] == "OpenDrawer"
    assert cfg["port"] == 8848
    assert cfg["state_dim"] == 16


def test_resolve_max_steps_uses_official_robocasa_registry(monkeypatch):
    import types

    registry_utils = types.ModuleType("robocasa.utils.dataset_registry_utils")
    registry_utils.get_task_horizon = lambda task: {"OpenDrawer": 750}[task]
    monkeypatch.setitem(sys.modules, "robocasa", types.ModuleType("robocasa"))
    monkeypatch.setitem(sys.modules, "robocasa.utils", types.ModuleType("robocasa.utils"))
    monkeypatch.setitem(sys.modules, "robocasa.utils.dataset_registry_utils", registry_utils)

    assert single_eval._resolve_max_steps({"task": "OpenDrawer"}) == 750


def test_resolve_max_steps_explicit_smoke_override_needs_no_robocasa():
    assert single_eval._resolve_max_steps({"task": "OpenDrawer", "max_steps_override": 12}) == 12
    with pytest.raises(ValueError, match="max_steps_override"):
        single_eval._resolve_max_steps({"task": "OpenDrawer", "max_steps_override": 0})


def test_rollout_accumulates_success_and_terminates():
    """_rollout drives env+policy via stubs (no sim/server): trial 0 succeeds at
    step 2, trial 1 never does -> 1 success; policy.act called the right #times.
    Mirrors robotwin's eval-loop test (stub env + stub policy)."""

    class _FakeEnv:
        def __init__(self):
            self.trial = -1
            self.t = 0

        def reset(self, seed=None):
            self.trial += 1
            self.t = 0
            return ({"annotation.human.task_description": "x"}, {"success": False})

        def step(self, action):
            self.t += 1
            done = self.trial == 0 and self.t >= 2  # only trial 0 succeeds, at step 2
            return ({"annotation.human.task_description": "x"}, 1.0 if done else 0.0, done, False, {"success": done})

    class _FakePolicy:
        def __init__(self):
            self.acts = 0

        def reset(self):
            pass

        def act(self, obs, prompt):
            self.acts += 1
            return {"action.dummy": 0}

    env, pol = _FakeEnv(), _FakePolicy()
    successes = single_eval._rollout(env, pol, num_trials=2, max_steps=5, seed=0)
    assert successes == 1
    assert pol.acts == 2 + 5  # trial 0: 2 steps then done; trial 1: 5 (max) steps


def test_rollout_uses_native_prompt():
    """The policy receives RoboCasa's task instruction without a wrapper."""
    seen = []

    class _FakeEnv:
        def reset(self, seed=None):
            return ({"annotation.human.task_description": "open the drawer"}, {})

        def step(self, action):
            return ({}, 0.0, True, False, {"success": False})

    class _FakePolicy:
        def reset(self):
            pass

        def act(self, obs, prompt):
            seen.append(prompt)
            return {}

    single_eval._rollout(_FakeEnv(), _FakePolicy(), num_trials=1, max_steps=3, seed=0)
    assert seen == ["open the drawer"]


def test_rollout_treats_info_success_as_terminal():
    class _FakeEnv:
        def __init__(self):
            self.steps = 0

        def reset(self, seed=None):
            return ({"annotation.human.task_description": "open the drawer"}, {})

        def step(self, action):
            self.steps += 1
            return ({}, 1.0, False, False, {"success": True})

    class _FakePolicy:
        def reset(self):
            pass

        def act(self, obs, prompt):
            return {}

    env = _FakeEnv()
    assert single_eval._rollout(env, _FakePolicy(), num_trials=1, max_steps=750, seed=0) == 1
    assert env.steps == 1


def test_prompt_template_preserves_native_instruction():
    """RoboCasa365 eval prompt handling must not add a prefix or suffix."""
    import prompt_template

    for s in ("open the drawer", "", "pick the apple from the counter and place it in the sink."):
        assert prompt_template.format_prompt_for_inference(s) == s


def test_build_policy_uses_compact_defaults(monkeypatch):
    captured = {}

    class _Capture:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(single_eval, "OpenWAMRoboCasa365Policy", _Capture)
    single_eval._build_policy({})
    assert captured["state_dim"] is None
    assert captured["right_wrist_camera_key"] == "video.robot0_agentview_right"
    assert "mobile_base" not in captured
    assert "base_proprio" not in captured
    assert "mask_torso_action" not in captured


def test_repo_template_declares_compact_contract():
    import yaml as _yaml

    tmpl = Path(single_eval.__file__).parent / "policy_config.yml"
    cfg = _yaml.safe_load(tmpl.read_text())
    assert cfg["state_dim"] == 19
    assert cfg["max_steps_override"] is None
    assert cfg["right_wrist_camera_key"] == "video.robot0_agentview_right"
    assert "max_steps" not in cfg
    assert "mask_torso_action" not in cfg


def test_hydra_compose_compact_maps_and_fixed_eval_semantics():
    import os

    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["dataloader=robocasa365"])
        assert cfg.dataloader.action_mode == "eef"
        assert list(cfg.dataloader.unify_action_map) == ["0-9", "68-72"]
        assert list(cfg.dataloader.unify_state_map) == ["0-9", "68-76"]
        assert list(cfg.dataloader.camera_layout) == [
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot0_agentview_right",
        ]
        assert "binary_action_dims" not in cfg.dataloader
        assert "gripper_convention" not in cfg.dataloader
        assert "mask_torso_action" not in cfg.dataloader
        # dataset_dir is user-mutable (the benchmark downloader rewrites it in
        # place), so only require a single umbrella string, not a specific value.
        assert isinstance(cfg.dataloader.dataset_dir, str) and cfg.dataloader.dataset_dir
        assert cfg.dataloader.normalization_stats_path is None


def test_only_canonical_robocasa365_dataset_is_registered():
    from openwam.dataloader.registry import DATASET_REGISTRY
    from openwam.dataloader.robocasa365 import MultiTaskRoboCasa365Dataset

    assert DATASET_REGISTRY["robocasa365"] is MultiTaskRoboCasa365Dataset
