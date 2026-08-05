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


def test_rollout_wraps_prompt_with_training_template():
    """The prompt sent to the policy must be the env instruction wrapped in the SAME template the
    dataloader applies at training time (format_prompt_for_inference) — robotwin parity; a raw
    instruction is out-of-distribution text conditioning."""
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
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference as train_fmt

    assert seen == [train_fmt("open the drawer")]


def test_prompt_template_matches_training_dataloader():
    """benchmarks/robocasa365/prompt_template.py must stay byte-for-byte identical to the
    training-time wrapper (same pinned contract as robotwin's prompt_template)."""
    import prompt_template

    from openwam.dataloader.transforms.multiview import format_prompt_for_inference as train_fmt

    for s in ("open the drawer", "", "pick the apple from the counter and place it in the sink."):
        assert prompt_template.format_prompt_for_inference(s) == train_fmt(s)


def test_build_policy_threads_mobile_base(monkeypatch):
    """_build_policy must forward dataloader.mobile_base to the policy and NOT pin state_dim (so the
    policy auto-derives 25 = arm20 + base5). Captures the ctor kwargs (the real ctor pings a live
    server)."""
    captured = {}

    class _Capture:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(single_eval, "OpenWAMRoboCasa365Policy", _Capture)
    single_eval._build_policy({"mobile_base": True, "osc_pos_scale": 0.05, "osc_rot_scale": 0.5})
    assert captured["mobile_base"] is True
    assert captured["state_dim"] is None  # auto → policy derives 25 from the flag


def test_build_policy_defaults_fixed_base(monkeypatch):
    captured = {}

    class _Capture:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(single_eval, "OpenWAMRoboCasa365Policy", _Capture)
    single_eval._build_policy({"osc_pos_scale": 0.05, "osc_rot_scale": 0.5})
    assert captured["mobile_base"] is False


def test_build_policy_threads_base_proprio(monkeypatch):
    """_build_policy must forward base_proprio (default 'velocity' — the historical representation);
    a global_pose ckpt evaluated with a template that omits the key would otherwise silently fall
    back to velocity (both 25-D, invisible to the width check)."""
    captured = {}

    class _Capture:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(single_eval, "OpenWAMRoboCasa365Policy", _Capture)
    single_eval._build_policy({"mobile_base": True, "base_proprio": "global_pose",
                               "osc_pos_scale": 0.05, "osc_rot_scale": 0.5})
    assert captured["base_proprio"] == "global_pose"
    captured.clear()
    single_eval._build_policy({"mobile_base": True, "osc_pos_scale": 0.05, "osc_rot_scale": 0.5})
    assert captured["base_proprio"] == "velocity"


def test_repo_template_declares_base_proprio():
    """The in-repo template must carry the key explicitly (PR #57 B5): the manual PR-body reminder is
    not a substitute for the default path being correct."""
    import yaml as _yaml

    tmpl = Path(single_eval.__file__).parent / "policy_config.yml"
    cfg = _yaml.safe_load(tmpl.read_text())
    assert cfg.get("base_proprio") == "velocity"


def test_hydra_compose_overrides_base_proprio_and_binary_dims():
    """PR #57: the training entry configs/dataloader/robocasa365.yaml must declare the new keys with
    historical defaults, so the STANDARD override syntax works (Hydra struct mode rejects overrides
    of undeclared keys with ConfigCompositionException — '+dataloader....' is not the documented
    launch command)."""
    import os

    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="train", overrides=["dataloader=robocasa365"])
        assert cfg.dataloader.base_proprio == "velocity"        # historical defaults declared
        assert cfg.dataloader.binary_action_dims is None
        cfg2 = compose(config_name="train", overrides=[
            "dataloader=robocasa365",
            "dataloader.base_proprio=global_pose",
            "dataloader.binary_action_dims=[9,24]",
        ])
        assert cfg2.dataloader.base_proprio == "global_pose"
        assert list(cfg2.dataloader.binary_action_dims) == [9, 24]
