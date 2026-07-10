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
