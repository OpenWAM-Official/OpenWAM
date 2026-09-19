from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _load_single_eval(monkeypatch):
    interface = types.ModuleType("openwam2libero_interface")
    interface.OpenWAMLiberoPolicy = object
    monkeypatch.setitem(sys.modules, "openwam2libero_interface", interface)

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "libero" / "single_eval.py"
    spec = importlib.util.spec_from_file_location("libero_single_eval_retry_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_env_retries_randomization_error(monkeypatch, capsys):
    single_eval = _load_single_eval(monkeypatch)
    expected_env = object()
    attempts = 0

    class RandomizationError(Exception):
        pass

    def make_env(task, cfg):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RandomizationError("placement sampling failed")
        return expected_env

    monkeypatch.setattr(single_eval, "_make_env", make_env)

    assert single_eval._make_env_with_randomization_retries(object(), {}) is expected_env
    assert attempts == 2
    assert "retrying (1/5)" in capsys.readouterr().out


def test_make_env_does_not_retry_unrelated_error(monkeypatch):
    single_eval = _load_single_eval(monkeypatch)
    attempts = 0

    def make_env(task, cfg):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("configuration is invalid")

    monkeypatch.setattr(single_eval, "_make_env", make_env)

    with pytest.raises(RuntimeError, match="configuration is invalid"):
        single_eval._make_env_with_randomization_retries(object(), {})
    assert attempts == 1


def test_ordinary_entry_keeps_native_result_schema(tmp_path, monkeypatch):
    single_eval = _load_single_eval(monkeypatch)
    monkeypatch.setattr(single_eval, "_write_libero_config", lambda: None)

    task = types.SimpleNamespace(name="task", language="instruction")

    class Suite:
        def get_task(self, task_id):
            return task

        def get_task_init_states(self, task_id):
            return [object()]

    libero_module = types.ModuleType("libero.libero")
    libero_module.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": Suite})
    package = types.ModuleType("libero")
    package.libero = libero_module
    monkeypatch.setitem(sys.modules, "libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)

    class Env:
        def seed(self, seed):
            self.seed_value = seed

        def reset(self):
            return {}

        def set_init_state(self, state):
            return {}

        def step(self, action):
            return {}, 1.0, True, {}

        def close(self):
            pass

    class Policy:
        def __init__(self, **kwargs):
            pass

        def reset(self):
            pass

        def act(self, observation, instruction):
            return [0] * 7

        def close(self):
            pass

    monkeypatch.setattr(single_eval, "_make_env_with_randomization_retries", lambda task, cfg: Env())
    monkeypatch.setattr(single_eval, "OpenWAMLiberoPolicy", Policy)
    result_dir = tmp_path / "result"
    assert (
        single_eval.run_eval(
            {
                "action_mode": "eef",
                "rng_mode": "environment",
                "reseed_each_trial": False,
                "suite": "libero_spatial",
                "task_id": 0,
                "num_trials": 1,
                "max_steps": 1,
                "settle_steps": 0,
                "settle_action": [0] * 7,
                "result_dir": str(result_dir),
            }
        )
        == 0
    )
    result = json.loads((result_dir / "results.json").read_text())
    assert set(result) == {
        "action_mode",
        "suite",
        "task_id",
        "task",
        "instruction",
        "trial_start",
        "trial_stop",
        "successes",
        "success_rate",
        "max_steps",
        "settle_steps",
        "seed",
        "trials",
    }
    assert set(result["trials"][0]) == {"trial", "success", "policy_steps", "last_reward"}
