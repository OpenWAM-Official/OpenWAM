from __future__ import annotations

import importlib.util
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
