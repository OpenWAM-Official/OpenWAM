"""Scheduling-only runtime switches (openwam.train.utils.runtime_tuning)."""

import copy

import pytest

from openwam.train.utils import runtime_tuning as rt

_CONFIG = {"gradient_accumulation_steps": 1, "zero_optimization": {"stage": 2}}


@pytest.fixture(autouse=True)
def _switches_off(monkeypatch):
    for name in ("OPENWAM_DS_TIMER_NOSYNC", "OPENWAM_DS_OVERLAP_COMM", "OPENWAM_GC_FREEZE"):
        monkeypatch.delenv(name, raising=False)


def test_deepspeed_config_untouched_by_default():
    config = copy.deepcopy(_CONFIG)
    rt.apply_deepspeed_runtime_settings(config)
    assert config == _CONFIG


def test_deepspeed_runtime_settings(monkeypatch):
    monkeypatch.setenv("OPENWAM_DS_TIMER_NOSYNC", "1")
    monkeypatch.setenv("OPENWAM_DS_OVERLAP_COMM", "1")
    config = copy.deepcopy(_CONFIG)
    rt.apply_deepspeed_runtime_settings(config)
    assert config["timers"] == {"throughput": {"synchronized": False}}
    assert config["zero_optimization"] == {"stage": 2, "overlap_comm": True}

    explicit = {"zero_optimization": {"stage": 2, "overlap_comm": False}, "timers": {"throughput": {"enabled": False}}}
    rt.apply_deepspeed_runtime_settings(explicit)
    assert explicit["zero_optimization"]["overlap_comm"] is False
    assert explicit["timers"]["throughput"] == {"enabled": False, "synchronized": False}

    stage1 = {"zero_optimization": {"stage": 1}}
    rt.apply_deepspeed_runtime_settings(stage1)
    assert "overlap_comm" not in stage1["zero_optimization"]


def test_gc_freeze_only_at_its_steps(monkeypatch):
    frozen = []
    monkeypatch.setattr(rt.gc, "freeze", lambda: frozen.append(True))
    for step in range(6):
        rt.maybe_freeze_gc(step)
    assert frozen == []
    monkeypatch.setenv("OPENWAM_GC_FREEZE", "1")
    for step in range(6):
        rt.maybe_freeze_gc(step)
    assert len(frozen) == 3


def test_parse_cpulist():
    assert rt._parse_cpulist("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
