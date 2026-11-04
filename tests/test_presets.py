"""presets/*.env: plain KEY=VALUE files whose OPENWAM_* switches exist in the code."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PRESETS = sorted((ROOT / "presets").glob("*.env"))
SOURCES = "\n".join(p.read_text() for d in ("openwam", "scripts") for p in (ROOT / d).rglob("*.py"))


def _entries(path):
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return [ln.split("=", 1) for ln in lines]


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.name)
def test_preset_switches_exist(path):
    entries = _entries(path)
    assert entries
    for key, value in entries:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", key) and value, f"{path.name}: {key}={value}"
        if key.startswith("OPENWAM_"):
            assert f'"{key}"' in SOURCES, f"{path.name}: {key} is not read anywhere"


def test_80gb_preset_extends_fast():
    fast = dict(_entries(ROOT / "presets/fast.env"))
    fast_80gb = dict(_entries(ROOT / "presets/fast_80gb.env"))
    assert fast_80gb == {**fast, "OPENWAM_TEXT_ENCODER_OFFLOAD": "1"}


@pytest.mark.parametrize("path", PRESETS, ids=lambda p: p.name)
def test_presets_do_not_override_checkpointed_layer_count(path):
    assert "OPENWAM_CKPT_LAYERS" not in dict(_entries(path))
