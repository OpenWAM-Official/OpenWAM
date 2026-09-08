"""Regression tests for the DIFFSYNTH_ATTENTION_IMPLEMENTATION override.

The override used to be taken verbatim, so a name whose library never imported
was returned anyway and failed later inside the kernel wrapper with a NameError.
These pin the contract against its sibling WAM_ATTENTION_IMPL in
``openwam/model/action_backbone/components.py``: unknown name raises, known but
unavailable warns and falls back, known and available is honoured.

The availability flags are faked, so every case runs on CPU-only CI.
"""

from __future__ import annotations

import logging

import pytest

ENV = "DIFFSYNTH_ATTENTION_IMPLEMENTATION"
FLAGS = {
    "flash_attention_3": "FLASH_ATTN_3_AVAILABLE",
    "flash_attention_2": "FLASH_ATTN_2_AVAILABLE",
    "sage_attention": "SAGE_ATTN_AVAILABLE",
    "xformers": "XFORMERS_AVAILABLE",
}


def _shared():
    from openwam.model.video_backbone.wan.shared.core.attention import attention as shared

    return shared


def _set_availability(monkeypatch, available: set):
    """Fake which libraries imported, so these run without any of them installed."""
    shared = _shared()
    for name, flag in FLAGS.items():
        monkeypatch.setattr(shared, flag, name in available)
    return shared


def test_unknown_override_raises_and_names_the_choices(monkeypatch):
    shared = _set_availability(monkeypatch, set(FLAGS))
    monkeypatch.setenv(ENV, "bogus_backend")

    with pytest.raises(ValueError) as excinfo:
        shared.initialize_attention_priority()

    message = str(excinfo.value)
    assert "bogus_backend" in message
    # the message must be actionable, not just a rejection
    for name in FLAGS:
        assert name in message


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_override_reads_as_no_override(monkeypatch, blank):
    """`DIFFSYNTH_ATTENTION_IMPLEMENTATION=` is reachable and must not be an error.

    WAM_ATTENTION_IMPL does `.strip().lower()` then `if override:`, so a blank value
    falls through to auto-detection there. An accidental `export VAR=` must not turn
    into a hard failure at import.
    """
    shared = _set_availability(monkeypatch, {"sage_attention"})
    monkeypatch.setenv(ENV, blank)

    assert shared.initialize_attention_priority() == "sage_attention"


def test_unavailable_override_warns_and_falls_back(monkeypatch, caplog):
    """The case that used to NameError inside the kernel wrapper."""
    shared = _set_availability(monkeypatch, {"flash_attention_2"})
    monkeypatch.setenv(ENV, "flash_attention_3")

    with caplog.at_level(logging.WARNING):
        resolved = shared.initialize_attention_priority()

    assert resolved == "flash_attention_2", "should fall back to auto-detect, not honour the request"
    assert "flash_attention_3" in caplog.text
    assert "not available" in caplog.text


def test_available_override_is_honoured(monkeypatch):
    shared = _set_availability(monkeypatch, {"flash_attention_2", "sage_attention"})
    monkeypatch.setenv(ENV, "sage_attention")

    assert shared.initialize_attention_priority() == "sage_attention"


def test_override_is_normalized(monkeypatch):
    """Match the sibling, which does .strip().lower() before its membership check."""
    shared = _set_availability(monkeypatch, {"flash_attention_2"})
    monkeypatch.setenv(ENV, "  FLASH_ATTENTION_2  ")

    assert shared.initialize_attention_priority() == "flash_attention_2"


def test_torch_is_always_selectable(monkeypatch):
    """Plain SDPA needs no library, so it must be honoured even with nothing installed."""
    shared = _set_availability(monkeypatch, set())
    monkeypatch.setenv(ENV, "torch")

    assert shared.initialize_attention_priority() == "torch"


def test_without_override_auto_detection_is_unchanged(monkeypatch):
    shared = _set_availability(monkeypatch, {"flash_attention_2", "sage_attention", "xformers"})
    monkeypatch.delenv(ENV, raising=False)

    assert shared.initialize_attention_priority() == "flash_attention_2"

    shared = _set_availability(monkeypatch, {"xformers"})
    assert shared.initialize_attention_priority() == "xformers"

    shared = _set_availability(monkeypatch, set())
    assert shared.initialize_attention_priority() == "torch"
