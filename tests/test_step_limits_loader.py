"""Unit tests for the per-task ``step_lim`` override loader.

Covers the fail-silent branches in ``benchmarks.robotwin.openwam2robotwin_interface
._load_step_lim_overrides`` so regressions (e.g. silently accepting ``bool``
values that collapse ``step_lim`` to 1) surface as test failures.
"""

import os
import textwrap

import pytest

try:
    from benchmarks.robotwin import openwam2robotwin_interface as iface
except Exception as exc:  # pragma: no cover — skip when adapter deps are missing
    pytest.skip(
        f"openwam2robotwin_interface not importable in this env: {exc}",
        allow_module_level=True,
    )

_load = iface._load_step_lim_overrides


def _write(tmp_path, body: str) -> str:
    path = os.path.join(tmp_path, "step_limits.yml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(body))
    return path


def test_missing_file_returns_empty(tmp_path):
    assert _load(os.path.join(tmp_path, "does_not_exist.yml")) == {}


def test_empty_file_returns_empty(tmp_path):
    assert _load(_write(tmp_path, "")) == {}


def test_comments_only_returns_empty(tmp_path):
    body = """
        # only comments
        # nothing to load
    """
    assert _load(_write(tmp_path, body)) == {}


def test_non_dict_root_is_rejected(tmp_path, capsys):
    # A top-level YAML list is not a task_name->int mapping.
    assert _load(_write(tmp_path, "- foo\n- bar\n")) == {}
    assert "must be a task_name->int mapping" in capsys.readouterr().out


def test_valid_ints_are_loaded(tmp_path):
    body = """
        adjust_bottle: 160
        open_laptop: 288
    """
    assert _load(_write(tmp_path, body)) == {"adjust_bottle": 160, "open_laptop": 288}


def test_bool_values_are_rejected(tmp_path, capsys):
    # ``int(True) == 1`` would silently cap step_lim at 1; reject explicitly.
    body = """
        adjust_bottle: true
        open_laptop: 288
    """
    assert _load(_write(tmp_path, body)) == {"open_laptop": 288}
    out = capsys.readouterr().out
    assert "adjust_bottle" in out
    assert "must be a plain int" in out


def test_float_values_are_rejected(tmp_path, capsys):
    # ``int(160.9) == 160`` would silently truncate; reject explicitly.
    body = """
        adjust_bottle: 160.9
        open_laptop: 288
    """
    assert _load(_write(tmp_path, body)) == {"open_laptop": 288}
    assert "adjust_bottle" in capsys.readouterr().out


def test_string_values_are_rejected(tmp_path, capsys):
    body = """
        adjust_bottle: "160"
        open_laptop: 288
    """
    assert _load(_write(tmp_path, body)) == {"open_laptop": 288}
    assert "adjust_bottle" in capsys.readouterr().out


def test_mixed_valid_and_invalid(tmp_path):
    body = """
        good_a: 32
        bad_bool: false
        bad_float: 1.5
        good_b: 64
    """
    assert _load(_write(tmp_path, body)) == {"good_a": 32, "good_b": 64}


def test_malformed_yaml_returns_empty(tmp_path, capsys):
    assert _load(_write(tmp_path, "adjust_bottle: [unclosed\n")) == {}
    assert "Failed to load step_lim overrides" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Override-application behavior (eval() side)
# ---------------------------------------------------------------------------


class _FakeTaskEnv:
    """Minimal TASK_ENV stand-in: eval() only reads a handful of attributes."""

    def __init__(self, task_name: str, step_lim: int, take_action_cnt: int = 0):
        self.task_name = task_name
        self.step_lim = step_lim
        self.take_action_cnt = take_action_cnt

    def get_instruction(self) -> str:
        return ""


def _run_override_block(task_env, overrides: dict) -> None:
    """Call ``iface._apply_step_lim_override`` with a swapped-in override table.

    The production ``iface.eval`` does two things: apply the override and then
    call into the HTTP client. We exercise only the first half via the
    extracted helper so tests stay pure — but we go through the *same* helper
    ``eval`` calls, so drift (someone editing the logic and forgetting to
    sync a copy) is impossible.
    """
    orig = iface._STEP_LIM_OVERRIDES
    iface._STEP_LIM_OVERRIDES = overrides
    try:
        iface._apply_step_lim_override(task_env)
    finally:
        iface._STEP_LIM_OVERRIDES = orig


@pytest.fixture(autouse=True)
def _reset_override_state():
    iface._LOGGED_OVERRIDES.clear()
    iface._MISSING_TASK_NAME_WARNED = False
    yield
    iface._LOGGED_OVERRIDES.clear()
    iface._MISSING_TASK_NAME_WARNED = False


def test_override_logs_once_and_applies_across_episodes(capsys):
    # RoboTwin re-seeds TASK_ENV.step_lim from its own YAML at every episode
    # start, so the override has to overwrite it each time — but only log the
    # first time to avoid 1000-line spam over a 50-task * 20-episode sweep.
    env = _FakeTaskEnv("adjust_bottle", step_lim=1000)
    _run_override_block(env, {"adjust_bottle": 160})
    assert env.step_lim == 160

    # Simulate RoboTwin resetting step_lim at the next episode start.
    env.step_lim = 1000
    _run_override_block(env, {"adjust_bottle": 160})
    assert env.step_lim == 160  # still overwritten

    out = capsys.readouterr().out
    assert out.count("step_lim override: adjust_bottle") == 1


def test_no_log_when_upstream_already_matches_override(capsys):
    # If RoboTwin's upstream step_lim already equals the YAML override, we
    # should neither rewrite nor log (the !=override short-circuit).
    env = _FakeTaskEnv("adjust_bottle", step_lim=160)
    _run_override_block(env, {"adjust_bottle": 160})
    assert env.step_lim == 160
    assert "step_lim override" not in capsys.readouterr().out


def test_no_log_for_task_not_in_overrides(capsys):
    env = _FakeTaskEnv("unlisted_task", step_lim=1000)
    _run_override_block(env, {"adjust_bottle": 160})
    assert env.step_lim == 1000
    assert "step_lim override" not in capsys.readouterr().out
