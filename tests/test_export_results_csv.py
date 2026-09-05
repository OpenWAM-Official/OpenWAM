"""Tests for benchmarks/robotwin/export_results_csv.py.

Focus: the step-limit-hit accounting that separates "model got it wrong" from
"rollout ran out of steps" in an otherwise opaque success rate.
"""

import csv
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "robotwin" / "export_results_csv.py"

_spec = importlib.util.spec_from_file_location("export_results_csv", _MODULE_PATH)
export_results_csv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_results_csv)


# ANSI color codes RoboTwin wraps its prints in — parsing must strip these.
_G, _R, _C, _P, _X = "\x1b[92m", "\x1b[91m", "\x1b[96m", "\x1b[95m", "\x1b[0m"


def _episode_success(step_lim: int, stop_at: int) -> str:
    lines = [f"step: {_G}{n} / {step_lim}{_X}\r" for n in range(1, stop_at + 1)]
    lines.append(f"{_G}Success!{_X}\n")
    return "".join(lines)


def _episode_fail(step_lim: int, stop_at: int) -> str:
    lines = [f"step: {_G}{n} / {step_lim}{_X}\r" for n in range(1, stop_at + 1)]
    lines.append(f"{_R}Fail!{_X}\n")
    return "".join(lines)


def _rate(suc: int, total: int) -> str:
    pct = round(suc / total * 100, 1)
    return f"Success rate: {_C}{suc}/{total}{_X} => {_P}{pct}%{_X}\n"


def _write_log(path: Path, episodes: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(episodes, encoding="utf-8")


def test_parse_episode_stats_distinguishes_step_limit_hits(tmp_path):
    body = (
        _episode_success(160, 40)
        + _rate(1, 1)  # success, not a hit
        + _episode_fail(160, 160)
        + _rate(1, 2)  # step-limit hit (160/160)
        + _episode_fail(160, 90)
        + _rate(1, 3)  # model failure (stopped early)
    )

    episodes, step_limit_hits = export_results_csv.parse_episode_stats_from_text(body)
    assert episodes == 3
    assert step_limit_hits == 1
    assert export_results_csv.parse_success_rate_from_text(body) == pytest.approx(33.3)


def test_parse_episode_stats_fail_without_step_line():
    # A Fail! with no preceding step line must not be miscounted as a limit hit.
    body = f"{_R}Fail!{_X}\n" + _rate(0, 1)
    assert export_results_csv.parse_episode_stats_from_text(body) == (1, 0)


def test_export_csv_reports_missing_log(tmp_path, monkeypatch):
    # A log referenced by summary.tsv but absent on disk: episodes/
    # step_limit_hits/success_rate must all blank consistently (not "0" for
    # some and blank for others), and main() must surface it as a failure.
    log_dir = tmp_path / "openwam_demo_clean_dlc_run_missing"
    task_log = log_dir / "node0" / "worker0" / "adjust_bottle_demo_clean.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "summary.tsv").write_text(
        f"task\tmode\tnode\tworker\tstatus\texit_code\tlog\nadjust_bottle\tdemo_clean\t0\t0\tok\t0\t{task_log}\n",
        encoding="utf-8",
    )
    (log_dir / "run.env").write_text(
        "run_id=run_missing\npolicy_name=openwam\nmode=demo_clean\ntotal_jobs=1\ntasks=adjust_bottle\n",
        encoding="utf-8",
    )

    out_csv = log_dir / "results.csv"
    monkeypatch.setattr("sys.argv", ["export_results_csv.py", str(log_dir), "-o", str(out_csv), "--strict"])
    assert export_results_csv.main() == 2

    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["success_rate"] == ""
    assert row["episodes"] == ""
    assert row["step_limit_hits"] == ""


def test_export_csv_reports_zero_episodes_not_blank_when_log_is_readable(tmp_path, monkeypatch):
    # A readable log with no verdict lines yet (e.g. crashed before the
    # first episode finished) must report "0", not blank — blank is
    # reserved for "log missing/unreadable" (see the test above), so the
    # two cases stay distinguishable in the CSV.
    log_dir = tmp_path / "openwam_demo_clean_dlc_run_zero"
    task_log = log_dir / "node0" / "worker0" / "adjust_bottle_demo_clean.log"
    _write_log(task_log, "booting policy server...\nstep: 1 / 160\r")

    (log_dir / "summary.tsv").write_text(
        f"task\tmode\tnode\tworker\tstatus\texit_code\tlog\nadjust_bottle\tdemo_clean\t0\t0\tfailed\t1\t{task_log}\n",
        encoding="utf-8",
    )
    (log_dir / "run.env").write_text(
        "run_id=run_zero\npolicy_name=openwam\nmode=demo_clean\ntotal_jobs=1\ntasks=adjust_bottle\n",
        encoding="utf-8",
    )

    out_csv = log_dir / "results.csv"
    monkeypatch.setattr("sys.argv", ["export_results_csv.py", str(log_dir), "-o", str(out_csv)])
    export_results_csv.main()

    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["episodes"] == "0"
    assert row["step_limit_hits"] == "0"


def test_export_csv_includes_step_limit_columns(tmp_path, monkeypatch):
    log_dir = tmp_path / "openwam_demo_clean_dlc_run1"
    task_log = log_dir / "node0" / "worker0" / "adjust_bottle_demo_clean.log"
    _write_log(
        task_log,
        _episode_success(160, 40) + _rate(1, 1) + _episode_fail(160, 160) + _rate(1, 2),
    )

    (log_dir).mkdir(parents=True, exist_ok=True)
    (log_dir / "summary.tsv").write_text(
        f"task\tmode\tnode\tworker\tstatus\texit_code\tlog\nadjust_bottle\tdemo_clean\t0\t0\tok\t0\t{task_log}\n",
        encoding="utf-8",
    )
    (log_dir / "run.env").write_text(
        "run_id=run1\npolicy_name=openwam\nmode=demo_clean\ntotal_jobs=1\ntasks=adjust_bottle\n",
        encoding="utf-8",
    )

    out_csv = log_dir / "results.csv"
    monkeypatch.setattr("sys.argv", ["export_results_csv.py", str(log_dir), "-o", str(out_csv)])
    assert export_results_csv.main() == 0

    with out_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["episodes"] == "2"
    assert row["step_limit_hits"] == "1"
    assert row["success_rate"] == "50.000000"
