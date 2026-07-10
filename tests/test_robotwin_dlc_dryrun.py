import os
import subprocess
from pathlib import Path


def test_robotwin_dlc_dryrun_assigns_each_job_once(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "robotwin" / "dlc_parallel_eval.sh"
    log_root = tmp_path / "logs"
    run_id = "pytest_dryrun"

    env = os.environ.copy()
    env.pop("ROBOTWIN_PATH", None)
    env.pop("ROBOTWIN_PYTHON", None)
    env["ROBOTWIN_LOG_ROOT"] = str(log_root)
    env["ROBOTWIN_RUN_ID"] = run_id
    env["DRY_RUN_SLEEP_SEC"] = "0"
    env["NNODES"] = "1"
    env["NODE_RANK"] = "0"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--fresh",
            "-m",
            "all",
            "-n",
            "dryrun",
            "-d",
            str(tmp_path / "missing_ckpt"),
            "-w",
            "2",
            "adjust_bottle",
            "open_laptop",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "shared-queue assignment validated: 4/4" in result.stdout

    log_dir = log_root / f"dryrun_all_dlc_{run_id}"
    summary = (log_dir / "summary.tsv").read_text(encoding="utf-8").splitlines()
    assert len(summary) == 5
    rows = summary[1:]
    assert {tuple(row.split("\t")[:2]) for row in rows} == {
        ("adjust_bottle", "demo_clean"),
        ("adjust_bottle", "demo_randomized"),
        ("open_laptop", "demo_clean"),
        ("open_laptop", "demo_randomized"),
    }
    assert all(row.split("\t")[4:6] == ["ok", "0"] for row in rows)
    assert not list((log_dir / "queue" / "pending").glob("*.job"))
    assert len(list((log_dir / "queue" / "claimed").glob("*.job*"))) == 4
    assert "dry_run=1" in (log_dir / "run.env").read_text(encoding="utf-8")


def test_robotwin_dlc_sentinel_published_on_pre_wait_abort(tmp_path):
    """A node that aborts before it ever reaches the worker-wait loop (e.g. a
    queue-ready timeout because rank0 never started) must still publish its
    done sentinel. Otherwise rank0's peer-sentinel poll would block for
    ALL_NODES_DONE_TIMEOUT_SEC (86400s) on a sentinel that is never written.
    """
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "robotwin" / "dlc_parallel_eval.sh"
    log_root = tmp_path / "logs"
    run_id = "pytest_sentinel_abort"

    env = os.environ.copy()
    env.pop("ROBOTWIN_PATH", None)
    env.pop("ROBOTWIN_PYTHON", None)
    env["ROBOTWIN_LOG_ROOT"] = str(log_root)
    env["ROBOTWIN_RUN_ID"] = run_id
    env["DRY_RUN_SLEEP_SEC"] = "0"
    env["NNODES"] = "2"
    env["NODE_RANK"] = "1"
    env["QUEUE_READY_TIMEOUT_SEC"] = "1"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--fresh",
            "-m",
            "all",
            "-n",
            "dryrun",
            "-d",
            str(tmp_path / "missing_ckpt"),
            "-w",
            "1",
            "adjust_bottle",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert "queue timeout" in (result.stdout + result.stderr)

    log_dir = log_root / f"dryrun_all_dlc_{run_id}"
    assert (log_dir / ".node1_done").is_file()
