import os
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarks" / "robotwin" / "dlc_parallel_eval.sh"

# The script prefers MLP_WORKER_NUM/MLP_ROLE_INDEX, then NNODES/NODE_RANK, then
# WORLD_SIZE/RANK. Running pytest inside a real DLC job would otherwise leave
# the MLP_*/WORLD_SIZE/RANK vars set from that job and silently override the
# NNODES/NODE_RANK topology these tests set explicitly, hanging on the wrong
# node count/rank instead of failing clearly.
_POPPED_ENV_VARS = (
    "ROBOTWIN_PATH",
    "ROBOTWIN_PYTHON",
    "MLP_WORKER_NUM",
    "MLP_ROLE_INDEX",
    "WORLD_SIZE",
    "RANK",
)


def _dlc_env(log_root: Path, run_id: str, *, nnodes: str, node_rank: str, **extra: str) -> dict:
    env = os.environ.copy()
    for var in _POPPED_ENV_VARS:
        env.pop(var, None)
    env["ROBOTWIN_LOG_ROOT"] = str(log_root)
    env["ROBOTWIN_RUN_ID"] = run_id
    env["NNODES"] = nnodes
    env["NODE_RANK"] = node_rank
    env.update(extra)
    return env


def _dlc_argv(*, mode: str, name: str, ckpt_dir: Path, workers: str, tasks: list) -> list:
    return [
        "bash",
        str(SCRIPT),
        "--dry-run",
        "--fresh",
        "-m",
        mode,
        "-n",
        name,
        "-d",
        str(ckpt_dir),
        "-w",
        workers,
        *tasks,
    ]


def _wait_for_path(path: Path, proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Poll for a path to appear instead of a fixed sleep before signaling a
    node under test, so the test's timing assumption is an observable file
    the script itself writes rather than a guess at how long setup takes.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        assert proc.poll() is None, f"process exited before {path} appeared"
        time.sleep(0.05)
    raise AssertionError(f"{path} did not appear within {timeout}s")


def _wait_for_node_dir(log_dir: Path, node_rank: str, proc: subprocess.Popen, timeout: float = 10.0) -> None:
    """Poll for ``node<rank>``'s directory instead of a fixed sleep before
    signaling a node under test: it's created right after the cleanup() trap
    is installed, so this is the earliest point at which a signal is
    guaranteed to be caught rather than racing the default disposition.
    """
    _wait_for_path(log_dir / f"node{node_rank}", proc, timeout)


def test_robotwin_dlc_dryrun_assigns_each_job_once(tmp_path):
    log_root = tmp_path / "logs"
    run_id = "pytest_dryrun"
    env = _dlc_env(log_root, run_id, nnodes="1", node_rank="0", DRY_RUN_SLEEP_SEC="0")

    result = subprocess.run(
        _dlc_argv(
            mode="all",
            name="dryrun",
            ckpt_dir=tmp_path / "missing_ckpt",
            workers="2",
            tasks=["adjust_bottle", "open_laptop"],
        ),
        cwd=REPO_ROOT,
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


def test_robotwin_dlc_two_nodes_join_and_complete(tmp_path):
    """Realistic multi-node happy path: two real node processes (rank0,
    rank1) both join the same shared queue and both complete normally, so
    rank0's peer-sentinel loop observes two genuine ``.node<rank>_done``
    files. Every other multi-node test in this file only ever gets as far
    as one node aborting before rank0 starts (NODE_JOINED gating) — this
    is the only one where both nodes actually finish and publish for real.
    """
    log_root = tmp_path / "logs"
    run_id = "pytest_two_nodes"
    common = dict(
        DRY_RUN_SLEEP_SEC="0",
        QUEUE_READY_TIMEOUT_SEC="15",
        DRY_RUN_BARRIER_TIMEOUT_SEC="15",
    )
    argv = _dlc_argv(
        mode="all",
        name="dryrun",
        ckpt_dir=tmp_path / "missing_ckpt",
        workers="1",
        tasks=["adjust_bottle", "open_laptop"],
    )

    procs = {}
    for rank in ("0", "1"):
        env = _dlc_env(log_root, run_id, nnodes="2", node_rank=rank, **common)
        procs[rank] = subprocess.Popen(
            argv, cwd=REPO_ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    stdouts = {}
    try:
        for rank, proc in procs.items():
            stdouts[rank], _ = proc.communicate(timeout=30)
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()
                proc.communicate(timeout=5)

    assert procs["0"].returncode == 0, stdouts["0"]
    assert procs["1"].returncode == 0, stdouts["1"]
    # Only rank0 prints the dry-run validation line, after its own
    # peer-sentinel-wait loop has observed both real sentinels.
    assert "shared-queue assignment validated: 4/4" in stdouts["0"]

    log_dir = log_root / f"dryrun_all_dlc_{run_id}"
    assert (log_dir / ".node0_done").is_file()
    assert (log_dir / ".node1_done").is_file()

    summary = (log_dir / "summary.tsv").read_text(encoding="utf-8").splitlines()
    assert len(summary) == 5  # header + 4 jobs, claimed exactly once across both nodes
    assert not list((log_dir / "queue" / "pending").glob("*.job"))
    assert len(list((log_dir / "queue" / "claimed").glob("*.job*"))) == 4


def test_robotwin_dlc_no_sentinel_published_on_pre_join_abort(tmp_path):
    """A node that gives up on the queue-ready wait (rank0 never started, or
    started too slowly) never joined this attempt's queue, so it must NOT
    publish a done sentinel.

    Publishing there was tried and reverted (see PR #36 review): a sentinel
    from a node that never joined would survive as long as READY_FILE is
    absent, so a same-RUN_ID retry without --fresh could see a stale "done"
    file for a node that hasn't even started the new attempt, and could race
    rank0's --fresh cleanup for a node that gives up moments before rank0
    finally comes up in the *same* attempt. This test guards against
    reintroducing that regression.
    """
    log_root = tmp_path / "logs"
    run_id = "pytest_no_sentinel_pre_join"
    env = _dlc_env(
        log_root, run_id, nnodes="2", node_rank="1",
        DRY_RUN_SLEEP_SEC="0", QUEUE_READY_TIMEOUT_SEC="1",
    )

    result = subprocess.run(
        _dlc_argv(
            mode="all",
            name="dryrun",
            ckpt_dir=tmp_path / "missing_ckpt",
            workers="1",
            tasks=["adjust_bottle"],
        ),
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert "queue timeout" in (result.stdout + result.stderr)

    log_dir = log_root / f"dryrun_all_dlc_{run_id}"
    assert not (log_dir / ".node1_done").is_file()


def test_robotwin_dlc_no_sentinel_published_on_pre_join_sigterm(tmp_path):
    """Same as above but via SIGTERM (e.g. DLC preemption) delivered while a
    node is still blocked in the queue-ready wait loop, not yet joined.
    """
    log_root = tmp_path / "logs"
    run_id = "pytest_no_sentinel_pre_join_sigterm"
    # rank0 never starts in this test, so this must not fire on its own before
    # the SIGTERM below does.
    env = _dlc_env(
        log_root, run_id, nnodes="2", node_rank="1",
        DRY_RUN_SLEEP_SEC="0", QUEUE_READY_TIMEOUT_SEC="300",
    )
    log_dir = log_root / f"dryrun_all_dlc_{run_id}"

    proc = subprocess.Popen(
        _dlc_argv(
            mode="all",
            name="dryrun",
            ckpt_dir=tmp_path / "missing_ckpt",
            workers="1",
            tasks=["adjust_bottle"],
        ),
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_node_dir(log_dir, "1", proc)
        proc.send_signal(signal.SIGTERM)
        stdout, _ = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)

    assert "cleanup complete" in stdout, stdout
    assert not (log_dir / ".node1_done").is_file()


def test_robotwin_dlc_sentinel_published_on_post_join_sigterm(tmp_path):
    """SIGTERM (e.g. DLC preemption) delivered after a node has joined the
    queue and started its workers must still publish the done sentinel via
    the cleanup() trap — the realistic mid-run preemption case the sentinel
    mechanism exists for.
    """
    log_root = tmp_path / "logs"
    run_id = "pytest_sentinel_post_join_sigterm"
    # Long enough that the worker is still mid-job when SIGTERM lands.
    env = _dlc_env(log_root, run_id, nnodes="1", node_rank="0", DRY_RUN_SLEEP_SEC="5")
    log_dir = log_root / f"dryrun_demo_clean_dlc_{run_id}"

    proc = subprocess.Popen(
        _dlc_argv(
            mode="demo_clean",
            name="dryrun",
            ckpt_dir=tmp_path / "missing_ckpt",
            workers="1",
            tasks=["adjust_bottle"],
        ),
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        # Poll for dryrun_node_ready/node0, not .queue_ready (READY_FILE):
        # the script sets NODE_JOINED=1 and touches READY_FILE as two
        # separate statements, so .queue_ready can become visible while
        # NODE_JOINED=1 hasn't run yet — a signal landing in that window
        # would make cleanup() see NODE_JOINED=0 and skip the sentinel,
        # flaking this test. dryrun_node_ready/node0 is only touched after
        # the join block (this test always runs --dry-run), so it's
        # strictly post-join and race-free.
        _wait_for_path(log_dir / "dryrun_node_ready" / "node0", proc)
        assert proc.poll() is None, "process exited before SIGTERM was sent"
        proc.send_signal(signal.SIGTERM)
        stdout, _ = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=5)

    assert "cleanup complete" in stdout, stdout
    assert (log_dir / ".node0_done").is_file()
