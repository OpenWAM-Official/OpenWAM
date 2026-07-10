import json
from types import SimpleNamespace


def load_console_module():
    import benchmarks.web_control as module

    return module


def load_compat_module():
    import benchmarks.robotwin.dlc_web_console as module

    return module


def make_builder(module, root):
    return module.SnapshotBuilder(
        root,
        max_logs=100,
        state_tail_bytes=100_000,
        max_task_log_bytes=100_000,
        max_error_snippets=10,
    )


def metric_by_id(snapshot, metric_id):
    metrics = {metric["id"]: metric for metric in snapshot.get("custom_metrics", [])}
    return metrics[metric_id]


def test_dlc_snapshot_merges_summary_queue_and_success_rates(tmp_path):
    console = load_console_module()
    root = tmp_path
    (root / "queue" / "pending").mkdir(parents=True)
    (root / "queue" / "claimed").mkdir(parents=True)
    (root / "node0" / "worker0").mkdir(parents=True)
    (root / "node0" / "servers").mkdir(parents=True)

    (root / "run.env").write_text(
        "run_id=smoke\n"
        "policy_name=openwam\n"
        "mode=all\n"
        "nnodes=1\n"
        "num_workers_per_node=1\n"
        "total_jobs=3\n"
        "tasks=adjust_bottle beat_block_hammer\n",
        encoding="utf-8",
    )
    (root / ".queue_ready").touch()
    (root / "queue" / "pending" / "000001_beat_block_hammer_demo_clean.job").write_text(
        "task=beat_block_hammer\nmode=demo_clean\n",
        encoding="utf-8",
    )
    (root / "queue" / "claimed" / "000002_beat_block_hammer_demo_randomized.job.node0.worker0").write_text(
        "task=beat_block_hammer\nmode=demo_randomized\n",
        encoding="utf-8",
    )
    task_log = root / "node0" / "worker0" / "adjust_bottle_demo_clean.log"
    task_log.write_text("start\nSuccess rate: 13/20 => 65.00%\n", encoding="utf-8")
    (root / "node0" / "worker0" / "beat_block_hammer_demo_randomized.log").write_text(
        "running\n",
        encoding="utf-8",
    )
    (root / "node0" / "worker0" / "worker.log").write_text("started\n", encoding="utf-8")
    (root / "summary.tsv").write_text(
        "task\tmode\tnode\tworker\tstatus\texit_code\tlog\n"
        f"adjust_bottle\tdemo_clean\t0\t0\tok\t0\t{task_log}\n",
        encoding="utf-8",
    )

    snapshot = make_builder(console, root).build()

    assert snapshot["progress"]["total"] == 3
    assert snapshot["progress"]["ok"] == 1
    assert snapshot["progress"]["running"] == 1
    assert snapshot["progress"]["pending"] == 1
    assert snapshot["rates"]["weighted_success"] == 13
    assert snapshot["rates"]["weighted_total"] == 20
    assert snapshot["rates"]["weighted_success_rate"] == 65.0
    assert metric_by_id(snapshot, "robotwin_demo_clean_success_rate")["raw_value"] == 65.0
    assert metric_by_id(snapshot, "robotwin_demo_randomized_success_rate")["raw_value"] is None
    assert len(snapshot["jobs"]) == 3
    assert any(job["status"] == "running" and job["log"].endswith("beat_block_hammer_demo_randomized.log") for job in snapshot["jobs"])
    assert snapshot["nodes"][0]["workers"][0]["running"] == 1


def test_dlc_snapshot_exposes_robotwin_mode_custom_metrics(tmp_path):
    console = load_console_module()
    root = tmp_path
    worker_dir = root / "node0" / "worker0"
    worker_dir.mkdir(parents=True)
    (root / "run.env").write_text(
        "run_id=modes\npolicy_name=openwam\nmode=all\ntotal_jobs=4\n"
        "tasks=clean_a clean_b random_a random_b\n",
        encoding="utf-8",
    )
    logs = {
        "clean_a_demo_clean.log": "Success rate: 3/5 => 60.00%\n",
        "clean_b_demo_clean.log": "Success rate: 4/5 => 80.00%\n",
        "random_a_demo_randomized.log": "Success rate: 40.00%\n",
        "random_b_demo_randomized.log": "Success rate: 80.00%\n",
    }
    for name, text in logs.items():
        (worker_dir / name).write_text(text, encoding="utf-8")
    (root / "summary.tsv").write_text(
        "task\tmode\tnode\tworker\tstatus\texit_code\tlog\n"
        f"clean_a\tdemo_clean\t0\t0\tok\t0\t{worker_dir / 'clean_a_demo_clean.log'}\n"
        f"clean_b\tdemo_clean\t0\t0\tok\t0\t{worker_dir / 'clean_b_demo_clean.log'}\n"
        f"random_a\tdemo_randomized\t0\t0\tok\t0\t{worker_dir / 'random_a_demo_randomized.log'}\n"
        f"random_b\tdemo_randomized\t0\t0\tok\t0\t{worker_dir / 'random_b_demo_randomized.log'}\n",
        encoding="utf-8",
    )

    snapshot = make_builder(console, root).build()
    clean = metric_by_id(snapshot, "robotwin_demo_clean_success_rate")
    randomized = metric_by_id(snapshot, "robotwin_demo_randomized_success_rate")

    assert clean["value"] == "70.00%"
    assert clean["raw_value"] == 70.0
    assert clean["source"] == "weighted_success"
    assert clean["weighted_success"] == 7
    assert clean["weighted_total"] == 10
    assert randomized["value"] == "60.00%"
    assert randomized["raw_value"] == 60.0
    assert randomized["source"] == "mean_success_rate"
    assert randomized["weighted_total"] == 0
    assert randomized["parsed_task_count"] == 2


def test_dlc_snapshot_falls_back_to_worker_files_without_summary(tmp_path):
    console = load_console_module()
    root = tmp_path
    worker_dir = root / "worker0"
    worker_dir.mkdir()
    (root / ".queue.txt").write_text("task_b|demo_clean\n", encoding="utf-8")
    (root / "run.env").write_text(
        "run_id=legacy\npolicy_name=openwam\nmode=demo_clean\ntotal_jobs=2\ntasks=task_a task_b\n",
        encoding="utf-8",
    )
    (worker_dir / "finished.txt").write_text("task_a|demo_clean\n", encoding="utf-8")
    (worker_dir / "worker.log").write_text("done task_a\n", encoding="utf-8")
    (worker_dir / "task_a_demo_clean.log").write_text(
        "Success rate: 4/5 => 80.00%\n",
        encoding="utf-8",
    )

    snapshot = make_builder(console, root).build()

    assert snapshot["summary"]["exists"] is False
    assert snapshot["progress"]["ok"] == 1
    assert snapshot["progress"]["pending"] == 1
    assert snapshot["queue"]["pending_count"] == 1
    assert snapshot["rates"]["weighted_success_rate"] == 80.0
    assert {job["status"] for job in snapshot["jobs"]} == {"ok", "pending"}


def test_dlc_snapshot_collects_failure_snippets_and_csv_rows(tmp_path):
    console = load_console_module()
    root = tmp_path
    worker_dir = root / "node0" / "worker0"
    worker_dir.mkdir(parents=True)
    (root / "run.env").write_text(
        "run_id=fail\npolicy_name=openwam\nmode=demo_clean\ntotal_jobs=1\ntasks=bad_task\n",
        encoding="utf-8",
    )
    task_log = worker_dir / "bad_task_demo_clean.log"
    task_log.write_text(
        "boot\nTraceback (most recent call last):\nRuntimeError: CUDA out of memory\n",
        encoding="utf-8",
    )
    (root / "summary.tsv").write_text(
        "task\tmode\tnode\tworker\tstatus\texit_code\tlog\n"
        f"bad_task\tdemo_clean\t0\t0\tfailed\t137\t{task_log}\n",
        encoding="utf-8",
    )

    builder = make_builder(console, root)
    snapshot = builder.build()
    rows = builder.build_results_rows()

    assert snapshot["progress"]["failed"] == 1
    assert "CUDA out of memory" in snapshot["failures"][0]["snippet"]
    assert rows[0]["status"] == "failed"
    assert rows[0]["exit_code"] == "137"


def test_results_csv_derives_all_columns_from_full_log_not_tail(tmp_path):
    """success_rate/success/episodes/step_limit_hits in /api/results.csv rows
    must all come from the same full-log read. A tiny state_tail_bytes here
    stands in for a log whose terminal ``success rate: X / Y`` line falls
    outside the tail window (e.g. behind a long traceback) — if any of these
    four columns were still sourced from the tail-windowed live-state job
    dict, that column would go blank/stale while its siblings stay accurate.
    """
    console = load_console_module()
    root = tmp_path
    worker_dir = root / "node0" / "worker0"
    worker_dir.mkdir(parents=True)
    (root / "run.env").write_text(
        "run_id=tailwindow\npolicy_name=openwam\nmode=demo_clean\ntotal_jobs=1\ntasks=bad_step_limit\n",
        encoding="utf-8",
    )
    task_log = worker_dir / "bad_step_limit_demo_clean.log"
    task_log.write_text(
        "step: 10 / 10\rFail!\n"
        + "Success rate: 3/5 => 60.00%\n"
        + ("padding to push the summary line out of a small tail window\n" * 50),
        encoding="utf-8",
    )
    (root / "summary.tsv").write_text(
        "task\tmode\tnode\tworker\tstatus\texit_code\tlog\n"
        f"bad_step_limit\tdemo_clean\t0\t0\tok\t0\t{task_log}\n",
        encoding="utf-8",
    )

    builder = console.SnapshotBuilder(
        root,
        max_logs=100,
        state_tail_bytes=64,
        max_task_log_bytes=100_000,
        max_error_snippets=10,
    )
    rows = builder.build_results_rows()

    assert len(rows) == 1
    row = rows[0]
    assert row["success_rate"] == "60.000000"
    assert row["success"] == 3
    assert row["episodes"] == 5
    assert row["step_limit_hits"] == 1


def test_dlc_snapshot_does_not_read_failure_snippet_outside_root(tmp_path):
    console = load_console_module()
    root = tmp_path / "logs"
    root.mkdir()
    outside = tmp_path / "outside.log"
    outside.write_text("secret should not be exposed\n", encoding="utf-8")
    (root / "run.env").write_text(
        "run_id=fail\npolicy_name=openwam\nmode=demo_clean\ntotal_jobs=1\ntasks=bad_task\n",
        encoding="utf-8",
    )
    (root / "summary.tsv").write_text(
        "task\tmode\tnode\tworker\tstatus\texit_code\tlog\n"
        f"bad_task\tdemo_clean\t0\t0\tfailed\t1\t{outside}\n",
        encoding="utf-8",
    )

    snapshot = make_builder(console, root).build()

    assert snapshot["failures"][0]["log"] == str(outside)
    assert snapshot["failures"][0]["snippet"] == ""
    assert "secret should not be exposed" not in json.dumps(snapshot)


def test_web_control_auto_detects_robotwin_and_compat_exports_builder(tmp_path):
    console = load_console_module()
    compat = load_compat_module()
    root = tmp_path
    (root / "run.env").write_text(
        "run_id=detect\npolicy_name=openwam\nmode=demo_clean\ntotal_jobs=0\ntasks=\n",
        encoding="utf-8",
    )

    adapter = console.create_adapter(
        root,
        "auto",
        max_logs=100,
        state_tail_bytes=100_000,
        max_task_log_bytes=100_000,
        max_error_snippets=10,
    )

    assert console.detect_benchmark(root) == "robotwin"
    assert isinstance(adapter, console.SnapshotBuilder)
    assert compat.SnapshotBuilder is console.SnapshotBuilder
    assert adapter.build_state()["benchmark"] == "robotwin"


def test_generic_adapter_lists_logs_and_csv_rows(tmp_path):
    console = load_console_module()
    root = tmp_path
    log_path = root / "plain.log"
    log_path.write_text("hello\nworld\n", encoding="utf-8")

    adapter = console.create_adapter(
        root,
        "generic",
        max_logs=100,
        state_tail_bytes=100_000,
        max_task_log_bytes=100_000,
        max_error_snippets=10,
    )
    state = adapter.build_state()
    rows = adapter.build_results_rows()

    assert state["benchmark"] == "generic"
    assert state["progress"]["total"] == 1
    assert state["custom_metrics"] == []
    assert state["logs"][0]["rel"] == "plain.log"
    assert rows[0]["log_path"] == "plain.log"


def test_resolve_requested_file_blocks_path_traversal(tmp_path):
    console = load_console_module()
    root = tmp_path / "logs"
    root.mkdir()
    (root / "safe.log").write_text("ok\n", encoding="utf-8")
    outside = tmp_path / "outside.log"
    outside.write_text("secret\n", encoding="utf-8")

    assert console.resolve_requested_file(root, "safe.log").name == "safe.log"
    try:
        console.resolve_requested_file(root, "../outside.log")
    except ValueError as exc:
        assert "inside log_dir" in str(exc)
    else:
        raise AssertionError("path traversal should be rejected")


def test_http_handler_serves_state_tail_and_csv(tmp_path):
    console = load_console_module()
    root = tmp_path
    (root / "plain.log").write_text("alpha\nbeta\n", encoding="utf-8")
    handler = console.build_handler(
        root,
        benchmark="generic",
        tail_bytes=100,
        state_tail_bytes=100_000,
        max_logs=100,
        max_task_log_bytes=100_000,
        max_error_snippets=10,
        refresh_sec=2.0,
    )

    class Harness(handler):
        def __init__(self):
            self.status = None
            self.headers = {}
            self.body = bytearray()
            self.path = "/"
            self.wfile = self
            self.server = SimpleNamespace()
            self.client_address = ("127.0.0.1", 0)

        def send_response(self, code, message=None):
            self.status = code

        def send_header(self, key, value):
            self.headers[key] = value

        def end_headers(self):
            pass

        def write(self, data):
            self.body.extend(data)

        def log_message(self, fmt, *args):
            pass

        def get(self, path):
            self.status = None
            self.headers = {}
            self.body = bytearray()
            self.path = path
            self.do_GET()
            return self.status, bytes(self.body), self.headers

    client = Harness()
    status, body, _ = client.get("/api/state")
    state = json.loads(body)
    assert status == 200
    assert state["benchmark"] == "generic"
    assert state["custom_metrics"] == []

    status, body, _ = client.get("/api/tail?file=plain.log&offset=-1&max_bytes=100")
    tail = json.loads(body)
    assert status == 200
    assert "beta" in tail["data"]

    status, body, _ = client.get("/api/results.csv")
    csv_body = body.decode("utf-8")
    assert status == 200
    assert "plain.log" in csv_body
