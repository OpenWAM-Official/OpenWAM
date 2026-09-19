"""CPU subprocess stand-ins for policy serving and the unmodified episode loop."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

args = sys.argv[1:]
if Path(args[0]).name == "deploy.py":
    if marker := os.environ.get("FAKE_SERVER_MARKER"):
        Path(marker).write_text(os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    if os.environ.get("FAKE_SERVER_FAIL_GPU") == os.environ.get("CUDA_VISIBLE_DEVICES", "unset"):
        raise SystemExit(1)
    port = int(args[args.index("--port") + 1])
    readiness_token = args[args.index("--readiness-token") + 1]
    import json

    from websockets.sync.server import serve

    def respond(ws):
        for message in ws:
            ws.send(json.dumps({"type": "pong", "readiness_token": readiness_token}))

    with serve(respond, "127.0.0.1", port) as server:
        server.serve_forever()

else:
    parser = argparse.ArgumentParser()
    for key in (
        "task",
        "mode",
        "policy-config",
        "host",
        "port",
        "render-device",
        "operation",
        "result-file",
        "progress-file",
        "total-episodes",
        "episode-start",
        "num-episodes",
        "manifest-file",
        "manifest-hash",
        "instruction-type",
    ):
        parser.add_argument("--" + key)
    evaluator_args = args[2:] if args[1:2] == ["labtasker"] else args[1:]
    options = parser.parse_args(evaluator_args)
    attempt = Path(os.environ["ROBOTWIN_RUNTIME_ROOT"]).parent
    (attempt / "client.pid").write_text(str(os.getpid()))
    if options.operation == "run_eval":
        (attempt / "port").write_text(options.port)
    child = None

    def stop(*args):
        if child is not None:
            child.terminate()
            child.wait()
        raise SystemExit(130)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if os.environ.get("FAKE_HANG"):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        (attempt / "grandchild.pid").write_text(str(child.pid))
        time.sleep(120)
    time.sleep(float(os.environ.get("FAKE_DELAY", ".01")))
    if os.environ.get("FAKE_FAIL_TASK") == options.task:
        raise SystemExit(1)
    marker = attempt.parent / "first-failure"
    if (
        os.environ.get("FAKE_FAIL_ONCE")
        and options.task == "adjust_bottle"
        and (not os.environ.get("FAKE_FAIL_OPERATION") or os.environ["FAKE_FAIL_OPERATION"] == options.operation)
        and (options.operation != "run_eval" or options.episode_start == "0")
        and not marker.exists()
    ):
        marker.touch()
        raise SystemExit(1)
    result_path = Path(options.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(options.total_episodes)
    if options.operation == "build_manifest":
        if os.environ.get("FAKE_MANIFEST_FAIL"):
            raise SystemExit(1)
        from benchmarks.utils.eval_manifest import seal_manifest
        from benchmarks.utils.task_progress import write_progress

        entries = []
        for index in range(total):
            entries.append(
                {
                    "episode": index,
                    "seed": 100000 + index * 2,
                    "instructions": {"seen": f"seen-{index}", "unseen": f"instruction-{index}"},
                }
            )
            write_progress(
                Path(options.progress_file),
                {"completed": index + 1, "total": total, "candidate_seed": 100000 + index * 2},
            )
        payload = seal_manifest(
            {"benchmark": "robotwin", "task": options.task, "mode": options.mode, "entries": entries}
        )
    else:
        import json

        from benchmarks.utils.task_progress import write_progress

        manifest = json.loads(Path(options.manifest_file).read_text())
        start = int(options.episode_start)
        count = int(options.num_episodes)
        entries = manifest["entries"][start : start + count]
        rows = []
        for index, entry in enumerate(entries):
            rows.append(
                {
                    "episode": start + index,
                    "seed": entry["seed"],
                    "instruction": entry["instructions"][options.instruction_type],
                    "success": not bool(os.environ.get("FAKE_ZERO")),
                }
            )
            write_progress(
                Path(options.progress_file),
                {
                    "completed": len(rows),
                    "total": count,
                    "episode": start + index,
                    "successes": sum(row["success"] for row in rows),
                },
            )
            time.sleep(float(os.environ.get("FAKE_BATCH_PROGRESS_DELAY", "0")))
        payload = {
            "operation": "run_eval",
            "instruction_type": options.instruction_type,
            "task": options.task,
            "mode": options.mode,
            "episode_start": start,
            "num_episodes": len(rows),
            "total_episodes": total,
            "manifest_hash": options.manifest_hash,
            "successes": sum(row["success"] for row in rows),
            "episodes": rows,
        }
    import json

    result_path.write_text(json.dumps(payload))
