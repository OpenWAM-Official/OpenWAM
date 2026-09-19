"""CPU stand-in for LIBERO's interpreter and a listening policy server."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if Path(sys.argv[1]).name == "labtasker_worker.py":
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
if Path(sys.argv[1]).name == "deploy.py":
    if os.environ.get("FAKE_SERVER_FAIL") or os.environ.get("FAKE_SERVER_FAIL_GPU") == os.environ.get(
        "CUDA_VISIBLE_DEVICES", "unset"
    ):
        sys.exit(1)
    sys.argv = [
        sys.argv[0],
        "--fake-server",
        sys.argv[sys.argv.index("--port") + 1],
        sys.argv[sys.argv.index("--readiness-token") + 1],
    ]
if sys.argv[1] == "-c":
    if "mujoco" in sys.argv[2]:
        print("3.3.2")
    else:
        print("OPENWAM_TASK_COUNTS=" + json.dumps({suite: 4 for suite in sys.argv[3:]}))
    sys.exit(0)
if sys.argv[1] == "--fake-server":
    from websockets.sync.server import serve

    def respond(ws):
        for message in ws:
            ws.send(json.dumps({"type": "pong", "readiness_token": sys.argv[3]}))

    with serve(respond, "127.0.0.1", int(sys.argv[2])) as server:
        server.serve_forever()
if Path(sys.argv[1]).name == "eval_manifest.py":
    if os.environ.get("FAKE_MANIFEST_FAIL"):
        raise SystemExit(1)
    planner = argparse.ArgumentParser()
    for name in ("config", "suite", "task-id", "total-trials", "result-file"):
        planner.add_argument("--" + name, required=True)
    planner.add_argument("--progress-file")
    options = planner.parse_args(sys.argv[2:])
    from benchmarks.utils.eval_manifest import seal_manifest, write_manifest
    from benchmarks.utils.rng_domain import encode_numpy_state
    from benchmarks.utils.task_progress import write_progress

    total = int(options.total_trials)
    np.random.seed(42)
    entries = []
    for index in range(total):
        entries.append(
            {
                "trial": index,
                "init_state_index": index,
                "pre_reset_numpy_state": encode_numpy_state(np.random.get_state()),
            }
        )
        np.random.random()
        if options.progress_file:
            write_progress(
                Path(options.progress_file),
                {"completed": index + 1, "total": total, "trial": index},
            )
    manifest = seal_manifest(
        {
            "benchmark": "libero",
            "suite": options.suite,
            "task_id": int(options.task_id),
            "seed": 42,
            "init_state_asset": {"name": "fake.pruned_init", "sha256": "sha256:" + "1" * 64},
            "bddl_asset": {"name": "fake.bddl", "sha256": "sha256:" + "2" * 64},
            "entries": entries,
        }
    )
    write_manifest(Path(options.result_file), manifest)
    sys.exit(0)
parser = argparse.ArgumentParser()
for name in (
    "suite",
    "task-id",
    "trial-start",
    "num-trials",
    "result-dir",
    "port",
    "manifest-file",
    "manifest-hash",
    "total-trials",
    "progress-file",
):
    parser.add_argument("--" + name, required=True)
args, _ = parser.parse_known_args(sys.argv[2:])
output = Path(args.result_dir)
output.mkdir(parents=True, exist_ok=True)
(output / "client.pid").write_text(str(os.getpid()))
if os.environ.get("FAKE_HANG"):
    time.sleep(120)
if os.environ.get("FAKE_FAIL_ALWAYS") and args.task_id == "0":
    sys.exit(1)
if os.environ.get("FAKE_FAIL") and args.task_id == "0" and args.trial_start == "0":
    marker = output.parent / "failed_once"
    if not marker.exists():
        marker.touch()
        sys.exit(1)
time.sleep(float(os.environ.get("FAKE_DELAY", "0.3")))
start, count = int(args.trial_start), int(args.num_trials)
manifest = json.loads(Path(args.manifest_file).read_text())
entries = manifest["entries"][start : start + count]
from benchmarks.utils.task_progress import write_progress  # noqa: E402

rows = []
for i, entry in zip(range(start, start + count), entries):
    rows.append(
        {
            "trial": i,
            "success": True,
            "init_state_index": entry["init_state_index"],
            "policy_steps": 1,
            "last_reward": 1.0,
        }
    )
    write_progress(
        Path(args.progress_file),
        {"completed": len(rows), "total": count, "trial": i, "successes": len(rows)},
    )
    time.sleep(float(os.environ.get("FAKE_BATCH_PROGRESS_DELAY", "0")))
(output / "results.json").write_text(
    json.dumps(
        {
            "suite": args.suite,
            "task_id": int(args.task_id),
            "trial_start": start,
            "trial_stop": start + count,
            "manifest_hash": args.manifest_hash,
            "total_trials": int(args.total_trials),
            "trials": rows,
            "successes": count,
            "num_trials": count,
            "port": int(args.port),
        }
    )
)
