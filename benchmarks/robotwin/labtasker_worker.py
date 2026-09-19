"""One native Worker and one exclusive policy server; no local retry loop."""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

import labtasker
import labtasker_runtime as rt

from benchmarks.utils.eval_manifest import load_cached_manifest, manifest_index_exists, publish_manifest
from benchmarks.utils.task_policy import WorkerPolicyServer, write_policy_config
from benchmarks.utils.task_progress import ProgressFileReporter


def interrupted(signum, frame):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt


def build_manifest(args: argparse.Namespace, task_info: labtasker.TaskInfo, inputs: dict[str, Any]) -> dict[str, Any]:
    task, mode = inputs["task"], inputs["mode"]
    total = inputs["total_episodes"]
    if not inputs["rebuild"]:
        cache_key = rt.manifest_cache_key(inputs)
        if manifest_index_exists(inputs["manifest_dir"], cache_key):
            path, cached = load_cached_manifest(inputs["manifest_dir"], cache_key)
            rt.validate_manifest(cached, inputs, minimum_entries=total)
            print(f"Using cached manifest {path}", flush=True)
            return {
                "operation": "build_manifest",
                "task": task,
                "mode": mode,
                "total_episodes": total,
                "manifest_hash": cached["manifest_hash"],
                "manifest_path": str(path),
            }

    attempt_dir = args.output_dir / "attempts" / task_info.id / task_info.run_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    policy_config = write_policy_config(inputs["policy_config"], attempt_dir)
    log_path = attempt_dir / "manifest.log"
    result_path = attempt_dir / "manifest.json"
    progress_path = attempt_dir / "progress.json"
    reporter = ProgressFileReporter(
        progress_path,
        base={"operation": "build_manifest", "task": task, "mode": mode},
    )
    command = [
        str(args.robotwin_python),
        str(rt.HERE / "eval_policy_wrapper.py"),
        "labtasker",
        "--task",
        task,
        "--mode",
        mode,
        "--policy-config",
        str(policy_config),
        "--operation",
        "build_manifest",
        "--result-file",
        str(result_path),
        "--progress-file",
        str(progress_path),
        "--total-episodes",
        str(total),
    ]
    if args.render_device is not None:
        command.extend(["--render-device", args.render_device])
    resources = rt.ProcessRegistry()
    try:
        with log_path.open("w") as log:
            process = rt.spawn_resource(
                command,
                cwd=rt.ROOT,
                stdout=log,
                env=rt.build_client_environment(args, inputs, attempt_dir),
            )
            resources.add(process)
            while process.poll() is None:
                if labtasker.cancellation_requested():
                    raise RuntimeError("Task ownership revoked")
                reporter.poll()
                time.sleep(0.1)
            reporter.poll()
            if process.returncode != 0:
                raise RuntimeError(f"RoboTwin manifest builder exited {process.returncode}; see {log_path}")
        payload = rt.read_client_result(result_path, inputs)
        rt.validate_manifest(payload, inputs, minimum_entries=total)
        manifest_path = publish_manifest(inputs["manifest_dir"], rt.manifest_cache_key(inputs), payload)
    finally:
        resources.terminate_all()

    result = {
        "operation": "build_manifest",
        "task": task,
        "mode": mode,
        "total_episodes": total,
        "manifest_hash": payload["manifest_hash"],
        "manifest_path": str(manifest_path),
    }
    return result


def evaluate_batch(
    args: argparse.Namespace, server: WorkerPolicyServer, task_info: labtasker.TaskInfo, inputs: dict[str, Any]
) -> dict[str, Any]:
    task, mode = inputs["task"], inputs["mode"]
    if (
        type(inputs.get("episode_start")) is not int
        or type(inputs.get("num_episodes")) is not int
        or inputs["episode_start"] < 0
        or inputs["num_episodes"] <= 0
        or inputs["episode_start"] + inputs["num_episodes"] > inputs["total_episodes"]
        or not all(isinstance(inputs.get(key), str) for key in ("manifest_hash", "manifest_path"))
    ):
        raise ValueError("invalid RoboTwin eval Task range")
    manifest = rt.load_manifest(
        Path(inputs["manifest_path"]),
        expected_manifest_hash=inputs["manifest_hash"],
    )
    rt.validate_manifest(manifest, inputs, minimum_entries=inputs["total_episodes"])
    rt.select_entries(manifest, inputs["episode_start"], inputs["num_episodes"])
    print(
        f"[eval] attempt={task_info.attempt} task={task} mode={mode} "
        f"episodes={inputs['episode_start']}:{inputs['episode_start'] + inputs['num_episodes']} "
        f"manifest={inputs['manifest_hash']}",
        flush=True,
    )
    server.load_or_reuse(inputs["model"])
    labtasker.report_worker_telemetry(
        {"model_id": inputs["model"]["identity"], "checkpoint": inputs["model"]["checkpoint"]}
    )
    attempt_dir = args.output_dir / "attempts" / task_info.id / task_info.run_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    policy_config = write_policy_config(inputs["policy_config"], attempt_dir)
    resources = rt.ProcessRegistry()
    log_path = attempt_dir / "client.log"
    result_path = attempt_dir / "result.json"
    progress_path = attempt_dir / "progress.json"
    reporter = ProgressFileReporter(
        progress_path,
        base={"operation": "run_eval", "task": task, "mode": mode, "episode_start": inputs["episode_start"]},
    )
    try:
        command = [
            str(args.robotwin_python),
            str(rt.HERE / "eval_policy_wrapper.py"),
            "labtasker",
            "--task",
            task,
            "--mode",
            mode,
            "--policy-config",
            str(policy_config),
            "--instruction-type",
            inputs["instruction_type"],
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--operation",
            "run_eval",
            "--result-file",
            str(result_path),
            "--progress-file",
            str(progress_path),
            "--total-episodes",
            str(inputs["total_episodes"]),
        ]
        command.extend(
            [
                "--episode-start",
                str(inputs["episode_start"]),
                "--num-episodes",
                str(inputs["num_episodes"]),
                "--manifest-file",
                inputs["manifest_path"],
                "--manifest-hash",
                inputs["manifest_hash"],
            ]
        )
        if args.render_device is not None:
            command.extend(["--render-device", args.render_device])
        with log_path.open("w") as log:
            process = rt.spawn_resource(
                command, cwd=rt.ROOT, stdout=log, env=rt.build_client_environment(args, inputs, attempt_dir)
            )
            resources.add(process)
            while process.poll() is None:
                if labtasker.cancellation_requested():
                    raise RuntimeError("Task ownership revoked")
                reporter.poll()
                time.sleep(0.1)
            reporter.poll()
            if process.returncode != 0:
                raise RuntimeError(f"RoboTwin client exited {process.returncode}; see {log_path}")
        payload = rt.read_client_result(result_path, inputs)
        payload["artifact_dir"] = str(attempt_dir)
    except Exception:
        server.close()
        raise
    finally:
        resources.terminate_all()
    return payload


def run_worker(args: argparse.Namespace, server: WorkerPolicyServer | None) -> None:
    @labtasker.loop(
        route=args.route,
        queue=args.queue,
        idle_timeout=args.idle_timeout,
        max_consecutive_failures=args.max_consecutive_failures,
        metadata={"node": socket.gethostname(), "gpu": args.gpu, "port": args.port},
    )
    def run_task() -> None:
        task_info = labtasker.task_info()
        inputs = dict(task_info.args)
        if (
            inputs.get("operation") != args.operation
            or inputs.get("task") not in rt.task_names()
            or inputs.get("mode") not in ("demo_clean", "demo_randomized")
            or type(inputs.get("total_episodes")) is not int
            or inputs["total_episodes"] <= 0
            or not isinstance(inputs.get("policy_config"), dict)
        ):
            raise ValueError("invalid RoboTwin Task input")
        if args.operation == "build_manifest":
            result = build_manifest(args, task_info, inputs)
        else:
            result = evaluate_batch(args, server, task_info, inputs)
        labtasker.finish(result)

    run_task()


def main(argv: list[str] | None = None) -> int:
    parser = rt.parser("worker")
    args = parser.parse_args(argv)
    args.route = args.route or f"openwam-robotwin-{args.operation}"
    if args.port is None:
        args.port = 8848 + args.gpu
    if args.gpu < 0 or not 0 < args.port < 65536:
        parser.error("invalid GPU or port")
    if args.robotwin_path is None or args.robotwin_python is None:
        parser.error("set ROBOTWIN_PATH and ROBOTWIN_PYTHON or their CLI options")
    args.output_dir = args.output_dir or rt.ROOT / "outputs/robotwin" / args.route
    rt.normalize_paths(args)
    if not (args.robotwin_path / "script/eval_policy.py").is_file() or not args.robotwin_python.is_file():
        parser.error("RoboTwin evaluator or Python is missing")
    server = None
    if args.operation == "run_eval":
        server = WorkerPolicyServer(args)
    try:
        if args.auto_start_local_server:
            with rt.project_context(), rt.create_client(queue=args.queue, auto_start_local_server=True) as client:
                client.count_tasks()
        print(f"[worker] route={args.route} operation={args.operation} gpu={args.gpu}", flush=True)
        with rt.project_context():
            run_worker(args, server)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
