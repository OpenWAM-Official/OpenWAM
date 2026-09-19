#!/usr/bin/env python3
"""One native Labtasker Worker with its own exclusive policy endpoint."""

from __future__ import annotations

import argparse
import signal
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import labtasker
import labtasker_runtime as runtime
from labtasker_runtime import project_context

from benchmarks.utils.eval_manifest import load_cached_manifest, manifest_index_exists
from benchmarks.utils.task_policy import WorkerPolicyServer, write_policy_config


def run_worker(
    args: argparse.Namespace, worker_resources: runtime.WorkerResources, server: WorkerPolicyServer | None
) -> None:
    clients = runtime.ProcessRegistry()

    @labtasker.loop(
        route=args.route,
        queue=args.queue,
        idle_timeout=args.idle_timeout,
        max_consecutive_failures=args.max_consecutive_failures,
        metadata={
            "node": socket.gethostname(),
            "gpu": worker_resources.gpu,
            "render_gpu": worker_resources.render_gpu,
            "port": worker_resources.port,
        },
    )
    def run_task() -> None:
        task_info = labtasker.task_info()
        inputs = dict(task_info.args)
        operation = inputs.get("operation")
        suite, task_id = inputs.get("suite"), inputs.get("task_id")
        total_trials = inputs.get("total_trials")
        if (
            operation != args.operation
            or not isinstance(suite, str)
            or type(task_id) is not int
            or task_id < 0
            or type(total_trials) is not int
            or total_trials <= 0
            or not isinstance(inputs.get("policy_config"), dict)
        ):
            raise ValueError("invalid LIBERO Task input")
        config = SimpleNamespace(**vars(args))
        config.seed = inputs.get("seed")
        config.total_trials, config.smoke = total_trials, total_trials == 1
        attempt_dir = args.output_dir / "attempts" / task_info.id / task_info.run_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        config.policy_config = write_policy_config(inputs["policy_config"], attempt_dir)
        if operation == "build_manifest":
            config.manifest_dir = Path(inputs["manifest_dir"])
            config.manifest_cache_key = runtime.manifest_cache_key(inputs)
            if not inputs["rebuild"]:
                if manifest_index_exists(config.manifest_dir, config.manifest_cache_key):
                    path, manifest = load_cached_manifest(config.manifest_dir, config.manifest_cache_key)
                    runtime.validate_manifest(
                        manifest,
                        runtime.BenchmarkTask(suite, task_id),
                        minimum_trials=total_trials,
                    )
                    print(f"Using cached manifest {path}", flush=True)
                    labtasker.finish(
                        dict(
                            operation="build_manifest",
                            suite=suite,
                            task_id=task_id,
                            total_trials=total_trials,
                            manifest_hash=manifest["manifest_hash"],
                            manifest_path=str(path),
                        )
                    )
                    return
            print(
                f"[manifest] attempt={task_info.attempt} suite={suite} task_id={task_id} total={total_trials}",
                flush=True,
            )
            try:
                path, payload = runtime._run_manifest(
                    config,
                    benchmark_task=runtime.BenchmarkTask(suite, task_id),
                    total_trials=total_trials,
                    render_gpu=worker_resources.render_gpu,
                    attempt_dir=attempt_dir,
                    registry=clients,
                )
            finally:
                clients.terminate_all()
            labtasker.finish(payload)
            return

        trial_start, num_trials = inputs.get("trial_start"), inputs.get("num_trials")
        if (
            type(trial_start) is not int
            or trial_start < 0
            or type(num_trials) is not int
            or num_trials <= 0
            or trial_start + num_trials > total_trials
            or not all(isinstance(inputs.get(key), str) for key in ("manifest_hash", "manifest_path"))
        ):
            raise ValueError("invalid LIBERO eval Task range")
        manifest = runtime.load_manifest(
            Path(inputs["manifest_path"]),
            expected_manifest_hash=inputs["manifest_hash"],
        )
        runtime.validate_manifest(
            manifest,
            runtime.BenchmarkTask(suite, task_id),
            minimum_trials=total_trials,
        )
        runtime.select_entries(manifest, trial_start, num_trials)
        print(
            f"[eval] attempt={task_info.attempt} suite={suite} task_id={task_id} "
            f"trials={trial_start}:{trial_start + num_trials} "
            f"manifest={inputs['manifest_hash']}",
            flush=True,
        )
        server.load_or_reuse(inputs["model"])
        labtasker.report_worker_telemetry(
            {"model_id": inputs["model"]["identity"], "checkpoint": inputs["model"]["checkpoint"]}
        )
        config.trial_start, config.num_trials = trial_start, num_trials
        config.manifest_path = Path(inputs["manifest_path"])
        config.manifest_hash = inputs["manifest_hash"]
        runtime.validate_libero_eval_config(inputs["policy_config"])
        try:
            path, payload = runtime._run_client(
                config,
                worker_resources=worker_resources,
                benchmark_task=runtime.BenchmarkTask(suite, task_id),
                trial_batch=runtime.TrialBatch(trial_start, num_trials),
                registry=clients,
                attempt=task_info.attempt,
                run_id=task_info.run_id,
                server=server,
            )
        except Exception:
            server.close()
            raise
        finally:
            clients.terminate_all()
        labtasker.finish(
            {
                key: payload[key]
                for key in (
                    "suite",
                    "task_id",
                    "trial_start",
                    "num_trials",
                    "successes",
                    "trials",
                    "manifest_hash",
                )
            }
            | {"total_trials": total_trials}
            | {"artifact_dir": str(path.parent)}
        )

    try:
        run_task()
    finally:
        clients.terminate_all()


def main(argv: list[str] | None = None) -> int:
    parser = runtime._build_parser("worker")
    args = parser.parse_args(argv)
    args.route = args.route or f"openwam-libero-{args.operation}"
    if args.port is None:
        args.port = 8920 + args.gpu
    if min(args.gpu, args.render_gpu if args.render_gpu is not None else 0) < 0:
        parser.error("GPU IDs must be non-negative")
    if not 0 < args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.libero_path is None or args.libero_python is None:
        parser.error("set LIBERO_PATH and LIBERO_PYTHON or their CLI options")
    args.output_dir = args.output_dir or runtime.DEFAULT_OUTPUT_ROOT / args.route
    runtime._normalize_paths(args)
    runtime._check_worker_environment(args)
    server = None
    if args.operation == "run_eval":
        server = WorkerPolicyServer(args)
    worker_resources = runtime.WorkerResources(
        args.gpu, args.render_gpu if args.render_gpu is not None else args.gpu, args.port
    )
    try:
        if args.auto_start_local_server:
            with project_context(), runtime.create_client(queue=args.queue, auto_start_local_server=True) as client:
                client.count_tasks()
        print(f"[worker] route={args.route} operation={args.operation} gpu={worker_resources.gpu}", flush=True)
        with project_context():
            run_worker(args, worker_resources, server)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        if server is not None:
            server.close()


def interrupt(signum, frame):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    sys.exit(main())
