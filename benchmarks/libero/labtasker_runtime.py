#!/usr/bin/env python3
"""Shared LIBERO runtime: native Tasks, evaluation resources and metric export."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import labtasker
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.utils.eval_manifest import (  # noqa: E402
    load_manifest,
    require_manifests,  # noqa: E402
    select_entries,  # noqa: E402, F401
)
from benchmarks.utils.eval_sharding import split_range  # noqa: E402
from benchmarks.utils.labtasker_utils import (  # noqa: E402
    count_task_statuses,
    create_client,
    list_submission_tasks,
    print_submission_id,
    print_summary_table,
    project_context,
    submit_tasks,
    validate_submission_coverage,
)
from benchmarks.utils.labtasker_utils import new_submission_id as new_submission_id  # noqa: E402, F401
from benchmarks.utils.owned_processes import ProcessRegistry  # noqa: E402, F401
from benchmarks.utils.owned_processes import spawn_resource as _spawn_resource  # noqa: E402
from benchmarks.utils.rng_domain import decode_numpy_state  # noqa: E402
from benchmarks.utils.task_policy import WorkerPolicyServer, load_policy_config, resolve_policy_model_spec  # noqa: E402
from benchmarks.utils.task_progress import ProgressFileReporter  # noqa: E402

DEFAULT_POLICY_CONFIG = SCRIPT_DIR / "policy_config.yml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "libero"
REQUIRED_MUJOCO_VERSION = "3.3.2"

SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "goal": "libero_goal",
    "object": "libero_object",
    # LIBERO calls the LIBERO-LONG suite ``libero_10`` in its Python API.
    "long": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_goal": "libero_goal",
    "libero_object": "libero_object",
    "libero_long": "libero_10",
    "libero_10": "libero_10",
}


@dataclass(frozen=True)
class BenchmarkTask:
    suite: str
    task_id: int


@dataclass(frozen=True)
class TrialBatch:
    trial_start: int
    num_trials: int

    @property
    def trial_stop(self) -> int:
        return self.trial_start + self.num_trials


@dataclass(frozen=True)
class WorkerResources:
    gpu: int
    render_gpu: int
    port: int


def validate_manifest(manifest: dict[str, Any], benchmark_task: BenchmarkTask, *, minimum_trials: int) -> None:
    entries = manifest.get("entries")
    assets = (manifest.get("init_state_asset"), manifest.get("bddl_asset"))
    valid_assets = all(
        isinstance(asset, dict)
        and isinstance(asset.get("name"), str)
        and bool(asset["name"])
        and isinstance(asset.get("sha256"), str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", asset["sha256"]) is not None
        for asset in assets
    )
    if (
        manifest.get("benchmark") != "libero"
        or manifest.get("suite") != benchmark_task.suite
        or manifest.get("task_id") != benchmark_task.task_id
        or manifest.get("seed") != 42
        or not isinstance(entries, list)
        or len(entries) < minimum_trials
        or not valid_assets
    ):
        raise ValueError("invalid LIBERO manifest; rebuild the cache with --operation build_manifest --rebuild")
    for trial, entry in enumerate(entries):
        if (
            entry.get("trial") != trial
            or isinstance(entry.get("init_state_index"), bool)
            or not isinstance(entry.get("init_state_index"), int)
            or entry["init_state_index"] < 0
        ):
            raise ValueError(f"invalid LIBERO manifest entry {trial}")
        try:
            decode_numpy_state(entry.get("pre_reset_numpy_state"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid LIBERO manifest RNG state at trial {trial}: {exc}") from exc


def _csv_items(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return items


def _resolve_suites(value: str) -> list[str]:
    suites = []
    for raw_name in _csv_items(value):
        key = raw_name.lower()
        if key not in SUITE_ALIASES:
            choices = ", ".join(("spatial", "goal", "object", "long"))
            raise argparse.ArgumentTypeError(f"unknown suite {raw_name!r}; use {choices}")
        suite = SUITE_ALIASES[key]
        if suite not in suites:
            suites.append(suite)
    return suites


def _resolve_task_ids(value: str) -> list[int] | None:
    if value == "all":
        return None
    try:
        task_ids = [int(item) for item in _csv_items(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("task IDs must be comma-separated integers or 'all'") from exc
    if any(task_id < 0 or task_id >= 10 for task_id in task_ids):
        raise argparse.ArgumentTypeError("LIBERO task IDs must be in 0..9")
    return list(dict.fromkeys(task_ids))


def _write_libero_config(config_root: Path, libero_path: Path) -> None:
    benchmark_root = libero_path / "libero" / "libero"
    config_root.mkdir(parents=True, exist_ok=True)
    values = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": libero_path / "datasets",
        "assets": benchmark_root / "assets",
    }
    (config_root / "config.yaml").write_text(
        yaml.safe_dump({key: str(path) for key, path in values.items()}), encoding="utf-8"
    )


def _build_client_environment(
    base_env: Mapping[str, str], *, libero_path: Path, config_root: Path, render_gpu: int
) -> dict[str, str]:
    """Set isolated LIBERO paths and conservative thread defaults."""
    env = dict(base_env)
    old_pythonpath = env.get("PYTHONPATH", "")
    pieces = [str(REPO_ROOT), str(libero_path), str(SCRIPT_DIR)]
    if old_pythonpath:
        pieces.append(old_pythonpath)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pieces),
            "LIBERO_PATH": str(libero_path),
            "LIBERO_CONFIG_ROOT": str(config_root),
            "LIBERO_CONFIG_PATH": str(config_root),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "CUDA_VISIBLE_DEVICES": str(render_gpu),
            "MUJOCO_EGL_DEVICE_ID": str(render_gpu),
            "PYTHONUNBUFFERED": "1",
            # Each task is a separate process.  Avoid multiplying BLAS/OpenMP
            # threads by the number of replicas on a GPU.
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def _get_mujoco_version(libero_python: Path) -> str:
    result = subprocess.run(
        [str(libero_python), "-c", "import mujoco; print(mujoco.__version__)"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to query MuJoCo version:\n{result.stdout}")
    return result.stdout.strip().splitlines()[-1]


def _read_valid_result(
    path: Path, benchmark_task: BenchmarkTask, trial_batch: TrialBatch, *, manifest_hash: str
) -> dict[str, Any]:
    """Validate only this execution's result, never search earlier attempts."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    trials = payload["trials"]
    valid = (
        payload["suite"] == benchmark_task.suite
        and all(
            type(payload[key]) is int for key in ("task_id", "trial_start", "trial_stop", "successes", "num_trials")
        )
        and payload["task_id"] == benchmark_task.task_id
        and payload["trial_start"] == trial_batch.trial_start
        and payload["trial_stop"] == trial_batch.trial_stop
        and isinstance(trials, list)
        and len(trials) == trial_batch.num_trials
        and all(type(t["trial"]) is int and type(t["success"]) is bool for t in trials)
        and {t["trial"] for t in trials} == set(range(trial_batch.trial_start, trial_batch.trial_stop))
        and payload["num_trials"] == len(trials)
        and payload["successes"] == sum(t["success"] for t in trials)
        and payload.get("manifest_hash") == manifest_hash
    )
    if not valid:
        raise ValueError(f"LIBERO client result is inconsistent: {path}")
    return payload


def _run_output_dir(output_dir: Path, benchmark_task: BenchmarkTask, trial_batch: TrialBatch) -> Path:
    return (
        output_dir
        / "videos"
        / benchmark_task.suite
        / f"task_{benchmark_task.task_id:02d}"
        / f"trials_{trial_batch.trial_start:03d}_{trial_batch.trial_stop - 1:03d}"
    )


def _build_client_command(
    args: argparse.Namespace,
    benchmark_task: BenchmarkTask,
    trial_batch: TrialBatch,
    port: int,
    result_dir: Path,
    progress_file: Path,
) -> list[str]:
    """Build a request for the canonical native-action client."""
    command = [
        str(args.libero_python),
        str(SCRIPT_DIR / "single_eval.py"),
        "--config",
        str(args.policy_config),
        "--suite",
        benchmark_task.suite,
        "--task-id",
        str(benchmark_task.task_id),
        "--host",
        args.host,
        "--port",
        str(port),
        "--trial-start",
        str(trial_batch.trial_start),
        "--num-trials",
        str(trial_batch.num_trials),
        "--result-dir",
        str(result_dir),
        "--progress-file",
        str(progress_file),
        "--manifest-file",
        str(args.manifest_path),
        "--manifest-hash",
        args.manifest_hash,
        "--total-trials",
        str(args.total_trials),
    ]
    if args.seed is not None:
        command.extend(("--seed", str(args.seed)))
    return command


def _run_manifest(
    args: argparse.Namespace,
    *,
    benchmark_task: BenchmarkTask,
    total_trials: int,
    render_gpu: int,
    attempt_dir: Path,
    registry: ProcessRegistry,
) -> tuple[Path, dict[str, Any]]:
    """Generate one reset-only LIBERO manifest in the pinned client runtime."""

    proposed = attempt_dir / "proposed-manifest.json"
    progress_path = attempt_dir / "progress.json"
    log_path = attempt_dir / "manifest.log"
    config_root = attempt_dir / "libero_config"
    _write_libero_config(config_root, args.libero_path)
    command = [
        str(args.libero_python),
        str(SCRIPT_DIR / "eval_manifest.py"),
        "--config",
        str(args.policy_config),
        "--suite",
        benchmark_task.suite,
        "--task-id",
        str(benchmark_task.task_id),
        "--total-trials",
        str(total_trials),
        "--result-file",
        str(proposed),
        "--progress-file",
        str(progress_path),
    ]
    reporter = ProgressFileReporter(
        progress_path,
        base={"operation": "build_manifest", "suite": benchmark_task.suite, "task_id": benchmark_task.task_id},
    )
    env = _build_client_environment(
        os.environ,
        libero_path=args.libero_path,
        config_root=config_root,
        render_gpu=render_gpu,
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = _spawn_resource(command, cwd=REPO_ROOT, env=env, stdout=log)
        registry.add(process)
        while process.poll() is None and not labtasker.cancellation_requested():
            reporter.poll()
            time.sleep(0.2)
        reporter.poll()
        if process.poll() is None:
            raise RuntimeError("Task ownership revoked")
        if process.returncode != 0:
            raise RuntimeError(f"LIBERO manifest builder exited {process.returncode}; see {log_path}")
    manifest = load_manifest(proposed)
    validate_manifest(manifest, benchmark_task, minimum_trials=total_trials)
    if len(manifest["entries"]) != total_trials:
        raise ValueError("LIBERO manifest builder produced too many entries")
    from benchmarks.utils.eval_manifest import publish_manifest

    sealed = publish_manifest(args.manifest_dir, args.manifest_cache_key, manifest)
    return sealed, {
        "operation": "build_manifest",
        "suite": benchmark_task.suite,
        "task_id": benchmark_task.task_id,
        "total_trials": total_trials,
        "manifest_hash": manifest["manifest_hash"],
        "manifest_path": str(sealed),
    }


def _run_client(
    args: argparse.Namespace,
    *,
    worker_resources: WorkerResources,
    benchmark_task: BenchmarkTask,
    trial_batch: TrialBatch,
    registry: ProcessRegistry,
    attempt: int,
    run_id: str,
    server: WorkerPolicyServer,
) -> tuple[Path, dict[str, Any]]:
    output_dir = args.output_dir
    run_dir = _run_output_dir(output_dir, benchmark_task, trial_batch)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        output_dir
        / "logs"
        / "clients"
        / (
            f"{benchmark_task.suite}_task{benchmark_task.task_id:02d}_"
            f"trials{trial_batch.trial_start:03d}-{trial_batch.trial_stop - 1:03d}_"
            f"attempt{attempt:02d}_{run_id}.log"
        )
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config_root = output_dir / "libero_configs" / run_id
    _write_libero_config(config_root, args.libero_path)
    result_dir = run_dir / run_id
    progress_path = result_dir / "progress.json"
    command = _build_client_command(args, benchmark_task, trial_batch, worker_resources.port, result_dir, progress_path)
    reporter = ProgressFileReporter(
        progress_path,
        base={
            "operation": "run_eval",
            "suite": benchmark_task.suite,
            "task_id": benchmark_task.task_id,
            "trial_start": trial_batch.trial_start,
        },
    )
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(f"command: {shlex.join(command)}\n")
    log_handle.write(f"policy_gpu: {worker_resources.gpu}\nrender_gpu: {worker_resources.render_gpu}\n")
    log_handle.flush()
    env = _build_client_environment(
        os.environ,
        libero_path=args.libero_path,
        config_root=config_root,
        render_gpu=worker_resources.render_gpu,
    )
    try:
        process = _spawn_resource(command, cwd=REPO_ROOT, env=env, stdout=log_handle)
    except BaseException:
        log_handle.close()
        raise
    registry.add(process)
    print(
        f"[start] gpu={worker_resources.gpu} port={worker_resources.port} "
        f"render_gpu={worker_resources.render_gpu} "
        f"{benchmark_task.suite}/task{benchmark_task.task_id:02d} "
        f"trials={trial_batch.trial_start}:{trial_batch.trial_stop} attempt={attempt}",
        flush=True,
    )

    try:
        while process.poll() is None and not labtasker.cancellation_requested():
            reporter.poll()
            time.sleep(0.2)
        reporter.poll()
        return_code = process.poll()
        if return_code is None:
            raise RuntimeError("LIBERO client cancelled")
        if return_code != 0:
            print(
                f"[FAIL] gpu={worker_resources.gpu} "
                f"{benchmark_task.suite}/task{benchmark_task.task_id:02d} "
                f"trials={trial_batch.trial_start}:{trial_batch.trial_stop} "
                f"exit={return_code} log={log_path}",
                flush=True,
            )
            raise RuntimeError(f"LIBERO client exited with code {return_code}: {log_path}")
        result = run_dir / run_id / "results.json"
        payload = _read_valid_result(result, benchmark_task, trial_batch, manifest_hash=args.manifest_hash)
        print(
            f"[done] gpu={worker_resources.gpu} "
            f"{benchmark_task.suite}/task{benchmark_task.task_id:02d} "
            f"trials={trial_batch.trial_start}:{trial_batch.trial_stop}",
            flush=True,
        )
        return result, payload
    finally:
        log_handle.close()


def _aggregate_task_results(group: Sequence[labtasker.Task]) -> dict[str, Any]:
    results = []
    for task in group:
        if task.status == "succeeded":
            result = validated_task_result(task)
            if result is None:
                raise ValueError(f"invalid LIBERO result: {task.id}")
            results.append(result)
    successes = sum(result["successes"] for result in results)
    trials = sum(result["num_trials"] for result in results)
    return {
        "successes": successes,
        "trials": trials,
        "success_rate": successes / trials if trials else None,
        **count_task_statuses(group),
    }


def _summarize(output_dir: Path, tasks: Sequence[labtasker.Task]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Write suite and task views from authoritative Labtasker state."""

    grouped: dict[tuple[str, int], list[labtasker.Task]] = {}
    for task in tasks:
        key = (task.args.get("suite"), task.args.get("task_id"))
        if not isinstance(key[0], str) or type(key[1]) is not int:
            raise ValueError(f"invalid LIBERO eval Task: {task.id}")
        grouped.setdefault(key, []).append(task)

    task_rows = [
        {"suite": suite, "task_id": task_id, **_aggregate_task_results(group)}
        for (suite, task_id), group in sorted(grouped.items())
    ]
    suite_rows = []
    for suite in dict.fromkeys(row["suite"] for row in task_rows):
        rows = [row for row in task_rows if row["suite"] == suite]
        successes = sum(row["successes"] for row in rows)
        trials = sum(row["trials"] for row in rows)
        suite_rows.append(
            {
                "suite": suite,
                "successes": successes,
                "trials": trials,
                "success_rate": successes / trials if trials else None,
                **{
                    key: sum(row[key] for row in rows)
                    for key in ("succeeded", "pending", "running", "failed", "expected")
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        "successes",
        "trials",
        "success_rate",
        "succeeded",
        "pending",
        "running",
        "failed",
        "expected",
    )
    for path, fields, rows in (
        (output_dir / "task_summary.csv", ("suite", "task_id", *metrics), task_rows),
        (output_dir / "suite_summary.csv", ("suite", *metrics), suite_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def rate(row: dict[str, Any]) -> str:
        return "-" if row["success_rate"] is None else f"{100 * row['success_rate']:.2f}%"

    print_summary_table(
        "Suite summary",
        ("Suite", "Success/Trials", "Success rate", "Done/Total", "Pending", "Running", "Failed"),
        [
            (
                row["suite"],
                f"{row['successes']}/{row['trials']}",
                rate(row),
                f"{row['succeeded']}/{row['expected']}",
                row["pending"],
                row["running"],
                row["failed"],
            )
            for row in suite_rows
        ],
    )
    print()
    print_summary_table(
        "Task summary",
        ("Suite", "Task", "Success/Trials", "Success rate", "Done/Total", "Pending", "Running", "Failed"),
        [
            (
                row["suite"],
                row["task_id"],
                f"{row['successes']}/{row['trials']}",
                rate(row),
                f"{row['succeeded']}/{row['expected']}",
                row["pending"],
                row["running"],
                row["failed"],
            )
            for row in task_rows
        ],
    )
    print("\nOutput files")
    print(f"  Suite summary: {output_dir / 'suite_summary.csv'}")
    print(f"  Task summary:  {output_dir / 'task_summary.csv'}")
    return task_rows, suite_rows


def _normalize_paths(args: argparse.Namespace) -> None:
    """Resolve user paths before selecting the project-local Labtasker instance."""
    for key in ("server_python", "libero_python", "libero_path", "policy_config", "output_dir"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(args, key, value.expanduser().absolute())


def _check_worker_environment(args: argparse.Namespace) -> None:
    required = [args.libero_python, SCRIPT_DIR / "single_eval.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if not args.libero_path.is_dir():
        missing.append(str(args.libero_path))
    if missing:
        raise FileNotFoundError("required paths are missing: " + ", ".join(missing))
    version = _get_mujoco_version(args.libero_python)
    if version != REQUIRED_MUJOCO_VERSION:
        raise RuntimeError(f"requires MuJoCo {REQUIRED_MUJOCO_VERSION}, found {version}")


def validate_libero_eval_config(config: Mapping[str, Any]) -> None:
    if str(config.get("action_mode", "")).strip().lower() != "eef":
        raise ValueError("canonical LIBERO requires action_mode: eef")
    expected_steps = {
        "libero_spatial": 600,
        "libero_object": 600,
        "libero_goal": 600,
        "libero_10": 700,
    }
    errors = []
    if config.get("max_steps") != 600 or config.get("max_steps_by_suite") != expected_steps:
        errors.append("SPATIAL/OBJECT/GOAL must use 600 steps and LONG must use 700 steps")
    if config.get("settle_steps") != 30 or config.get("settle_action") != [0, 0, 0, 0, 0, 0, -1]:
        errors.append("settling must use 30 actions with gripper=-1")
    if config.get("seed") != 42:
        errors.append(f"seed must be 42, got {config.get('seed')!r}")
    if config.get("rng_mode") != "environment":
        errors.append("rng_mode must be 'environment'")
    if config.get("reseed_each_trial") is not False:
        errors.append("reseed_each_trial must be false")
    if errors:
        raise ValueError("invalid canonical LIBERO evaluation config:\n  - " + "\n  - ".join(errors))


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value: str) -> float:
    number = float(value)
    if number < 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def _build_parser(mode: str = "submit") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    worker = mode == "worker"
    parser.add_argument("--route", help="Task route (default: openwam-libero-<operation>)")
    parser.add_argument("--queue")
    parser.add_argument(
        "--auto-start-local-server",
        action="store_true",
        help="explicitly authorize startup of the project-local Labtasker Server",
    )
    parser.add_argument("--operation", choices=("build_manifest", "run_eval"), default="run_eval")
    if not worker:
        parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
        parser.add_argument("--suites", type=_resolve_suites, default=_resolve_suites("spatial,goal,object,long"))
        parser.add_argument("--task-ids", type=_resolve_task_ids, default=None)
        parser.add_argument("--num-trials", type=_positive)
        parser.add_argument("--trial-batch-size", type=_positive, default=5)
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--manifest-dir", type=Path, default=REPO_ROOT / "outputs/manifest-cache/libero")
        parser.add_argument("--rebuild", action="store_true")
        parser.add_argument("--max-attempts", type=_positive, default=3)
        parser.add_argument("--priority", type=int, default=0)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--submission-id")
        parser.add_argument("--note")
        parser.set_defaults(deploy_args=[])
        parser.epilog = "Use -- followed by native OpenWAM deploy arguments (run_eval only)."
        return parser
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--libero-python", type=Path, default=os.environ.get("LIBERO_PYTHON"))
    parser.add_argument("--libero-path", type=Path, default=os.environ.get("LIBERO_PATH"))
    parser.add_argument("--server-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--server-start-timeout", type=_positive, default=1200)
    parser.add_argument("--idle-timeout", type=_nonnegative, default=300)
    parser.add_argument("--max-consecutive-failures", type=_positive, default=5)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--render-gpu", type=int)
    parser.add_argument("--port", type=int, help="Policy port (default: 8920 + GPU ID)")
    return parser


def manifest_cache_key(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": "libero",
        "suite": inputs["suite"],
        "task_id": inputs["task_id"],
        "seed": 42,
    }


def build_task_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    _normalize_paths(args)
    args.manifest_dir = args.manifest_dir.expanduser().resolve()
    if args.operation == "build_manifest" and args.deploy_args:
        raise ValueError("deploy arguments after -- are only accepted for run_eval")
    if args.operation == "run_eval" and args.rebuild:
        raise ValueError("--rebuild is only accepted for build_manifest")
    if args.smoke and (args.num_trials is not None or args.task_ids is not None):
        raise ValueError("--smoke cannot be combined with --num-trials or --task-ids")
    suites = args.suites[:1] if args.smoke else args.suites
    # Canonical pinned suites; submission does not import the simulator.
    counts = {"libero_spatial": 10, "libero_goal": 10, "libero_object": 10, "libero_10": 10}
    config = load_policy_config(args.policy_config)
    validate_libero_eval_config(config)
    total_trials = 1 if args.smoke else (args.num_trials if args.num_trials is not None else 50)
    manifests = [
        dict(
            operation="build_manifest",
            suite=suite,
            task_id=task_id,
            total_trials=total_trials,
            seed=42,
            policy_config=config,
            manifest_dir=str(args.manifest_dir),
            rebuild=args.rebuild,
        )
        for suite in suites
        for task_id in (
            [0] if args.smoke else (args.task_ids if args.task_ids is not None else list(range(counts[suite])))
        )
    ]
    if args.operation == "build_manifest":
        return manifests
    task_selector = "" if args.task_ids is None else " --task-ids " + ",".join(map(str, args.task_ids))
    trial_count = "" if args.smoke else f" --num-trials {total_trials}"
    cached = require_manifests(
        args.manifest_dir,
        [manifest_cache_key(item) for item in manifests],
        "python benchmarks/libero/labtasker_submit.py --operation build_manifest --suites "
        + ",".join(suites)
        + (" --smoke" if args.smoke else "")
        + task_selector
        + trial_count
        + f" --manifest-dir {str(args.manifest_dir)!r}",
    )
    model = resolve_policy_model_spec(args.deploy_args)
    inputs = []
    for item, (path, manifest) in zip(manifests, cached, strict=True):
        validate_manifest(
            manifest,
            BenchmarkTask(item["suite"], item["task_id"]),
            minimum_trials=item["total_trials"],
        )
        print(f"Using cached manifest {path}")
        for batch in split_range(0, item["total_trials"], args.trial_batch_size):
            inputs.append(
                {key: item[key] for key in ("suite", "task_id", "total_trials", "seed", "policy_config")}
                | dict(
                    operation="run_eval",
                    trial_start=batch.start,
                    num_trials=batch.count,
                    manifest_hash=manifest["manifest_hash"],
                    manifest_path=str(path),
                    model=model,
                )
            )
    return inputs


def list_libero_submission_tasks(
    client: labtasker.Client, submission_id: str, operation: str | None = None
) -> list[labtasker.Task]:
    return list_submission_tasks(client, "libero", submission_id, operation)


def submit(
    inputs: Iterable[dict[str, Any]],
    *,
    submission_id: str,
    route: str,
    queue: str | None = None,
    max_attempts: int = 3,
    priority: int = 0,
    note: str | None = None,
    auto_start_local_server: bool = False,
) -> list[str]:
    """Retry the original definition; never rewrite or requeue an existing Task."""
    inputs = list(inputs)
    print(
        f"[submission] id={submission_id} operation={inputs[0]['operation'] if inputs else '-'} "
        f"queue={queue or 'default'} route={route}",
        flush=True,
    )
    print_submission_id(submission_id)
    grouped_counts: dict[tuple[str, int], int] = {}
    for item in inputs:
        if item["operation"] == "run_eval":
            key = (item["suite"], item["task_id"])
            grouped_counts[key] = grouped_counts.get(key, 0) + 1
    grouped_indices: dict[tuple[str, int], int] = {}
    names: list[str] = []
    for item in inputs:
        if item["operation"] == "build_manifest":
            display = f"libero({item['suite']}):task{item['task_id']:02d}_build_manifest"
        else:
            key = (item["suite"], item["task_id"])
            grouped_indices[key] = grouped_indices.get(key, 0) + 1
            total = grouped_counts[key]
            width = max(2, len(str(total)))
            display = (
                f"libero({item['suite']}):task{item['task_id']:02d}_{grouped_indices[key]:0{width}d}_{total:0{width}d}"
            )
        names.append(display + (f"_{note}" if note else ""))
    with project_context(), create_client(queue=queue, auto_start_local_server=auto_start_local_server) as client:
        task_ids = submit_tasks(
            client,
            inputs,
            benchmark="libero",
            submission_id=submission_id,
            route=route,
            names=names,
            max_attempts=max_attempts,
            priority=priority,
        )
    print(f"[submitted] id={submission_id} confirmed={len(task_ids)}/{len(inputs)}", flush=True)
    print_submission_id(submission_id)
    return task_ids


def validated_task_result(task: labtasker.Task) -> dict[str, Any] | None:
    result = task.result
    if task.status != "succeeded" or not isinstance(result, dict) or task.args.get("operation") != "run_eval":
        return None
    args = task.args
    if (
        not isinstance(args.get("suite"), str)
        or not all(isinstance(args.get(key), str) for key in ("manifest_hash", "manifest_path"))
        or any(type(args.get(key)) is not int for key in ("task_id", "trial_start", "num_trials", "total_trials"))
    ):
        return None
    if not all(type(result.get(key)) is int for key in ("successes", "num_trials")):
        return None
    if not 0 <= result["successes"] <= result["num_trials"] or result["num_trials"] <= 0:
        return None
    trials = result.get("trials")
    if not isinstance(trials, list) or len(trials) != result["num_trials"]:
        return None
    if not all(
        type(row.get("trial")) is int
        and type(row.get("init_state_index")) is int
        and type(row.get("success")) is bool
        and type(row.get("policy_steps")) is int
        and isinstance(row.get("last_reward"), (int, float))
        for row in trials
    ):
        return None
    expected = set(range(args["trial_start"], args["trial_start"] + args["num_trials"]))
    if {row["trial"] for row in trials} != expected or result["successes"] != sum(row["success"] for row in trials):
        return None
    if any(
        result.get(key) != args.get(key)
        for key in (
            "suite",
            "task_id",
            "trial_start",
            "num_trials",
            "total_trials",
            "manifest_hash",
        )
    ):
        return None
    return result


def summarize(
    submission_id: str,
    output: str | Path,
    *,
    queue: str | None = None,
    auto_start_local_server: bool = False,
    skip_coverage_check: bool = False,
) -> int:
    with project_context(), create_client(queue=queue, auto_start_local_server=auto_start_local_server) as client:
        tasks = list_libero_submission_tasks(client, submission_id, "run_eval")
    if not tasks:
        raise ValueError(f"LIBERO eval submission not found: {submission_id}")
    validate_submission_coverage(tasks, skip_coverage_check=skip_coverage_check)
    _summarize(Path(output), tasks)
    return 0


def parse_submit_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse submission options strictly and forward only arguments after --."""
    argv = list(sys.argv[1:] if argv is None else argv)
    separator = argv.index("--") if "--" in argv else len(argv)
    args = _build_parser("submit").parse_args(argv[:separator])
    args.deploy_args = argv[separator + 1 :]
    return args
