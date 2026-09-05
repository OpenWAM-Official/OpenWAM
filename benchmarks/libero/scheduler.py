#!/usr/bin/env python3
"""Shared scheduler for the canonical OpenWAM LIBERO benchmark.

Every policy-server replica pulls its next task from one shared request queue.
A failed client request is placed at the back of that queue (up to the configured
attempt limit), so fast replicas keep working and a retry is not tied to the GPU
on which it first failed. A task's contiguous trial range still runs in one
client/environment, preserving its RNG stream.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import queue
import random
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_DIR = Path("/path/to/openwam_checkpoints/new-openwam-libero-sft-10epoch-final")
DEFAULT_CKPT_NAME = "checkpoint_step_10850.safetensors"
DEFAULT_LIBERO_PATH = Path("/path/to/LIBERO")
DEFAULT_LIBERO_PYTHON = Path("/path/to/miniconda3/envs/libero/bin/python")
DEFAULT_SERVER_PYTHON = Path("/usr/bin/python3.12")
DEFAULT_POLICY_CONFIG = SCRIPT_DIR / "policy_config.yml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "libero"
REQUIRED_MUJOCO_VERSION = "3.3.2"
LIBERO_PROTOCOL_VERSION = "openwam-libero-native-action-seed42-settle30-v1"

# Scheduling-only fields may change across a checkpointed evaluation. They do
# not alter simulator state, observations, policy settings, or task selection.
RESUME_OPERATIONAL_FIELDS = frozenset({"render_gpus"})

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
class TaskJob:
    suite: str
    task_id: int


@dataclass(frozen=True)
class TrialRun:
    trial_start: int
    num_trials: int

    @property
    def trial_stop(self) -> int:
        return self.trial_start + self.num_trials


@dataclass(frozen=True)
class ReplicaSlot:
    gpu: int
    gpu_slot: int
    replica: int
    port: int


@dataclass(frozen=True)
class QueuedTask:
    job: TaskJob
    attempt: int = 1


@dataclass
class ServerProcess:
    gpu: int
    replica: int
    port: int
    process: subprocess.Popen
    log_handle: object


class ProcessRegistry:
    """Track only subprocesses created by this scheduler for scoped cleanup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: list[subprocess.Popen] = []

    def add(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.append(process)

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and any(p.poll() is None for p in processes):
            time.sleep(0.2)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass


def _csv_items(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return items


def _parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(item) for item in _csv_items(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid GPU list: {value!r}") from exc
    if any(gpu < 0 for gpu in gpus):
        raise argparse.ArgumentTypeError("GPU ids must be non-negative")
    if len(gpus) != len(set(gpus)):
        raise argparse.ArgumentTypeError("GPU ids must be unique")
    return gpus


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


def _parse_task_ids(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    try:
        task_ids = [int(item) for item in _csv_items(value)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid task id list: {value!r}") from exc
    if any(task_id < 0 for task_id in task_ids):
        raise argparse.ArgumentTypeError("task ids must be non-negative")
    if len(task_ids) != len(set(task_ids)):
        raise argparse.ArgumentTypeError("task ids must be unique")
    return task_ids


def _parse_task_sample_ratio(value: str) -> float | None:
    """Parse an optional per-suite task sampling ratio."""
    ratio = float(value)
    if ratio <= 0.0 or ratio > 1.0:
        raise argparse.ArgumentTypeError("task sample ratio must be in (0, 1]")
    return None if ratio >= 1.0 else ratio


def _server_ports(base_port: int, gpu_slot: int, replicas_per_gpu: int = 2) -> tuple[int, ...]:
    start = base_port + replicas_per_gpu * gpu_slot
    return tuple(start + replica for replica in range(replicas_per_gpu))


def _build_jobs(
    suites: Iterable[str],
    task_counts: dict[str, int],
    selected_task_ids: list[int] | None,
    *,
    sample_ratio: float | None = None,
    sample_seed: int = 42,
) -> list[TaskJob]:
    if selected_task_ids is not None and sample_ratio is not None:
        raise ValueError("--task-ids and --task-sample-ratio cannot be used together")
    jobs = []
    for suite in suites:
        count = task_counts[suite]
        if sample_ratio is not None:
            # Adapted from ImageWAM's MIT-licensed LIBERO task sampler. Sample
            # independently per suite and restore task-id order for stable output.
            sample_count = max(1, int(math.ceil(count * sample_ratio)))
            rng = random.Random(f"{sample_seed}:{suite}")
            task_ids = sorted(rng.sample(range(count), sample_count))
        else:
            task_ids = range(count) if selected_task_ids is None else selected_task_ids
        invalid = [task_id for task_id in task_ids if task_id >= count]
        if invalid:
            raise ValueError(f"{suite} has {count} tasks; invalid task ids: {invalid}")
        jobs.extend(TaskJob(suite, task_id) for task_id in task_ids)
    return jobs


def _build_replica_slots(gpus: list[int], base_port: int, replicas_per_gpu: int = 2) -> list[ReplicaSlot]:
    return [
        ReplicaSlot(gpu=gpu, gpu_slot=gpu_slot, replica=replica, port=port)
        for gpu_slot, gpu in enumerate(gpus)
        for replica, port in enumerate(_server_ports(base_port, gpu_slot, replicas_per_gpu))
    ]


def _render_device_for_slot(args: argparse.Namespace, slot: ReplicaSlot) -> int:
    """Choose an EGL device independently of the policy-server CUDA device."""
    render_gpus = args.render_gpus if args.render_gpus is not None else args.gpus
    return render_gpus[slot.gpu_slot % len(render_gpus)]


def _assign_jobs(jobs: list[TaskJob], slots: list[ReplicaSlot]) -> dict[ReplicaSlot, list[TaskJob]]:
    """Balance tasks across GPUs, then alternate them between each GPU's replicas."""
    if not slots:
        raise ValueError("at least one replica slot is required")
    assignments = {slot: [] for slot in slots}
    gpu_order = list(dict.fromkeys(slot.gpu for slot in slots))
    slots_by_gpu = {
        gpu: sorted((slot for slot in slots if slot.gpu == gpu), key=lambda slot: slot.replica) for gpu in gpu_order
    }
    assigned_per_gpu = {gpu: 0 for gpu in gpu_order}
    for index, job in enumerate(jobs):
        gpu = gpu_order[index % len(gpu_order)]
        gpu_slots = slots_by_gpu[gpu]
        slot = gpu_slots[assigned_per_gpu[gpu] % len(gpu_slots)]
        assignments[slot].append(job)
        assigned_per_gpu[gpu] += 1
    return assignments


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
    # All current paths are plain absolute POSIX paths, so JSON strings are
    # valid YAML scalars and avoid adding a PyYAML dependency to this launcher.
    text = "".join(f"{key}: {json.dumps(str(path))}\n" for key, path in values.items())
    (config_root / "config.yaml").write_text(text, encoding="utf-8")


def _client_env(
    base_env: dict[str, str],
    *,
    libero_path: Path,
    config_root: Path,
    render_gpu: int,
) -> dict[str, str]:
    env = dict(base_env)
    old_pythonpath = env.get("PYTHONPATH", "")
    pieces = [str(libero_path), str(SCRIPT_DIR)]
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
            # The client performs no model inference locally. Rendering can be
            # kept off a saturated policy GPU, but CUDA visibility and MuJoCo's
            # physical EGL selection must still name the same device.
            "CUDA_VISIBLE_DEVICES": str(render_gpu),
            "MUJOCO_EGL_DEVICE_ID": str(render_gpu),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _enumerate_task_counts(suites: list[str], *, libero_python: Path, libero_path: Path) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="openwam-libero-enumerate-") as raw_tmp:
        config_root = Path(raw_tmp)
        _write_libero_config(config_root, libero_path)
        env = _client_env(
            os.environ,
            libero_path=libero_path,
            config_root=config_root,
            render_gpu=0,
        )
        code = (
            "import json,sys\n"
            "from libero.libero import benchmark\n"
            "names=sys.argv[1:]\n"
            "mapping=benchmark.get_benchmark_dict()\n"
            "counts={name:mapping[name]().get_num_tasks() for name in names}\n"
            "print('OPENWAM_TASK_COUNTS='+json.dumps(counts, sort_keys=True))\n"
        )
        result = subprocess.run(
            [str(libero_python), "-c", code, *suites],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
    marker = "OPENWAM_TASK_COUNTS="
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(marker):
            return {key: int(value) for key, value in json.loads(line[len(marker) :]).items()}
    raise RuntimeError(f"failed to enumerate LIBERO tasks (exit={result.returncode}):\n{result.stdout}")


def _mujoco_version(libero_python: Path) -> str:
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


def _port_is_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _websocket_server_is_ready(host: str, port: int) -> bool:
    """Perform a valid application-level ping without polluting server logs."""
    try:
        from websockets.exceptions import WebSocketException
        from websockets.sync.client import connect

        with connect(
            f"ws://{host}:{port}",
            open_timeout=1,
            close_timeout=1,
            ping_interval=None,
            proxy=None,
        ) as connection:
            connection.send(json.dumps({"type": "ping"}))
            response = json.loads(connection.recv(timeout=1))
        return response.get("type") == "pong"
    except (OSError, TimeoutError, ValueError, WebSocketException):
        return False


def _wait_for_servers(servers: list[ServerProcess], host: str, timeout: int) -> None:
    pending = {(server.gpu, server.replica): server for server in servers}
    started = time.monotonic()
    last_report = 0.0
    while pending:
        for key, server in list(pending.items()):
            return_code = server.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"server gpu={server.gpu} replica={server.replica} port={server.port} "
                    f"exited with code {return_code}; see its server log"
                )
            if _websocket_server_is_ready(host, server.port):
                print(
                    f"[ready] gpu={server.gpu} replica={server.replica} port={server.port}",
                    flush=True,
                )
                pending.pop(key)
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            waiting = ", ".join(f"gpu{s.gpu}/:{s.port}" for s in pending.values())
            raise TimeoutError(f"servers did not become ready within {timeout}s: {waiting}")
        if pending and elapsed - last_report >= 15:
            print(f"[wait] loading {len(pending)} server(s), elapsed={elapsed:.0f}s", flush=True)
            last_report = elapsed
        time.sleep(1)


def _latest_valid_result(run_dir: Path, job: TaskJob, trial_run: TrialRun) -> Path | None:
    candidates = sorted(
        run_dir.glob("*/results.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    expected_trials = set(range(trial_run.trial_start, trial_run.trial_stop))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            actual_trials = {int(trial["trial"]) for trial in payload["trials"]}
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if (
            payload.get("suite") == job.suite
            and int(payload.get("task_id", -1)) == job.task_id
            and int(payload.get("trial_start", -1)) == trial_run.trial_start
            and int(payload.get("trial_stop", -1)) == trial_run.trial_stop
            and actual_trials == expected_trials
        ):
            return path
    return None


def _run_output_dir(output_dir: Path, job: TaskJob, trial_run: TrialRun) -> Path:
    return (
        output_dir
        / "videos"
        / job.suite
        / f"task_{job.task_id:02d}"
        / f"trials_{trial_run.trial_start:03d}_{trial_run.trial_stop - 1:03d}"
    )


def _unfinished_jobs(output_dir: Path, jobs: Iterable[TaskJob], trial_run: TrialRun) -> list[TaskJob]:
    """Return unfinished jobs so resumed evaluations can rebalance their queues."""
    return [
        job for job in jobs if _latest_valid_result(_run_output_dir(output_dir, job, trial_run), job, trial_run) is None
    ]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resume_signature(
    args: argparse.Namespace,
    *,
    jobs: list[TaskJob],
    trial_run: TrialRun,
    mujoco_version: str,
) -> dict:
    policy_config = yaml.safe_load(args.policy_config.read_text(encoding="utf-8")) or {}
    effective_seed = int(args.seed if args.seed is not None else policy_config.get("seed", 42))
    jobs_payload = [job.__dict__ for job in jobs]
    jobs_sha256 = hashlib.sha256(
        json.dumps(jobs_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "protocol_version": LIBERO_PROTOCOL_VERSION,
        "checkpoint": str(args.ckpt_dir / args.ckpt_name),
        "policy_config_sha256": _file_sha256(args.policy_config),
        "seed_override": args.seed,
        "effective_seed": effective_seed,
        "mujoco_version": mujoco_version,
        "inference_mode": args.inference_mode,
        "inference_horizon": args.inference_horizon,
        "denoise_mode": args.denoise_mode,
        "denoise_steps": args.denoise_steps,
        "trial_start": trial_run.trial_start,
        "num_trials": trial_run.num_trials,
        "jobs_sha256": jobs_sha256,
        "jobs_count": len(jobs),
        "render_gpus": args.render_gpus if args.render_gpus is not None else args.gpus,
    }


def _validate_resume_signature(output_dir: Path, expected: dict) -> None:
    """Refuse to combine results produced by different evaluation protocols."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        videos_dir = output_dir / "videos"
        if videos_dir.exists() and any(videos_dir.rglob("results.json")):
            raise RuntimeError(
                f"cannot resume {output_dir}: results exist but manifest.json is missing; use a new output directory"
            )
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read resume manifest: {manifest_path}") from exc
    actual = manifest.get("resume_signature")
    if actual == expected:
        return
    if not isinstance(actual, dict):
        raise RuntimeError(
            f"cannot resume legacy evaluation at {output_dir}: it has no protocol signature; use a new output directory"
        )
    differing = sorted(
        key
        for key in set(actual) | set(expected)
        if key not in RESUME_OPERATIONAL_FIELDS and actual.get(key) != expected.get(key)
    )
    if not differing:
        return
    details = ", ".join(f"{key}={actual.get(key)!r}->{expected.get(key)!r}" for key in differing)
    raise RuntimeError(f"cannot resume {output_dir} with a different evaluation protocol: {details}")


def _client_command(
    args: argparse.Namespace,
    job: TaskJob,
    trial_run: TrialRun,
    port: int,
    run_dir: Path,
) -> list[str]:
    command = [
        str(args.libero_python),
        str(SCRIPT_DIR / "single_eval.py"),
        "--config",
        str(args.policy_config),
        "--suite",
        job.suite,
        "--task-id",
        str(job.task_id),
        "--host",
        args.host,
        "--port",
        str(port),
        "--trial-start",
        str(trial_run.trial_start),
        "--num-trials",
        str(trial_run.num_trials),
        "--result-dir",
        str(run_dir / "attempt_01"),
    ]
    if args.seed is not None:
        command.extend(("--seed", str(args.seed)))
    return command


def _run_client(
    args: argparse.Namespace,
    *,
    slot: ReplicaSlot,
    job: TaskJob,
    trial_run: TrialRun,
    output_dir: Path,
    registry: ProcessRegistry,
    stop_event: threading.Event,
    attempt: int = 1,
) -> bool:
    run_dir = _run_output_dir(output_dir, job, trial_run)
    previous = _latest_valid_result(run_dir, job, trial_run)
    if previous is not None:
        print(
            f"[skip] gpu={slot.gpu} replica={slot.replica} "
            f"{job.suite}/task{job.task_id:02d} "
            f"trials={trial_run.trial_start}:{trial_run.trial_stop} result={previous}",
            flush=True,
        )
        return True

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        output_dir
        / "logs"
        / "clients"
        / (
            f"{job.suite}_task{job.task_id:02d}_"
            f"trials{trial_run.trial_start:03d}-{trial_run.trial_stop - 1:03d}_"
            f"attempt{attempt:02d}.log"
        )
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config_root = output_dir / "libero_configs" / f"gpu{slot.gpu}_replica{slot.replica}"
    _write_libero_config(config_root, args.libero_path)
    command = _client_command(args, job, trial_run, slot.port, run_dir)
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(f"command: {shlex.join(command)}\n")
    log_handle.write(f"policy_gpu: {slot.gpu}\nrender_gpu: {_render_device_for_slot(args, slot)}\n")
    log_handle.flush()
    env = _client_env(
        os.environ,
        libero_path=args.libero_path,
        config_root=config_root,
        render_gpu=_render_device_for_slot(args, slot),
    )
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    registry.add(process)
    print(
        f"[start] gpu={slot.gpu} replica={slot.replica} port={slot.port} "
        f"render_gpu={_render_device_for_slot(args, slot)} "
        f"{job.suite}/task{job.task_id:02d} "
        f"trials={trial_run.trial_start}:{trial_run.trial_stop} attempt={attempt}",
        flush=True,
    )

    try:
        while process.poll() is None and not stop_event.wait(1):
            pass
        return_code = process.poll()
        if return_code is None:
            return False
        if return_code != 0:
            print(
                f"[FAIL] gpu={slot.gpu} replica={slot.replica} "
                f"{job.suite}/task{job.task_id:02d} "
                f"trials={trial_run.trial_start}:{trial_run.trial_stop} "
                f"exit={return_code} log={log_path}",
                flush=True,
            )
            return False
        result = _latest_valid_result(run_dir, job, trial_run)
        if result is None:
            print(f"[FAIL] client exited 0 but result is missing: {log_path}", flush=True)
            return False
        print(
            f"[done] gpu={slot.gpu} replica={slot.replica} "
            f"{job.suite}/task{job.task_id:02d} "
            f"trials={trial_run.trial_start}:{trial_run.trial_stop}",
            flush=True,
        )
        return True
    finally:
        log_handle.close()


def _dynamic_worker(
    args: argparse.Namespace,
    *,
    slot: ReplicaSlot,
    work_queue: queue.Queue[QueuedTask | None],
    trial_run: TrialRun,
    output_dir: Path,
    registry: ProcessRegistry,
    stop_event: threading.Event,
) -> list[TaskJob]:
    """Pull tasks on demand and return failed requests to the shared queue.

    Consecutive client crashes open a per-worker circuit breaker.  The worker
    pauses before pulling another request, then rejoins the queue instead of
    retiring permanently.  Native simulator / EGL failures can be transient;
    permanently removing a healthy policy endpoint after a short burst leaves
    the rest of a long evaluation needlessly under-provisioned.
    """
    failed = []
    consecutive_failures = 0
    initial_delay = (slot.port - args.base_port) * args.client_start_stagger
    if initial_delay > 0 and stop_event.wait(initial_delay):
        return failed
    while True:
        if stop_event.is_set():
            return failed
        try:
            queued = work_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            if queued is None:
                return failed
            job = queued.job
            attempt = queued.attempt
            if stop_event.is_set():
                failed.append(job)
                continue
            try:
                succeeded = _run_client(
                    args,
                    slot=slot,
                    job=job,
                    trial_run=trial_run,
                    output_dir=output_dir,
                    registry=registry,
                    stop_event=stop_event,
                    attempt=attempt,
                )
            except Exception as exc:
                print(
                    f"[FAIL] gpu={slot.gpu} replica={slot.replica} "
                    f"{job.suite}/task{job.task_id:02d} attempt={attempt} "
                    f"scheduler_exception={type(exc).__name__}: {exc}",
                    flush=True,
                )
                succeeded = False
            if succeeded:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if stop_event.is_set():
                failed.append(job)
                continue
            if attempt >= args.client_max_attempts:
                print(
                    f"[exhausted] {job.suite}/task{job.task_id:02d} attempts={attempt}",
                    flush=True,
                )
                failed.append(job)
                if consecutive_failures >= args.worker_max_consecutive_failures:
                    print(
                        f"[cooldown] gpu={slot.gpu} replica={slot.replica} "
                        f"port={slot.port} consecutive_failures={consecutive_failures} "
                        f"delay={args.worker_recovery_delay:g}s",
                        flush=True,
                    )
                    if stop_event.wait(args.worker_recovery_delay):
                        return failed
                    consecutive_failures = 0
                    print(
                        f"[recovered] gpu={slot.gpu} replica={slot.replica} port={slot.port} pulling from shared queue",
                        flush=True,
                    )
                continue
            next_attempt = attempt + 1
            if stop_event.wait(args.client_retry_delay):
                failed.append(job)
                continue
            work_queue.put(QueuedTask(job=job, attempt=next_attempt))
            print(
                f"[requeue] gpu={slot.gpu} replica={slot.replica} "
                f"{job.suite}/task{job.task_id:02d} "
                f"next_attempt={next_attempt}/{args.client_max_attempts}",
                flush=True,
            )
            if consecutive_failures >= args.worker_max_consecutive_failures:
                print(
                    f"[cooldown] gpu={slot.gpu} replica={slot.replica} "
                    f"port={slot.port} consecutive_failures={consecutive_failures} "
                    f"delay={args.worker_recovery_delay:g}s",
                    flush=True,
                )
                if stop_event.wait(args.worker_recovery_delay):
                    return failed
                consecutive_failures = 0
                print(
                    f"[recovered] gpu={slot.gpu} replica={slot.replica} port={slot.port} pulling from shared queue",
                    flush=True,
                )
        finally:
            work_queue.task_done()


def _wait_for_dynamic_queue(
    work_queue: queue.Queue[QueuedTask | None],
    futures: list[concurrent.futures.Future],
    *,
    poll_interval: float = 0.5,
) -> None:
    """Wait for all requests without hanging after every worker retires."""
    drained = threading.Event()

    def wait_until_drained() -> None:
        work_queue.join()
        drained.set()

    waiter = threading.Thread(target=wait_until_drained, name="libero-queue-waiter", daemon=True)
    waiter.start()
    while not drained.wait(poll_interval):
        if not futures or not all(future.done() for future in futures):
            continue

        # No worker can consume the remaining requests. Drain only queued
        # entries (all workers are already done) so the waiter can terminate,
        # then fail explicitly instead of blocking forever in Queue.join().
        abandoned = 0
        while True:
            try:
                work_queue.get_nowait()
            except queue.Empty:
                break
            else:
                abandoned += 1
                work_queue.task_done()
        drained.wait(max(poll_interval, 0.1))
        raise RuntimeError(
            f"all {len(futures)} dynamic worker(s) exited with {abandoned} queued request(s) still pending"
        )


def _server_command(args: argparse.Namespace, port: int) -> list[str]:
    command = [
        str(args.server_python),
        str(REPO_ROOT / "scripts" / "deploy.py"),
        "--ckpt-dir",
        str(args.ckpt_dir),
        "--ckpt-name",
        args.ckpt_name,
        "--device",
        "cuda:0",
        "--host",
        args.host,
        "--port",
        str(port),
        "--denoise-steps",
        str(args.denoise_steps),
        "--denoise-mode",
        args.denoise_mode,
        "--inference-mode",
        args.inference_mode,
        "--inference-horizon",
        str(args.inference_horizon),
    ]
    if args.compile_enabled is not None:
        command.extend(("--compile-enabled", args.compile_enabled))
    command.extend(args.extra_deploy_arg)
    return command


def _start_servers(
    args: argparse.Namespace,
    *,
    slots: list[ReplicaSlot],
    output_dir: Path,
    registry: ProcessRegistry,
) -> list[ServerProcess]:
    server_dir = output_dir / "logs" / "servers"
    server_dir.mkdir(parents=True, exist_ok=True)
    servers = []
    for slot in slots:
        command = _server_command(args, slot.port)
        log_path = server_dir / f"gpu{slot.gpu}_replica{slot.replica}_port{slot.port}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        log_handle.write(f"command: {shlex.join(command)}\n")
        log_handle.flush()
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": str(slot.gpu), "PYTHONUNBUFFERED": "1"})
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        registry.add(process)
        servers.append(ServerProcess(slot.gpu, slot.replica, slot.port, process, log_handle))
        print(
            f"[server] gpu={slot.gpu} replica={slot.replica} port={slot.port} pid={process.pid}",
            flush=True,
        )
    _wait_for_servers(servers, args.host, args.server_start_timeout)
    return servers


def _read_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize(
    output_dir: Path,
    jobs: list[TaskJob],
    trial_run: TrialRun,
    *,
    write_files: bool,
) -> tuple[dict, bool]:
    task_rows = []
    missing = []
    for job in jobs:
        result_path = _latest_valid_result(_run_output_dir(output_dir, job, trial_run), job, trial_run)
        payload = None
        if result_path is None:
            missing.append(
                {
                    "suite": job.suite,
                    "task_id": job.task_id,
                    "trial_start": trial_run.trial_start,
                    "trial_stop": trial_run.trial_stop,
                }
            )
        else:
            payload = _read_result(result_path)
        successes = 0 if payload is None else int(payload["successes"])
        trials = 0 if payload is None else int(payload["num_trials"])
        task_rows.append(
            {
                "suite": job.suite,
                "task_id": job.task_id,
                "task_name": None if payload is None else payload.get("task"),
                "successes": successes,
                "trials": trials,
                "success_rate": successes / trials if trials else None,
                "complete": payload is not None,
                "result_paths": [] if result_path is None else [str(result_path)],
            }
        )

    suite_rows = []
    for suite in dict.fromkeys(job.suite for job in jobs):
        rows = [row for row in task_rows if row["suite"] == suite]
        successes = sum(row["successes"] for row in rows)
        trials = sum(row["trials"] for row in rows)
        suite_rows.append(
            {
                "suite": suite,
                "successes": successes,
                "trials": trials,
                "success_rate": successes / trials if trials else None,
                "tasks_complete": sum(bool(row["complete"]) for row in rows),
                "tasks_expected": len(rows),
            }
        )
    total_successes = sum(row["successes"] for row in suite_rows)
    total_trials = sum(row["trials"] for row in suite_rows)
    summary = {
        "task_results": task_rows,
        "suite_results": suite_rows,
        "overall": {
            "successes": total_successes,
            "trials": total_trials,
            "success_rate": total_successes / total_trials if total_trials else None,
            "tasks_complete": sum(bool(row["complete"]) for row in task_rows),
            "tasks_expected": len(task_rows),
        },
        "missing_runs": missing,
    }
    if write_files:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "suite",
                    "task_id",
                    "task_name",
                    "successes",
                    "trials",
                    "success_rate",
                    "complete",
                ),
            )
            writer.writeheader()
            for row in task_rows:
                writer.writerow({key: row[key] for key in writer.fieldnames})

    if len(task_rows) <= 200:
        print("\n=== LIBERO task results ===")
        for row in task_rows:
            rate = "n/a" if row["success_rate"] is None else f"{100 * row['success_rate']:.1f}%"
            state = "ok" if row["complete"] else "INCOMPLETE"
            print(f"[{state:10}] {row['suite']}/task{row['task_id']:02d}  {row['successes']}/{row['trials']} ({rate})")
    else:
        print(f"\n=== LIBERO task results: {len(task_rows)} rows written to summary files ===")
    print("\n=== LIBERO suite results ===")
    for row in suite_rows:
        rate = "n/a" if row["success_rate"] is None else f"{100 * row['success_rate']:.2f}%"
        print(
            f"{row['suite']:15} {row['successes']}/{row['trials']} ({rate}), "
            f"tasks={row['tasks_complete']}/{row['tasks_expected']}"
        )
    overall = summary["overall"]
    overall_rate = "n/a" if overall["success_rate"] is None else f"{100 * overall['success_rate']:.2f}%"
    print(
        f"overall         {overall['successes']}/{overall['trials']} ({overall_rate}), "
        f"tasks={overall['tasks_complete']}/{overall['tasks_expected']}"
    )
    if write_files:
        print(f"summary         {output_dir / 'summary.json'}")
    return summary, not missing


def _preflight(args: argparse.Namespace) -> None:
    required_files = [
        args.server_python,
        args.libero_python,
        args.policy_config,
        args.ckpt_dir / args.ckpt_name,
        args.ckpt_dir / "config.yaml",
        args.ckpt_dir / "normalization_stats.npy",
        SCRIPT_DIR / "single_eval.py",
        REPO_ROOT / "scripts" / "deploy.py",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if not args.libero_path.is_dir():
        missing.append(str(args.libero_path))
    if missing:
        raise FileNotFoundError("required paths are missing:\n  " + "\n  ".join(missing))
    highest_port = args.base_port + args.replicas_per_gpu * len(args.gpus) - 1
    if args.base_port <= 0 or highest_port > 65535:
        raise ValueError(f"invalid port range {args.base_port}..{highest_port}")
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if args.trial_start < 0:
        raise ValueError("--trial-start must be non-negative")
    if args.denoise_steps <= 0:
        raise ValueError("--denoise-steps must be positive")
    if args.inference_horizon <= 0:
        raise ValueError("--inference-horizon must be positive")
    if args.server_start_timeout <= 0:
        raise ValueError("--server-start-timeout must be positive")
    if args.client_max_attempts <= 0:
        raise ValueError("--client-max-attempts must be positive")
    if args.client_retry_delay < 0:
        raise ValueError("--client-retry-delay must be non-negative")
    if args.worker_max_consecutive_failures <= 0:
        raise ValueError("--worker-max-consecutive-failures must be positive")
    if args.worker_recovery_delay < 0:
        raise ValueError("--worker-recovery-delay must be non-negative")
    if args.client_start_stagger < 0:
        raise ValueError("--client-start-stagger must be non-negative")


def _validate_protocol(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(args.policy_config.read_text(encoding="utf-8")) or {}
    expected_steps = {
        "libero_spatial": 600,
        "libero_object": 600,
        "libero_goal": 600,
        "libero_10": 700,
    }
    effective_seed = args.seed if args.seed is not None else cfg.get("seed")
    errors = []
    effective_num_trials = 1 if args.smoke else args.num_trials
    expected_num_trials = 1 if args.smoke else 50
    if effective_num_trials != expected_num_trials or args.trial_start != 0:
        errors.append(
            f"trial range must be 0:{expected_num_trials}, "
            f"got {args.trial_start}:{args.trial_start + effective_num_trials}"
        )
    if cfg.get("max_steps") != 600 or cfg.get("max_steps_by_suite") != expected_steps:
        errors.append("SPATIAL/OBJECT/GOAL must use 600 steps and LONG must use 700 steps")
    if cfg.get("settle_steps") != 30 or cfg.get("settle_action") != [0, 0, 0, 0, 0, 0, -1]:
        errors.append("settling must use 30 actions with gripper=-1")
    if effective_seed != 42:
        errors.append(f"effective seed must be 42, got {effective_seed!r}")
    if cfg.get("rng_mode") != "environment":
        errors.append("rng_mode must be 'environment'")
    if cfg.get("reseed_each_trial") is not False:
        errors.append("reseed_each_trial must be false")
    if errors:
        raise ValueError("LIBERO pinned protocol violation:\n  - " + "\n  - ".join(errors))


def _print_plan(
    args: argparse.Namespace,
    jobs: list[TaskJob],
    pending_jobs: list[TaskJob],
    slots: list[ReplicaSlot],
    active_slots: list[ReplicaSlot],
    trial_run: TrialRun,
    mujoco_version: str,
) -> None:
    print(f"checkpoint : {args.ckpt_dir / args.ckpt_name}")
    print(f"MuJoCo     : {mujoco_version} ({args.libero_python})")
    print(f"mode       : {args.inference_mode}, inference_horizon={args.inference_horizon}")
    print(f"seed       : {args.seed if args.seed is not None else 'from policy config'}")
    if args.task_sample_ratio is not None:
        sampled_by_suite = {suite: sum(job.suite == suite for job in jobs) for suite in args.suites}
        print(f"sampling   : ratio={args.task_sample_ratio:g}, seed={args.task_sample_seed}, tasks={sampled_by_suite}")
    print(f"trials     : {trial_run.trial_start}:{trial_run.trial_stop} continuous in one environment per task")
    print(f"scheduler  : dynamic shared queue ({len(pending_jobs)} pending request(s))")
    print(
        f"servers    : {len(active_slots)} active replica(s) "
        f"({len(args.gpus)} GPUs, up to {args.replicas_per_gpu} replicas each)"
    )
    for gpu in args.gpus:
        for slot in sorted(
            (slot for slot in slots if slot.gpu == gpu),
            key=lambda slot: slot.replica,
        ):
            state = "pulling from shared queue" if slot in active_slots else "idle"
            print(
                f"gpu {gpu} replica {slot.replica}: port={slot.port} "
                f"render_gpu={_render_device_for_slot(args, slot)} {state}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--ckpt-name", default=DEFAULT_CKPT_NAME)
    parser.add_argument("--gpus", type=_parse_gpus, default=_parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument(
        "--render-gpus",
        type=_parse_gpus,
        help="physical EGL devices assigned by policy-GPU slot (default: same as --gpus)",
    )
    parser.add_argument(
        "--replicas-per-gpu",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="policy server/client replicas per GPU (default: 2; maximum: 3)",
    )
    parser.add_argument("--base-port", type=int, default=8920)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--server-python", type=Path, default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--libero-python", type=Path)
    parser.add_argument("--libero-path", type=Path)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--suites", type=_resolve_suites, default=_resolve_suites("spatial,goal,object,long"))
    parser.add_argument("--task-ids", type=_parse_task_ids, default=None, help="all or comma-separated ids")
    parser.add_argument(
        "--task-sample-ratio",
        type=_parse_task_sample_ratio,
        default=None,
        help="sample this fraction independently within every suite; mutually exclusive with --task-ids",
    )
    parser.add_argument(
        "--task-sample-seed",
        type=int,
        default=42,
        help="seed used by --task-sample-ratio",
    )
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="trials per task (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the seed in policy_config.yml for every task environment",
    )
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--denoise-mode", choices=("sync", "async"), default="sync")
    parser.add_argument("--inference-mode", choices=("sync", "async"), default="sync")
    parser.add_argument("--inference-horizon", type=int, default=10)
    parser.add_argument("--compile-enabled", choices=("true", "false"), default=None)
    parser.add_argument(
        "--extra-deploy-arg",
        action="append",
        default=[],
        help="repeatable OmegaConf deploy override, e.g. optimization.dit_cache.enabled=false",
    )
    parser.add_argument("--server-start-timeout", type=int, default=1200)
    parser.add_argument(
        "--client-max-attempts",
        type=int,
        default=3,
        help="maximum attempts for each client task (default: 3)",
    )
    parser.add_argument(
        "--client-retry-delay",
        type=float,
        default=2.0,
        help="seconds to wait before retrying a failed client task (default: 2)",
    )
    parser.add_argument(
        "--worker-max-consecutive-failures",
        type=int,
        default=3,
        help="cool down an endpoint after this many consecutive client failures (default: 3)",
    )
    parser.add_argument(
        "--worker-recovery-delay",
        type=float,
        default=30.0,
        help="seconds before a cooled-down endpoint rejoins the shared queue (default: 30)",
    )
    parser.add_argument(
        "--client-start-stagger",
        type=float,
        default=0.5,
        help="seconds of startup staggering per port offset (default: 0.5)",
    )
    parser.add_argument("--server-mode", choices=("managed", "external"), default="managed")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only task 0 of the first suite for one trial",
    )
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.libero_python is None:
        args.libero_python = DEFAULT_LIBERO_PYTHON
    if args.libero_path is None:
        args.libero_path = DEFAULT_LIBERO_PATH
    if args.num_trials is None:
        args.num_trials = 50
    args.ckpt_dir = args.ckpt_dir.expanduser().resolve()
    args.server_python = args.server_python.expanduser().resolve()
    # Preserve the public default aliases in logs and manifests instead of
    # exposing an implementation-specific physical environment directory.
    args.libero_python = args.libero_python.expanduser().absolute()
    args.libero_path = args.libero_path.expanduser().absolute()
    args.policy_config = args.policy_config.expanduser().resolve()
    _preflight(args)
    _validate_protocol(args)
    mujoco_version = _mujoco_version(args.libero_python)
    if mujoco_version != REQUIRED_MUJOCO_VERSION:
        raise RuntimeError(
            f"this evaluation requires MuJoCo {REQUIRED_MUJOCO_VERSION}, but {args.libero_python} has {mujoco_version}"
        )

    task_counts = _enumerate_task_counts(
        args.suites,
        libero_python=args.libero_python,
        libero_path=args.libero_path,
    )
    selected_task_ids = args.task_ids
    if args.smoke:
        args.suites = args.suites[:1]
        task_counts = {args.suites[0]: task_counts[args.suites[0]]}
        selected_task_ids = [0]
        args.num_trials = 1
    jobs = _build_jobs(
        args.suites,
        task_counts,
        selected_task_ids,
        sample_ratio=args.task_sample_ratio,
        sample_seed=args.task_sample_seed,
    )
    trial_run = TrialRun(args.trial_start, args.num_trials)
    resume_signature = _resume_signature(
        args,
        jobs=jobs,
        trial_run=trial_run,
        mujoco_version=mujoco_version,
    )
    assignment_jobs = jobs
    if args.output_dir is not None:
        existing_output_dir = args.output_dir.expanduser().resolve()
        _validate_resume_signature(existing_output_dir, resume_signature)
        assignment_jobs = _unfinished_jobs(existing_output_dir, jobs, trial_run)
        skipped = len(jobs) - len(assignment_jobs)
        if skipped:
            print(
                f"[resume] preserving {skipped} completed task(s); "
                f"rebalancing {len(assignment_jobs)} unfinished task(s)",
                flush=True,
            )
    slots = _build_replica_slots(args.gpus, args.base_port, args.replicas_per_gpu)
    pull_order = sorted(slots, key=lambda slot: (slot.replica, slot.gpu_slot))
    active_slots = pull_order[: min(len(pull_order), len(assignment_jobs))]
    _print_plan(args, jobs, assignment_jobs, slots, active_slots, trial_run, mujoco_version)

    if args.dry_run:
        print("[dry-run] no servers or clients were started")
        return 0
    if args.output_dir is None:
        if args.summarize_only:
            raise ValueError("--summarize-only requires --output-dir")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"10epoch_all_suites_{stamp}"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "checkpoint": str(args.ckpt_dir / args.ckpt_name),
        "created_at": datetime.now().isoformat(),
        "server_mode": args.server_mode,
        "inference_mode": args.inference_mode,
        "inference_horizon": args.inference_horizon,
        "denoise_mode": args.denoise_mode,
        "denoise_steps": args.denoise_steps,
        "replicas_per_gpu": args.replicas_per_gpu,
        "gpus": args.gpus,
        "render_gpus": args.render_gpus if args.render_gpus is not None else args.gpus,
        "render_assignments": [
            {
                "policy_gpu": slot.gpu,
                "replica": slot.replica,
                "render_gpu": _render_device_for_slot(args, slot),
            }
            for slot in slots
        ],
        "client_max_attempts": args.client_max_attempts,
        "client_retry_delay": args.client_retry_delay,
        "worker_max_consecutive_failures": args.worker_max_consecutive_failures,
        "worker_recovery_delay": args.worker_recovery_delay,
        "client_start_stagger": args.client_start_stagger,
        "policy_config": str(args.policy_config),
        "seed_override": args.seed,
        "effective_seed": resume_signature["effective_seed"],
        "task_sample_ratio": args.task_sample_ratio,
        "task_sample_seed": args.task_sample_seed,
        "libero_python": str(args.libero_python),
        "libero_path": str(args.libero_path),
        "mujoco_version": mujoco_version,
        "scheduler": "dynamic_shared_queue",
        "resume_signature": resume_signature,
        "trial_run": trial_run.__dict__ | {"trial_stop": trial_run.trial_stop},
        "initial_queue": [job.__dict__ for job in assignment_jobs],
        "workers": [
            {
                "gpu": slot.gpu,
                "gpu_slot": slot.gpu_slot,
                "replica": slot.replica,
                "port": slot.port,
                "active": slot in active_slots,
                "jobs": [],
            }
            for slot in slots
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.summarize_only:
        _, complete = _summarize(output_dir, jobs, trial_run, write_files=True)
        return 0 if complete else 1
    if not active_slots:
        _, complete = _summarize(output_dir, jobs, trial_run, write_files=True)
        print(f"output          {output_dir}")
        return 0 if complete else 1

    ports = [slot.port for slot in active_slots]
    occupied = [port for port in ports if _port_is_open(args.host, port)]
    if args.server_mode == "managed" and occupied:
        raise RuntimeError(f"managed-server ports are already occupied: {occupied}")
    if args.server_mode == "external":
        # Validate the application protocol rather than opening and immediately
        # dropping a raw TCP connection, which websockets logs as a malformed
        # HTTP handshake on an otherwise healthy external policy server.
        unavailable = [port for port in ports if not _websocket_server_is_ready(args.host, port)]
        if unavailable:
            raise RuntimeError(f"external servers are not listening on ports: {unavailable}")

    registry = ProcessRegistry()
    stop_event = threading.Event()
    work_queue: queue.Queue[QueuedTask | None] = queue.Queue()
    for job in assignment_jobs:
        work_queue.put(QueuedTask(job=job))
    servers: list[ServerProcess] = []
    failed_jobs = []
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    futures: list[concurrent.futures.Future] = []
    try:
        if args.server_mode == "managed":
            servers = _start_servers(
                args,
                slots=active_slots,
                output_dir=output_dir,
                registry=registry,
            )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(active_slots))
        futures = [
            executor.submit(
                _dynamic_worker,
                args,
                slot=slot,
                work_queue=work_queue,
                trial_run=trial_run,
                output_dir=output_dir,
                registry=registry,
                stop_event=stop_event,
            )
            for slot in active_slots
        ]
        _wait_for_dynamic_queue(work_queue, futures)
        for _ in active_slots:
            work_queue.put(None)
        for future in concurrent.futures.as_completed(futures):
            failed_jobs.extend(future.result())
    except KeyboardInterrupt:
        print("\n[interrupt] stopping scheduler-owned clients and servers", file=sys.stderr, flush=True)
        stop_event.set()
        return_code = 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        stop_event.set()
        return_code = 1
    else:
        return_code = 1 if failed_jobs else 0
    finally:
        # Terminate first: ThreadPoolExecutor.shutdown(wait=True) otherwise
        # waits for long-running simulator subprocesses after Ctrl-C.
        registry.terminate_all()
        if executor is not None:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        for server in servers:
            server.log_handle.close()

    _, complete = _summarize(output_dir, jobs, trial_run, write_files=True)
    if not complete:
        return_code = return_code or 1
    print(f"output          {output_dir}")
    return return_code


if __name__ == "__main__":
    sys.exit(main())
