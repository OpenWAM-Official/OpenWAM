#!/usr/bin/env python3
"""Run the canonical native-action LIBERO benchmark with a shared queue.

It reuses the scheduler implementation in this directory for server lifecycle,
retries, resume, and dynamic queue handling while selecting the canonical
native-action client.

By default two policy-server/client replicas are created per requested GPU.
Every replica pulls the next ``suite/task`` request from one shared queue;
when a request finishes, that same worker immediately pulls another one.
Consequently, a fast GPU/replica is not left idle while another one is still
processing a long task.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parent


def _native_client_command(args, job, trial_run, port, run_dir):
    """Build a request for the canonical native-action client."""
    command = [
        str(args.libero_python),
        str(BENCHMARK_DIR / "single_eval.py"),
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
        # The shared scheduler treats each child directory as an attempt and
        # searches ``run_dir/*/results.json`` when resuming.  Keep that
        # layout even though the canonical client writes one results
        # file directly into the directory it receives.
        str(run_dir / "attempt_01"),
    ]
    if args.seed is not None:
        command.extend(("--seed", str(args.seed)))
    return command


def _native_client_env(base_env, *, libero_path, config_root, render_gpu):
    """Set isolated LIBERO paths and conservative thread defaults."""
    env = dict(base_env)
    old_pythonpath = env.get("PYTHONPATH", "")
    pieces = [str(REPO_ROOT), str(libero_path), str(BENCHMARK_DIR)]
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


def _native_read_result(path: Path) -> dict:
    """Normalize the standalone client's result schema for the scheduler."""
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("num_trials", len(payload.get("trials", [])))
    return payload


def main(argv: list[str] | None = None) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.libero import scheduler

    scheduler.SCRIPT_DIR = BENCHMARK_DIR
    scheduler.DEFAULT_CKPT_DIR = Path("/path/to/openwam_checkpoints/new-openwam-libero-sft-10epoch-final")
    scheduler.DEFAULT_CKPT_NAME = "checkpoint_step_10690.safetensors"
    scheduler.DEFAULT_POLICY_CONFIG = BENCHMARK_DIR / "policy_config.yml"
    scheduler.DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "libero"
    scheduler.LIBERO_PROTOCOL_VERSION = "openwam-libero-native-action-seed42-settle30-v1"
    scheduler._client_command = _native_client_command
    scheduler._client_env = _native_client_env
    scheduler._read_result = _native_read_result
    # The lightweight launcher interpreter may not have the ``websockets``
    # package installed even though the isolated LIBERO client does. A
    # listening TCP socket is sufficient here: deploy.py binds only after its
    # policy is initialized, and the client performs the real websocket ping.
    scheduler._websocket_server_is_ready = scheduler._port_is_open

    args = list(sys.argv[1:] if argv is None else argv)
    # A full run is a strict superset of a previously sampled run.
    # In this standalone entry point we may intentionally reuse valid results
    # from that subset; the checkpoint/config/trial protocol is still checked
    # normally, while only the task-list hash/count are allowed to change.
    if "--resume-superset" in args:
        args.remove("--resume-superset")
        scheduler.RESUME_OPERATIONAL_FIELDS = frozenset(
            set(scheduler.RESUME_OPERATIONAL_FIELDS) | {"jobs_sha256", "jobs_count"}
        )
    # Keep the canonical config as the default while allowing an explicit config
    # for experiments.
    if "--policy-config" not in args:
        args.extend(("--policy-config", str(BENCHMARK_DIR / "policy_config.yml")))
    # Require the canonical action mode before launching any workers.
    config_index = (
        args.index("--policy-config")
        if "--policy-config" in args
        else next((i for i, value in enumerate(args) if value.startswith("--policy-config=")), -1)
    )
    config_path = (
        Path(args[config_index + 1])
        if config_index >= 0 and args[config_index] == "--policy-config"
        else Path(args[config_index].split("=", 1)[1])
        if config_index >= 0
        else None
    )
    if config_path is not None:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if str(config.get("action_mode", "")).strip().lower() != "eef":
            raise ValueError(f"{config_path} is not a canonical LIBERO config; expected action_mode: eef")
    return scheduler.main(args)


if __name__ == "__main__":
    sys.exit(main())
