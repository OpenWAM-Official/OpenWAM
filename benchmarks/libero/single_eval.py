#!/usr/bin/env python3
"""Evaluate a canonical native-action LIBERO checkpoint through the OpenWAM server.

It imports the canonical native-action policy adapter in this directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openwam2libero_interface import OpenWAMLiberoPolicy  # noqa: E402


def _repo_root() -> Path:
    raw_root = os.environ.get("LIBERO_PATH", "")
    if not raw_root:
        raise SystemExit("LIBERO_PATH is not set")
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"LIBERO_PATH does not point to a LIBERO repo: {root}")
    return root


def _write_libero_config() -> None:
    repo_root = _repo_root()
    benchmark_root = repo_root / "libero" / "libero"
    config_root = Path(os.environ.get("LIBERO_CONFIG_ROOT", Path.home() / ".libero-openwam")).expanduser()
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(repo_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    with (config_root / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _require_bool(value, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a YAML boolean, got {value!r}")
    return value


def _resolve_trial_range(cfg: dict) -> tuple[int, int]:
    start = cfg.get("trial_start", 0)
    count = cfg.get("num_trials", 1)
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError(f"trial_start must be a non-negative YAML integer, got {start!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"num_trials must be a positive YAML integer, got {count!r}")
    return start, start + count


def _resolve_max_steps(cfg: dict, suite_name: str) -> int:
    values = cfg.get("max_steps_by_suite") or {}
    value = values.get(suite_name, cfg.get("max_steps", 600))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"max_steps for {suite_name} must be a positive YAML integer, got {value!r}")
    return int(value)


def _parse_settle_action(value, action_dim: int = 7) -> np.ndarray:
    if value is None:
        return np.zeros(action_dim, dtype=np.float32)
    if not isinstance(value, (list, tuple)) or len(value) != action_dim:
        raise ValueError(f"settle_action must contain exactly {action_dim} values, got {value!r}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise TypeError(f"settle_action values must be YAML numbers, got {value!r}")
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("settle_action values must be finite")
    return result


def _make_env(task, cfg: dict):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    assets_dir = Path(get_libero_path("assets"))
    if not assets_dir.is_dir():
        raise SystemExit(f"LIBERO assets are missing: {assets_dir}")
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=int(cfg.get("camera_height", 256)),
        camera_widths=int(cfg.get("camera_width", 256)),
    )


def _make_env_with_randomization_retries(task, cfg: dict, max_attempts: int = 5):
    for attempt in range(1, max_attempts + 1):
        try:
            return _make_env(task, cfg)
        except Exception as exc:
            if exc.__class__.__name__ != "RandomizationError" or attempt == max_attempts:
                raise
            print(f"[warning] LIBERO placement sampling failed; retrying ({attempt}/{max_attempts})", flush=True)
    raise AssertionError("unreachable")


def run_eval(cfg: dict) -> int:
    if str(cfg.get("action_mode", "")).strip().lower() != "eef":
        raise ValueError("LIBERO runner requires action_mode: eef")
    if cfg.get("rng_mode", "environment") != "environment":
        raise ValueError("LIBERO runner requires rng_mode: environment")
    _write_libero_config()

    seed = int(cfg.get("seed", 42))

    from libero.libero import benchmark

    suite_name = str(cfg.get("suite", "libero_spatial"))
    task_id = int(cfg.get("task_id", 0))
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise SystemExit(f"Unknown LIBERO suite: {suite_name}. Available: {sorted(benchmark_dict)}")
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    trial_start, trial_stop = _resolve_trial_range(cfg)
    max_steps = _resolve_max_steps(cfg, suite_name)
    settle_steps = int(cfg.get("settle_steps", 30))
    settle_action = _parse_settle_action(cfg.get("settle_action"), 7)
    reseed_each_trial = _require_bool(cfg.get("reseed_each_trial", False), "reseed_each_trial")
    fail_on_incomplete = _require_bool(cfg.get("fail_on_incomplete", False), "fail_on_incomplete")

    init_states = task_suite.get_task_init_states(task_id)
    env = _make_env_with_randomization_retries(task, cfg)
    if not reseed_each_trial:
        env.seed(seed)
    policy = OpenWAMLiberoPolicy(
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 8848)),
        request_timeout=int(cfg.get("request_timeout", 300)),
        action_mode=cfg["action_mode"],
        head_camera_key=cfg.get("head_camera_key", "agentview_image"),
        left_wrist_camera_key=cfg.get("left_wrist_camera_key", "robot0_eye_in_hand_image"),
        right_wrist_camera_key=cfg.get("right_wrist_camera_key"),
        image_transform=cfg.get("image_transform", "rotate_180"),
        send_state=_require_bool(cfg.get("send_state", True), "send_state"),
        state_dim=int(cfg.get("state_dim", 10)),
        action_dim=7,
        action_indices=cfg.get("action_indices"),
        action_clip=cfg.get("action_clip"),
        debug=_require_bool(cfg.get("debug", False), "debug"),
        debug_dir=cfg.get("debug_dir", "./debug_libero"),
    )

    successes = 0
    trial_results = []
    try:
        for trial in range(trial_start, trial_stop):
            if reseed_each_trial:
                env.seed(seed + trial)
            result = {"trial": trial, "success": False, "policy_steps": 0}
            try:
                obs = env.reset()
                if len(init_states) > 0:
                    obs = env.set_init_state(init_states[trial % len(init_states)])
                for _ in range(settle_steps):
                    obs, _, _, _ = env.step(settle_action)
                policy.reset()
                done = False
                for step in range(max_steps):
                    obs, reward, done, _ = env.step(policy.act(obs, task.language))
                    result["policy_steps"] = step + 1
                    result["last_reward"] = float(reward)
                    if done:
                        successes += 1
                        result["success"] = True
                        break
            finally:
                trial_results.append(result)
            print(
                f"[RESULT] trial={trial} success={result['success']} steps={result['policy_steps']}",
                flush=True,
            )
    finally:
        env.close()
        policy.close()

    count = trial_stop - trial_start
    success_rate = successes / max(count, 1)
    print(f"Success rate: {successes}/{count} => {success_rate * 100:.1f}%")
    result_dir = cfg.get("result_dir")
    if result_dir:
        output = Path(result_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        result = {
            "action_mode": "eef",
            "suite": suite_name,
            "task_id": task_id,
            "task": task.name,
            "instruction": task.language,
            "trial_start": trial_start,
            "trial_stop": trial_stop,
            "successes": successes,
            "success_rate": success_rate,
            "max_steps": max_steps,
            "settle_steps": settle_steps,
            "seed": seed,
            "trials": trial_results,
        }
        path = output / "results.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[libero] results={path}")
    return 1 if fail_on_incomplete and successes != count else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--suite")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--trial-start", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--result-dir", type=Path)
    args = parser.parse_args(argv)
    cfg = _load_config(args.config)
    cfg["policy_config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    for key in ("host", "port", "suite", "task_id", "num_trials", "trial_start", "seed", "result_dir"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = str(value) if key == "result_dir" else value
    return run_eval(cfg)


if __name__ == "__main__":
    sys.exit(main())
