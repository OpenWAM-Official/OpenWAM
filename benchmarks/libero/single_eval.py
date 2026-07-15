#!/usr/bin/env python3
"""Run one LIBERO task against an already-running OpenWAM policy server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from openwam2libero_interface import OpenWAMLiberoPolicy


def _repo_root() -> Path:
    env_name = "LIBERO_PATH"
    raw_root = os.environ.get(env_name, "")
    if not raw_root:
        raise SystemExit(f"{env_name} is not set")
    root = Path(raw_root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"{env_name} does not point to a LIBERO repo: {root}")
    return root.resolve()


def _config_root() -> Path:
    default = Path.home() / (".libero-openwam")
    env_name = "LIBERO_CONFIG_ROOT"
    return Path(os.environ.get(env_name, default)).expanduser().resolve()


def _write_libero_config() -> None:
    repo_root = _repo_root()
    benchmark_root = repo_root / "libero" / "libero"
    config_root = _config_root()
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(repo_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    with (config_root / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _require_bool(value, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a YAML boolean, got {value!r}")
    return value


def _parse_optional_int(value, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a YAML integer or null, got {value!r}")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive or null, got {value!r}")
    return parsed


def _parse_optional_float(value, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a YAML number or null, got {value!r}")
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive or null, got {value!r}")
    return parsed


def _make_env(task, cfg: dict):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    assets_dir = Path(get_libero_path("assets"))
    if not assets_dir.is_dir():
        raise SystemExit(f"LIBERO assets are missing: {assets_dir}")
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=int(cfg.get("camera_height", 128)),
        camera_widths=int(cfg.get("camera_width", 128)),
    )


def run_eval(cfg: dict) -> int:
    _write_libero_config()

    from libero.libero import benchmark

    suite_name = cfg.get("suite", "libero_spatial")
    task_id = int(cfg.get("task_id", 0))
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise SystemExit(f"Unknown LIBERO suite: {suite_name}. Available: {sorted(benchmark_dict)}")
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    print(f"[libero-eval] suite={suite_name} task_id={task_id} task={task.name}")
    print(f"[libero-eval] instruction={task.language}")

    policy = OpenWAMLiberoPolicy(
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 8848)),
        request_timeout=int(cfg.get("request_timeout", 300)),
        head_camera_key=cfg.get("head_camera_key", "agentview_image"),
        left_wrist_camera_key=cfg.get("left_wrist_camera_key"),
        right_wrist_camera_key=cfg.get("right_wrist_camera_key"),
        image_transform=cfg.get("image_transform", "rotate_180"),
        send_state=_require_bool(cfg.get("send_state", False), "send_state"),
        state_keys=list(cfg.get("state_keys") or []),
        state_dim=_parse_optional_int(cfg.get("state_dim"), "state_dim"),
        action_dim=int(cfg.get("action_dim", 7)),
        action_indices=cfg.get("action_indices"),
        action_clip=_parse_optional_float(cfg.get("action_clip"), "action_clip"),
        debug=_require_bool(cfg.get("debug", False), "debug"),
        debug_dir=cfg.get("debug_dir", "./debug_libero"),
    )

    num_trials = int(cfg.get("num_trials", 1))
    max_steps = int(cfg.get("max_steps", 600))
    settle_steps = int(cfg.get("settle_steps", 10))
    fail_on_incomplete = _require_bool(cfg.get("fail_on_incomplete", False), "fail_on_incomplete")
    init_states = task_suite.get_task_init_states(task_id)
    env = _make_env(task, cfg)
    successes = 0
    try:
        for trial in range(num_trials):
            env.seed(int(cfg.get("seed", 0)) + trial)
            obs = env.reset()
            if len(init_states) > 0:
                obs = env.set_init_state(init_states[trial % len(init_states)])
            settle_action = np.zeros(int(cfg.get("action_dim", 7)), dtype=np.float32)
            for _ in range(settle_steps):
                obs, _, _, _ = env.step(settle_action)
            policy.reset()

            done = False
            for step in range(max_steps):
                action = policy.act(obs, task.language)
                obs, reward, done, info = env.step(action)
                if done:
                    successes += 1
                    print(f"[RESULT] trial={trial} success step={step + 1} reward={reward}")
                    break
            if not done:
                print(f"[RESULT] trial={trial} failed max_steps={max_steps}")
    finally:
        env.close()
        policy.close()

    rate = successes / max(num_trials, 1)
    print(f"Success rate: {successes}/{num_trials} => {rate * 100:.1f}%")
    return 1 if fail_on_incomplete and successes != num_trials else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--suite")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--num-trials", type=int)
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    for key in ["host", "port", "suite", "task_id", "num_trials"]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return run_eval(cfg)


if __name__ == "__main__":
    sys.exit(main())
