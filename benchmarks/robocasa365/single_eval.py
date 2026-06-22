#!/usr/bin/env python3
"""Run one RoboCasa365 task against an already-running OpenWAM policy server.

The OpenWAM server owns the model / checkpoint / preprocessing; this script only
drives the RoboCasa365 robosuite sim and forwards observations over WebSocket.
Start the server first, e.g. ``scripts/deploy.sh --ckpt-dir <ckpt> --port 8848``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from openwam2robocasa365_interface import (  # noqa: E402
    DEFAULT_STATE_KEYS,
    OpenWAMRoboCasa365Policy,
)


def _load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_optional(value):
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null"):
        return None
    return value


def _parse_bool(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off", "none", "null", ""):
            return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}")


def _parse_optional_int(value, field_name: str):
    value = _normalize_optional(value)
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive or null, got {value!r}")
    return parsed


def _make_env(cfg: dict):
    import gymnasium as gym
    import robocasa  # noqa: F401  (registers robosuite envs)
    import robocasa.wrappers.gym_wrapper  # noqa: F401  (registers robocasa/<Task> gym ids)

    task = cfg.get("task", "OpenDrawer")
    return gym.make(
        f"robocasa/{task}",
        split=cfg.get("split", "target"),
        enable_render=True,
    )


def _build_policy(cfg: dict) -> OpenWAMRoboCasa365Policy:
    return OpenWAMRoboCasa365Policy(
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 8848)),
        request_timeout=int(cfg.get("request_timeout", 300)),
        head_camera_key=cfg.get("head_camera_key", "video.robot0_agentview_left"),
        left_wrist_camera_key=_normalize_optional(
            cfg.get("left_wrist_camera_key", "video.robot0_eye_in_hand")
        ),
        right_wrist_camera_key=_normalize_optional(
            cfg.get("right_wrist_camera_key")  # default null: server black-fills the unused 2nd-wrist slot
        ),
        image_transform=cfg.get("image_transform", "none"),
        state_keys=list(cfg.get("state_keys") or DEFAULT_STATE_KEYS),
        state_dim=_parse_optional_int(cfg.get("state_dim", 16), "state_dim"),
        action_dim=int(cfg.get("action_dim", 12)),
        osc_pos_scale=cfg.get("osc_pos_scale"),
        osc_rot_scale=cfg.get("osc_rot_scale"),
        debug=_parse_bool(cfg.get("debug", False), "debug"),
        debug_dir=cfg.get("debug_dir", "./debug_robocasa365"),
    )


def _rollout(env, policy, *, num_trials: int, max_steps: int, seed: int) -> int:
    """Run ``num_trials`` episodes; return the success count.

    ``env`` / ``policy`` are injected so the loop is unit-testable with stubs
    (no sim, no server) — one obs in, one action dict out, success OR-accumulated.
    """
    successes = 0
    for trial in range(num_trials):
        obs, info = env.reset(seed=seed + trial)
        instruction = obs.get("annotation.human.task_description", "")
        policy.reset()
        success = bool(info.get("success", False))
        for _ in range(max_steps):
            obs, reward, done, truncated, info = env.step(policy.act(obs, instruction))
            success = success or bool(info.get("success", False))
            if done or truncated:
                break
        successes += int(success)
        print(f"[RESULT] trial={trial} success={success}")
    return successes


def run_eval(cfg: dict) -> int:
    num_trials = int(cfg.get("num_trials", 5))
    max_steps = int(cfg.get("max_steps", 500))
    seed = int(cfg.get("seed", 0))

    # Nested try/finally so the already-connected policy is closed even if
    # _make_env raises (e.g. robocasa not installed), and so a failing
    # env.close() does not skip policy.close().
    policy = _build_policy(cfg)
    try:
        env = _make_env(cfg)
        try:
            successes = _rollout(env, policy, num_trials=num_trials, max_steps=max_steps, seed=seed)
        finally:
            env.close()
    finally:
        policy.close()

    rate = successes / max(num_trials, 1)
    print(f"Success rate: {successes}/{num_trials} => {rate * 100:.1f}%")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--task")
    parser.add_argument("--split")
    parser.add_argument("--num-trials", type=int)
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    for key, value in (
        ("host", args.host),
        ("port", args.port),
        ("task", args.task),
        ("split", args.split),
        ("num_trials", args.num_trials),
    ):
        if value is not None:
            cfg[key] = value
    return run_eval(cfg)


if __name__ == "__main__":
    sys.exit(main())
