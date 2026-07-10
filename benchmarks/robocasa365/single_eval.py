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

    # Full RoboCasa365: any task (mobile or fixed) is evaluable — the model commands the base via
    # the mobile_base channel (no fixed-base manifest gate). Eval scope (the official 50 target
    # tasks) is chosen by the task list passed to multi_eval, not enforced here.
    task = cfg.get("task", "OpenDrawer")
    return gym.make(
        f"robocasa/{task}",
        split=cfg.get("split", "target"),
        enable_render=True,
    )


def _resolve_max_steps(cfg: dict) -> int:
    """Per-task eval horizon: a step_limits.yml override for the task, else cfg['max_steps'] (500).

    Mirrors robotwin's per-task step_lim overrides — a single global max_steps under-/over-runs
    tasks of different lengths. The env still terminates early on done/truncated, so this is a cap.
    """
    import os

    task = cfg.get("task", "OpenDrawer")
    default = int(cfg.get("max_steps", 500))
    path = os.path.join(os.path.dirname(__file__), "step_limits.yml")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        if task in overrides:
            return int(overrides[task])
    return default


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
        # Default None → the policy auto-derives the expected proprio width (20-D EEF, or 25-D when
        # mobile_base appends the base5 proprio). An explicit state_dim still overrides.
        state_dim=_parse_optional_int(cfg.get("state_dim"), "state_dim"),
        action_dim=int(cfg.get("action_dim", 12)),
        osc_pos_scale=cfg.get("osc_pos_scale"),
        osc_rot_scale=cfg.get("osc_rot_scale"),
        # mobile ckpts: the client derives the base5 proprio (A′-rescaled body-frame base velocity)
        # from the sim obs and sends 25-D [arm20, base5]. Must match the ckpt's dataloader.mobile_base.
        mobile_base=_parse_bool(cfg.get("mobile_base", False), "mobile_base"),
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
        if "annotation.human.task_description" not in obs:
            raise KeyError(
                "reset obs has no 'annotation.human.task_description'; refusing to send an empty "
                f"prompt to the policy. Got keys: {sorted(obs)}. Check the env's annotation key."
            )
        instruction = obs["annotation.human.task_description"]
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
    max_steps = _resolve_max_steps(cfg)
    seed = int(cfg.get("seed", 0))
    print(f"[eval] task={cfg.get('task')} max_steps={max_steps} (per-task step_limits.yml or cfg default)")

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
