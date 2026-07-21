#!/usr/bin/env python3
"""Run one RoboCasa GR1 tabletop task against an OpenWAM policy server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from openwam2robocasa_gr1_interface import OpenWAMRoboCasaGR1Policy, zero_action


def _repo_root() -> Path:
    raw_root = os.environ.get("ROBOCASA_GR1_PATH", "")
    if not raw_root:
        raise SystemExit("ROBOCASA_GR1_PATH is not set")
    root = Path(raw_root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"ROBOCASA_GR1_PATH does not point to a repo: {root}")
    return root.resolve()


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


def _make_env(cfg: dict):
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

    repo_root = _repo_root()
    print(f"[robocasa-gr1] repo={repo_root}")
    env_id = cfg.get("env_id", "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env")
    return gym.make(
        env_id,
        enable_render=_require_bool(cfg.get("enable_render", True), "enable_render"),
    )


def run_eval(cfg: dict) -> int:
    env = _make_env(cfg)
    policy = OpenWAMRoboCasaGR1Policy(
        action_space=env.action_space,
        env=env,
        action_mode=cfg.get("action_mode", "eef"),
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 8848)),
        request_timeout=int(cfg.get("request_timeout", 300)),
        head_camera_key=cfg.get("head_camera_key", "video.ego_view_pad_res256_freq20"),
        left_wrist_camera_key=cfg.get("left_wrist_camera_key"),
        right_wrist_camera_key=cfg.get("right_wrist_camera_key"),
        prompt_key=cfg.get("prompt_key", "annotation.human.coarse_action"),
        fallback_prompt_key=cfg.get("fallback_prompt_key", "annotation.human.action.task_description"),
        send_state=_require_bool(cfg.get("send_state", True), "send_state"),
        state_dim=_parse_optional_int(cfg.get("state_dim"), "state_dim"),
        debug=_require_bool(cfg.get("debug", False), "debug"),
        debug_dir=cfg.get("debug_dir", "./debug_robocasa_gr1"),
    )

    num_episodes = int(cfg.get("num_episodes", 1))
    max_steps = int(cfg.get("max_steps", 720))
    settle_steps = int(cfg.get("settle_steps", 0))
    fail_on_incomplete = _require_bool(cfg.get("fail_on_incomplete", False), "fail_on_incomplete")
    successes = 0

    try:
        for episode in range(num_episodes):
            obs, _info = env.reset(seed=int(cfg.get("seed", 0)) + episode)
            for _ in range(settle_steps):
                obs, _, _, _, _ = env.step(zero_action(env.action_space))
            policy.reset()

            success = False
            for step in range(max_steps):
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                success = bool(info.get("success", False) or reward > 0)
                if success or terminated or truncated:
                    print(
                        f"[RESULT] episode={episode} success={success} "
                        f"step={step + 1} reward={reward} terminated={terminated} truncated={truncated}"
                    )
                    break
            if success:
                successes += 1
            elif not success:
                print(f"[RESULT] episode={episode} failed max_steps={max_steps}")
    finally:
        policy.close()
        env.close()

    rate = successes / max(num_episodes, 1)
    print(f"Success rate: {successes}/{num_episodes} => {rate * 100:.1f}%")
    return 1 if fail_on_incomplete and successes != num_episodes else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--env-id")
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    for key in ["host", "port", "env_id", "num_episodes", "max_steps"]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return run_eval(cfg)


if __name__ == "__main__":
    sys.exit(main())
