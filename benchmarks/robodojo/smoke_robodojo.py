#!/usr/bin/env python3
"""Progressive contract, server, debug, and live Isaac smokes for RoboDojo."""

from __future__ import annotations

import argparse
import copy
import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.robodojo.openwam_model_client import OpenWAMRoboDojoModelClient
from benchmarks.robodojo.single_eval import (
    PhysXRestartRequired,
    _positive_integer,
    configure_runtime_import_paths,
    construct_with_no_network_client,
    load_runner_config,
    run_eval,
    validate_runner_config,
    validate_server_endpoint,
    verify_runtime_import_provenance,
)
from benchmarks.utils import WSPolicyClient
from openwam.robodojo import (
    arx_x5_calibration,
    discover_episodes,
    robot_base_to_env_relative_world,
    validate_calibration,
)

_OPENWAM_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = Path(__file__).with_name("policy_config.yml")


def contract_smoke(
    calibration_path: str | os.PathLike[str] | None = None,
    *,
    dataset_root: str | os.PathLike[str] | None = None,
    task: str | None = None,
) -> None:
    """Check calibration and, optionally, the formal HDF5 conversion path."""

    if calibration_path is None:
        calibration = arx_x5_calibration()
    else:
        from openwam.robodojo import load_calibration

        calibration = load_calibration(calibration_path)
    print(
        "contract=ok embodiment="
        f"{calibration['embodiment']} endpoint={calibration['endpoint']['link_name']}"
    )
    if (dataset_root is None) != (task is None):
        raise ValueError("dataset_root and task must be provided together")
    if dataset_root is None:
        return

    from openwam.dataloader.robodojo import (
        read_calibrated_eef20,
        validate_robodojo_episode,
    )

    episodes = discover_episodes(dataset_root, task)
    for episode in episodes:
        validate_robodojo_episode(episode)
    raw = read_calibrated_eef20(episodes[0], calibration)
    if raw.ndim != 2 or raw.shape[1] != 20 or not np.all(np.isfinite(raw)):
        raise ValueError(
            f"formal RoboDojo conversion must produce finite (T, 20), got {raw.shape}"
        )
    print(
        f"formal_data=ok task={task} episodes={len(episodes)} "
        f"first_shape={tuple(raw.shape)}"
    )


def ping_smoke(
    host: str,
    port: int,
    timeout: float,
    *,
    transport_factory=WSPolicyClient,
) -> None:
    client = transport_factory(
        f"ws://{host}:{int(port)}",
        timeout=float(timeout),
        open_timeout=min(10.0, float(timeout)),
    )
    try:
        response = client.ping()
        if not isinstance(response, dict) or response.get("type") != "pong":
            raise RuntimeError(f"expected OpenWAM pong, got {response!r}")
        print(f"ping=ok server=ws://{host}:{int(port)}")
    finally:
        client.close()


def make_synthetic_debug_observation(
    observation: dict[str, Any],
    calibration: dict[str, Any],
    *,
    env_idx: int = 0,
) -> dict[str, Any]:
    """Replace debug-env placeholder state with valid native ``xyz+wxyz`` state."""

    calibration = validate_calibration(calibration)
    if not isinstance(observation, dict):
        raise TypeError("debug observation must be a mapping")
    synthetic = copy.deepcopy(observation)
    state = synthetic.setdefault("state", {})
    base_poses = {
        "left": np.array([0.10, 0.00, 0.20, 1.0, 0.0, 0.0, 0.0]),
        "right": np.array([0.10, 0.00, 0.20, 1.0, 0.0, 0.0, 0.0]),
    }
    for side in ("left", "right"):
        arm = calibration["arms"][side]
        state[f"{side}_ee_pose"] = robot_base_to_env_relative_world(
            base_poses[side],
            arm["base_pos_relative_to_env_origin"],
            arm["base_quat_wxyz"],
        )
        state[f"{side}_ee_joint_state"] = np.array([0.5], dtype=np.float64)
    synthetic["env_idx"] = int(env_idx)
    return synthetic


def run_debug_protocol_rollout(
    test_env: Any,
    model_client: OpenWAMRoboDojoModelClient,
    calibration: dict[str, Any],
    *,
    steps: int = 1,
) -> int:
    """Exercise reset/update/get/take using a no-Isaac TestEnv-like object."""

    if isinstance(steps, bool) or int(steps) <= 0:
        raise ValueError(f"steps must be positive, got {steps!r}")
    reset = getattr(test_env, "reset", None)
    if callable(reset):
        reset()
    else:
        model_client.call(func_name="reset")
        test_env.episode_step = 0
    completed = 0
    for _ in range(int(steps)):
        observation = make_synthetic_debug_observation(
            test_env.get_obs(), calibration, env_idx=0
        )
        model_client.call(func_name="update_obs", obs=observation)
        actions = model_client.call(func_name="get_action")
        for action in actions:
            test_env.take_action(action)
            completed += 1
    return completed


def debug_smoke(
    config: dict[str, Any],
    calibration_path: str | os.PathLike[str] | None = None,
    *,
    steps: int = 1,
    obs_encoded: bool = True,
    transport_factory=WSPolicyClient,
) -> None:
    """Construct RoboDojo's debug env without its MsgPack client, then roll out."""

    config = validate_runner_config({**config, "eval_count": 1})
    configure_runtime_import_paths(_OPENWAM_ROOT, config["robodojo_root"])
    verify_runtime_import_provenance(_OPENWAM_ROOT, config["robodojo_root"])
    if calibration_path is None:
        calibration = arx_x5_calibration()
    else:
        from openwam.robodojo import load_calibration

        calibration = load_calibration(calibration_path)

    ws_module = importlib.import_module("client_server.ws")
    debug_module = importlib.import_module("XPolicyLab.debug_env_client")
    deploy_cfg = {
        "bench_name": "RoboDojo",
        "task_name": config["task"],
        "env_cfg_type": "arx_x5",
        "policy_name": "demo_policy",
        "protocol": "ws",
        "host": config["host"],
        "port": config["port"],
        "policy_server_url": f"ws://{config['host']}:{config['port']}",
        "evaluation_id": "openwam-debug",
        "trial_id": "openwam-debug-trial",
        "action_case_id": "openwam_debug_case",
        "repeat_index": None,
        "eval_episode_num": 1,
        "eval_batch": False,
        "obs_encoded": bool(obs_encoded),
    }
    test_env = construct_with_no_network_client(
        ws_module, lambda: debug_module.TestEnv(deploy_cfg)
    )
    model_client = OpenWAMRoboDojoModelClient(
        task_env=None,
        host=config["host"],
        port=config["port"],
        timeout=config["timeout"],
        num_envs=1,
        env_config="arx_x5",
        robot_action_dim_info=test_env.robot_action_dim_info,
        transport_factory=transport_factory,
        frame_provider=lambda: calibration,
        debug=config["debug"],
    )
    test_env.model_client = model_client
    try:
        completed = run_debug_protocol_rollout(
            test_env, model_client, calibration, steps=steps
        )
        print(f"debug=ok steps={completed} env_type={os.environ.get('EVAL_ENV_TYPE')}")
    finally:
        try:
            model_client.close()
        finally:
            test_env.model_client = None


def isaac_smoke(config: dict[str, Any], *, steps: int = 1) -> int:
    smoke_config = dict(config)
    smoke_config["eval_count"] = 1
    smoke_config["max_steps"] = _positive_integer(steps, "steps")
    try:
        return run_eval(smoke_config)
    except PhysXRestartRequired as restart:
        print(
            f"[OpenWAM RoboDojo] Isaac smoke requests fresh-process restart "
            f"after PhysX failure ({restart}); exiting 99"
        )
        return 99


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("contract", "ping", "debug", "isaac"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--calibration")
    parser.add_argument("--dataset-root")
    parser.add_argument("--task")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--robodojo-root")
    args = parser.parse_args(argv)

    config = load_runner_config(args.config)
    for key, value in (
        ("robodojo_root", args.robodojo_root),
        ("host", args.host),
        ("port", args.port),
        ("task", args.task),
        ("timeout", args.timeout),
    ):
        if value is not None:
            config[key] = value

    if args.mode == "contract":
        contract_smoke(
            args.calibration,
            dataset_root=args.dataset_root,
            task=args.task,
        )
    elif args.mode == "ping":
        host, port, timeout = validate_server_endpoint(
            config.get("host", "127.0.0.1"),
            config.get("port", 8848),
            config.get("timeout", 300.0),
        )
        ping_smoke(host, port, timeout)
    elif args.mode == "debug":
        debug_smoke(
            config,
            args.calibration,
            steps=args.steps,
        )
    else:
        return isaac_smoke(config, steps=args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "contract_smoke",
    "debug_smoke",
    "isaac_smoke",
    "main",
    "make_synthetic_debug_observation",
    "ping_smoke",
    "run_debug_protocol_rollout",
]
