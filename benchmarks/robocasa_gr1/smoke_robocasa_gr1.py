#!/usr/bin/env python3
"""Smoke checks for RoboCasa GR1 tabletop benchmark installations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from openwam2robocasa_gr1_interface import zero_action

DEFAULT_ENV_ID = "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"


def _repo_root() -> Path:
    raw_root = os.environ.get("ROBOCASA_GR1_PATH", "")
    if not raw_root:
        raise SystemExit("ROBOCASA_GR1_PATH is not set")
    root = Path(raw_root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"ROBOCASA_GR1_PATH does not exist: {root}")
    return root.resolve()


def import_smoke() -> None:
    repo_root = _repo_root()
    import gymnasium as gym  # noqa: PLC0415
    import robocasa  # noqa: F401, PLC0415
    import robosuite  # noqa: PLC0415
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401, PLC0415

    print(f"repo={repo_root}")
    print(f"robocasa_package={Path(robocasa.__file__).resolve()}")
    print(f"robosuite_version={robosuite.__version__}")
    registered = sorted(env_id for env_id in gym.envs.registry if env_id.startswith("gr1_unified/"))
    print(f"gr1_unified_envs={len(registered)}")
    if registered:
        print(f"first_env={registered[0]}")


def task_smoke(env_id: str) -> None:
    import_smoke()
    import gymnasium as gym  # noqa: PLC0415

    if env_id not in gym.envs.registry:
        available = sorted(env_id for env_id in gym.envs.registry if env_id.startswith("gr1_unified/"))
        raise SystemExit(f"Unknown RoboCasa GR1 env_id: {env_id}. Available count: {len(available)}")
    spec = gym.spec(env_id)
    print(f"env_id={env_id}")
    print(f"entry_point={spec.entry_point}")


def env_smoke(env_id: str, steps: int, enable_render: bool) -> None:
    task_smoke(env_id)
    import gymnasium as gym  # noqa: PLC0415

    env = gym.make(env_id, enable_render=enable_render)
    try:
        obs, info = env.reset(seed=0)
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(zero_action(env.action_space))
        obs_keys = ",".join(sorted(obs.keys()))
        action_dims = {
            key: int(np.prod(space.shape))
            for key, space in env.action_space.spaces.items()
            if getattr(space, "shape", None) is not None
        }
        print(
            f"env_smoke=ok reward={reward} terminated={terminated} truncated={truncated} "
            f"success={info.get('success', False)} obs_keys={obs_keys} action_dims={action_dims}"
        )
    finally:
        env.close()


def dataset_smoke(dataset: Path) -> None:
    import h5py  # noqa: PLC0415

    dataset = dataset.expanduser()
    if not dataset.is_file():
        raise SystemExit(f"Dataset file does not exist: {dataset}")
    with h5py.File(dataset, "r") as f:
        keys = list(f.keys())
        if "data" not in f:
            raise SystemExit(f"Expected HDF5 group 'data' in {dataset}; found {keys}")
        episodes = list(f["data"].keys())
        if not episodes:
            raise SystemExit(f"Dataset contains no episodes: {dataset}")
        ep = f["data"][episodes[0]]
        print(f"dataset={dataset}")
        print(f"episodes={len(episodes)} first_episode={episodes[0]}")
        for key in ["actions", "actions_abs", "states"]:
            if key in ep:
                print(f"{key}_shape={ep[key].shape}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["import", "task", "env", "dataset"], default="import")
    parser.add_argument("--env-id", default=os.environ.get("ROBOCASA_GR1_ENV_ID", DEFAULT_ENV_ID))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("ROBOCASA_GR1_SMOKE_STEPS", "1")))
    parser.add_argument("--enable-render", action="store_true")
    parser.add_argument("--dataset", type=Path, default=os.environ.get("ROBOCASA_GR1_DATASET", ""))
    args = parser.parse_args(argv)

    if args.mode == "import":
        import_smoke()
    elif args.mode == "task":
        task_smoke(args.env_id)
    elif args.mode == "env":
        env_smoke(args.env_id, args.steps, args.enable_render)
    else:
        if not args.dataset:
            raise SystemExit("--dataset or ROBOCASA_GR1_DATASET is required for dataset smoke")
        dataset_smoke(args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
