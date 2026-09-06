#!/usr/bin/env python3
"""Canonical smoke checks for a RoboCasa365 install + the OpenWAM WebSocket path.

Modes:
  import     - import robocasa + gym wrapper; confirm robocasa/<Task> registration
  env        - gym.make + reset + step a zero action dict (sim plumbing, no policy)
  roundtrip  - ping + predict a dummy obs against a running OpenWAM server (no sim)

robocasa / robosuite / MuJoCo live in a separate env; run with
``ROBOCASA365_PYTHON``. ``roundtrip`` only needs
``benchmarks.utils`` dependencies.

For per-step obs/action inspection during a real eval, set ``debug: true`` in
``policy_config.yml`` (the adapter dumps ``ep{N}/step_{N}/`` bundles).
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def import_smoke() -> None:
    import gymnasium as gym  # noqa: PLC0415
    import robocasa  # noqa: F401,PLC0415
    import robocasa.wrappers.gym_wrapper  # noqa: F401,PLC0415  (registers robocasa/<Task>)

    ids = sorted(k for k in gym.envs.registry.keys() if k.startswith("robocasa/"))
    print(f"robocasa_env_count={len(ids)}")
    print(f"sample_envs={','.join(ids[:5])}")


def env_smoke(task: str, split: str, steps: int) -> None:
    import gymnasium as gym  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import robocasa  # noqa: F401,PLC0415
    import robocasa.wrappers.gym_wrapper  # noqa: F401,PLC0415
    from openwam2robocasa365_interface import ACTION_SLICES  # noqa: PLC0415

    env = gym.make(f"robocasa/{task}", split=split, enable_render=True)
    try:
        obs, info = env.reset(seed=0)
        zero_action = {key: np.zeros(end - start, dtype=np.float32) for key, (start, end) in ACTION_SLICES.items()}
        # control_mode=-1 = "achieved" mode (no base driving) for this zero-action plumbing smoke; a
        # zero here binarizes to -1 at the env's 0.5 threshold anyway, but -1 is explicit.
        zero_action["action.control_mode"][:] = -1.0
        reward = done = None
        for _ in range(steps):
            obs, reward, done, truncated, info = env.step(zero_action)
        print(f"env_smoke=ok task={task} reward={reward} done={done}")
        print(f"obs_keys={','.join(sorted(obs.keys()))}")
    finally:
        env.close()


def roundtrip_smoke(host: str, port: int, state_dim: int) -> None:
    import numpy as np  # noqa: PLC0415

    from benchmarks.utils import (  # noqa: PLC0415
        WSPolicyClient,
        build_payload,
        encode_numpy_b64,
        transport,
    )

    black = np.zeros((256, 256, 3), dtype=np.uint8)
    client = WSPolicyClient(f"ws://{host}:{port}")
    try:
        pong = client.ping()
        if pong.get("type") != transport.PONG:
            raise SystemExit(f"unexpected ping response: {pong}")
        payload = build_payload(
            head=encode_numpy_b64(black),
            left_wrist=encode_numpy_b64(black),
            # The real client sends robot0_agentview_right in this fixed
            # transport slot; the smoke image is intentionally synthetic.
            right_wrist=encode_numpy_b64(black),
            prompt="smoke",
            state=[0.0] * state_dim,
        )
        action = client.predict(payload)["action"]
        print(f"roundtrip=ok action_dim={len(action)}")
    finally:
        client.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["import", "env", "roundtrip"], default="import")
    parser.add_argument("--task", default=os.environ.get("ROBOCASA365_SMOKE_TASK", "OpenDrawer"))
    parser.add_argument("--split", default=os.environ.get("ROBOCASA365_SMOKE_SPLIT", "pretrain"))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("ROBOCASA365_SMOKE_STEPS", "1")))
    parser.add_argument("--host", default=os.environ.get("ROBOCASA365_POLICY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ROBOCASA365_PORT", "8848")))
    parser.add_argument("--state-dim", type=int, default=19)  # compact EEF10 + world base pose9
    args = parser.parse_args(argv)

    if args.mode == "import":
        import_smoke()
    elif args.mode == "env":
        env_smoke(args.task, args.split, args.steps)
    else:
        roundtrip_smoke(args.host, args.port, args.state_dim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
