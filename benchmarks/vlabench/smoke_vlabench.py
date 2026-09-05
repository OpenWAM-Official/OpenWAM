#!/usr/bin/env python3
"""Smoke checks for the VLABench <-> OpenWAM integration.

No GPU, no checkpoint, no trained policy required.

Modes
-----
``env``
    Load one VLABench task and assert the observation contract this client
    depends on: camera count and ordering, ``ee_state`` width, and the presence
    of ``instruction`` / robot base position.

``loop``
    Full closed loop against an in-process **mock** OpenWAM server that echoes
    the proprio back as the action (a hold-still policy). Exercises the real
    ``Evaluator``, the real adapter, and the real IK — so it catches wiring,
    frame and dtype breakage that unit tests cannot.

Usage (inside the VLABench env, via run_smoke.sh):
    python smoke_vlabench.py --mode env  --task select_fruit
    python smoke_vlabench.py --mode loop --task select_fruit --max-steps 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

EXPECTED_CAMERAS = {
    0: "right",
    1: "left",
    2: "forward",
    3: "franka/Franka_wrist_cam",
}


def _load_env(task: str, seed: int = 42):
    import random

    # Importing these modules is what registers the robot / task classes that
    # load_env resolves by name; the star-import upstream uses is only needed at
    # module level, and `import *` is a SyntaxError inside a function.
    import VLABench.robots  # noqa: F401
    import VLABench.tasks  # noqa: F401
    from VLABench.envs import load_env

    np.random.seed(seed)
    random.seed(seed)
    env = load_env(task, random_init=True, eval=False, run_mode="eval")
    env.reset()
    return env


def _camera_names(env, ncam: int) -> list:
    try:
        import mujoco

        model = getattr(env.physics.model, "_model", env.physics.model)
        return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(ncam)]
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"  (camera names unavailable: {exc})")
        return []


def check_env(task: str) -> int:
    """Assert the obs contract the adapter relies on."""
    from openwam2vlabench_interface import (
        HEAD_CAMERA_INDEX,
        LEFT_WRIST_CAMERA_INDEX,
        RIGHT_WRIST_CAMERA_INDEX,
    )

    env = _load_env(task)
    try:
        obs = env.get_observation(require_pcd=False)
        obs["instruction"] = env.task.get_instruction()
        base = np.asarray(env.get_robot_frame_position(), np.float64).reshape(-1)

        failures = []
        rgb = np.asarray(obs["rgb"])
        print(f"  rgb            {rgb.shape} {rgb.dtype}")
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            failures.append(f"rgb must be (ncam, H, W, 3), got {rgb.shape}")
        ncam = rgb.shape[0]
        for idx, label in (
            (HEAD_CAMERA_INDEX, "head"),
            (LEFT_WRIST_CAMERA_INDEX, "left_wrist"),
            (RIGHT_WRIST_CAMERA_INDEX, "right_wrist"),
        ):
            if not (0 <= idx < ncam):
                failures.append(f"{label} camera index {idx} out of range (ncam={ncam})")

        names = _camera_names(env, ncam)
        for i, nm in enumerate(names):
            print(f"  cam[{i}]         {nm}")
        for i, expected in EXPECTED_CAMERAS.items():
            if i < len(names) and names[i] != expected:
                failures.append(f"cam[{i}] is {names[i]!r}, expected {expected!r} — camera order changed upstream")

        ee = np.asarray(obs["ee_state"], np.float64).reshape(-1)
        print(f"  ee_state       {ee.shape}  (need >= 8: pos3 + quat_wxyz4 + grip1)")
        if ee.shape[0] < 8:
            failures.append(f"ee_state must be >= 8-D, got {ee.shape[0]}")

        print(f"  robot_frame    {base.tolist()}")
        if base.shape[0] != 3:
            failures.append(f"robot base position must be 3-D, got {base.shape[0]}")

        print(f"  instruction    {obs['instruction']!r}")
        if not obs.get("instruction"):
            failures.append("task.get_instruction() returned empty")

        # The adapter's own conversion, on live data.
        from benchmarks.utils import vlabench_obs_to_eef10

        eef10 = vlabench_obs_to_eef10(ee, base)
        print(f"  -> EEF10       {np.round(eef10, 4).tolist()}")
        if eef10.shape != (10,):
            failures.append(f"vlabench_obs_to_eef10 produced {eef10.shape}, expected (10,)")

        for f in failures:
            print(f"  [FAIL] {f}")
        return 1 if failures else 0
    finally:
        env.close()


class _MockServer:
    """Minimal in-process OpenWAM policy server that echoes proprio as action."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.calls = 0
        self.resets = 0
        self._server = None
        self._thread = None
        self._ready = threading.Event()

    def _handler(self, ws):
        from benchmarks.utils import transport

        for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == transport.PING:
                ws.send(json.dumps({"type": transport.PONG}))
            elif kind == transport.RESET:
                self.resets += 1
                ws.send(json.dumps({"type": transport.RESET_ACK}))
            elif kind == transport.OBS:
                self.calls += 1
                state = msg.get("state")
                if not state or len(state) != 10:
                    ws.send(
                        json.dumps(
                            {
                                "type": transport.ERROR,
                                "code": transport.ERR_OBS_VALIDATION,
                                "message": f"expected a 10-D state, got {state and len(state)}",
                            }
                        )
                    )
                    continue
                # Hold still: command the pose we are already at.
                ws.send(json.dumps({"type": transport.ACTION, "action": list(state), "step": self.calls}))
            else:
                ws.send(json.dumps({"type": transport.ERROR, "code": transport.ERR_UNKNOWN_TYPE, "message": kind}))

    def start(self) -> int:
        from websockets.sync.server import serve

        self._server = serve(self._handler, self.host, self.port, max_size=None)
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()


def check_loop(task: str, max_steps: int, server_addr: str | None = None) -> int:
    """Run the real adapter against the live env.

    Without ``server_addr`` an in-process mock stands in for the policy server
    (echo-proprio hold-still), which keeps the check GPU- and checkpoint-free.
    Pass ``host:port`` to drive a REAL OpenWAM server instead — that also
    exercises the wire's action width, denormalization and latency.
    """
    from openwam2vlabench_interface import OpenWAMVLABenchPolicy

    server = None
    if server_addr:
        host, _, port_s = server_addr.rpartition(":")
        host, port = (host or "127.0.0.1"), int(port_s)
        timeout = 300
        print(f"  live server    ws://{host}:{port}")
    else:
        server = _MockServer()
        port, host, timeout = server.start(), "127.0.0.1", 30
        print(f"  mock server    ws://127.0.0.1:{port} (echo-proprio hold-still policy)")

    env = None
    try:
        policy = OpenWAMVLABenchPolicy(host=host, port=port, request_timeout=timeout)
        policy.reset()

        env = _load_env(task)
        from VLABench.utils.utils import euler_to_quaternion

        base = env.get_robot_frame_position()
        for step in range(max_steps):
            obs = env.get_observation(require_pcd=False)
            obs["instruction"] = env.task.get_instruction()
            obs["robot_frame"] = base

            pos, euler, gripper = policy.predict(obs)
            if step == 0:
                if server is not None:
                    # The mock commands the pose we are already at, so the whole
                    # world -> base -> EEF10 -> wire -> base -> world round trip
                    # must land back on the current position. Any drift here is a
                    # base-offset or rotation-convention error.
                    world_pos = np.asarray(obs["ee_state"], np.float64)[:3]
                    drift = float(np.linalg.norm(np.asarray(pos) - world_pos))
                    print(f"  hold-still drift {drift:.2e} m  (frame round-trip; should be ~0)")
                    if drift > 1e-4:
                        print(f"  [FAIL] frame round-trip drifted {drift:.3e} m — base offset mismatch")
                        return 1
                print(f"  target_pos     {np.round(pos, 4).tolist()}")
                print(f"  target_euler   {np.round(euler, 4).tolist()}")
                suffix = "  (mock echoes proprio; polarity differs)" if server is not None else ""
                print(f"  gripper_state  {np.round(gripper, 4).tolist()}{suffix}")

            # Same path the evaluator takes: euler -> quat -> IK -> env.step.
            quat = euler_to_quaternion(*euler)
            _, qpos = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=quat)
            env.step(np.concatenate([qpos, gripper]))

        if server is not None:
            print(f"  stepped        {max_steps} steps, {server.calls} predicts, {server.resets} resets")
            if server.calls != max_steps:
                print(f"  [FAIL] server saw {server.calls} predicts, expected {max_steps}")
                return 1
        else:
            print(f"  stepped        {max_steps} steps against the live server")
        policy.close()
        return 0
    finally:
        if env is not None:
            env.close()
        if server is not None:
            server.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("env", "loop"), default="env")
    parser.add_argument("--task", default="select_fruit")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--server",
        default=None,
        help="loop mode: drive a REAL OpenWAM server at host:port instead of the in-process mock",
    )
    args = parser.parse_args()

    if not os.environ.get("VLABENCH_ROOT"):
        raise SystemExit("VLABENCH_ROOT is not set (run_smoke.sh sets it)")

    print(f"[smoke:{args.mode}] task={args.task}")
    rc = check_env(args.task) if args.mode == "env" else check_loop(args.task, args.max_steps, server_addr=args.server)
    print("PASS" if rc == 0 else "FAIL")
    sys.exit(rc)


if __name__ == "__main__":
    main()
