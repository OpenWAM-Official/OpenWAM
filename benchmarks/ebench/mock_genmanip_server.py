"""Offline mock of the GenManip eval server for EBench bridge validation.

Replays real EBench dataset episodes (parquet states + mp4 frames) over the
exact EvalClient wire contract — pickle bodies on ``/reset`` / ``/step``,
``GET /docs`` health, ``POST /kill`` — and validates every incoming action
dict the way the real server would consume it
(``genmanip/core/evaluator/utils.py::parse_embodiment_action`` +
``env.step`` base handling):

* ``control_type == "ee_pose"``, two ``(pos, quat_wxyz, grip)`` arm entries;
* ``position + orientation`` must be *list* concatenation producing 7 floats
  (the real server passes that straight to cuRobo ``ik_single`` — numpy
  arrays would broadcast-add and crash it);
* unit-norm quaternion (else real IK silently falls back to current joints);
* per-finger gripper inside GenManip's invalid-state guard ``[-0.01, 0.054]``;
* ``base_motion`` 3 finite floats; with ``base_is_rel=True`` the step counts
  against the real server's ±0.015 m / ±1° per-step clamps (over-limit steps
  are tallied and reported, mirroring the silent clipping the sim applies).

The mock is OPEN-LOOP: actions are validated and logged (JSONL), obs replay
real demo frames regardless. That exercises the full
EvalClient → bridge → OpenWAM-server → bridge → wire path without Isaac Sim.
Real closed-loop scoring still requires the GenManip server (see the
"实地验证" step in benchmarks/ebench/README.md).

Run inside an env with numpy / pandas / pyarrow / av (e.g. the training env):

    python benchmarks/ebench/mock_genmanip_server.py \
        --dataset-dir /path/to/EBench-Dataset --bucket simple_pnp/task1 \
        --episodes 2 --steps-per-episode 8 --port 8087
"""

import argparse
import json
import logging
import pickle
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

logger = logging.getLogger("mock_genmanip")

CAMS = (
    "video.overlook_camera_view",
    "video.left_camera_view",
    "video.right_camera_view",
)
# GenManip invalid-state guard + per-step base clamps (env.py / dualarm_manip.py)
GRIPPER_GUARD = (-0.01, 0.054)
BASE_STEP_CLAMP_M = 0.015
BASE_STEP_CLAMP_DEG = 1.0


def _decode_frames(video_path: Path, indices: list) -> list:
    """Decode specific frame indices from an mp4 as RGB uint8 arrays."""
    import av

    wanted = sorted(set(int(i) for i in indices))
    out = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i > wanted[-1]:
                break
            if i in wanted:
                out[i] = frame.to_ndarray(format="rgb24")
    missing = [i for i in wanted if i not in out]
    if missing:
        raise RuntimeError(f"{video_path} missing frames {missing}")
    return [out[int(i)] for i in indices]


class EpisodeReplay:
    """Preloaded obs stream for one dataset episode."""

    def __init__(self, bucket: Path, episode_index: int, num_steps: int):
        import pandas as pd

        with (bucket / "meta" / "info.json").open() as f:
            info = json.load(f)
        data_path = bucket / info["data_path"].format(
            episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
            episode_index=episode_index,
            chunk_index=episode_index // int(info.get("chunks_size", 1000)),
        )
        df = pd.read_parquet(data_path)
        self.num_steps = min(int(num_steps), len(df))
        self.states = {
            key: np.stack(df[key].to_numpy())[: self.num_steps].astype(np.float32)
            for key in ("state.ee_pose", "state.gripper", "state.base")
        }
        task_idx = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
        self.instruction = None
        tasks_path = bucket / "meta" / "tasks.jsonl"
        with tasks_path.open() as f:
            for line in f:
                row = json.loads(line)
                if int(row["task_index"]) == task_idx:
                    self.instruction = str(row["task"])
                    break
        if not self.instruction:
            raise ValueError(f"no task text for task_index={task_idx} in {tasks_path}")

        indices = list(range(self.num_steps))
        self.frames = {}
        for cam in CAMS:
            video_path = bucket / info["video_path"].format(
                episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
                episode_index=episode_index,
                chunk_index=episode_index // int(info.get("chunks_size", 1000)),
                video_key=cam,
            )
            self.frames[cam] = _decode_frames(video_path, indices)
        logger.info(
            "episode %d loaded: %d steps, instruction=%r",
            episode_index,
            self.num_steps,
            self.instruction,
        )

    def obs_at(self, t: int, episode_id: str) -> dict:
        obs = {
            "reset": t == 0,
            "timestep": t,
            "episode_id": episode_id,
            "robot_id": "manip/lift2/R5a",
            "instruction": self.instruction,
            "state.joints": np.zeros(12, dtype=np.float32),  # unused by the bridge
            "state.ee_pose": [
                [self.states["state.ee_pose"][t][0:3].tolist(), self.states["state.ee_pose"][t][3:7].tolist()],
                [self.states["state.ee_pose"][t][7:10].tolist(), self.states["state.ee_pose"][t][10:14].tolist()],
            ],
            "state.gripper": self.states["state.gripper"][t],
            "state.base": self.states["state.base"][t],
        }
        for cam in CAMS:
            obs[cam] = self.frames[cam][t]
        return obs


class MockState:
    def __init__(self, replays: list, log_path: Path):
        self.replays = replays
        self.episode = 0
        self.t = 0
        self.worker_ids: list = []
        self.actions_logged = 0
        self.validation_errors: list = []
        self.base_overlimit_steps = 0
        self.log_file = log_path.open("w")
        self.lock = threading.Lock()

    def validate_action(self, action: dict) -> None:
        """Emulate the real server's consumption; raise on contract violations."""
        if action.get("control_type") != "ee_pose":
            raise ValueError(f"control_type must be 'ee_pose', got {action.get('control_type')!r}")
        if action.get("is_rel") is not False:
            raise ValueError("is_rel must be False (absolute EE targets)")
        arms = action.get("action")
        if not isinstance(arms, list) or len(arms) != 2:
            raise ValueError(
                f"'action' must be a list of 2 arm tuples, got {type(arms)} len {len(arms) if isinstance(arms, list) else '?'}"
            )
        for i, arm in enumerate(arms):
            pos, quat, grip = arm
            # Real server: planner.ik_single(position + orientation, ...) — list concat.
            if not isinstance(pos, list) or not isinstance(quat, list):
                raise ValueError(f"arm {i}: position/orientation must be Python lists (server does list concat)")
            combined = pos + quat
            if len(combined) != 7:
                raise ValueError(f"arm {i}: position+orientation must concat to 7 values, got {len(combined)}")
            if not all(isinstance(v, float) for v in combined):
                raise ValueError(f"arm {i}: pose values must be plain floats")
            norm = float(np.linalg.norm(quat))
            if abs(norm - 1.0) > 0.05:
                raise ValueError(f"arm {i}: quaternion norm {norm:.4f} not unit (real IK would silently hold joints)")
            if len(grip) != 2:
                raise ValueError(f"arm {i}: gripper must have 2 finger values")
            for g in grip:
                if not (GRIPPER_GUARD[0] <= float(g) <= GRIPPER_GUARD[1]):
                    raise ValueError(f"arm {i}: gripper {g} outside guard {GRIPPER_GUARD}")
        base = action.get("base_motion")
        if base is None or len(base) != 3 or not np.isfinite(np.asarray(base, dtype=np.float64)).all():
            raise ValueError(f"base_motion must be 3 finite floats, got {base!r}")
        if not isinstance(action.get("base_is_rel"), bool):
            raise ValueError("base_is_rel must be a bool")
        if action["base_is_rel"]:
            if (
                abs(base[0]) > BASE_STEP_CLAMP_M
                or abs(base[1]) > BASE_STEP_CLAMP_M
                or abs(base[2]) > BASE_STEP_CLAMP_DEG
            ):
                self.base_overlimit_steps += 1  # real server clips silently; tally it

    def current_replay(self) -> "EpisodeReplay":
        return self.replays[self.episode]

    def obs_payload(self) -> dict:
        replay = self.current_replay()
        episode_id = f"ebench-mock/run/ep{self.episode}/{self.episode:03d}"
        inner = replay.obs_at(self.t, episode_id)
        return {wid: {"obs": inner, "metric": None} for wid in self.worker_ids}

    def final_payload(self) -> dict:
        metric = {"mock_replay": {"score": 0.0, "sr": 0.0}}
        return {wid: {"obs": None, "metric": metric} for wid in self.worker_ids}

    def advance(self) -> dict:
        self.t += 1
        if self.t >= self.current_replay().num_steps:
            self.episode += 1
            self.t = 0
        if self.episode >= len(self.replays):
            return self.final_payload()
        return self.obs_payload()


class Handler(BaseHTTPRequestHandler):
    state: MockState = None  # injected

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, content_type: str = "application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/docs"):
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        path = self.path.split("?")[0]
        st = self.state
        try:
            if path == "/kill" or path == "/create_workers":
                self._send(200, json.dumps({"ok": True}).encode(), "application/json")
                return
            if path == "/reset":
                req = pickle.loads(raw)
                with st.lock:
                    st.worker_ids = [str(w) for w in req["worker_ids"]]
                    st.episode, st.t = 0, 0
                    payload = st.obs_payload()
                self._send(200, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                return
            if path == "/step":
                actions = pickle.loads(raw)
                with st.lock:
                    for wid, action in actions.items():
                        try:
                            st.validate_action(action)
                        except ValueError as e:
                            st.validation_errors.append(str(e))
                            logger.error("ACTION CONTRACT VIOLATION (worker %s): %s", wid, e)
                        st.log_file.write(
                            json.dumps(
                                {
                                    "episode": st.episode,
                                    "t": st.t,
                                    "worker": str(wid),
                                    "base_motion": [float(v) for v in action.get("base_motion", [])],
                                    "base_is_rel": action.get("base_is_rel"),
                                    "grip": [float(action["action"][0][2][0]), float(action["action"][1][2][0])],
                                    "l_pos": [float(v) for v in action["action"][0][0]],
                                }
                            )
                            + "\n"
                        )
                        st.log_file.flush()
                        st.actions_logged += 1
                    payload = st.advance()
                    if all(v["obs"] is None for v in payload.values()):
                        logger.info(
                            "replay finished: %d actions, %d contract violations, %d base over-limit steps",
                            st.actions_logged,
                            len(st.validation_errors),
                            st.base_overlimit_steps,
                        )
                self._send(200, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                return
            self._send(404, b"not found", "text/plain")
        except Exception as e:  # noqa: BLE001 — mirror the real server's 500 body
            logger.exception("mock server error")
            self._send(500, json.dumps({"detail": str(e)}).encode(), "application/json")


def main():
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--bucket", default="simple_pnp/task1")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--first-episode-index", type=int, default=0)
    p.add_argument("--steps-per-episode", type=int, default=8)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8087)
    p.add_argument("--log-file", default="/tmp/ebench_mock_actions.jsonl")
    args = p.parse_args()

    bucket = Path(args.dataset_dir) / args.bucket
    replays = [
        EpisodeReplay(bucket, args.first_episode_index + i, args.steps_per_episode) for i in range(args.episodes)
    ]
    Handler.state = MockState(replays, Path(args.log_file))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info(
        "mock GenManip server on %s:%d (%d episodes × %d steps)",
        args.host,
        args.port,
        args.episodes,
        args.steps_per_episode,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
