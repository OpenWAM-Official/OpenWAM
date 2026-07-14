"""Offline mock of the GenManip eval server for EBench bridge validation.

Replays real EBench dataset episodes over the EvalClient wire contract and
acts as a FAILURE-TYPE ORACLE: any action-contract violation returns HTTP 500
(the real chain propagates parse errors worker→ray→FastAPI 500), so a broken
bridge FAILS a mock run instead of logging-and-passing.

Wire fidelity (mirrors ``genmanip/utils/standalone/io_utils.py`` +
``env.get_obs``):

* cameras ship as ``{"type": "jpeg_bytes", "dtype", "shape", "data"}`` (JPEG
  q85) and numpy states as ``{"type": "numpy_array", base64}`` — the
  EvalClient decode paths (TurboJPEG, frombuffer) are actually exercised;
* ``/reset`` returns ``reset_pending`` and the first obs is delivered via the
  ``/reset_result`` poll, exercising the client's async-reset path;
* every step carries a non-None running ``metric`` dict and episode rollovers
  carry an ``episode_result``, matching the real per-worker wrapper.

Action validation emulates the real consumption path
(``parse_embodiment_action`` + ``env.step``): ``ee_pose`` structure, LIST
``position + orientation`` concat to 7 plain floats, finiteness (NaN-safe:
``not (|norm-1| <= tol)``), unit quaternion, 2 finger values inside the
achieved-state guard band, ``base_motion`` finiteness; relative-base steps
are tallied against the real ±0.015 m / ±1° clamps and absolute-base steps
against the ±0.2 m / 20° jump guard.

KNOWN DIVERGENCES from the real server (inherent to an open-loop replay —
see tests/benchmarks and the gate-3 review for the full list):

* no IK / physics / achieved-state simulation: unreachable poses, cuRobo
  IK-hold, invalid-state early termination, and success scoring do not exist;
  episodes always run exactly ``--steps-per-episode`` frames;
* ingress checks are STRICTER than real in places (real accepts tuples,
  missing base keys, non-unit quats — cuRobo normalizes): this mock is a
  bridge-compliance checker, not a permissiveness emulator;
* the real gripper guard applies to the ACHIEVED state post-physics; here it
  is applied to the command (the bridge clips to [0, 0.044] anyway);
* single global replay state: run ONE EvalClient against one mock instance.

Run inside an env with numpy / pandas / pyarrow / av (e.g. the training env):

    python benchmarks/ebench/mock_genmanip_server.py \
        --dataset-dir /path/to/EBench-Dataset --bucket simple_pnp/task1 \
        --episodes 2 --steps-per-episode 8 --port 8087
"""

import argparse
import base64
import io
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
BASE_JUMP_GUARD_M = 0.2
BASE_JUMP_GUARD_DEG = 20.0


def _encode_jpeg_dict(rgb: np.ndarray) -> dict:
    """Camera frame → the real server's jpeg_bytes typed dict (JPEG q85)."""
    try:
        from turbojpeg import TJPF_RGB, TurboJPEG

        data = TurboJPEG().encode(rgb, quality=85, pixel_format=TJPF_RGB)
    except Exception:  # noqa: BLE001 — PIL JPEG decodes fine through TurboJPEG
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
    return {"type": "jpeg_bytes", "dtype": str(rgb.dtype), "shape": rgb.shape, "data": data}


def _encode_numpy_dict(arr: np.ndarray) -> dict:
    """State vector → the real server's numpy_array typed dict."""
    arr = np.ascontiguousarray(arr)
    return {
        "type": "numpy_array",
        "dtype": str(arr.dtype),
        "shape": arr.shape,
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
    }


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
        chunk = episode_index // int(info.get("chunks_size", 1000))
        data_path = bucket / info["data_path"].format(
            episode_chunk=chunk, episode_index=episode_index, chunk_index=chunk
        )
        df = pd.read_parquet(data_path)
        self.num_steps = min(int(num_steps), len(df))
        self.states = {
            key: np.stack(df[key].to_numpy())[: self.num_steps].astype(np.float32)
            for key in ("state.ee_pose", "state.gripper", "state.base", "state.joints")
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
                episode_chunk=chunk, episode_index=episode_index, chunk_index=chunk, video_key=cam
            )
            self.frames[cam] = _decode_frames(video_path, indices)
        logger.info(
            "episode %d loaded: %d steps, instruction=%r",
            episode_index,
            self.num_steps,
            self.instruction,
        )

    def obs_at(self, t: int, episode_id: str) -> dict:
        ee = self.states["state.ee_pose"][t]
        obs = {
            "reset": t == 0,
            "timestep": t,
            "episode_id": episode_id,
            "robot_id": "manip/lift2/R5a",
            "instruction": self.instruction,
            "state.joints": _encode_numpy_dict(self.states["state.joints"][t]),
            "state.ee_pose": [
                [ee[0:3].tolist(), ee[3:7].tolist()],
                [ee[7:10].tolist(), ee[10:14].tolist()],
            ],
            "state.gripper": _encode_numpy_dict(self.states["state.gripper"][t]),
            "state.base": _encode_numpy_dict(self.states["state.base"][t]),
        }
        for cam in CAMS:
            obs[cam] = _encode_jpeg_dict(self.frames[cam][t])
        return obs


class ContractViolation(ValueError):
    """Action violates the wire contract — mapped to HTTP 500 like the real chain."""


class MockState:
    def __init__(self, replays: list, log_path: Path):
        self.replays = replays
        self.episode = 0
        self.t = 0
        self.worker_ids: list = []
        self.pending_reset = False
        self.actions_logged = 0
        self.violations = 0
        self.base_overlimit_steps = 0
        self.base_jump_guard_steps = 0
        self.log_file = log_path.open("w")
        self.lock = threading.Lock()

    def validate_action(self, action: dict) -> None:
        """Emulate the real server's consumption; raise ContractViolation."""
        if action.get("control_type") != "ee_pose":
            raise ContractViolation(f"control_type must be 'ee_pose', got {action.get('control_type')!r}")
        if action.get("is_rel") is not False:
            raise ContractViolation("is_rel must be False (absolute EE targets)")
        arms = action.get("action")
        if not isinstance(arms, list) or len(arms) != 2:
            raise ContractViolation(f"'action' must be a list of 2 arm tuples, got {type(arms).__name__}")
        for i, arm in enumerate(arms):
            pos, quat, grip = arm
            # Real server: planner.ik_single(position + orientation, ...) — list concat.
            if not isinstance(pos, list) or not isinstance(quat, list):
                raise ContractViolation(f"arm {i}: position/orientation must be Python lists (server list-concats)")
            combined = pos + quat
            if len(combined) != 7:
                raise ContractViolation(f"arm {i}: position+orientation must concat to 7 values, got {len(combined)}")
            if not all(isinstance(v, float) for v in combined):
                raise ContractViolation(f"arm {i}: pose values must be plain floats")
            if not np.isfinite(combined).all():
                raise ContractViolation(f"arm {i}: non-finite pose values {combined}")
            norm = float(np.linalg.norm(quat))
            # NaN-safe: `not (<= tol)` also catches norm=NaN, which a plain
            # `abs(norm-1) > tol` comparison would silently accept.
            if not (abs(norm - 1.0) <= 0.05):
                raise ContractViolation(f"arm {i}: quaternion norm {norm:.4f} not unit (real IK would hold joints)")
            if len(grip) != 2:
                raise ContractViolation(f"arm {i}: gripper must have 2 finger values")
            for g in grip:
                if not (GRIPPER_GUARD[0] <= float(g) <= GRIPPER_GUARD[1]):
                    raise ContractViolation(f"arm {i}: gripper {g} outside guard {GRIPPER_GUARD}")
        base = action.get("base_motion")
        if base is None or len(base) != 3 or not np.isfinite(np.asarray(base, dtype=np.float64)).all():
            raise ContractViolation(f"base_motion must be 3 finite floats, got {base!r}")
        if not isinstance(action.get("base_is_rel"), bool):
            raise ContractViolation("base_is_rel must be a bool")
        if action["base_is_rel"]:
            if (
                abs(base[0]) > BASE_STEP_CLAMP_M
                or abs(base[1]) > BASE_STEP_CLAMP_M
                or abs(base[2]) > BASE_STEP_CLAMP_DEG
            ):
                self.base_overlimit_steps += 1  # real server clips silently; tally it
        else:
            # Absolute mode: the real server has NO per-step clamp — a large
            # jump trips the achieved-state guard and zero-scores the episode.
            prev = self.replays[self.episode].states["state.base"][max(self.t - 1, 0)]
            if (
                abs(base[0] - float(prev[0])) > BASE_JUMP_GUARD_M
                or abs(base[1] - float(prev[1])) > BASE_JUMP_GUARD_M
                or abs(base[2] - float(np.degrees(prev[2]))) > BASE_JUMP_GUARD_DEG
            ):
                self.base_jump_guard_steps += 1

    def current_replay(self) -> "EpisodeReplay":
        return self.replays[self.episode]

    def _episode_id(self) -> str:
        return f"ebench-mock/run/ep{self.episode}/{self.episode:03d}"

    def _running_metric(self) -> dict:
        # Real pool sends a non-None running metric dict on EVERY step.
        return {f"*({self.episode}/{len(self.replays)})mock_replay": {"score": 0.0, "sr": 0.0}}

    def obs_payload(self, episode_result: dict = None) -> dict:
        inner = self.current_replay().obs_at(self.t, self._episode_id())
        wrapper = {"obs": inner, "metric": self._running_metric(), "episode_result": episode_result}
        return {wid: wrapper for wid in self.worker_ids}

    def final_payload(self) -> dict:
        metric = {"mock_replay": {"score": 0.0, "sr": 0.0}}
        return {wid: {"obs": None, "metric": metric, "episode_result": None} for wid in self.worker_ids}

    def advance(self) -> dict:
        self.t += 1
        episode_result = None
        if self.t >= self.current_replay().num_steps:
            episode_result = {
                "episode_id": self._episode_id(),
                "task_name": "mock_replay",
                "seed": self.episode,
                "score": 0.0,
                "sr": 0.0,
                "worker_id": self.worker_ids[0] if self.worker_ids else "0",
            }
            self.episode += 1
            self.t = 0
        if self.episode >= len(self.replays):
            logger.info(
                "replay finished: %d actions, %d violations (500'd), %d rel-base over-limit, %d abs-base jump-guard",
                self.actions_logged,
                self.violations,
                self.base_overlimit_steps,
                self.base_jump_guard_steps,
            )
            return self.final_payload()
        return self.obs_payload(episode_result)


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

    def _send_pickle(self, payload) -> None:
        self._send(200, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

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
            if path in ("/kill", "/create_workers"):
                self._send(200, json.dumps({"ok": True}).encode(), "application/json")
                return
            if path == "/reset":
                req = pickle.loads(raw)
                with st.lock:
                    st.worker_ids = [str(w) for w in req["worker_ids"]]
                    st.episode, st.t = 0, 0
                    st.pending_reset = True
                # Async-reset protocol: first response only flags the pending
                # reset; the client must poll /reset_result for the real obs.
                self._send_pickle({wid: {"obs": None, "metric": None, "reset_pending": True} for wid in st.worker_ids})
                return
            if path == "/reset_result":
                with st.lock:
                    if not st.pending_reset:
                        self._send_pickle({wid: {"obs": None, "metric": None} for wid in st.worker_ids})
                        return
                    st.pending_reset = False
                    self._send_pickle(st.obs_payload())
                return
            if path == "/step":
                actions = pickle.loads(raw)
                with st.lock:
                    for wid, action in actions.items():
                        try:
                            st.validate_action(action)
                        except ContractViolation as e:
                            st.violations += 1
                            logger.error("ACTION CONTRACT VIOLATION (worker %s): %s", wid, e)
                            # Failure-type oracle: the real chain 500s on parse
                            # errors — a broken bridge must FAIL the mock run.
                            self._send(500, json.dumps({"detail": str(e)}).encode(), "application/json")
                            return
                        st.log_file.write(
                            json.dumps(
                                {
                                    "episode": st.episode,
                                    "t": st.t,
                                    "worker": str(wid),
                                    "base_motion": [float(v) for v in action["base_motion"]],
                                    "base_is_rel": action["base_is_rel"],
                                    "grip": [float(action["action"][0][2][0]), float(action["action"][1][2][0])],
                                    "l_pos": [float(v) for v in action["action"][0][0]],
                                }
                            )
                            + "\n"
                        )
                        st.log_file.flush()
                        st.actions_logged += 1
                    payload = st.advance()
                self._send_pickle(payload)
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
