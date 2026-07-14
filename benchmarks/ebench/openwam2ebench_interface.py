"""EBench (GenManip) eval driver for the OpenWAM Policy Server.

Unlike RoboTwin/RoboCasa365 (where the sim loads our adapter), EBench inverts
control: the *policy side* is a client that polls the GenManip Isaac Sim eval
server with ``genmanip_client.EvalClient``. This module is therefore a
standalone driver — the EvalClient loop on the north side, a
``benchmarks.utils.WSPolicyClient`` to the OpenWAM policy server on the south
side::

    GenManip eval server (Isaac Sim / online endpoint, :8087)
        ▲ EvalClient  (pickle wire; obs down, action dicts up)
    THIS DRIVER  — one process per worker_id
        ▼ WSPolicyClient (JSON-WS)
    OpenWAM policy server (:8848, one server per worker — stateful executor)

Per sim step: obs → (images + wrapped prompt + RAW-23 proprio) → server
normalizes/infers/denormalizes → RAW-23 physical action → EBench ``ee_pose``
action dict (GenManip runs cuRobo IK server-side). Single-step ``step()`` only:
``/step_chunk`` returns just the final obs, but the OpenWAM executor consumes
one obs per popped action.

Conversions live in ``benchmarks/utils/action_conversion.py`` and mirror the
trainer's rendering byte-for-byte (pinned by tests/benchmarks/test_ebench_bridge.py).
``--base-mode`` must match the checkpoint's ``dataloader.base_action_source``
(default ``delta``).

Env: the GenManip client env (``pip install -e genmanip-client``) plus
``numpy, Pillow, websockets>=15``. No torch, no openwam import.
"""

# benchmarks.utils lives two levels up; make it importable from the EBench
# client env without installing the repo.
import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402

from benchmarks.ebench.prompt_template import format_prompt_for_inference  # noqa: E402
from benchmarks.utils import (  # noqa: E402
    WSPolicyClient,
    build_payload,
    encode_numpy_b64,
)
from benchmarks.utils.action_conversion import (  # noqa: E402
    EBENCH_BASE_SOURCES,
    EBENCH_RAW_DIM,
    ebench_obs_to_raw23,
    ebench_render_state_base,
    raw23_to_ebench_action,
)

logger = logging.getLogger("openwam2ebench")

# GenManip obs keys (see genmanip/core/evaluator/env.py::get_obs). The trainer
# composes overlook as the multiview head slot and left/right wrists as hand
# slots (configs/dataloader/ebench.yaml camera_layout) — same mapping here.
HEAD_KEY = "video.overlook_camera_view"
LEFT_WRIST_KEY = "video.left_camera_view"
RIGHT_WRIST_KEY = "video.right_camera_view"
STATE_EE_KEY = "state.ee_pose"
STATE_GRIPPER_KEY = "state.gripper"
STATE_BASE_KEY = "state.base"
INSTRUCTION_KEY = "instruction"


def wait_until_healthy(south: WSPolicyClient, deadline_s: float = 300.0) -> None:
    """Block until the OpenWAM server answers a ping (compile warmup can be slow)."""
    start = time.time()
    last_err: Optional[Exception] = None
    while time.time() - start < deadline_s:
        try:
            south.ping()
            return
        except Exception as e:  # noqa: BLE001 — retried until deadline
            last_err = e
            time.sleep(2.0)
    raise RuntimeError(f"OpenWAM server not healthy after {deadline_s}s: {last_err}")


class EBenchOpenWAMDriver:
    """One worker's EvalClient loop bridged to one OpenWAM policy server."""

    def __init__(
        self,
        south: WSPolicyClient,
        *,
        base_mode: str = "delta",
        send_state: bool = True,
    ):
        if base_mode not in EBENCH_BASE_SOURCES:
            raise ValueError(f"base_mode must be one of {EBENCH_BASE_SOURCES}, got {base_mode!r}")
        self._south = south
        self._base_mode = base_mode
        self._send_state = send_state
        self._prev_base: Optional[np.ndarray] = None
        self._episode_active = False
        self._prompt: Optional[str] = None
        self.steps = 0
        self.episodes = 0

    def on_episode_start(self, inner_obs: dict) -> None:
        """Reset south server state + bridge base bookkeeping, re-read the prompt."""
        reply = self._south.reset()
        if reply.get("type") != "reset_ack":
            raise RuntimeError(f"OpenWAM server reset not acknowledged: {reply}")
        self._prev_base = None
        instruction = inner_obs.get(INSTRUCTION_KEY)
        if not instruction or not str(instruction).strip():
            raise ValueError(
                "EBench obs carries no 'instruction' — a language-conditioned checkpoint "
                "must not run with an empty prompt"
            )
        self._prompt = format_prompt_for_inference(str(instruction))
        self._episode_active = True
        self.episodes += 1
        logger.info("episode %d start; prompt=%r", self.episodes, self._prompt)

    def act(self, inner_obs: dict) -> dict:
        """One sim step: obs dict in → EBench action dict out."""
        if inner_obs.get("reset", False) or not self._episode_active:
            self.on_episode_start(inner_obs)

        head = inner_obs.get(HEAD_KEY)
        if head is None:
            raise KeyError(f"EBench obs missing required camera {HEAD_KEY!r}")
        left = inner_obs.get(LEFT_WRIST_KEY)
        right = inner_obs.get(RIGHT_WRIST_KEY)

        state_list = None
        cur_base = None
        if self._send_state:
            cur_base = np.asarray(inner_obs[STATE_BASE_KEY], dtype=np.float64).reshape(3)
            rendered = ebench_render_state_base(cur_base, self._prev_base, self._base_mode)
            raw23 = ebench_obs_to_raw23(inner_obs[STATE_EE_KEY], inner_obs[STATE_GRIPPER_KEY], rendered)
            state_list = [float(v) for v in raw23]

        payload = build_payload(
            head=encode_numpy_b64(np.asarray(head)),
            left_wrist=encode_numpy_b64(np.asarray(left)) if left is not None else None,
            right_wrist=encode_numpy_b64(np.asarray(right)) if right is not None else None,
            prompt=self._prompt,
            state=state_list,
        )
        reply = self._south.predict(payload)
        action = np.asarray(reply["action"], dtype=np.float64).reshape(-1)
        if action.shape[0] != EBENCH_RAW_DIM:
            raise ValueError(
                f"OpenWAM server returned {action.shape[0]}-D action; expected raw "
                f"{EBENCH_RAW_DIM}-D — is the checkpoint an EBench (action_mode=ebench, "
                "unify_action=true) checkpoint?"
            )

        # Update AFTER building this step's proprio: proprio(t) differences
        # state(t) against state(t-1), exactly like training rows.
        if cur_base is not None:
            self._prev_base = cur_base
        self.steps += 1
        return raw23_to_ebench_action(action, self._base_mode)


def run_worker(args) -> None:
    from genmanip_client import EvalClient  # imported here: only needed at runtime

    south = WSPolicyClient(f"ws://{args.south_host}:{args.south_port}", timeout=float(args.request_timeout))
    wait_until_healthy(south)
    driver = EBenchOpenWAMDriver(south, base_mode=args.base_mode, send_state=not args.no_send_state)

    def make_client():
        return EvalClient(
            args.url,
            worker_ids=[str(args.worker_id)],
            token=args.token or None,
            run_id=args.run_id,
            save_process=args.save_process,
            verbose=True,
        )

    client = make_client()
    try:
        obs = client.reset()
        done = False
        while not done:
            actions = {}
            for wid, entry in obs.items():
                inner = (entry or {}).get("obs")
                if inner is None:
                    continue  # worker finished or reset pending
                actions[wid] = driver.act(inner)
            if not actions:
                break
            for attempt in range(args.client_reinit_retries + 1):
                try:
                    obs, done = client.step(actions)
                    break
                except Exception as e:  # noqa: BLE001 — north-side transport recovery
                    if attempt == args.client_reinit_retries:
                        raise
                    logger.warning(
                        "EvalClient.step failed (%s); rebuilding client (attempt %d)",
                        e,
                        attempt + 1,
                    )
                    try:
                        client.close()
                    except Exception:  # noqa: BLE001
                        pass
                    time.sleep(args.client_reinit_backoff * (attempt + 1))
                    client = make_client()
                    obs = client.reset()
                    driver._episode_active = False  # fresh episode after reconnect
        for wid, entry in (obs or {}).items():
            metric = (entry or {}).get("metric")
            if metric:
                logger.info("worker %s final metrics: %s", wid, metric)
        logger.info("run complete: %d episodes, %d steps bridged", driver.episodes, driver.steps)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        south.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8087", help="GenManip eval server / online endpoint")
    p.add_argument("--token", default="", help="Bearer token (online evaluation)")
    p.add_argument("--run-id", default="", help="run_id (== online task_id)")
    p.add_argument("--worker-id", default="0", help="single worker id served by this process")
    p.add_argument("--south-host", default="127.0.0.1")
    p.add_argument("--south-port", type=int, default=8848)
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument(
        "--base-mode",
        default="delta",
        choices=list(EBENCH_BASE_SOURCES),
        help="must match the checkpoint's dataloader.base_action_source",
    )
    p.add_argument("--no-send-state", action="store_true", help="for non-proprio checkpoints")
    p.add_argument("--save-process", action="store_true", help="client-side per-episode video/log dump")
    p.add_argument("--client-reinit-retries", type=int, default=3)
    p.add_argument("--client-reinit-backoff", type=float, default=5.0)
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    run_worker(build_parser().parse_args())


if __name__ == "__main__":
    main()
