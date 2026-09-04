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
    resize_for_lshape_slot,
)
from benchmarks.utils.action_conversion import (  # noqa: E402
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
        send_state: bool = True,
    ):
        self._south = south
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

    def invalidate_episode(self) -> None:
        """Force on_episode_start at the next act() — used after a north
        reconnect, whose reset() started a fresh sim episode."""
        self._episode_active = False

    def act(self, inner_obs: dict) -> dict:
        """One sim step: obs dict in → EBench action dict out."""
        if inner_obs.get("reset", False) or not self._episode_active:
            self.on_episode_start(inner_obs)

        head = inner_obs.get(HEAD_KEY)
        if head is None:
            raise KeyError(f"EBench obs missing required camera {HEAD_KEY!r}")
        left = inner_obs.get(LEFT_WRIST_KEY)
        right = inner_obs.get(RIGHT_WRIST_KEY)
        for name, img in ((HEAD_KEY, head), (LEFT_WRIST_KEY, left), (RIGHT_WRIST_KEY, right)):
            if img is None:
                continue
            arr = np.asarray(img)
            if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
                # Fail fast with the camera name: a float/RGBA/BGR frame would
                # either crash deep inside Pillow or silently swap channels.
                raise ValueError(f"camera {name!r} must be HxWx3 uint8 RGB, got dtype={arr.dtype} shape={arr.shape}")

        state_list = None
        cur_base = None
        if self._send_state:
            cur_base = np.asarray(inner_obs[STATE_BASE_KEY], dtype=np.float64).reshape(3)
            rendered = ebench_render_state_base(cur_base, self._prev_base)
            raw23 = ebench_obs_to_raw23(inner_obs[STATE_EE_KEY], inner_obs[STATE_GRIPPER_KEY], rendered)
            state_list = [float(v) for v in raw23]

        payload = build_payload(
            head=encode_numpy_b64(resize_for_lshape_slot(np.asarray(head), "head_camera")),
            left_wrist=(
                encode_numpy_b64(resize_for_lshape_slot(np.asarray(left), "left_wrist_camera"))
                if left is not None
                else None
            ),
            right_wrist=(
                encode_numpy_b64(resize_for_lshape_slot(np.asarray(right), "right_wrist_camera"))
                if right is not None
                else None
            ),
            prompt=self._prompt,
            state=state_list,
        )
        reply = self._south.predict(payload)
        action = np.asarray(reply["action"], dtype=np.float64).reshape(-1)
        if action.shape[0] != EBENCH_RAW_DIM:
            raise ValueError(
                f"OpenWAM server returned {action.shape[0]}-D action; expected raw "
                f"{EBENCH_RAW_DIM}-D — is the checkpoint an EBench (action_mode=eef, "
                "unify_action=true) checkpoint?"
            )
        if not np.isfinite(action).all():
            # NaN survives every hop (JSON emits NaN tokens; np.clip(nan)=nan;
            # a NaN quat passes norm checks because abs(nan-1)>tol is False) —
            # stop it here, before the sim consumes a NaN pose target.
            raise ValueError(f"OpenWAM server returned non-finite action: {action.tolist()}")

        out = raw23_to_ebench_action(action)
        for i, (_pos, quat, _grip) in enumerate(out["action"]):
            norm = float(np.linalg.norm(quat))
            if not abs(norm - 1.0) <= 0.05:
                # Degenerate rot6d (e.g. all-zeros from an early/diverged model:
                # rot6d stats are identity-pinned, so a zero normalized output
                # denormalizes to a zero rot6d) → non-unit quat → real cuRobo IK
                # silently holds the current joints. Fail loudly instead.
                raise ValueError(
                    f"arm {i} rot6d degenerated to a non-unit quaternion (norm={norm:.4f}); "
                    f"raw rot6d={action[3:9] if i == 0 else action[13:19]}"
                )

        # Update AFTER building this step's proprio: proprio(t) differences
        # state(t) against state(t-1), exactly like training rows.
        if cur_base is not None:
            self._prev_base = cur_base
        self.steps += 1
        return out


def run_worker(args) -> None:
    from genmanip_client import EvalClient  # imported here: only needed at runtime

    south = WSPolicyClient(f"ws://{args.south_host}:{args.south_port}", timeout=float(args.request_timeout))
    wait_until_healthy(south)
    driver = EBenchOpenWAMDriver(south, send_state=not args.no_send_state)

    def make_client():
        return EvalClient(
            args.url,
            worker_ids=[str(args.worker_id)],
            token=args.token or None,
            run_id=args.run_id,
            save_process=args.save_process,
            verbose=True,
        )

    def reconnect(last_err: Exception):
        """Rebuild the EvalClient with retries; return the fresh reset obs.

        The whole rebuild+reset lives INSIDE the retried scope: the most
        likely trigger is a sim-server hiccup/restart, so the first reinit is
        likely to fail too — it must consume a retry, not abort the run.
        """
        nonlocal client
        err = last_err
        for attempt in range(args.client_reinit_retries):
            logger.warning(
                "EvalClient.step failed (%s); rebuilding client (attempt %d/%d)",
                err,
                attempt + 1,
                args.client_reinit_retries,
            )
            time.sleep(args.client_reinit_backoff * (attempt + 1))
            try:
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
                client = make_client()
                return client.reset()
            except Exception as e:  # noqa: BLE001 — reinit itself failed; burn a retry
                err = e
        raise RuntimeError(f"EvalClient recovery failed after {args.client_reinit_retries} attempts") from err

    client = make_client()
    try:
        obs = client.reset()
        done = False
        while not done:
            actions = {}
            for wid, entry in obs.items():
                if (entry or {}).get("lock_lost"):
                    logger.warning("worker %s lost its episode lock; server will requeue", wid)
                inner = (entry or {}).get("obs")
                if inner is None:
                    continue  # worker finished or reset pending
                actions[wid] = driver.act(inner)
            if not actions:
                logger.warning(
                    "no actionable obs (workers finished or wedged) — leaving the loop; "
                    "verify the final metrics below before trusting this run"
                )
                break
            try:
                obs, done = client.step(actions)
            except Exception as e:  # noqa: BLE001 — north-side transport recovery
                # DISCARD the stale actions: client.reset() kills the workers
                # and starts a fresh episode, so replaying an action computed
                # for the old episode's obs would corrupt the new one. Resume
                # the outer loop from the fresh reset obs instead.
                obs = reconnect(e)
                driver.invalidate_episode()
                first = next(
                    ((v or {}).get("obs") for v in obs.values() if (v or {}).get("obs") is not None),
                    None,
                )
                if first is not None and not first.get("reset", False):
                    raise RuntimeError(
                        "post-reconnect obs is mid-episode (reset=False): the sim episode was "
                        "NOT restarted — aborting instead of silently splicing policy state"
                    )
                continue
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
    p.add_argument(
        "--config",
        default=None,
        help="optional YAML (e.g. benchmarks/ebench/policy_config.yml) supplying defaults; CLI flags win",
    )
    p.add_argument("--url", default="http://127.0.0.1:8087", help="GenManip eval server / online endpoint")
    p.add_argument("--token", default="", help="Bearer token (online evaluation)")
    p.add_argument("--run-id", default="", help="run_id (== online task_id)")
    p.add_argument("--worker-id", default="0", help="single worker id served by this process")
    p.add_argument("--south-host", default="127.0.0.1")
    p.add_argument("--south-port", type=int, default=8848)
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument(
        "--ckpt-config",
        default=None,
        help="path to the served checkpoint's config.yaml — when the ckpt dir is reachable from this "
        "host, pass it so base-mode/action-mode mismatches fail at startup instead of scoring garbage",
    )
    p.add_argument("--no-send-state", action="store_true", help="for non-proprio checkpoints")
    p.add_argument("--save-process", action="store_true", help="client-side per-episode video/log dump")
    p.add_argument("--client-reinit-retries", type=int, default=3)
    p.add_argument("--client-reinit-backoff", type=float, default=5.0)
    return p


def parse_args_with_config(argv=None) -> argparse.Namespace:
    """Two-phase parse: --config YAML supplies defaults, explicit CLI flags win."""
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        import yaml

        with open(pre.config) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in parser._actions}
        mapped = {}
        for key, value in cfg.items():
            if key == "send_state":
                mapped["no_send_state"] = not bool(value)
            elif key in known:
                mapped[key] = value
            else:
                raise ValueError(f"unknown key {key!r} in {pre.config}")
        parser.set_defaults(**mapped)
    return parser.parse_args(argv)


def verify_ckpt_config(args) -> None:
    """Hard-verify the bridge's action settings against the checkpoint config."""
    if not args.ckpt_config:
        logger.warning(
            "--ckpt-config not given: the bridge contract is UNVERIFIED against the checkpoint. "
            "A mismatch scores garbage silently — pass the ckpt's config.yaml when reachable."
        )
        return
    import yaml

    with open(args.ckpt_config) as f:
        cfg = yaml.safe_load(f)
    dl = cfg.get("dataloader", {}) or {}
    problems = []
    if dl.get("type") != "ebench" or dl.get("action_mode") != "eef":
        problems.append(f"dataloader type/action_mode is {dl.get('type')}/{dl.get('action_mode')}, not ebench/eef")
    if not dl.get("unify_action", False):
        problems.append("checkpoint was not trained with unify_action=true")
    if problems:
        raise ValueError("checkpoint/bridge contract mismatch: " + "; ".join(problems))
    logger.info("ckpt-config verified: action_mode=eef, unify_action=true")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    args = parse_args_with_config()
    verify_ckpt_config(args)
    run_worker(args)


if __name__ == "__main__":
    main()
