"""BEHAVIOR-1K (OmniGibson) eval bridge for the OpenWAM Policy Server.

OmniGibson's challenge eval driver (``omnigibson/learning/eval.py policy=websocket``)
drives an **openpi** ``WebsocketClientPolicy`` (msgpack-numpy over WebSocket). The
OpenWAM policy server, by contrast, speaks a JSON-over-WebSocket protocol (port
8848). The model trains in the unified 80-D space, but the server's
``_UnifyAwareNormalizer`` (PR #17) gathers the 80-D output back to the reader's
RAW-27 layout and unnormalizes there — so it RETURNS and EXPECTS RAW-27. This
module is the **bridge** between the two:

    OmniGibson ──(openpi msgpack-numpy)──▶  THIS bridge  ──(OpenWAM JSON-WS)──▶  OpenWAM 8848 server
       eval.py                              (north server)     WSPolicyClient        (model, unchanged)

Per step the bridge:

  1. receives the openpi obs (R1Pro camera RGBs + 256-D proprio + ``task_id``),
  2. forwards cameras + a synthesized prompt + the RAW-27 proprio to the OpenWAM
     server (reusing ``benchmarks.utils`` exactly like the RoboTwin adapter does),
  3. converts the server's denormalized RAW-27 action into the R1Pro 21-D
     controller vector (IK ``absolute_pose`` arms), and
  4. returns ``{"action": (21,)}`` — the single ready-to-execute native action.

Wire contract with the OmniGibson client (verified against the challenge
``network_utils.py``):

  * metadata: the server sends ONE msgpack frame (``{}``) on connect, before the
    loop — the client blocks on it in ``__init__``.
  * ``act``: client sends the obs dict → server replies with exactly one msgpack
    frame ``{"action": ndarray}``. (The official openpi server also attaches a
    ``"server_timing"`` field; this bridge omits it, and the official client
    ignores it when absent.)
  * ``reset``: client sends ``{"reset": True}`` **fire-and-forget** (no recv) →
    server resets south state and sends NOTHING back (preserves 1-send-1-recv on
    ``act`` only).
  * error: server sends a TEXT frame (traceback) then closes with code 1011.
  * health: HTTP ``GET /healthz`` → ``200 OK`` (no upgrade).

Heavy deps live only on the OpenWAM server. This bridge needs ``numpy``,
``Pillow``, ``websockets``, ``msgpack`` — it never imports ``openwam`` or ``openpi``.
"""

# benchmarks.utils lives one level up; mirror the RoboTwin adapter and put the
# project root on sys.path so the shared client/transport/conversions import
# from this process too.
import os as _os
import sys as _sys

_PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)

import argparse  # noqa: E402
import http  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402

from benchmarks.behavior import msgpack_numpy  # noqa: E402
from benchmarks.utils import (  # noqa: E402
    WSPolicyClient,
    build_payload,
    encode_numpy_b64,
    r1pro_proprio_to_raw27,
    raw27_to_r1pro_action,
    resize_for_lshape_slot,
    transport,  # noqa: E402
)

logger = logging.getLogger("behavior_bridge")

# Literal R1Pro RGB obs keys as they arrive on the wire (post double-colon
# flatten in eval.py); verified against learning/policies.py::IMAGE_KEYS.
HEAD_KEY = "robot_r1::robot_r1:zed_link:Camera:0::rgb"
LEFT_WRIST_KEY = "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"
RIGHT_WRIST_KEY = "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"
PROPRIO_KEY = "robot_r1::proprio"  # 256-D
TASK_ID_KEY = "task_id"  # int64 tensor [idx]


class BehaviorBridge:
    """Stateful obs→action bridge backed by a running OpenWAM policy server.

    The OpenWAM server (south) manages action chunking and replanning
    internally, so the bridge sends one obs per step and returns the one action
    the server pops — matching the OmniGibson client's one-action-per-step
    contract.
    """

    def __init__(
        self,
        south_host: str = "127.0.0.1",
        south_port: int = 8848,
        *,
        send_state: bool = True,
        task_names: Optional[dict] = None,
        default_prompt: Optional[str] = None,
        request_timeout: int = 300,
        south_client: Optional[WSPolicyClient] = None,
    ) -> None:
        self._send_state = send_state
        self._task_names = {int(k): str(v) for k, v in (task_names or {}).items()}
        self._default_prompt = default_prompt
        self._ws_url = f"ws://{south_host}:{south_port}"
        # south_client injectable for unit tests (no live server needed).
        self._south = (
            south_client if south_client is not None else WSPolicyClient(self._ws_url, timeout=request_timeout)
        )
        self._step = 0
        self._last_prompt: Optional[str] = None

    # ----- south-server lifecycle ------------------------------------------

    def wait_until_healthy(self, timeout_s: int = 300, poll_interval: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        last_exc: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                if self._south.ping().get("type") == transport.PONG:
                    logger.info("OpenWAM server healthy at %s", self._ws_url)
                    return
            except Exception as exc:  # noqa: BLE001 — retry until the deadline
                last_exc = exc
                self._south.close()
            time.sleep(poll_interval)
        raise RuntimeError(f"OpenWAM server not healthy within {timeout_s}s at {self._ws_url}. Last error: {last_exc}")

    def reset(self) -> None:
        """Clear south-server episode state (called on a client ``{"reset": True}``)."""
        self._step = 0
        self._last_prompt = None
        result = self._south.reset()
        if result.get("type") != transport.RESET_ACK:
            raise RuntimeError(f"OpenWAM server reset failed: {result}")

    # ----- obs helpers ------------------------------------------------------

    def _resolve_prompt(self, obs: dict) -> str:
        """Synthesize the language instruction the south model was trained on.

        Priority: an explicit ``prompt`` on the wire (set only if eval.py's
        ``cfg.prompt`` is non-null) → the ``task_id`` mapping (training-verbatim
        sentences from ``gen_task_prompts.py``; ``replace`` keeps a raw
        activity-name mapping usable) → ``default_prompt``. Fails fast otherwise
        so a language-conditioned checkpoint is never silently fed an empty prompt.
        """
        p = obs.get("prompt")
        if p is not None and not isinstance(p, np.ndarray):
            text = str(p).strip()
            if text:
                return text
        tid = obs.get(TASK_ID_KEY)
        idx: Optional[int] = None
        if tid is not None and self._task_names:
            idx = int(np.asarray(tid).reshape(-1)[0])
            name = self._task_names.get(idx)
            if name:
                return name.replace("_", " ").strip()
        if self._default_prompt:
            if idx is not None:
                # A map was provided but lacks this task — the default is a
                # DIFFERENT prompt than training. Loud, so a truncated
                # task_prompts.json can't silently degrade an eval.
                logger.warning(
                    "task_id %d missing from --task-names; falling back to --default-prompt %r",
                    idx,
                    self._default_prompt,
                )
            return self._default_prompt
        raise ValueError(
            "could not resolve a task prompt: obs has no 'prompt' and no usable 'task_id' mapping. "
            "Pass --task-names <json> (task_id→training prompt, generated by "
            "benchmarks/behavior/gen_task_prompts.py) or --default-prompt."
        )

    def _extract_camera(self, obs: dict, key: str, *, required: bool) -> Optional[np.ndarray]:
        img = obs.get(key)
        if img is None:
            if required:
                raise KeyError(
                    f"head camera {key!r} missing from obs. Available keys: {sorted(obs.keys())}. "
                    "Check the R1Pro camera key constants against the eval build."
                )
            return None
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError(f"camera {key!r} expected HxWx3 RGB, got shape {arr.shape}")
        return np.ascontiguousarray(arr[..., :3], dtype=np.uint8)

    def _build_state(self, obs: dict) -> Optional[list]:
        if not self._send_state:
            return None
        proprio = obs.get(PROPRIO_KEY)
        if proprio is None:
            raise KeyError(
                f"send_state=True but obs has no {PROPRIO_KEY!r}. The BEHAVIOR checkpoint is "
                "proprio-conditioned; pass --no-send-state only for a non-proprio checkpoint."
            )
        # Send RAW-27 proprio; the server's _UnifyAwareNormalizer normalizes it and
        # scatters it into the unified space the model wants (PR #17).
        raw = r1pro_proprio_to_raw27(np.asarray(proprio, dtype=np.float32))
        return [float(v) for v in raw]

    # ----- per-step inference ----------------------------------------------

    def infer(self, obs: dict) -> dict:
        """Run one obs through the south server and return ``{"action": (21,)}``."""
        head = self._extract_camera(obs, HEAD_KEY, required=True)
        left = self._extract_camera(obs, LEFT_WRIST_KEY, required=False)
        right = self._extract_camera(obs, RIGHT_WRIST_KEY, required=False)
        prompt = self._resolve_prompt(obs)
        if prompt != self._last_prompt:
            # One line per episode/task switch (not per 30 Hz step) so a sim-box
            # run can confirm the model gets training-verbatim sentences.
            logger.info("resolved prompt: %r", prompt)
            self._last_prompt = prompt
        state = self._build_state(obs)

        payload = build_payload(
            head=encode_numpy_b64(resize_for_lshape_slot(head, "head_camera")),
            left_wrist=(
                encode_numpy_b64(resize_for_lshape_slot(left, "left_wrist_camera"))
                if left is not None
                else None
            ),
            right_wrist=(
                encode_numpy_b64(resize_for_lshape_slot(right, "right_wrist_camera"))
                if right is not None
                else None
            ),
            prompt=prompt,
            state=state,
        )
        response = self._south.predict(payload)
        # Server returns the RAW-27 physical action (the _UnifyAwareNormalizer
        # gathered the model's 80-D output back to raw before unnormalizing).
        action_raw = np.asarray(response["action"], dtype=np.float32)
        action21 = raw27_to_r1pro_action(action_raw)
        self._step += 1
        return {"action": action21}

    def handle_message(self, obs: dict) -> Optional[dict]:
        """Dispatch one decoded client frame.

        Returns the reply dict for an ``act`` frame, or ``None`` for a ``reset``
        frame (which must NOT be answered — the client does not ``recv`` after it).
        """
        if "reset" in obs:
            self.reset()
            return None
        return self.infer(obs)


# ---------------------------------------------------------------------------
# WebSocket server (openpi wire-compatible) — websockets.sync
# ---------------------------------------------------------------------------

# Metadata frame sent on connect. The client only stores it (no schema); an
# empty dict is valid and sufficient.
SERVER_METADATA: dict = {}


def _health_check(connection, request):
    """HTTP ``GET /healthz`` → 200, without upgrading to a websocket."""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _make_handler(bridge: BehaviorBridge):
    from websockets.frames import CloseCode

    def handler(ws):
        packer = msgpack_numpy.Packer()
        # 1) Metadata FIRST — the client blocks on this in its constructor.
        ws.send(packer.pack(SERVER_METADATA))
        while True:
            try:
                raw = ws.recv()
            except Exception:  # noqa: BLE001 — ConnectionClosed on client disconnect: exit cleanly
                return
            try:
                obs = msgpack_numpy.unpackb(raw, strict_map_key=False)
                reply = bridge.handle_message(obs)
                if reply is not None:  # reset path sends nothing
                    ws.send(packer.pack(reply))
            except Exception:  # noqa: BLE001 — signal failure openpi-style: text frame + close 1011
                ws.send(traceback.format_exc())
                ws.close(code=CloseCode.INTERNAL_ERROR, reason="Internal server error; traceback in previous frame.")
                return

    return handler


def serve_forever(bridge: BehaviorBridge, host: str, port: int) -> None:
    from websockets.sync.server import serve

    bridge.wait_until_healthy()
    with serve(
        _make_handler(bridge),
        host,
        port,
        compression=None,
        max_size=None,
        process_request=_health_check,
    ) as server:
        logger.info("BEHAVIOR bridge listening on ws://%s:%d → OpenWAM %s", host, port, bridge._ws_url)
        server.serve_forever()


def _load_task_names(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path) as f:
        data = json.load(f)
    mapping = {int(k): str(v) for k, v in data.items()}
    underscored = sorted(k for k, v in mapping.items() if "_" in v)
    if underscored:
        logger.warning(
            "--task-names values for task_id(s) %s contain '_' and will be de-underscored before "
            "reaching the model. Legacy activity-name maps rely on this; training-verbatim prompts "
            "should not (regenerate with gen_task_prompts.py).",
            underscored,
        )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0", help="bind host for the north (OmniGibson-facing) server")
    parser.add_argument("--port", type=int, default=8000, help="bind port for the north server")
    parser.add_argument("--south-host", default="127.0.0.1", help="OpenWAM policy server host")
    parser.add_argument("--south-port", type=int, default=8848, help="OpenWAM policy server port")
    parser.add_argument(
        "--task-names",
        default=None,
        help="JSON file mapping task_id→prompt. Use gen_task_prompts.py so the values are the "
        "training-verbatim dataset sentences; a raw activity-name mapping also works "
        "(de-underscored) but is off the training text distribution.",
    )
    parser.add_argument("--default-prompt", default=None, help="fallback prompt when task_id is unmapped")
    parser.add_argument(
        "--no-send-state", action="store_true", help="do not send proprio (non-proprio checkpoints only)"
    )
    parser.add_argument("--request-timeout", type=int, default=300, help="south-server round-trip timeout (s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    bridge = BehaviorBridge(
        south_host=args.south_host,
        south_port=args.south_port,
        send_state=not args.no_send_state,
        task_names=_load_task_names(args.task_names),
        default_prompt=args.default_prompt,
        request_timeout=args.request_timeout,
    )
    serve_forever(bridge, args.host, args.port)


if __name__ == "__main__":
    main()
