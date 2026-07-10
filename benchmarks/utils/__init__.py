"""Client-side toolkit for talking to an OpenWAM policy server.

You do **not** need to know anything about the server — its model,
preprocessing, multi-view composition, or checkpoint. Just speak the WebSocket
protocol: send raw camera frames + the prompt the model should see, get an
action back already in physical units. The server does the rest — except prompt
formatting: it forwards ``prompt`` verbatim, so wrap it in your benchmark's
template first (RoboTwin does this in ``benchmarks/robotwin/prompt_template.py``).

Quick start::

    from benchmarks.utils import WSPolicyClient, build_payload, encode_numpy_b64, ServerError

    with WSPolicyClient("ws://<host>:8848") as client:
        client.ping()                       # verify the server is up (raises if unreachable)
        client.reset()                      # call once at the start of every episode
        for rgb_frame, proprio in robot_loop():          # rgb_frame: HxWx3 RGB uint8
            payload = build_payload(
                head=encode_numpy_b64(rgb_frame),        # head_camera is REQUIRED
                # left_wrist=..., right_wrist=...,        # optional; server black-fills if absent
                prompt="pick up the bottle",             # sent verbatim; wrap it in your template first
                state=list(proprio),                     # optional; required for proprio checkpoints
            )
            try:
                action = client.predict(payload)["action"]   # physical units, feed to controller
            except ServerError as e:
                ...                                       # e.status / e.code / e.message

Public surface — the only names you need:

    WSPolicyClient        one persistent WS connection: ``predict`` / ``reset`` / ``ping``
    build_payload         assemble the obs message
    encode_numpy_b64      HxWx3 RGB uint8 array  -> base64 JPEG
    encode_path_b64       JPEG/PNG file path     -> base64
    ServerError           structured server error (``.status`` / ``.code`` / ``.message``)

Minimal deps: ``numpy``, ``Pillow``, ``websockets`` (``opencv-python`` only if you
decode camera frames yourself). Full step-by-step guide: ``benchmarks/README.md``.
The wire contract (message types) lives in ``benchmarks/utils/transport.py``
(client mirror; the server-side source of truth is ``openwam/deploy/server.py``).
"""

from benchmarks.utils.action_conversion import (
    base_velocity_body,
    eef20d_to_ee16d,
    eef20d_to_robocasa12d,
    quat_xyzw_to_axis_angle,
    quat_xyzw_to_rot6d,
    r1pro_proprio_to_raw27,
    raw27_to_r1pro_action,
    robocasa_state_to_eef20d,
    robotwin_endpose_to_eef20d,
    rot6d_to_axis_angle,
    rot6d_to_quat_xyzw,
)
from benchmarks.utils.client import (
    ServerError,
    build_payload,
    encode_numpy_b64,
    encode_path_b64,
    server_error_from_body,
)
from benchmarks.utils.transport import WSPolicyClient

__all__ = [
    "ServerError",
    "WSPolicyClient",
    "base_velocity_body",
    "build_payload",
    "eef20d_to_ee16d",
    "eef20d_to_robocasa12d",
    "encode_numpy_b64",
    "encode_path_b64",
    "quat_xyzw_to_axis_angle",
    "quat_xyzw_to_rot6d",
    "r1pro_proprio_to_raw27",
    "raw27_to_r1pro_action",
    "robocasa_state_to_eef20d",
    "robotwin_endpose_to_eef20d",
    "rot6d_to_axis_angle",
    "rot6d_to_quat_xyzw",
    "server_error_from_body",
]
