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
    encode_numpy_b64      HxWx3 RGB uint8 array  -> lossless base64 PNG
    resize_for_lshape_slot  reproduce a training reader's LANCZOS tile resize
    encode_path_b64       JPEG/PNG file path     -> base64
    ServerError           structured server error (``.status`` / ``.code`` / ``.message``)

Minimal deps: ``numpy``, ``Pillow``, ``websockets`` (``opencv-python`` only if you
decode camera frames yourself). Full step-by-step guide: ``benchmarks/README.md``.
The wire contract (message types) lives in ``benchmarks/utils/transport.py``
(client mirror; the server-side source of truth is ``openwam/deploy/server.py``).
"""

from benchmarks.utils.action_conversion import (
    base_pose_planar5,
    base_velocity_body,
    base_velocity_cmd,
    binarize_robocasa_action12,
    ebench_obs_to_raw23,
    ebench_quat_wxyz_to_rot6d,
    ebench_render_state_base,
    ebench_wrap_angle_rad,
    eef10_to_robocasa12d,
    eef10_to_vlabench_ee,
    eef20d_to_ee16d,
    eef20d_to_robocasa12d,
    libero_gripper_qpos_to_cmd,
    libero_obs_to_eef10,
    libero_open_scale_to_gripper_cmd,
    quat_xyzw_to_axis_angle,
    quat_xyzw_to_rot6d,
    raw23_to_ebench_action,
    robocasa_state_to_eef10,
    robocasa_state_to_eef20d,
    robotwin_endpose_to_eef20d,
    rot6d_to_axis_angle,
    rot6d_to_euler_xyz,
    rot6d_to_quat_xyzw,
    vlabench_obs_to_eef10,
)
from benchmarks.utils.client import (
    ServerError,
    build_payload,
    encode_numpy_b64,
    encode_path_b64,
    resize_for_lshape_slot,
    server_error_from_body,
)
from benchmarks.utils.transport import WSPolicyClient

__all__ = [
    "ServerError",
    "WSPolicyClient",
    "base_velocity_body",
    "base_pose_planar5",
    "base_velocity_cmd",
    "binarize_robocasa_action12",
    "build_payload",
    "ebench_obs_to_raw23",
    "ebench_quat_wxyz_to_rot6d",
    "ebench_render_state_base",
    "ebench_wrap_angle_rad",
    "eef10_to_robocasa12d",
    "eef10_to_vlabench_ee",
    "eef20d_to_ee16d",
    "eef20d_to_robocasa12d",
    "encode_numpy_b64",
    "encode_path_b64",
    "resize_for_lshape_slot",
    "libero_gripper_qpos_to_cmd",
    "libero_obs_to_eef10",
    "libero_open_scale_to_gripper_cmd",
    "quat_xyzw_to_axis_angle",
    "quat_xyzw_to_rot6d",
    "raw23_to_ebench_action",
    "robocasa_state_to_eef10",
    "robocasa_state_to_eef20d",
    "robotwin_endpose_to_eef20d",
    "rot6d_to_axis_angle",
    "rot6d_to_euler_xyz",
    "rot6d_to_quat_xyzw",
    "server_error_from_body",
    "vlabench_obs_to_eef10",
]
