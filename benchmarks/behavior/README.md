# BEHAVIOR-1K (OmniGibson) Benchmark Evaluation

Closed-loop eval bridge for the BEHAVIOR-1K 2025 Challenge (robot **R1Pro**),
mirroring `benchmarks/robotwin/` but for OmniGibson's eval driver.

OmniGibson's challenge eval driver speaks the **openpi** websocket protocol
(msgpack-numpy); the OpenWAM policy server speaks **JSON-over-WebSocket** (port
8848). The model trains in the unified 80-D space, but the server's
`_UnifyAwareNormalizer` (PR #17) gathers its output back to the reader's **RAW-27**
layout and unnormalizes there — so the wire carries RAW-27. This directory is the
**bridge** between them — the OpenWAM server and checkpoint run **unchanged**.

```
OmniGibson eval.py ──(openpi msgpack-numpy)──▶  bridge  ──(OpenWAM JSON-WS)──▶  OpenWAM server (8848)
  policy=websocket                          (this dir)      WSPolicyClient        model, normalizer
```

Like the RoboTwin client, the bridge needs only `numpy`, `Pillow`, `websockets`,
`msgpack` — it never imports `openwam` or `openpi`, so it runs inside the
OmniGibson conda env.

## Files

| File | Description |
|---|---|
| `openwam2behavior_bridge.py` | The bridge: openpi-protocol north server → OpenWAM 8848 south client. Entry point. |
| `msgpack_numpy.py` | openpi-byte-compatible msgpack+numpy codec (hand-rolled; no openpi dep). |
| `configs/r1pro.yaml` | R1Pro controller config to ship with the submission (IK `absolute_pose` arms). |
| `run_bridge.sh` | Launch the bridge. |

Pure-numpy conversions live in `benchmarks/utils/action_conversion.py`
(`raw27_to_r1pro_action`, `r1pro_proprio_to_raw27`, `rot6d_to_axis_angle`);
offline tests in `tests/benchmarks/test_behavior_bridge.py`.

## Action space

The OpenWAM checkpoint predicts **end-effector poses** (rot6d), so both arms use
an `InverseKinematicsController` in `absolute_pose` mode and OmniGibson runs the
IK. The server returns the **RAW-27** action `[L_pos3, L_rot6d6, L_grip1, R_pos3,
R_rot6d6, R_grip1, base3, trunk4]`; the bridge maps it to the executed R1Pro
**21-D** vector, in the robot's `_raw_controller_order` (grippers interleaved):

```
[ base(3), trunk(4), arm_left(6: xyz+axisangle), gripper_left(1),
  arm_right(6: xyz+axisangle), gripper_right(1) ]
```

| Channel | Source (RAW-27, denormalized) | Transform |
|---|---|---|
| `base` | `[20:23]` `[vx,vy,vyaw]` | pass-through, clip `[-1,1]` |
| `trunk` | `[23:27]` 4 torso joints | pass-through, clip `[-1,1]` |
| `arm_left` | `[0:3]` xyz + `[3:9]` rot6d | xyz (metric) + rot6d→**axis-angle** (base frame) |
| `gripper_left` | `[9]` | pass-through, clip `[-1,1]` |
| `arm_right` | `[10:13]` xyz + `[13:19]` rot6d | xyz + rot6d→axis-angle |
| `gripper_right` | `[19]` | pass-through, clip `[-1,1]` |

Only the arms change representation (the model outputs EEF, not joints).
`base`/`trunk`/`gripper` are the model's own native recorded commands, fed to the
demo controllers unchanged (those controllers keep `command_input_limits:
default`). The proprio (`state`) sent south is the **RAW-27** proprio assembled
from the R1Pro 256-D `robot_r1::proprio` (EEF pose + base/trunk/gripper); the
server's `_UnifyAwareNormalizer` normalizes it and scatters it into the unified
space the model wants.

## Protocol (verified against the challenge `network_utils.py`)

- **metadata**: on connect the bridge sends one msgpack frame `{}` (the client
  blocks on it in its constructor).
- **act**: client sends the obs dict → bridge replies with exactly one msgpack
  frame `{"action": (21,) float}`. (The official openpi server also attaches a
  `"server_timing"` field; this bridge omits it, and the official client ignores
  it when absent.)
- **reset**: client sends `{"reset": True}` **fire-and-forget** (no `recv`) → the
  bridge resets south state and sends **nothing** back.
- **error**: bridge sends a TEXT frame (traceback) then closes with code 1011.
- **health**: HTTP `GET /healthz` → `200 OK`.

Wire obs keys (literal): `robot_r1::robot_r1:zed_link:Camera:0::rgb` (head),
`...left_realsense_link...` / `...right_realsense_link...` (wrists, HWC uint8),
`robot_r1::proprio` (256-D), `task_id` (int64). A natural-language `prompt` is on
the wire only if `eval.py`'s `cfg.prompt` is set; otherwise the bridge synthesizes
the instruction from `task_id` (see below).

## Setup

### 1. Install OmniGibson + the challenge eval code

Follow the [BEHAVIOR-1K / OmniGibson](https://behavior.stanford.edu/) install
(Isaac Sim + OmniGibson + the `omnigibson/learning/` challenge module). The
bridge runs in that env.

### 2. Ship the controller config

Copy `configs/r1pro.yaml` over OmniGibson's
`omnigibson/learning/configs/robot/r1pro.yaml` (the rules permit — and require —
shipping the robot controller config with the submission). It switches the arms
to IK `absolute_pose`; base/trunk/grippers keep the demo controllers.

### 3. Generate the task_id → instruction map (once)

The OpenWAM checkpoint is language-conditioned, but the wire obs carries only
`task_id`. Generate the mapping from the installed OmniGibson and let the bridge
de-underscore the activity name into the prompt:

```bash
python -c "from omnigibson.learning.utils.eval_utils import TASK_INDICES_TO_NAMES; \
  import json; json.dump({int(k): v for k, v in TASK_INDICES_TO_NAMES.items()}, \
  open('task_names.json', 'w'), indent=2)"
```

### 4. Start the OpenWAM server (south)

On a GPU box (can be remote):

```bash
bash scripts/deploy.sh --ckpt-dir /path/to/behavior_ckpt --port 8848
```

### 5. Start the bridge (north)

```bash
BRIDGE_PYTHON=$(which python) bash benchmarks/behavior/run_bridge.sh \
    --port 8000 --south-host 127.0.0.1 --south-port 8848 \
    --task-names task_names.json
```

### 6. Run the OmniGibson eval

```bash
python omnigibson/learning/eval.py policy=websocket \
    task.name=turning_on_radio \
    websockets_host=127.0.0.1 websockets_port=8000
```

## ⚠️ Validate on a sim box before scoring

OmniGibson/Isaac Sim is **not installed** on the dev box, so the closed loop is
unvalidated. The pure-numpy conversions, the msgpack codec, and the bridge
dispatch (metadata-first / act-reply / reset-no-reply / error framing) are
covered by `tests/benchmarks/test_behavior_bridge.py`. Confirm these on a
sim-capable box, in order of risk:

1. **Proprio offsets.** `r1pro_proprio_to_raw27` (`action_conversion.py`) decodes
   ALL proprio channels from the robot's `proprio_obs` layout (eef pose, arm/trunk
   qpos, 2-finger gripper qpos → open-scale, `base_qvel`), verified numerically on
   the dataset's `observation.state` and byte-checked against the trainer's
   `_state_to_raw_proprio_eef` by `test_behavior_bridge`. Confirm the offsets still
   hold against the live `robot_r1::proprio` on a sim box (a re-upload could shift
   the packing), or run `--no-send-state` to A/B.
2. **Base velocity frame.** The proprio base is `base_qvel` (world-frame
   `d(base_qpos)/dt`) rotated into the robot's LOCAL/base frame by `-yaw` and scaled
   by `1/[0.75,0.75,1.0]` (the controller output limits) — so it lands in the SAME
   frame and scale as the local-frame base action command, and both share the pooled
   normalization stats. Confirm the live `robot_r1::proprio` base velocity is
   world-frame and the base yaw offset is correct on a sim box.
3. **Prompt text.** The model was trained on the dataset's `tasks[0]` strings;
   the bridge uses the de-underscored activity name. If training used a different
   phrasing, adjust the mapping (or pass `--default-prompt`).
4. **Action-vector width.** With IK arms the executed vector is 21-D;
   `apply_action` asserts `len(action) == sum(controller.command_dim)`. If you
   keep JointController arms instead, the model's EEF output would need a
   different conversion (joint-space) — keep arms on IK.

## Offline test status

```bash
make test  # includes tests/benchmarks/test_behavior_bridge.py  → 24 passed
```
