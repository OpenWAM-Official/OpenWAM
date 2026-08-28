# RoboCasa365 native-delta evaluation

This client connects the RoboCasa simulator to an OpenWAM server trained on the
independent RoboCasa365 native-delta LeRobot v3 conversion.

## Representation contract

The simulator exposes native state16 and consumes native action12. The policy
uses the converted dataset's compact representation:

- state19: achieved EEF `xyz3 + rot6d6 + gripper1`, followed by world base
  `xyz3 + rot6d6`;
- action15: native normalized EEF delta
  `xyz3 + rot6d(Exp(delta_rotvec3)) + gripper1`, followed by
  `base_vx + base_vy + base_vyaw + torso + control_mode`.

Training scatters state and action independently into the 80-D shared model
space:

```text
state19  [0:10] -> [0:10], [10:19] -> [68:77]
action15 [0:10] -> [0:10], [10:15] -> [68:73]
```

The server gathers and de-normalizes action15. The client directly reconstructs
the native OSC command:

```text
native_delta_xyz    = action15[0:3]
native_delta_rotvec = Log(rot6d_to_matrix(action15[3:9]))
native_gripper      = -action15[9]
```

No current state, previous target, `0.05` position scale, or `0.5` rotation
scale participates in the action bridge. Base velocity, torso, and control mode
pass through. Gripper convention is `-1=closed, +1=open` on the policy side and
is sign-inverted back to RoboCasa's native close command.

## Cameras and prompt

- `robot0_agentview_left` is sent as the head camera;
- `robot0_eye_in_hand` is sent as the left wrist camera;
- `robot0_agentview_right` is sent through the fixed `right_wrist_camera`
  transport field and fills the bottom-right slot;
- eval reproduces the unchanged training reader's LANCZOS slot resize
  (`320x256` head, `160x128` bottom views) and transports all three views as
  lossless PNG;
- the native task instruction is sent unchanged, with no prompt prefix or
  suffix, matching the training dataloader.

## Running

Start the server with a `robocasa365_delta_action` checkpoint, then run:

```bash
ROBOCASA365_DELTA_ACTION_PYTHON=/path/to/robocasa/env/bin/python \
  bash benchmarks/robocasa365_delta_action/single_eval.sh OpenDrawer target 8848 127.0.0.1
```

For the official target list:

```bash
ROBOCASA365_DELTA_ACTION_PYTHON=/path/to/robocasa/env/bin/python \
  bash benchmarks/robocasa365_delta_action/multi_eval.sh target
```

Smoke checks:

```bash
ROBOCASA365_DELTA_ACTION_PYTHON=... bash benchmarks/robocasa365_delta_action/run_smoke.sh import
ROBOCASA365_DELTA_ACTION_PYTHON=... bash benchmarks/robocasa365_delta_action/run_smoke.sh env OpenDrawer
ROBOCASA365_DELTA_ACTION_PYTHON=... bash benchmarks/robocasa365_delta_action/run_smoke.sh roundtrip
```

Set `debug: true` in `policy_config.yml` to write per-step camera, state19, and
native action12 inspection bundles.

The rollout horizon is read directly from RoboCasa's official
`robocasa.utils.dataset_registry_utils.get_task_horizon(task)` registry. Leave
`max_steps_override: null` for benchmark evaluation; set it only to deliberately
shorten a smoke/debug run. A successful `info["success"]` terminates the episode,
matching the official RoboCasa evaluators.

## Training data

The default dataloader config reads the two independent converted repos:

```text
/path/to/robocasa365_delta_action_v3/robocasa365-pretrain-atomic
/path/to/robocasa365_delta_action_v3/robocasa365-pretrain-composite
```

Both have identical LeRobot v3 schemas. The shared statistics file is
`/path/to/robocasa365_delta_action_v3/robocasa365_delta_action_multitask_compact_stats.npy`.
