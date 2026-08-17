# RoboCasa365 evaluation

This client connects the RoboCasa simulator to an OpenWAM server trained on the
compact RoboCasa365 LeRobot v3 conversion.

## Representation contract

The simulator exposes native state16 and consumes native action12. The policy
uses the converted dataset's physical representation:

- state19: achieved EEF `xyz3 + rot6d6 + gripper1`, followed by world base
  `xyz3 + rot6d6`;
- action15: absolute EEF target `xyz3 + rot6d6 + gripper1`, followed by
  `base_vx + base_vy + base_vyaw + torso + control_mode`.

Training scatters state and action independently into the 80-D shared model
space:

```text
state19  [0:10] -> [0:10], [10:19] -> [68:77]
action15 [0:10] -> [0:10], [10:15] -> [68:73]
```

The server gathers and de-normalizes action15. The client converts its absolute
EEF target back to native OSC using the current achieved EEF observation:

```text
delta_xyz    = (target_xyz - current_xyz) / 0.05
delta_rotvec = Log(target_R @ current_R.T) / 0.5
```

This conversion never uses the previous commanded target, including when
`control_mode=+1`. The base velocity, torso, and control mode are passed through
to the native action. Gripper convention is `-1=closed, +1=open` on the policy
side and is flipped to RoboCasa's native close command at the bridge.

The scales `0.05 m` and `0.5 rad` are the `OSC_POSE.output_max` values recorded
in every source dataset. They are configured in `policy_config.yml`.

## Cameras and prompt

- `robot0_agentview_left` is sent as the head camera;
- `robot0_eye_in_hand` is sent as the left wrist camera;
- the unused right-wrist slot is black-filled;
- eval reproduces the unchanged training reader's LANCZOS slot resize
  (`320x256` head, `160x128` wrist) and transports both views as lossless PNG;
- the native task instruction is sent unchanged, with no prompt prefix or
  suffix, matching the training dataloader.

## Running

Start the OpenWAM server with a compact RoboCasa365 checkpoint, then run:

```bash
ROBOCASA365_PYTHON=/path/to/robocasa/env/bin/python \
  bash benchmarks/robocasa365/single_eval.sh OpenDrawer target 8848 127.0.0.1
```

For the official target list:

```bash
ROBOCASA365_PYTHON=/path/to/robocasa/env/bin/python \
  bash benchmarks/robocasa365/multi_eval.sh target
```

Smoke checks:

```bash
ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/run_smoke.sh import
ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/run_smoke.sh env OpenDrawer
ROBOCASA365_PYTHON=... bash benchmarks/robocasa365/run_smoke.sh roundtrip
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
/path/to/robocasa365_openwam_v3/robocasa365-pretrain-atomic
/path/to/robocasa365_openwam_v3/robocasa365-pretrain-composite
```

Both have identical LeRobot v3 schemas. The shared statistics file is
`/path/to/robocasa365_openwam_v3/robocasa365_multitask_compact_stats.npy`.
