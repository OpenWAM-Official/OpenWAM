# RoboCasa365 evaluation

This client connects the RoboCasa simulator to an OpenWAM server trained on the
canonical RoboCasa365 native-action LeRobot v3 conversion.

## Workflow inventory

| Stage | Files |
|---|---|
| Reproducible isolated client environment | `environment.yml`, `setup_env.sh` |
| Simulator/client preflight | `run_smoke.sh`, `smoke_robocasa365.py` |
| Policy-client configuration | `policy_config.yml` |
| OpenWAM WebSocket/action adapter | `openwam2robocasa365_interface.py` |
| One-task client/evaluator | `single_eval.sh`, `single_eval.py` |
| Official target-list evaluation + CSV | `multi_eval.sh`, `target_tasks.txt` |
| Managed server → client → evaluation | `run_eval.sh` |

The simulator and thin client run in their own RoboCasa365 environment; the
model server runs in the normal OpenWAM environment. This separation prevents
the simulator's pinned NumPy/MuJoCo stack from changing the model runtime.

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

## 1. Create and verify the client environment

The setup pins RoboCasa `1.0.1` and the compatible robosuite master revision,
creates the conda environment, installs both repositories, sets up macros, and
downloads the official kitchen assets (about 10 GB):

```bash
CONDA_BIN=/path/to/conda \
ROBOCASA365_ENV_PREFIX=/path/to/envs/robocasa365 \
ROBOCASA365_PATH=/path/to/robocasa \
ROBOSUITE_PATH=/path/to/robosuite \
  bash benchmarks/robocasa365/setup_env.sh
```

Set `ROBOCASA365_DOWNLOAD_ASSETS=0` only when assets are already available or
when intentionally preparing dependencies first. Then run the preflight from
weakest to strongest:

```bash
ROBOCASA365_PYTHON=/path/to/envs/robocasa365/bin/python \
  bash benchmarks/robocasa365/run_smoke.sh import
ROBOCASA365_PYTHON=/path/to/envs/robocasa365/bin/python \
  bash benchmarks/robocasa365/run_smoke.sh env OpenDrawer
```

## 2. Manual server and client startup

Start the server in the OpenWAM environment:

```bash
bash scripts/deploy.sh \
  --ckpt-dir /path/to/robocasa365_checkpoint \
  --ckpt-name checkpoint_step_10000.safetensors \
  --device cuda:0 --port 8848
```

Optionally verify the client/server wire path without creating a simulator:

```bash
ROBOCASA365_PYTHON=/path/to/envs/robocasa365/bin/python \
  bash benchmarks/robocasa365/run_smoke.sh roundtrip
```

Then run one task:

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

## 3. Managed end-to-end evaluation

`run_eval.sh` owns the complete lifecycle for one policy server: it starts the
server, waits for its TCP endpoint, launches the isolated client, evaluates one
task or the official target list, and stops the server on success, failure, or
interrupt.

One task:

```bash
SERVER_PYTHON=/path/to/openwam/bin/python \
ROBOCASA365_PYTHON=/path/to/envs/robocasa365/bin/python \
  bash benchmarks/robocasa365/run_eval.sh \
    /path/to/checkpoint checkpoint_step_10000.safetensors OpenDrawer
```

All official target tasks:

```bash
SERVER_PYTHON=/path/to/openwam/bin/python \
ROBOCASA365_PYTHON=/path/to/envs/robocasa365/bin/python \
OUTPUT_DIR=outputs/robocasa365/full \
  bash benchmarks/robocasa365/run_eval.sh \
    /path/to/checkpoint checkpoint_step_10000.safetensors target
```

Useful overrides are `SERVER_DEVICE`, `HOST`, `PORT`, `SPLIT`,
`SERVER_START_TIMEOUT`, `OUTPUT_DIR`, and `ROBOCASA365_POLICY_CONFIG`. Server
and client logs are preserved under `OUTPUT_DIR`; multi-task evaluation also
writes `tasks/summary_<split>.csv`.

The rollout horizon is read directly from RoboCasa's official
`robocasa.utils.dataset_registry_utils.get_task_horizon(task)` registry. Leave
`max_steps_override: null` for benchmark evaluation; set it only to deliberately
shorten a smoke/debug run. A successful `info["success"]` terminates the episode,
matching the official RoboCasa evaluators.

## Training data

The default dataloader config reads the two independent converted repos:

```text
/path/to/benchmark_data/robocasa365/robocasa365-pretrain-atomic
/path/to/benchmark_data/robocasa365/robocasa365-pretrain-composite
```

Both have identical LeRobot v3 schemas. The shared statistics file is
`/path/to/benchmark_data/robocasa365/robocasa365_multitask_compact_stats.npy`.
