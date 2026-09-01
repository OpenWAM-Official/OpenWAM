# LIBERO evaluation

This directory contains the complete OpenWAM evaluation path for
LIBERO: a reproducible client environment, simulator preflight,
policy-client configuration, manual single-task evaluation, and managed
multi-GPU evaluation that starts and stops its own OpenWAM servers.

## Workflow inventory

| Stage | Files |
|---|---|
| Environment lock | `environment.yml` |
| Install repo, patch, dependencies, assets | `setup_env.sh` |
| Import/task/simulator preflight | `run_smoke.sh` + `smoke_libero.py` |
| Policy client config | `policy_config.yml` |
| OpenWAM WebSocket adapter | `openwam2libero_interface.py` |
| One-task client/evaluator | `single_eval.sh` + `single_eval.py` |
| Managed server + client queue + summary | `run_eval.sh` → `run_all_suites.py` |
| Scheduling/resume implementation | `scheduler.py` |

The compatibility change required by the pinned upstream checkout lives under
`patches/`; the setup script applies it idempotently and rejects an unexpected
checkout instead of silently evaluating different code.

## Representation contract

The OpenWAM server returns raw 10-D native actions:

```text
[native_delta_xyz3, rot6d(Exp(native_delta_axis_angle3)), gripper_open_command]
```

The canonical reader uses separate action/state normalization blocks and keeps
the six rot6d dimensions as an identity mapping. The client sends achieved
EEF10 proprioception and converts the response to LIBERO's runtime 7-D OSC
command:

```text
[native_delta_xyz3, native_delta_axis_angle3, native_close_command]
```

Only rot6d decoding and the gripper sign conversion happen at this boundary.
There is no absolute-goal composition and no `0.05 m` / `0.5 rad` controller
scaling. The trained gripper convention is `-1 = closed, +1 = open`; LIBERO's
runtime command is the negation (`+1 = close, -1 = open`).

## 1. Create the client environment

The simulator/client environment is intentionally separate from the OpenWAM
server environment.

```bash
CONDA_BIN=/path/to/conda \
LIBERO_ENV_PREFIX=/path/to/envs/libero \
LIBERO_PATH=/path/to/LIBERO \
  bash benchmarks/libero/setup_env.sh
```

The setup script pins the upstream commit and MuJoCo `3.3.2`.

## 2. Verify the environment before loading a model

Run progressively stronger checks:

```bash
bash benchmarks/libero/run_smoke.sh import
bash benchmarks/libero/run_smoke.sh task
bash benchmarks/libero/run_smoke.sh env
```

Useful overrides are `LIBERO_PYTHON`, `LIBERO_PATH`, `LIBERO_CONFIG_ROOT`,
`LIBERO_SMOKE_SUITE`, `LIBERO_SMOKE_TASK_ID`, and `LIBERO_SMOKE_GPU`.

## 3. Manual server and single-task client

Start the policy server in the OpenWAM environment:

```bash
bash scripts/deploy.sh \
  --ckpt-dir /path/to/libero_checkpoint \
  --ckpt-name checkpoint_step_10000.safetensors \
  --device cuda:0 --port 8848
```

Then start the simulator/client in the isolated LIBERO environment:

```bash
LIBERO_PATH=/path/to/LIBERO \
LIBERO_PYTHON=/path/to/envs/libero/bin/python \
  bash benchmarks/libero/single_eval.sh libero_spatial 0 8848 127.0.0.1
```

The client performs a WebSocket ping and checks the server-advertised
`representation` before beginning a rollout. Results are written to
`result_dir/results.json` when configured or when `--result-dir` is supplied to
`single_eval.py`.

## 4. Managed end-to-end evaluation

`run_eval.sh` is the complete launcher. It starts policy-server replicas, waits
for readiness, starts isolated LIBERO clients, dynamically schedules tasks,
retries failed requests, writes results and summaries, and always tears the
servers down:

```bash
SERVER_PYTHON=/path/to/openwam/bin/python \
LIBERO_PYTHON=/path/to/envs/libero/bin/python \
LIBERO_PATH=/path/to/LIBERO \
GPUS=0,1 REPLICAS_PER_GPU=1 \
  bash benchmarks/libero/run_eval.sh \
    /path/to/libero_checkpoint checkpoint_step_10000.safetensors \
    --compile-enabled false
```

For a one-task end-to-end validation, append `--smoke`.

The underlying Python entry point can also be called directly for task
sampling, custom suites, external servers, or resume:

```bash
python benchmarks/libero/run_all_suites.py \
  --ckpt-dir /path/to/checkpoint \
  --ckpt-name checkpoint_step_10000.safetensors \
  --gpus 0,1,2,3 --replicas-per-gpu 2 \
  --task-sample-ratio 0.2 --task-sample-seed 42 \
  --output-dir outputs/libero/my_run
```

Reusing an existing `--output-dir` resumes valid completed tasks. When
expanding a sampled run to the full task list, pass `--resume-superset`.
Checkpoint, config, seed, trial range, MuJoCo version, and protocol checks stay
strict. `manifest.json`, per-task `results.json`, `summary.json`, `summary.csv`,
client logs, and server logs make the evaluation auditable.

## Training

The matching training configuration is:

```bash
bash scripts/train.sh dataloader=libero
```

Its required invariants are `type: libero` and `action_mode: libero`.
