# LIBERO Benchmark Evaluation

These scripts connect LIBERO to an already-running
OpenWAM policy server, mirroring the RoboTwin benchmark pattern: the benchmark
process owns simulation and observations, while OpenWAM serving stays in the
main model environment.

Use official LIBERO assets and datasets. Ordinary LIBERO
evaluation is pinned to the validated MuJoCo 3.3.2 environment described below.
Because LIBERO both install the top-level package name
`libero`, keep them in separate Python environments when using both.

## Files

| File | Description |
| --- | --- |
| `openwam2libero_interface.py` | WebSocket policy adapter from LIBERO observations to OpenWAM server payloads. |
| `policy_config.yml` | Eval client config template. |
| `single_eval.py` | Run one LIBERO task against an OpenWAM policy server. |
| `single_eval.sh` | Shell wrapper that patches host/port/suite/task at runtime. |
| `run_10epoch_all_suites.py` | Shared multi-GPU launcher for LIBERO. |
| `run_10epoch.sh` | Run the complete evaluation with the default client environment. |
| `run_smoke.sh` | Preflight `import`, `task`, or `env` checks for a prepared LIBERO install. |
| `smoke_libero.py` | Preflight implementation; writes an isolated `config.yaml` before importing LIBERO. |
| `environment.yml` | Reproducible conda and pip dependency pins for the default client environment. |
| `setup_env.sh` | Create/update the default environment and LIBERO checkout. |

## Prerequisites

Ordinary LIBERO is fixed to the following tested installation:

- Python 3.11.15
- MuJoCo 3.3.2
- robosuite 1.4.0
- bddl 1.0.1
- LIBERO commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- environment: `/path/to/miniconda3/envs/libero`
- checkout: `/path/to/LIBERO`

Create or reconcile that installation with:

```bash
bash benchmarks/libero/setup_env.sh
```

The setup script uses `environment.yml`, installs LIBERO editable,
and applies the included compatibility patch needed by PyTorch 2.6 and newer.
After installation, download the official assets/datasets required by LIBERO.



- Python 3.11.15
- MuJoCo 3.3.2
- robosuite 1.4.0
- bddl 1.0.1


On a new machine, install the required rendering/archive libraries once and
then run the reproducible setup. The setup downloads the official asset
archive and verifies its SHA-256 checksum before extraction.


Start an OpenWAM policy server separately for a single-task evaluation:

```bash
bash scripts/deploy.sh --ckpt-dir /path/to/openwam_ckpt --port 8848
```

Environment paths used by the benchmark client:

| Paths |
| --- |
| Defaults to the pinned paths above; `LIBERO_PATH` and `LIBERO_PYTHON` may override them. |

Optional config roots:

| Variable | Purpose |
| --- | --- |
| `LIBERO_CONFIG_ROOT` | Ordinary LIBERO config root. Defaults to `~/.libero-openwam`. |

The wrappers write `config.yaml` under these roots before importing LIBERO, so
they do not prompt interactively and do not clobber `~/.libero/config.yaml`.

## Training Data

OpenWAM's `libero` dataloader consumes the standalone canonical LeRobot v3
dataset generated from the Fast-WAM source:

```bash
python scripts/convert_libero_to_absolute_eef10_v3.py \
  --source /path/to/libero-fastwam \
  --output /path/to/libero
```

`configs/dataloader/libero.yaml` points to that output by default. Normalization statistics load from
`<dataset_dir>/meta/libero_normalization_stats.npy` and are auto-computed
there on first use if the file is missing. To pre-compute them instead:

```bash
python -m openwam.dataloader.utils.stats_computation.libero_stats_computation \
  --config configs/dataloader/libero.yaml \
  --output /path/to/libero/meta/libero_normalization_stats.npy
```

### EEF10 data contract

The reader accepts only row-aligned single-arm **EEF10**
`[xyz3, rot6d6, gripper_open_scale1]` columns:

- **Gripper direction.** The trained channel (dim 9) is an **open-scale**:
  `-1 = closed, +1 = open`. The eval client negates it into LIBERO's native
  robosuite command convention on the way out.
- **Proprio** is the achieved EEF10 `observation.state` at window frame 0.
  Its scalar gripper channel is the clipped finger aperture
  `qpos[0]-qpos[1]` mapped to open-scale. This is the required 1-DoF EEF10
  projection; the original two finger-joint values are not stored separately.
- **Action target** at step `t` is the absolute OSC EEF10 goal stored in
  `action[t]`. It is consumed from the same row without reconstruction or a
  `t+1` shift; the final episode row remains a valid supervised action.
- With `unify_action: true` and `unify_action_map: ["0-9"]` the 10 physical
  dims scatter into the unified 80-D pretraining space (left-arm slots); all
  other slots stay masked, so `model.architecture.action_dim=80` needs no
  LIBERO-specific override.
- Action targets and achieved proprio are pooled into one global `eef`
  normalization block and both use that same transform; rot6d dims are pinned
  to identity and never normalized. The gripper dim IS normalized, so the block
  records its `gripper_convention` and the reader refuses a mismatched file.

The stored videos already follow the 180-degree-rotated LIBERO convention. The
evaluation client applies the same transform to live simulator observations.

## Single-Task Evaluation

Ordinary LIBERO:

```bash
bash benchmarks/libero/single_eval.sh ordinary libero_spatial 0 8848 127.0.0.1
```




`policy_config.yml` controls camera mapping, number of trials, max steps,
proprioception, and action handling. By default (`action_mode: eef`) the client
sends `agentview_image` as `head_camera`, `robot0_eye_in_hand_image` as
`left_wrist_camera`, and the live 10-D EEF proprio assembled from
`robot0_eef_pos` / `robot0_eef_quat` / `robot0_gripper_qpos` — byte-consistent
with the dataloader. The server returns the raw EEF10 full-pose target; the
client converts it to the env's native 7-D OSC delta using the live controller
`output_max` scales (probed automatically, falls back to 0.05 m / 0.5 rad).

Use `POLICY_CONFIG_PATH=/path/to/custom.yml` to run with a copied config.

Important defaults:

- `image_transform: rotate_180` matches the standard LIBERO/OpenVLA convention
  for robosuite offscreen images. Set it to `none` only for checkpoints trained
  on raw unrotated LIBERO frames.
- `settle_steps: 30` runs the configured settle action after `set_init_state()` before querying
  the policy, allowing objects to settle into a physical state.
- `fail_on_incomplete: false` means the script exits successfully after a
  completed benchmark run even when success rate is below 100%. Set it to
  `true` for smoke tests that should fail unless every trial succeeds.

## Full 10-epoch evaluation on eight GPUs

The following command starts two independent copies of the 10-epoch policy on
each GPU (ports 8920–8935). Every task is evaluated by one client/environment
that runs trials 0–49 continuously. The two replicas on a GPU receive disjoint
task queues, so they evaluate different tasks concurrently without splitting a
task's RNG stream. The 40 tasks from LIBERO-SPATIAL, LIBERO-GOAL,
LIBERO-OBJECT, and LIBERO-LONG (`libero_10` in the Python API) are distributed
evenly, five tasks per GPU (three on one replica and two on the other). All
videos, client/server logs, a run manifest, `summary.csv`, and `summary.json`
are retained under the printed run directory.

```bash
/usr/bin/python3.12 benchmarks/libero/run_10epoch_all_suites.py
```

The launcher requires MuJoCo 3.3.2 from the default LIBERO environment,
synchronous inference, and an inference horizon of 32. The
default output root is the persistent data path
`/path/to/OpenWAM/outputs/libero`. Useful preflight and
recovery commands are shown below. The default rollout limit is 600 policy
steps for SPATIAL, OBJECT, and GOAL, and 700 for LONG (`libero_10`).

```bash
# Enumerate and display the complete assignment without starting processes.
/usr/bin/python3.12 benchmarks/libero/run_10epoch_all_suites.py --dry-run

# One rollout of spatial task 0; only its assigned policy replica is started.
/usr/bin/python3.12 benchmarks/libero/run_10epoch_all_suites.py --smoke --gpus 0

# Resume an interrupted output directory; complete 50-trial task runs are skipped.
/usr/bin/python3.12 benchmarks/libero/run_10epoch_all_suites.py \
  --output-dir /path/to/existing/run

# Rebuild statistics without starting servers or clients.
/usr/bin/python3.12 benchmarks/libero/run_10epoch_all_suites.py \
  --summarize-only --output-dir /path/to/existing/run
```

Each server is owned by the launcher and is stopped when the run finishes or
is interrupted. The launcher never terminates unrelated server processes.

Use `run_10epoch.sh` inside tmux for a persistent full evaluation:

```bash
tmux new-session -d -s libero_10ep \
  "cd /path/to/OpenWAM && bash benchmarks/libero/run_10epoch.sh"

# Override the sync/async action-execution horizon for this run.
tmux new-session -d -s libero_10ep_h10 \
  "cd /path/to/OpenWAM && INFERENCE_HORIZON=10 \
   bash benchmarks/libero/run_10epoch.sh"
```

## Smoke Tests

```bash
bash benchmarks/libero/run_smoke.sh task
```

Smoke knobs:

| Variable | Default | Description |
| --- | ---: | --- |
| `LIBERO_SMOKE_SUITE` | `libero_spatial` | Benchmark suite name. |
| `LIBERO_SMOKE_TASK_ID` | `0` | Task id within the suite. |
| `LIBERO_SMOKE_CAMERA_SIZE` | `128` | Camera height/width for `env` smoke. |
| `LIBERO_SMOKE_STEPS` | `1` | Number of dummy zero-action env steps. |
| `LIBERO_SMOKE_GPU` | `0` | GPU id for EGL/MuJoCo rendering in `env` mode. |

## Assets

`import` and `task` smokes only need code plus BDDL/init files. `env` smoke also
needs the corresponding assets directory:

```text
<LIBERO repo>/libero/libero/assets/
```

Ordinary LIBERO uses the standard LIBERO assets/datasets described in its upstream README.

`env` smoke also needs a working EGL/MuJoCo rendering stack. On headless
containers where NVIDIA EGL libraries are missing or are empty placeholders, the
smoke exits before importing robosuite and prints the offending library path.
