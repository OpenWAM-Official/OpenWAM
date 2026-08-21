# LIBERO Benchmark Evaluation

These scripts connect LIBERO to an already-running
OpenWAM policy server, mirroring the RoboTwin benchmark pattern: the benchmark
process owns simulation and observations, while OpenWAM serving stays in the
main model environment.

Use official LIBERO assets and datasets. Ordinary LIBERO mirrors
the official direct dependency versions and pins the validated MuJoCo 3.3.2
resolution described below.
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

Ordinary LIBERO is fixed to the following tested installation. Its direct
dependencies mirror the official `requirements.txt`; MuJoCo 3.3.2 is retained
as the compatible resolution for the unbounded `robosuite==1.4.0` dependency:

- Python 3.10 (required by the official `numpy==1.22.4` pin)
- NumPy 1.22.4
- OpenCV 4.6.0.66
- robomimic 0.2.0
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

Its environment mirrors the versions in the official `requirements.txt`. The official `robosuite==1.4.0` requirement leaves MuJoCo unbounded above, but current MuJoCo 3.11 is incompatible with it; the environment therefore locks the known-good 3.3.2 resolution. The resolved version is also recorded in every run manifest:

- Python 3.10 (required by the official `numpy==1.22.4` pin)
- NumPy 1.22.4
- OpenCV 4.6.0.66
- robomimic 0.2.0
- robosuite 1.4.0
- bddl 1.0.1
- MuJoCo 3.3.2 (compatibility resolution for an unpinned transitive dependency)
- wand 0.7.2 and scikit-image 0.19.3 (reproducible resolutions for the
  two packages left unpinned by `extra_requirements.txt`)


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

OpenWAM's `libero` dataloader defaults to the standalone canonical LeRobot v3
dataset converted from the official `lerobot/libero` snapshot:

```bash
python scripts/convert_lerobot_libero_to_absolute_eef10_v3.py \
  --source /path/to/libero-lerobot \
  --output /path/to/libero
```

The independent 20 FPS Fast-WAM conversion is retained under an explicit name:

```bash
python scripts/convert_libero_to_absolute_eef10_v3.py \
  --source /path/to/libero-fastwam \
  --output /path/to/libero-fastwam-absolute-eef10-v3
```

`configs/dataloader/libero.yaml` points to the official-snapshot conversion at
`/path/to/libero` by default. The single
training/deployment normalization artifact is
`<dataset_dir>/meta/normalization_stats.npy`; it is auto-computed there on first
use if missing. To pre-compute it instead:

```bash
python -m openwam.dataloader.utils.stats_computation.libero_stats_computation \
  --config configs/dataloader/libero.yaml \
  --output /path/to/libero/meta/normalization_stats.npy
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
- `fail_on_incomplete: false` means the script exits successfully after a
  completed benchmark run even when success rate is below 100%. Set it to
  `true` for smoke tests that should fail unless every trial succeeds.

## Full 10-epoch evaluation on eight GPUs

The following command starts two independent copies of the 10-epoch policy on
each GPU (ports 8920–8935). Every task is evaluated by one client/environment
that runs trials 0–49 continuously. Replicas pull whole tasks from one dynamic
queue, so an idle replica immediately receives the next task without splitting
a task's RNG stream. A failed task is returned to the queue up to three times;
an endpoint is retired after three consecutive client failures. All videos,
client/server logs, a run manifest, `summary.csv`, and `summary.json` are
retained under the printed run directory.

```bash
/usr/bin/python3.12 benchmarks/libero/run_10epoch_all_suites.py
```

The launcher requires MuJoCo 3.3.2 from the default LIBERO environment. It
defaults to synchronous inference with horizon 10 and synchronous 10-step
denoising, but these runtime parameters are user-configurable. The default
output root is the persistent data path
`/path/to/OpenWAM/outputs/libero`. Useful preflight and
recovery commands are shown below. The ordinary-LIBERO rollout limit is 600
policy steps for SPATIAL, GOAL, and OBJECT, and 700 for LONG (`libero_10`).

```bash
# Enumerate and display the complete assignment without starting processes.
/usr/bin/python3.12 benchmarks/libero/run_10epoch_all_suites.py --dry-run

# One rollout of spatial task 0; only its assigned policy replica is started.
/usr/bin/python3.12 benchmarks/libero/run_10epoch_all_suites.py --smoke --gpus 0

# Resume an interrupted output directory. Completed task/trial ranges are preserved,
# and only unfinished jobs are rebalanced across the available workers.
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

# The wrapper defaults to synchronous horizon 10; this can be overridden.
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
