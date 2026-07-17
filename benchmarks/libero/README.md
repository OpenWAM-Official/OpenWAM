# LIBERO Benchmark Evaluation

These scripts connect LIBERO to an already-running
OpenWAM policy server, mirroring the RoboTwin benchmark pattern: the benchmark
process owns simulation and observations, while OpenWAM serving stays in the
main model environment.

Use official LIBERO repositories, assets, and datasets. This
directory does not vendor benchmark assets or require project-specific paths.
Because LIBERO both install the top-level package name
`libero`, install them into separate Python environments when using both.

## Files

| File | Description |
| --- | --- |
| `openwam2libero_interface.py` | WebSocket policy adapter from LIBERO observations to OpenWAM server payloads. |
| `policy_config.yml` | Eval client config template. |
| `single_eval.py` | Run one LIBERO task against an OpenWAM policy server. |
| `single_eval.sh` | Shell wrapper that patches host/port/suite/task at runtime. |
| `run_smoke.sh` | Preflight `import`, `task`, or `env` checks for a prepared LIBERO install. |
| `smoke_libero.py` | Preflight implementation; writes an isolated `config.yaml` before importing LIBERO. |

## Prerequisites

1. Install LIBERO following their official README.
2. Download the official assets/datasets required by that benchmark.
3. Start an OpenWAM policy server separately:

```bash
bash scripts/deploy.sh --ckpt-dir /path/to/openwam_ckpt --port 8848
```

Required environment variables for the benchmark client:

| Required variables |
| --- |
| `LIBERO_PATH=/path/to/LIBERO`, `LIBERO_PYTHON=/path/to/libero/env/bin/python` |

Optional config roots:

| Variable | Purpose |
| --- | --- |
| `LIBERO_CONFIG_ROOT` | Ordinary LIBERO config root. Defaults to `~/.libero-openwam`. |

The wrappers write `config.yaml` under these roots before importing LIBERO, so
they do not prompt interactively and do not clobber `~/.libero/config.yaml`.

## Training Data

OpenWAM's `libero` dataloader consumes LeRobot v3 parquet/MP4 data. The
recommended source is:

```bash
hf download nvidia/LIBERO_LeRobot_v3 \
  --repo-type dataset \
  --local-dir /path/to/LIBERO_LeRobot_v3
```

Configure `configs/dataloader/libero.yaml`, then generate action statistics:

```bash
python scripts/libero_compute_stats.py \
  --config configs/dataloader/libero.yaml \
  --output /path/to/LIBERO_LeRobot_v3/libero_normalization_stats.npy
```

LIBERO's 7-D action is a delta command while its 8-D state is an absolute
EEF/gripper observation. They must not share normalization statistics. The
current reader therefore trains image+language→action and masks proprioception
out. Required model overrides:

```text
model.architecture.action_dim=7
model.architecture.use_proprioception=false
```

The stored videos already follow the 180-degree-rotated LIBERO convention. The
evaluation client applies the same transform to live simulator observations.

## Single-Task Evaluation

Ordinary LIBERO:

```bash
LIBERO_PATH=/path/to/LIBERO \
LIBERO_PYTHON=/path/to/libero/bin/python \
bash benchmarks/libero/single_eval.sh ordinary libero_spatial 0 8848 127.0.0.1
```




`policy_config.yml` controls camera mapping, number of trials, max steps,
optional proprioception, and action handling. By default the client sends
`agentview_image` as `head_camera`, `robot0_eye_in_hand_image` as
`left_wrist_camera`, and forwards the server action directly as a 7D LIBERO
action.

Use `POLICY_CONFIG_PATH=/path/to/custom.yml` to run with a copied config.

Important defaults:

- `image_transform: rotate_180` matches the standard LIBERO/OpenVLA convention
  for robosuite offscreen images. Set it to `none` only for checkpoints trained
  on raw unrotated LIBERO frames.
- `settle_steps: 10` runs zero actions after `set_init_state()` before querying
  the policy, allowing objects to settle into a physical state.
- `fail_on_incomplete: false` means the script exits successfully after a
  completed benchmark run even when success rate is below 100%. Set it to
  `true` for smoke tests that should fail unless every trial succeeds.

## Smoke Tests

```bash
LIBERO_PATH=/path/to/LIBERO \
LIBERO_PYTHON=/path/to/libero/bin/python \
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
