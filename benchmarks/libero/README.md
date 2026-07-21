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



```bash
python -m openwam.dataloader.utils.stats_computation.libero_stats_computation \
  --config configs/dataloader/libero.yaml \
  --output /path/to/LIBERO_LeRobot_v3/libero_normalization_stats.npy
```

### EEF10 data contract

The reader trains on the repo-standard single-arm **EEF10** representation
`[xyz3, rot6d6, gripper1]` (world frame, full pose), not on LIBERO's native 7-D
OSC delta:

- **Proprio** at window frame 0 is the achieved 8-D `observation.state`
  rendered to EEF10 (axis-angle → rot6d; finger separation → [-1, +1] command
  space, +1 = close).
- **Action target** at step `t` is the **next frame's achieved pose**
  (`state[t+1]` → xyz + rot6d) plus the recorded gripper command `action[t][6]`
  — a full absolute pose target. The final window step has no `t+1` and is
  masked out of the loss.
- With `unify_action: true` and `unify_action_map: ["0-9"]` the 10 physical
  dims scatter into the unified 80-D pretraining space (left-arm slots); all
  other slots stay masked, so `model.architecture.action_dim=80` needs no
  LIBERO-specific override.
- Action targets and achieved proprio are pooled into one global `eef`
  normalization block and both use that same transform; rot6d dims are pinned
  to identity and never normalized.

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
proprioception, and action handling. By default (`action_mode: eef`) the client
sends `agentview_image` as `head_camera`, `robot0_eye_in_hand_image` as
`left_wrist_camera`, and the live 10-D EEF proprio assembled from
`robot0_eef_pos` / `robot0_eef_quat` / `robot0_gripper_qpos` — byte-consistent
with the dataloader. The server returns the raw EEF10 full-pose target; the
client converts it to the env's native 7-D OSC delta using the live controller
`output_max` scales (probed automatically, falls back to 0.05 m / 0.5 rad).
`action_mode: native` keeps the legacy 7-D passthrough for checkpoints trained
directly on raw OSC actions.

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
