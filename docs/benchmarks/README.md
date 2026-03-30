# Benchmark Workflows

This document describes the current benchmark-facing evaluation workflows in OpenWAM.

## Support Matrix

| Benchmark | Mode | Config | Status | Notes |
|---|---|---|---|---|
| RoboTwin | Offline | `eval=robotwin_offline` | Supported | Main documented validation path |
| RoboTwin | Online | `eval=robotwin_online` | Supported | Requires RoboTwin simulator/environment |
| SimplerEnv | Online | `eval=simpler_env` | Supported in code | Requires external `simpler-env` setup |
| LIBERO | Online | `eval=libero` | Supported in code | Requires external LIBERO setup |

## Common Evaluation Entry Point

All benchmark workflows use the same Hydra entrypoint:

```bash
python scripts/eval.py eval=<CONFIG_NAME> eval.ckpt_path=/path/to/checkpoint.safetensors
```

## RoboTwin Offline

Use this for quick validation against held-out dataset samples.

```bash
python scripts/eval.py \
  eval=robotwin_offline \
  eval.ckpt_path=/path/to/checkpoint.safetensors
```

Outputs:

- `action_mse`
- `action_mae`
- optional video metrics when ground-truth video is available

## RoboTwin Online

Use this for closed-loop policy evaluation in the RoboTwin environment.

```bash
python scripts/eval.py \
  eval=robotwin_online \
  eval.ckpt_path=/path/to/checkpoint.safetensors \
  eval.num_episodes=10
```

## SimplerEnv

Install the required environment first.

```bash
pip install simpler-env
```

Then run:

```bash
python scripts/eval.py \
  eval=simpler_env \
  eval.ckpt_path=/path/to/checkpoint.safetensors \
  eval.robot=google_robot
```

Optional overrides:

- `eval.robot=google_robot|widowx`
- `eval.tasks=[google_robot_pick_coke_can]`
- `eval.num_episodes=5`

## LIBERO

Install the required environment first.

```bash
pip install libero
```

Then run:

```bash
python scripts/eval.py \
  eval=libero \
  eval.ckpt_path=/path/to/checkpoint.safetensors \
  eval.task_suites=[libero_spatial]
```

Optional overrides:

- `eval.task_suites=[libero_spatial,libero_object]`
- `eval.num_episodes=5`
- `eval.max_steps_per_episode=300`

## Result Artifacts

Evaluation results are saved to `eval.output_dir` when provided.

Default output paths:

- RoboTwin: `eval_results/`
- SimplerEnv: `eval_results/simpler_env/`
- LIBERO: `eval_results/libero/`

Each run writes a `results.json` summary.

## Current Boundaries

- RoboTwin is the most mature evaluation path.
- SimplerEnv and LIBERO are integrated into the main evaluation entrypoint, but still depend on external simulator stacks.
- Benchmark orchestration scripts and environment installation guides can be expanded further in later phases.
