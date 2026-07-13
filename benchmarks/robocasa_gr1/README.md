# RoboCasa GR1 Tabletop Benchmark

These scripts connect the official
[`robocasa/robocasa-gr1-tabletop-tasks`](https://github.com/robocasa/robocasa-gr1-tabletop-tasks)
simulation benchmark to an already-running OpenWAM policy server. The benchmark
process owns RoboCasa / MuJoCo simulation, observations, and success metrics;
OpenWAM serving stays in the main model environment and is accessed through the
shared WebSocket protocol.

Use the official benchmark repository, assets, and NVIDIA dataset. This
directory does not vendor external assets or demonstrations.

For a short machine-migration setup flow, start with
[`SETUP_QUICKSTART.md`](SETUP_QUICKSTART.md).

## Prerequisites

1. Install the official RoboCasa GR1 tabletop repository following its README:

   ```bash
   conda create -c conda-forge -n robocasa-gr1 python=3.10 -y
   conda activate robocasa-gr1

   git clone https://github.com/ARISE-Initiative/robosuite.git
   pip install -e robosuite

   git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
   pip install -e robocasa-gr1-tabletop-tasks
   cd robocasa-gr1-tabletop-tasks
   python robocasa/scripts/download_tabletop_assets.py -y
   ```

2. Optionally download the official demonstrations for dataset inspection or
   training:

   ```bash
   huggingface-cli download \
     --repo-type dataset nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim \
     --local-dir ./datasets/
   ```

3. Start an OpenWAM policy server separately:

   ```bash
   bash scripts/deploy.sh --ckpt-dir /path/to/openwam_ckpt --port 8848
   ```

## Smoke Tests

Run import / task-registry smoke checks:

```bash
ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks \
ROBOCASA_GR1_PYTHON=/path/to/robocasa-gr1/bin/python \
bash benchmarks/robocasa_gr1/run_smoke.sh import

ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks \
ROBOCASA_GR1_PYTHON=/path/to/robocasa-gr1/bin/python \
bash benchmarks/robocasa_gr1/run_smoke.sh task
```

Run a lightweight env reset/step smoke after assets are installed:

```bash
ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks \
ROBOCASA_GR1_PYTHON=/path/to/robocasa-gr1/bin/python \
ROBOCASA_GR1_SMOKE_STEPS=1 \
bash benchmarks/robocasa_gr1/run_smoke.sh env
```

For render-enabled smoke tests and machine migration notes, see
[`RENDERING.md`](RENDERING.md).

Inspect an HDF5 file from the NVIDIA dataset:

```bash
ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks \
ROBOCASA_GR1_PYTHON=/path/to/robocasa-gr1/bin/python \
ROBOCASA_GR1_DATASET=/path/to/demo.hdf5 \
bash benchmarks/robocasa_gr1/run_smoke.sh dataset
```

## Evaluation

```bash
ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks \
ROBOCASA_GR1_PYTHON=/path/to/robocasa-gr1/bin/python \
bash benchmarks/robocasa_gr1/single_eval.sh \
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
  8848 \
  127.0.0.1
```

`policy_config.yml` controls camera keys, state forwarding, action splitting,
episode count, and max steps. By default:

- `head_camera_key` is `video.ego_view_pad_res256_freq20`, the official
  processed ego-view stream from `GrootRoboCasaEnv`.
- `send_state: true` forwards the GR00T-style `state.*` fields. Set an explicit
  ordered `state_keys` list matching the conversion job; `null` uses sorted keys
  for deterministic diagnostics but cannot prove semantic alignment.
- Set an explicit ordered `action_keys` list matching the conversion job.
  `action_keys: null` uses sorted `env.action_space` keys.
- The NVIDIA dataset card documents 44D state/action, while the default
  `gr1_unified/*GR1ArmsAndWaistFourierHands_Env` gym wrapper currently reports
  29D state/action (`6+6+7+7+3`). Treat the env smoke output as the source of
  truth for the checkpoint you evaluate. The shipped 29D environment directly
  consumes only a matching joint-mode checkpoint; 20D EEF and 80D unified
  checkpoints require a robot-specific EEF-to-joint controller/conversion.
- `fail_on_incomplete: false` means the process exits successfully after a
  completed benchmark run even when success rate is below 100%. Set it to
  `true` for pass/fail smoke gates.

## Supported Task IDs

The official repo registers all 24 tabletop tasks under `gr1_unified/*`, for
example:

```text
gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
```

See the official RoboCasa GR1 README for the full list.
