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

2. Optionally download the official 24k GR1 demonstrations:

   ```bash
   hf download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
     --repo-type dataset \
     --include "gr1_unified.*/**" \
     --local-dir /path/to/robocasa-gr1-24k
   ```

   These folders are LeRobot **v2.0** with native 44-D joint/body vectors.
   This integration intentionally supports only bimanual EEF20, so the public
   files are not directly trainable. First use a trusted simulator/FK export to
   add these four columns:

   ```text
   eef_sim_pose_action[12] + gripper_open_scale_action[2]
   eef_sim_pose_state[12]  + gripper_open_scale_state[2]
   ```

   Then re-index the enriched v2.0 buckets non-destructively:

   ```bash
   python scripts/convert_robocasa_gr1_v20_to_v30.py \
     --input /path/to/robocasa-gr1-eef-v20 \
     --output /path/to/robocasa-gr1-eef-v30
   ```

   The converter rejects joint-only buckets rather than assigning false EEF
   semantics. It validates pose/gripper shapes for every episode.
   The converter hard-links payloads by default, so conversion is fast and does
   not duplicate the large parquet/MP4 data. Use `--link-mode symlink` across
   mount layouts, or `--link-mode copy` only when duplication is intended.

   `configs/dataloader/robocasa_gr1.yaml` accepts only
   `[L xyz3, rot6d6, grip1, R xyz3, rot6d6, grip1]`. Its explicit map places
   the two per-arm 10-D blocks into unified slots `0-9` and `34-43`. Generated
   stats pin all rot6d dimensions to identity so only xyz/gripper are normalized.

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

Inspect a converted training sample, save the composed image and ColorJitter
comparison, and validate action/prompt contracts:

```bash
python scripts/inspect_robocasa_gr1_dataloader.py \
  --config configs/dataloader/robocasa_gr1.yaml \
  --dataset-dir /path/to/robocasa-gr1-eef-v30 \
  --sample-index 0 \
  --output-dir validation_outputs/robocasa_gr1_sample0
```

The output directory contains plain/jittered composed PNGs, a temporal contact
sheet, a difference image, and `inspection_report.json`. The report checks that
the single real ego view occupies the top 256×320 slot, both absent wrist slots
are black, ColorJitter changes pixels, and the resolved task prompt is non-empty.
For EEF/unify configs it additionally checks rot6d orthonormality, identity
normalization on rotation components, and the EEF20↔unified80 map round-trip.

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
- The NVIDIA dataset card documents 44D joint state/action, while the default
  `gr1_unified/*GR1ArmsAndWaistFourierHands_Env` gym wrapper currently reports
  29D joint state/action (`6+6+7+7+3`). The websocket transport is integrated,
  but a 20D EEF checkpoint cannot drive this 29D joint environment until a
  robot-specific EEF-to-joint controller is supplied. The client fails on the
  dimension mismatch instead of silently slicing actions; full simulator
  train/deploy closure remains blocked by that controller, not by websocket IO.
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
