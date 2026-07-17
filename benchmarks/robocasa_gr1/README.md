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
   This integration trains base-frame EEF33, so run the checked-in MuJoCo FK
   enrichment and v3 conversion:

   ```bash
   export ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks
   export ROBOCASA_PYTHON=/path/to/robocasa-gr1/bin/python
   export OPENWAM_PYTHON=/path/to/openwam/bin/python
   bash scripts/prepare_robocasa_gr1_eef33.sh \
     /path/to/robocasa-gr1-24k \
     /path/to/robocasa-gr1-eef33-v20 \
     /path/to/robocasa-gr1-eef33-v30
   ```

   EEF33 is
   `[L xyz3, L rot6d6, L hand6, R xyz3, R rot6d6, R hand6, waist3]`.
   Poses are relative to `robot0_base`, making enrichment deterministic across
   randomized world placements. The converter rejects joint-only buckets.
   The converter hard-links payloads by default, so conversion is fast and does
   not duplicate the large parquet/MP4 data. Use `--link-mode symlink` across
   mount layouts, or `--link-mode copy` only when duplication is intended.

   `configs/dataloader/robocasa_gr1.yaml` accepts only
   EEF33. Its exact unified map is
   `["0-8", "10-15", "34-42", "44-49", "68-70"]`; gripper slots 9/43 stay
   masked because Fourier hands use six joint commands. Action and achieved
   state have separate stats, while all rot6d dimensions remain identity-pinned.

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
  --dataset-dir /path/to/robocasa-gr1-eef33-v30 \
  --sample-index 0 \
  --output-dir validation_outputs/robocasa_gr1_sample0
```

The output directory contains plain/jittered composed PNGs, a temporal contact
sheet, a difference image, and `inspection_report.json`. The report checks that
the single real ego view occupies the top 256×320 slot, both absent wrist slots
are black, ColorJitter changes pixels, and the resolved task prompt is non-empty.
For EEF/unify configs it additionally checks rot6d orthonormality, identity
normalization on rotation components, and the EEF33↔unified80 map round-trip.

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
- `send_state: true` converts the live 29-D GR00T state through the same
  base-frame FK as offline enrichment and sends EEF33 proprio.
- The returned EEF33 action is solved by damped-least-squares dual-arm IK;
  Fourier hand6 and waist3 are passed through unchanged. Joint limits,
  residual thresholds, and hold-current fallback are enforced.
- The NVIDIA dataset card documents 44D joint state/action, while the default
  `gr1_unified/*GR1ArmsAndWaistFourierHands_Env` gym wrapper currently reports
  29D joint state/action (`6+6+7+7+3`). The EEF33 FK/IK bridge supplies the
  required conversion in both directions.
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
