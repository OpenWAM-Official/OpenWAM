# Benchmark Statistics

This note keeps only high-level benchmark facts: dataset scale, public training
cost references, planning GPU-hour estimates, and reported baseline scores.

## Dataset Scale

### RoboCasa GR1

| Dataset / split | Tasks | Trajectories | Storage | Notes |
|---|---:|---:|---:|---|
| NVIDIA `PhysicalAI-Robotics-GR00T-Teleop-Sim` HDF5 | 24 | 24,000 | ~14 GB | 1,000 teleoperation trajectories per task. |
| NVIDIA `PhysicalAI-Robotics-GR00T-Teleop-Sim` LeRobot | 24 | 24,000 | ~39 GB | LeRobot-style release of the same simulation data. |
| DIAL full robot-only setting | 24 | 24,000 | same as above | Uses all 1,000 trajectories per task. |
| DIAL few-shot robot-only setting | 24 | 2,400 | subset | 100 trajectories per task. |
| DIAL human+robot setting | 24 GR1 tasks + EgoDex | 2,400 GR1 + EgoDex | mixed | Uses few-shot GR1 plus EgoDex basic-pick-place human demonstrations. |
| Online eval, DIAL in-distribution | 24 | 1,200 episodes | generated | 50 episodes per task. |

RoboCasa GR1 action/state conventions reported by DIAL:

- 29D joint-space GR1 control: dual arms 14D, hands 12D, waist 3D.
- 18D EEF pose: left wrist `(xyz3 + rot6d6)` and right wrist `(xyz3 + rot6d6)`.
- 47D combined simulation state/action reference: 29D joints + 18D EEF.

### LIBERO

| Suite | Tasks | Demonstrations | Notes |
|---|---:|---:|---|
| LIBERO-Spatial | 10 | ~500 | Common public setup uses 50 demonstrations per task. |
| LIBERO-Object | 10 | ~500 | Common public setup uses 50 demonstrations per task. |
| LIBERO-Goal | 10 | ~500 | Common public setup uses 50 demonstrations per task. |
| LIBERO-Long | 10 | ~500 | Common public setup uses 50 demonstrations per task. |

## Training Cost References

GPU-hour values below are planning estimates unless the source publishes wall
time. Use `GPU-hours = number_of_GPUs * wall_clock_hours` once measured on the
target cluster.

| Benchmark / source | Hardware | Steps | Data scale | Estimated GPU-hours | Notes |
|---|---:|---:|---:|---:|---|
| OpenWAM debug config | any viable GPU node | 20 max steps | tiny smoke | <1 | Plumbing only, not a performance run. |
| RoboCasa GR1, NVIDIA GR00T N1.7 finetune | 8 GPUs | 60,000 | 24 tasks | ~500-1,300 | Official example uses `NUM_GPUS=8`, `GLOBAL_BATCH_SIZE=512`, `MAX_STEPS=60000`, `SAVE_STEPS=2000`; wall time not public. |
| RoboCasa GR1, DIAL full robot-only | not public | 160,000 | 24,000 trajectories | ~1,300-3,500 | 80K decoupled warmup + 80K end-to-end. |
| RoboCasa GR1, DIAL few-shot robot-only | not public | 40,000 | 2,400 trajectories | ~300-900 | 100 trajectories per task. |
| RoboCasa GR1, DIAL human+robot | not public | 60,000 | EgoDex + 2,400 GR1 | ~500-1,300 | 40K co-training + 20K GR1 finetune. |
| LIBERO, OpenVLA LoRA | 8x A100 80GB | 50K / 50K / 60K / 80K | 10 tasks per suite | ~400-1,400 per suite | Spatial/Object/Goal/Long respectively, batch size 128, LoRA rank 32. |

## Public Baseline Scores

### RoboCasa GR1

| Method / checkpoint | Average success | Details |
|---|---:|---|
| DIAL full data | 70.2% | 24 tasks, 50 episodes per task; 68.9% Pick & Place, 74.3% Articulated. |
| DIAL few-shot | 58.3% | 100 trajectories per task. |
| FLARE | 55.0% | Reported by DIAL as strongest prior full-data baseline. |
| GR00T-N1.6 | 47.6% | Reported by DIAL. |
| GR00T N1.7 `ROBOCASA_GR1_TABLETOP` | 44.5% | NVIDIA example README, 20 trials per task except one 22-trial task. |

NVIDIA N1.7 per-task examples:

| Task | Success |
|---|---:|
| `PnPBottleToCabinetClose` | 70.0% |
| `PnPCanToDrawerClose` | 70.0% |
| `PnPCupToDrawerClose` | 35.0% |
| `PnPWineToCabinetClose` | 65.0% |
| `PosttrainPnPNovelFromPlateToPlateSplitA` | 75.0% |

### LIBERO

OpenVLA public fine-tuning table:

| Method | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|
| Diffusion Policy from scratch | 78.3% | 92.5% | 68.3% | 50.5% | 72.4% |
| Octo fine-tuned | 78.9% | 85.7% | 84.6% | 51.1% | 75.1% |
| OpenVLA fine-tuned | 84.7% | 88.4% | 79.2% | 53.7% | 76.5% |

Additional published LIBERO references:

| Method | Spatial | Object | Goal | Long | Average |
|---|---:|---:|---:|---:|---:|
| DiT Policy fine-tuned | 84.2% | 96.3% | 85.4% | 63.8% | 82.4% |
| OpenVLA-OFT PD&AC Cont-Diffusion | 96.9% | 98.1% | 95.5% | 91.1% | 95.4% |
| OpenVLA-OFT, original unfiltered data | 95.2% | 94.2% | 95.2% | 93.2% | 94.5% |
| OpenVLA-OFT, latest comparison table | 97.6% | 98.4% | 97.9% | 94.5% | 97.1% |
