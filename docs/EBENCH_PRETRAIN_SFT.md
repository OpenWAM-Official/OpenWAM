# EBench Pretrain-SFT

This setup fine-tunes an OpenWAM 80-D pretrain checkpoint on EBench without
changing the model head. The code path is:

1. Download EBench into `/path/to/data_lake/EBench-Dataset`.
2. Use `dataloader=ebench`, which emits 80-D `action`, `action_mask`,
   `proprio`, and `proprio_mask`.
3. Load an 80-D OpenWAM checkpoint via `training.pretrained_checkpoint_path`.
4. Run full SFT through `scripts/train_ebench_sft.sh`.

## Data

Dataset link:

```bash
https://huggingface.co/datasets/InternRobotics/EBench-Dataset
```

Full download:

```bash
huggingface-cli download InternRobotics/EBench-Dataset \
  --repo-type dataset \
  --local-dir /path/to/data_lake/EBench-Dataset
```

Small feasibility subset for simplePNP and table teleop tasks:

```bash
huggingface-cli download InternRobotics/EBench-Dataset \
  --repo-type dataset \
  --local-dir /path/to/data_lake/EBench-Dataset \
  --include 'simple_pnp/task1/**' \
  --include 'teleop_tasks/peg_in_hole/**'
```

The default config discovers:

```yaml
groups: [simple_pnp, teleop_tasks]
```

For a smoke subset, pass:

```bash
EBENCH_BUCKETS='[simple_pnp/task1,teleop_tasks/peg_in_hole]'
```

## 80-D Action Mapping

EBench raw control is 19-D:

```text
[0:6]   left arm joints
[6:12]  right arm joints
[12:14] left two-finger gripper
[14:16] right two-finger gripper
[16:19] base velocity command
```

OpenWAM 80-D placement:

```text
raw joints[0:6]      -> 80D[10:16]
raw gripper[0:2]    -> 80D[16:18], mean -> 80D[9]
raw joints[6:12]    -> 80D[42:48]
raw gripper[2:4]    -> 80D[48:50], mean -> 80D[41]
raw base[0:3]       -> 80D[64:67]
```

All EEF xyz/rot6d slots and unused hand/reserved slots stay zero and masked
out. The action loss mask is `(T, 80)` and has 21 valid dimensions per valid
timestep.

The default base source is `action.base`, matching the EBench paper's mobile
base interface: a 3-D velocity command `[vx, vy, yaw_rate]` for planar x/y
motion and yaw rate. It is placed in the reserved 80-D slots `[64:67)`.
For an ablation with EBench's alternate delta-base field:

```bash
dataloader.base_action_source=delta
```

## Normalization

`configs/dataloader/ebench.yaml` defaults to:

```yaml
normalize_mode: z-score
normalization_stats_path: /path/to/data_lake/EBench-Dataset/meta/ebench80_stats.npy
```

On first load, the dataloader builds this cache from each bucket's
`meta/episodes_stats.jsonl`. Action stats use `action.*` columns; proprio stats
use `state.*` columns. The checkpoint directory gets `normalization_stats.npy`
copied automatically for deployment-time action denormalization.

`scripts/train_ebench_sft.sh` avoids polluting the full-dataset stats cache
during subset runs: if `EBENCH_BUCKETS` is set and `EBENCH_STATS_PATH` is not
set, stats are written under `$OUTPUT_DIR/ebench80_stats.npy`.

## Check Dataloader Only

This does not construct the model:

```bash
cd /path/to/OpenWAM
python scripts/check_ebench_dataloader.py \
  --dataset-dir /path/to/data_lake/EBench-Dataset \
  --buckets simple_pnp/task1 teleop_tasks/peg_in_hole \
  --samples 2
```

Expected contracts:

```text
action:       (32, 80)
action_mask:  (32, 80)
proprio:      (1, 80)
proprio_mask: (1, 80)
```

## Start Full SFT

Set the latest 80-D OpenWAM pretrain checkpoint:

```bash
cd /path/to/OpenWAM
export PRETRAIN_CKPT=/path/to/openwam_80d_pretrain/checkpoint_step_x.safetensors
export EBENCH_DATASET_DIR=/path/to/data_lake/EBench-Dataset
export OUTPUT_DIR=/path/to/train_runs/openwam_ebench_sft
export WAN22_PATH=/path/to/Wan2.2-TI2V-5B

bash scripts/train_ebench_sft.sh
```

Smoke-run on two buckets:

```bash
EBENCH_BUCKETS='[simple_pnp/task1,teleop_tasks/peg_in_hole]' \
BATCH_SIZE=1 \
bash scripts/train_ebench_sft.sh training.max_steps=20 training.save_steps=20
```

The script forces:

```text
dataloader=ebench
model.architecture.action_dim=80
model.architecture.state_dim=80
training.pretrained_checkpoint_path=$PRETRAIN_CKPT
```

No LoRA is enabled. The run is full SFT over the trainable OpenWAM modules
defined by the selected model config.
