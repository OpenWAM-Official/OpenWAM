# EBench Pretrain-SFT

This setup fine-tunes an OpenWAM 80-D pretrain checkpoint on EBench without
changing the model head. The code path is:

1. Download EBench into `/path/to/data_lake/EBench-Dataset`.
2. Use `dataloader=ebench`, which builds raw 23-D EEF/base actions and, by
   default, scatters them to 80-D with `unify_action_map`.
3. Load an 80-D OpenWAM checkpoint via `training.finetune_ckpt_path`.
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

The default config discovers all three EBench training families:

```yaml
groups: [long_horizon, simple_pnp, teleop_tasks]
```

For the original simplePNP + table teleop smoke subset, pass:

```bash
EBENCH_BUCKETS='[simple_pnp/task1,teleop_tasks/peg_in_hole]'
```

## Action Mapping

EBench raw control is converted to 23-D:

```text
[0:10]  left  xyz(3) + rot6d(6) + scalar gripper(1)
[10:20] right xyz(3) + rot6d(6) + scalar gripper(1)
[20:23] base x, y, yaw
```

The reader uses `action.ee_pose` / `state.ee_pose` (`xyz + quaternion(wxyz)`
per arm), converts quaternion to rot6d, and averages each hand's two gripper
finger values into one scalar gripper.

With `unify_action: false`, the dataloader emits this raw 23-D vector and all
23 dimensions are visible. With `unify_action: true` (default for pretrain-SFT),
`unify_action_map` scatters the raw vector into OpenWAM's 80-D layout:

```text
raw[0:10]   -> 80D[0:10]    left xyz + rot6d + gripper
raw[10:20]  -> 80D[34:44]   right xyz + rot6d + gripper
raw[20:23]  -> 80D[68:71]   base x, y, yaw
```

Dexterous hand slots and unused reserved slots stay zero and masked out. The
80-D action loss mask is `(T, 80)` and has 23 valid dimensions per valid
timestep.

The default base source is `action.base`, matching the EBench paper's mobile
base interface: a 3-D planar command `[x, y, yaw]`. It is placed in reserved
80-D slots `[68:71)`.
For an ablation with EBench's alternate delta-base field:

```bash
dataloader.base_action_source=delta
```

## Normalization

`configs/dataloader/ebench.yaml` defaults to:

```yaml
normalize_mode: z-score
normalization_stats_path: /path/to/data_lake/EBench-Dataset/meta/ebench_stats.npy
```

On first load, the dataloader builds this cache from each bucket's
`meta/episodes_stats.jsonl`. The cache stores raw 23-D stats under the
`action_mode` key (`ebench` by default). Both action and proprio use the same
action stats, matching deployment. Rot6d dimensions are pinned to identity
stats because they cannot be derived exactly from quaternion summary moments.

The checkpoint directory gets `normalization_stats.npy` copied automatically.
With `unify_action=true`, deploy gathers the model's 80-D output back to raw
23-D via `unify_action_map` and then applies this raw-space denormalizer.

`scripts/train_ebench_sft.sh` avoids polluting the full-dataset stats cache
during subset runs: if `EBENCH_BUCKETS` is set and `EBENCH_STATS_PATH` is not
set, stats are written under `$OUTPUT_DIR/ebench_stats.npy`.

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
export FINETUNE_CKPT=/path/to/openwam_80d_pretrain/checkpoint_step_x.safetensors
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
dataloader.unify_action=true
dataloader.unify_action_map=["0-9","34-43","68-70"]
model.architecture.action_dim=80
model.architecture.state_dim=80
training.finetune_ckpt_path=$FINETUNE_CKPT
```

No LoRA is enabled. The run is full SFT over the trainable OpenWAM modules
defined by the selected model config.
