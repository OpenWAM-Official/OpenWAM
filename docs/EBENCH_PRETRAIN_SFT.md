# EBench Pretrain-SFT

This setup fine-tunes an OpenWAM 80-D pretrain checkpoint on EBench without
changing the model head. The code path is:

1. Download EBench into `/path/to/data_lake/EBench-Dataset`.
2. Use `dataloader=ebench`, which builds raw 23-D EEF/base actions and, by
   default, scatters them to 80-D with `unify_action_map`.
3. Load an 80-D OpenWAM checkpoint via `training.finetune_ckpt_path`.
4. Run full SFT through `scripts/train.sh` with the overrides below.

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
raw[20:23]  -> 80D[68:71]   base x, y, yaw(deg)
```

Dexterous hand slots and unused reserved slots stay zero and masked out. The
80-D action loss mask is `(T, 80)` and has 23 valid dimensions per valid
timestep.

### Base action semantics

GenManip's dataset converter (`genmanip2lerobot.py`) defines the two base
fields precisely — neither is a velocity:

* `action.base_delta` — the per-step commanded displacement
  `[dx_m, dy_m, dyaw_deg]` in the robot's spawn/odom axes (the raw
  `base_motion` sent each step; GenManip clips it to ±0.015 m / ±1°).
* `action.base` — the running cumsum of `action.base_delta` since episode
  start, i.e. an episode-cumulative commanded pose with **degree** yaw.

The default `base_action_source=delta` supervises `action.base_delta`: a
per-step command is the closest analogue of the instantaneous base command
BEHAVIOR keeps in shared slots `[68:71)` (BEHAVIOR stores local-frame
*velocity*; EBench deltas are odom-frame *displacements* with degree yaw —
same role, different frame/unit, so do not pool their stats in mixtures).
Proprio renders the *measured* per-step displacement
(`state.base[t] - state.base[t-1]`, yaw wrapped, rad→deg) into the same
space, mirroring BEHAVIOR's measured-state-into-command-space rendering.

The `cumulative` ablation supervises `action.base` instead; its proprio is
`state.base` with yaw rad→deg:

```bash
dataloader.base_action_source=cumulative
```

## Normalization

`configs/dataloader/ebench.yaml` defaults to:

```yaml
normalize_mode: min-max
```

Stats live at the fixed location `<dataset_dir>/meta/ebench_stats.npy`. When
the file is missing, the dataloader auto-builds it on first construction via
the offline parquet scan (`ebench_stats_computation`): rank 0 runs the scan
and writes atomically, other ranks poll for the file (timeout/interval
overridable via `OPENWAM_STATS_WAIT_TIMEOUT_S` / `OPENWAM_STATS_POLL_INTERVAL_S`).
The scan carries true q01/q99, so all modes work with no manual pre-step.
To force a rebuild (e.g. after re-exporting episodes), delete the cache file.
The offline module can still be run standalone to prebuild:

```bash
python -m openwam.dataloader.utils.stats_computation.ebench_stats_computation \
    --dataset_dir /path/to/data_lake/EBench-Dataset
```

`min-max` keeps action targets in the bounded `[-1, 1]` distribution the 80-D
pretrain checkpoint was trained on (the family convention). `z-score`
(unbounded) remains available for ablations. `quantile` uses the scan's true
q01/q99; a legacy summary-built cache without them is rejected with a hard
error, never a silent min/max alias.

The cache stores raw 23-D stats under the `action_mode` key (`ebench` by
default) plus a fingerprint (schema version, action keys, bucket paths, and a
sha256 over each bucket's dataset-relative path + its `episodes_stats.jsonl`
bytes — re-downloading a bucket invalidates the cache, while moving the whole
dataset to another mount does not). A `dataloader.buckets`/`groups` subset run
fingerprints its own bucket set, so it conflicts with a full-set cache at the
same fixed path — delete the cache when switching bucket sets.
Both action and proprio use the same action stats, matching deployment. Rot6d
dimensions are pinned to identity stats because they cannot be derived exactly
from quaternion summary moments.

The checkpoint directory gets `normalization_stats.npy` copied automatically.
With `unify_action=true`, deploy gathers the model's 80-D output back to raw
23-D via `unify_action_map` and then applies this raw-space denormalizer.

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

Point `scripts/train.sh` at the latest 80-D OpenWAM pretrain checkpoint:

```bash
bash scripts/train.sh \
    dataloader=ebench \
    dataloader.action_mode=ebench \
    dataloader.unify_action=true \
    'dataloader.unify_action_map=["0-9","34-43","68-70"]' \
    model=dual_system \
    model.architecture.action_dim=80 \
    model.architecture.state_dim=80 \
    model.video_backbone.model_path=/path/to/Wan2.2-TI2V-5B \
    training.finetune_ckpt_path=/path/to/openwam_80d_pretrain/checkpoint_step_x.safetensors \
    training.output_path=/path/to/train_runs/openwam_ebench_sft
```

Smoke-run on two buckets (note: the bucket subset fingerprints its own stats
cache at the fixed location — see the normalization section above):

```bash
bash scripts/train.sh \
    dataloader=ebench \
    ... \
    dataloader.groups=null \
    'dataloader.buckets=[simple_pnp/task1,teleop_tasks/peg_in_hole]' \
    training.batch_size=1 training.max_steps=20 training.save_steps=20
```

No LoRA is enabled. The run is full SFT over the trainable OpenWAM modules
defined by the selected model config.
