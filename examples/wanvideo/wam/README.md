# WAM: Video-Action Model

Joint video + action generation for robot policy learning using video diffusion as a world model.
The model generates a video of the robot's motion and simultaneously predicts the corresponding
joint-space actions.

---

## Architecture

```
                       ┌──────────────────────────────────────────────────┐
                       │              Video DiT (Wan2.x)                  │
                       │  ┌────────┐  ┌────────┐        ┌────────┐       │
                       │  │  DiT   │─►│  DiT   │─► ··· ─┤  DiT   │─► v̂  │
                       │  │Block 0 │  │Block 1 │        │Block N │       │
                       │  └───┬────┘  └───┬────┘        └───┬────┘       │
                       └──────┼────────────┼─────────────────┼───────────┘
                              │            │                 │
                        bridge features (selected layers)    │
                              ▼            ▼                 ▼
                       ┌──────────────────────────────────────────────────┐
                       │             ActionDiT (~84M)                     │
              a_noisy─►│  │Action 0│─►│Action 1│─► ··· ─┤Action K│─► â   │
                       └──────────────────────────────────────────────────┘
```

**Bridge types** (`--bridge_type`):
- `cross_attn` — unidirectional cross-attention, video → action
- `cross_attn_detach` *(default)* — same, gradients detached at bridge
- `joint_self_attn` — MMDiT-style bidirectional joint self-attention

**Supported video backbones** (`--backbone`):

| Backbone | Model | Conditioning | `video_dim` | Resolution |
|---|---|---|---|---|
| `vace` | Wan2.1-VACE-1.3B | Context Adapter (frozen DiT + Context Blocks) | 1536 | 480×832, 720×1280 |
| `ti2v` | Wan2.2-TI2V-5B | Per-token timestep (`seperated_timestep`) | 3072 | any (h%32==0, w%32==0) |

---

## File Overview

```
wam/
├── train_video_action.py          # Training module + loss (VideoActionTrainingModule)
├── video_action_dataset.py        # RoboTwinDataset, MultiTaskRoboTwinDataset
├── joint_inference.py             # Joint video-action denoising loop + schedule generators
├── eval_robotwin.py               # VAMPolicy adapter for offline/online evaluation
├── compute_action_stats.py        # Compute action normalization stats from HDF5 files
├── prepare_robotwin.py            # Convert RoboTwin episodes to training-ready HDF5
├── view_cameras.py                # Gradio viewer for RoboTwin camera angles
├── accelerate_config_1gpu.yaml    # Accelerate config for single-GPU training
└── accelerate_config_4gpu.yaml    # Accelerate config for 4-GPU training
```

---

## Dataset

**RoboTwin 2.0** — 50 bimanual manipulation tasks, 5 robot embodiments:
`aloha-agilex`, `arx-x5`, `franka`, `ur5`, `airbot`

Data layout:
```
/path/to/dataset/
└── <task_name>/
    └── <robot>_<variant>/
        └── data/
            ├── episode0.hdf5
            ├── episode1.hdf5
            └── ...
```

Each HDF5 contains JPEG-encoded camera frames and `joint_action/vector` (T, 14) actions.

**Task split** (42 train / 8 holdout):
```python
from video_action_dataset import ROBOTWIN_TRAIN_TASKS, ROBOTWIN_HOLDOUT_TASKS, ROBOTWIN_ALL_TASKS
```

---

## Training

### Key training flags

| Flag | Description |
|---|---|
| `--backbone vace\|ti2v` | Video backbone (drives resolution validation) |
| `--trainable_models vace\|dit` | Which video model weights to train |
| `--extra_inputs` | `vace_video,vace_reference_image,action_trajectory` (VACE) or `vace_reference_image,action_trajectory` (TI2V) |
| `--multiview` | Assemble head/left/right cameras into 2×2 grid |
| `--bridge_type` | `cross_attn_detach` (default), `cross_attn`, `joint_self_attn` |
| `--lambda_video` | Video loss weight (set to 0 for action-only training) |
| `--lambda_action` | Action loss weight |
| `--dataset_type` | `robotwin` (single-task) or `robotwin_multitask` |

### Multi-task training

```bash
accelerate launch examples/wanvideo/wam/train_video_action.py \
  --dataset_type robotwin_multitask \
  --dataset_dir /path/to/robotwin_2_0/dataset \
  --robot arx-x5 --variant clean_50 \
  --backbone vace \
  --trainable_models vace \
  --extra_inputs "vace_video,vace_reference_image,action_trajectory" \
  --bridge_type cross_attn_detach \
  --lambda_video 1.0 --lambda_action 1.0
```

---

## Inference & Evaluation

### Offline evaluation (reads HDF5, no simulator)

```bash
python examples/wanvideo/wam/eval_robotwin.py \
  --action_checkpoint /path/to/checkpoint.safetensors \
  --model_paths '["models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors","models/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth","models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth"]' \
  --tokenizer_path models/Wan-AI/Wan2.1-VACE-1.3B/google/umt5-xxl \
  --task_name adjust_bottle --robot arx-x5 \
  --hdf5_data_root /path/to/dataset/adjust_bottle/arx-x5_clean_50/data \
  --offline --num_eval_samples 5
```

### Denoising schedules

```python
from joint_inference import make_schedule, generate_video_and_actions

schedule = make_schedule("sync", num_steps=20)         # video + action in sync
schedule = make_schedule("action_only", num_steps=20)  # video clean, action only
schedule = make_schedule("video_leading", num_steps=20, lead_steps=10)
schedule = make_schedule("cascade", video_steps=20, action_steps=20)
```

---

## Data Utilities

### Compute action normalization stats

```bash
# Single task
python examples/wanvideo/wam/compute_action_stats.py \
  --data_root /path/to/dataset/adjust_bottle/arx-x5_clean_50/data \
  --output data/robotwin/arx-x5_action_stats.npy

# All training tasks combined
python examples/wanvideo/wam/compute_action_stats.py \
  --format robotwin_multitask \
  --dataset_dir /path/to/robotwin_2_0/dataset \
  --robot arx-x5 --variant clean_50
```

### Browse camera angles

```bash
python examples/wanvideo/wam/view_cameras.py \
  --src /path/to/robotwin_2_0/dataset --port 7860
```

---

## Precomputed Action Stats

| File | Description |
|---|---|
