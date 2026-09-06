# OpenWAM-α downstream fine-tuning

OpenWAM-α uses a fixed action and state contract. A downstream reader may keep any native representation internally, but the sample passed to OpenWAM must follow the slot layout, temporal settings, and preprocessing rules below.

## 1. Build a compatible dataloader

Register a reader with [openwam/dataloader/registry.py](../../openwam/dataloader/registry.py), implement `from_config`, and return the fields consumed by the trainer:

~~~text
# Video and prompt
video              list[PIL.Image]
first_frame_image  list[PIL.Image]
prompt             str

# Action and state; both use the fixed α width
# action: (T - 1, 80), action_mask: (T - 1, 80)
# proprio: (1, 80), proprio_mask: (1, 80)
action             float Tensor
action_mask        bool Tensor
proprio            float Tensor
proprio_mask       bool Tensor

# Valid video positions
video_mask         bool Tensor, shape (num_video_frames,)
~~~

Use [BaseDataset](../../openwam/dataloader/bases/dataset.py) for a custom source or [LeRobotV3Reader](../../openwam/dataloader/bases/lerobot_v3_reader.py) for LeRobot v3 data. Add a Hydra file under [configs/dataloader/](../../configs/dataloader/):

~~~yaml
# configs/dataloader/my_task.yaml
# Dataset and action mapping
type: my_task
dataset_dir: /path/to/dataset
unify_action: true
unify_action_map: [0-9, 34-43]       # source dimensions mapped to α slots

# α preprocessing and temporal contract
normalize_mode: min-max              # fixed for α fine-tuning
num_frames: 33                       # 32-action chunk: num_frames - 1
video_stride: 4                      # 9 sampled video frames from 33 source frames
window_stride: 1
~~~

`num_frames=33`, `video_stride=4`, and the derived 32-action chunk are fixed. There is no separate `action_chunk` YAML key for the standard sampler; `inference_horizon` only limits how many generated actions are consumed before the next generation.

### The α action and preprocessing contract

The foundation model uses `action_dim=80` and `state_dim=80`. The canonical slots are:

~~~text
# Left arm
0:3    left EEF position (x, y, z)
3:9    left EEF rotation (rot6d)
9      left gripper
10:34  left dexterous hand joints (24)

# Right arm
34:37  right EEF position (x, y, z)
37:43  right EEF rotation (rot6d)
43     right gripper
44:68  right dexterous hand joints (24)

# Reserved
68:80  reserved slots
~~~

Native layouts may differ internally, but final action and proprio tensors must agree in width, slot order, and physical meaning. Both gripper slots use:

~~~text
-1 = fully closed
+1 = fully open
~~~

Convert quaternions, axis-angle, or Euler rotations to rot6d at the reader boundary. The rot6d blocks at slots `3:9` and `37:43` must pass through normalization unchanged. Any other rotation representation is normalized with the other non-rot6d dimensions.

The α protocol fixes min-max normalization for action and proprio:

~~~text
# Normalized range
normalized = clip(2 * (x - min) / (max - min) - 1, -1, 1)

# Rot6d identity mapping
min = -1
max = +1
~~~

Fit statistics on the training split. Set the rot6d statistics to the identity values above so those components are unchanged. If normalization runs before unified scatter, pin native rot6d dimensions before `map_to_unify`; if it runs after scatter, pin both α slot ranges. Apply the same rule to proprioception. Keep `normalization_stats.npy` with the run for deployment.

Relevant helpers are [openwam/dataloader/utils/normalization.py](../../openwam/dataloader/utils/normalization.py), [openwam/dataloader/utils/unify_action.py](../../openwam/dataloader/utils/unify_action.py), and [openwam/dataloader/transforms/rotation.py](../../openwam/dataloader/transforms/rotation.py).

## 2. Download the foundation checkpoint

Run the repository downloader:

~~~bash
python scripts/download_assets/download_openwam_checkpoints.py
~~~

Choose **OpenWAM_Alpha → OpenWAM-Alpha-Pretrain-Foundation-Model**. The Hugging Face repository is:

~~~text
OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model
~~~

Use the downloaded directory as `<foundation_ckpt_path>` and pass it to `training.finetune_ckpt_path`. A fine-tuning run writes its own output directory and config.

## 3. Launch fine-tuning

Select the reader and foundation checkpoint:

~~~bash
# Dataset, foundation model, run length, and output
NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=my_task \
  training.finetune_ckpt_path=<foundation_ckpt_path> \
  training.num_epochs=5 \
  project.output_dir=<output_dir_path>
~~~

The launcher uses `torchrun`.

Relevant memory and precision fields are in [configs/train.yaml](../../configs/train.yaml):

~~~yaml
training:
  # Precision and distributed optimizer
  mixed_precision: bf16                 # bf16, fp16, or no
  zero_stage: 2                         # ZeRO stage
  gradient_accumulation_steps: 1        # optimizer update every N micro-batches

  # Activation memory
  use_gradient_checkpointing: true      # recompute activations during backward
  use_gradient_checkpointing_offload: false  # offload checkpointed activations to CPU

  # Parameter and optimizer placement
  offload_optimizer_device: none        # none or cpu
  batch_size: 24                        # per-process micro-batch size
~~~

`finetune_ckpt_path` starts a new run at step 0 and loads weights from the source directory. `resume_ckpt_path` continues the original run and restores optimizer, scheduler, RNG, and model state; it requires `save_full_states_for_resume: true`. Keep the two paths mutually exclusive.

## 4. Deploy the fine-tuned policy

A deployable output contains its config, weights, tokenizer/component assets, and normalization artifact:

~~~bash
bash scripts/deploy.sh <ckpt_dir_path>
~~~

The benchmark client must gather the model 80-D output back to native dimensions, apply the saved inverse normalization, and preserve the same slot map and gripper convention. Runtime execution and denoising controls are documented in [train-and-deploy.md](train-and-deploy.md).
