# Training and deployment

OpenWAM composes each run from Hydra YAML. The top-level configuration is [configs/train.yaml](../../configs/train.yaml); it selects the model and dataloader, while each model group selects its component backbones. The trainer saves the resolved configuration with each checkpoint.

## 1. Choose and assemble a model

### Architecture

Select the model group in [configs/train.yaml](../../configs/train.yaml), then edit the `architecture` block in the selected model file:

- [configs/model/dual_system.yaml](../../configs/model/dual_system.yaml)
- [configs/model/single_system.yaml](../../configs/model/single_system.yaml)
- [configs/model/tri_system.yaml](../../configs/model/tri_system.yaml)

~~~yaml
# configs/train.yaml
defaults:
  # Model and data groups
  - model: dual_system       # choose dual_system, single_system, or tri_system
  - dataloader: robotwin     # choose a supported dataloader group
~~~

~~~text
# Choose one model group
model=dual_system
model=single_system
model=tri_system
~~~

All architecture options are under `architecture:`. Hydra exposes the composed values as `model.architecture.*` for CLI overrides.

~~~yaml
# configs/model/tri_system.yaml
architecture:
  # Architecture identity
  framework: tri_system                 # single_system, dual_system, or tri_system
  variant: joint_self_attn               # dual: joint_self_attn | joint_cross_attn | idm; single: vanilla | moe

  # Token geometry and inputs
  action_dim: 80                         # action-token width; match dataloader output
  use_proprioception: true               # require a proprio field in each sample
  state_dim: 80                          # proprioception width

  # Attention visibility
  attention_mask_mode: mutual            # mutual | action_sees_video | video_sees_action | isolated
  video_attention_mask_mode: first_frame_causal  # first_frame_causal | per_frame_causal | bidirectional

  # Bridge and memory controls
  bridge_layers: null                    # explicit layers; null uses bridge_interval
  bridge_interval: 1                     # use every layer when set to 1
  mot_checkpoint_mixed_attn: true        # checkpoint mixed attention to reduce GPU memory

  # Variant-specific options
  detach_bridge: false                   # DualSystem joint_cross_attn only
  idm_video_cond_noise_prob: 0.5         # IDM video-condition noise probability
  understanding_expert:                  # TriSystem understanding stream
    dim: 512                              # projected VLM feature width
    ffn_dim: 2048
    vlm_projector_type: mlp3x_silu        # linear or mlp{N}x_silu
~~~

`idm` uses teacher-forced inverse dynamics and two-stage inference. `tri_system` adds a frozen Qwen3-VL stream. Implementations and registries are in [openwam/model/architectures/](../../openwam/model/architectures/). The selected model file composes the video, action, and (for TriSystem) VLM groups through its `defaults` list.

### Video backbone

Choose a Hydra group from [configs/model/video_backbone/](../../configs/model/video_backbone/); implementations are in [openwam/model/video_backbone/](../../openwam/model/video_backbone/).

~~~text
# Choose one video group
model/video_backbone=wan22_ti2v_5b
model/video_backbone=wan21_vace_1_3b
model/video_backbone=wan21_i2v_14b_480p
model/video_backbone=cosmos_predict25_2b
model/video_backbone=cosmos3_edge
~~~

Edit the selected group file, for example [wan22_ti2v_5b.yaml](../../configs/model/video_backbone/wan22_ti2v_5b.yaml), [cosmos_predict25_2b.yaml](../../configs/model/video_backbone/cosmos_predict25_2b.yaml), or [cosmos3_edge.yaml](../../configs/model/video_backbone/cosmos3_edge.yaml):

~~~yaml
# configs/model/video_backbone/<selected_group>.yaml
# Identity and weights
name: wan22_ti2v_5b                 # registry key; use the selected backbone name
model_path: /path/to/video_backbone_checkpoint  # weights or bundle root

# Initialization and scheduler
from_scratch: false                 # rebuild Wan projections around an encoder
shift_video: 5.0                    # flow-matching scheduler shift
# Cosmos3 additionally defines fps, prompt_duration_template, and freeze_und.
~~~

The selected backbone supplies video geometry to the architecture and action backbone. Keep `model_path` consistent with the assets in [Assets Preparation](../../README.md#assets-preparation).

### VLM backbone

TriSystem selects a VLM group from [configs/model/vlm_backbone/](../../configs/model/vlm_backbone/); implementation and registry code are in [openwam/model/vlm_backbone/](../../openwam/model/vlm_backbone/).

~~~yaml
# configs/model/vlm_backbone/qwen3_vl_2b.yaml
# Identity and weights
name: qwen3_vl_2b                       # Hydra group and registry key
checkpoint_path: /path/to/qwen3_vl_2b    # pretrained VLM directory

# Loading and sequence length
load_pretrained: true                    # load checkpoint weights
max_length: 512                          # processor text limit
~~~

Override with the merged path `model.vlm_backbone.*`:

~~~text
# Select the group
model/vlm_backbone=qwen3_vl_2b

# Override its checkpoint
model.vlm_backbone.checkpoint_path=/path/to/qwen3_vl_2b
~~~

### Action backbone

DualSystem and TriSystem construct [separate_action_dit.yaml](../../configs/model/action_backbone/separate_action_dit.yaml); SingleSystem constructs [shared_action_backbone.yaml](../../configs/model/action_backbone/shared_action_backbone.yaml). Implementations are in [openwam/model/action_backbone/](../../openwam/model/action_backbone/), and there is no separate action-backbone registry.

~~~yaml
# configs/model/action_backbone/separate_action_dit.yaml
# ActionDiT dimensions
dim: 1024                        # ActionDiT hidden width
ffn_dim: 4096                    # ActionDiT feed-forward width
shift_action: 5.0                # action scheduler shift
# num_layers, video_dim, num_heads, and attn_head_dim resolve from the video backbone

---
# configs/model/action_backbone/shared_action_backbone.yaml
# Shared action decoder dimensions
action_decoder_hidden_dim: 1024  # shared action decoder width
expert_ffn_dim: 4096             # MoE expert width
~~~

Select a group with `model/action_backbone=separate_action_dit` or `model/action_backbone=shared_action_backbone`; keep its output aligned with `architecture.action_dim` and the dataloader.

### Visual encoder and S-VAE

Edit the encoder group file under [configs/model/video_backbone/encoder/](../../configs/model/video_backbone/encoder/); the selected encoder fields are merged into model.video_backbone.encoder.
When from_scratch=true, select an external encoder under configs/model/video_backbone/encoder/: wan22_vae, flux2_vae, dinov3, or vjepa21. DINOv3, FLUX.2, and V-JEPA require this path; the encoder metadata supplies the latent geometry.

~~~yaml
# Select an external encoder when rebuilding the video backbone
model:
  video_backbone:
    from_scratch: true
    encoder:
      name: dinov3                    # wan22_vae, flux2_vae, dinov3, or vjepa21
      model_path: /path/to/visual_encoder

      # Optional S-VAE reducer
      svae_path: /path/to/svae_checkpoint
      svae_target_dim: 48              # reducer output width
~~~

S-VAE is an offline feature reducer. Collect features with the complete data chain, train the reducer, then point svae_path at the resulting file. The scripts and reducer implementation are under scripts/svae_train/ and openwam/model/video_backbone/encoder/svae/.

### Dataloader selection and loading

Built-in registry names are:

~~~text
robotwin, robodojo, robocasa_gr1, ebench, libero, robocasa365, vlabench
~~~

The matching dataloader fields are in [configs/dataloader/](../../configs/dataloader/), for example [configs/dataloader/libero.yaml](../../configs/dataloader/libero.yaml).
Readers are under openwam/dataloader/ and configs under configs/dataloader/. Select one with Hydra:

~~~bash
# Select the dataloader and dataset
bash scripts/train.sh \
  dataloader=libero \
  dataloader.dataset_dir=<dataset_dir_path> \
  dataloader.unify_action=true \
  dataloader.num_frames=33 \
  dataloader.video_stride=4
~~~

Each registered class implements from_config(config, split) and returns window samples. The trainer keeps samples as a list of dictionaries. num_frames is the raw state/action window length; the action horizon is num_frames - 1. Video is subsampled by video_stride, so the standard 33/4 setting yields 9 video frames. Camera layout, transforms, normalization, action maps, and masks belong to the dataloader.

## 2. Start training

### Fresh, from-scratch, and fine-tuning runs

With a pretrained video backbone, a fresh run loads the selected components and trains the unfrozen modules from step zero. from_scratch rebuilds the Wan2.2 DiT input/output projections around an external encoder.

~~~bash
# Model and dataloader
bash scripts/train.sh model=dual_system dataloader=robotwin \
  dataloader.dataset_dir=<dataset_dir_path> \
  training.output_path=<output_dir_path> \
  training.batch_size=1 \
  training.max_steps=20
~~~

These training fields come from [configs/train.yaml](../../configs/train.yaml); CLI overrides take precedence over the file.
Use training.finetune_ckpt_path to warm-start weights into a new run. Use training.resume_ckpt_path only to continue a run that saved full Accelerate state:

~~~yaml
training:
  # Start a new run from existing weights
  finetune_ckpt_path: /path/to/base_checkpoint

  # Continue a run with its saved optimizer and scheduler state
  resume_ckpt_path: null
  save_full_states_for_resume: true
~~~

The two paths are mutually exclusive. Fine-tuning starts at step 0 in a new output directory; resume restores optimizer, scheduler, RNG, and model state in the original run.

### Memory, precision, and distributed launch

The distributed and memory fields below are defined in [configs/train.yaml](../../configs/train.yaml). `scripts/train.sh` invokes `torchrun` and detects visible GPUs.

~~~yaml
training:
  # Precision and distributed optimizer
  mixed_precision: bf16                 # bf16, fp16, or no
  zero_stage: 2                         # ZeRO stage; stage 2 also shards gradients
  gradient_accumulation_steps: 1        # optimizer update every N micro-batches

  # Activation memory
  use_gradient_checkpointing: true      # recompute activations during backward
  use_gradient_checkpointing_offload: false  # offload checkpointed activations to CPU

  # Parameter and optimizer placement
  initialize_model_on_cpu: false        # initialize on CPU before accelerator placement
  offload_optimizer_device: none        # none or cpu
~~~

Set `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, and `MASTER_ADDR` for multi-node launch.

## 3. Deploy a checkpoint

Deployment defaults are defined in [configs/deploy.yaml](../../configs/deploy.yaml). A training output contains weights, `config.yaml`, normalization data, tokenizer/component assets, and the saved model specification. Start a WebSocket server with:

~~~bash
bash scripts/deploy.sh <ckpt_dir_path>
~~~

The newest checkpoint is selected automatically. To pin a file or launch one server per GPU:

~~~bash
bash scripts/deploy.sh <ckpt_dir_path> --ckpt-name <ckpt_file_name>
NUM_GPUS=4 PORT_BASE=8848 bash scripts/deploy.sh <ckpt_dir_path>
~~~

The server accepts `obs`, `reset`, and `ping` JSON messages. Payload details are in [Benchmark integration](benchmark-integration.md).

### Execution and denoising modes

All inference fields below belong to the `inference` section of [configs/deploy.yaml](../../configs/deploy.yaml). The two mode switches are independent:

~~~yaml
inference:
  # Denoising trajectory
  denoise_steps: 10                    # denoising steps per generation
  denoise_mode: sync                   # sync or async noise trajectory

  # Async denoising alignment
  lead_modality: video                 # lead stream: video or action
  variance_shift_alpha: 1.0            # lead-curve shift; must be >= 1
  linear_offset: 0.0                   # lag; 0 <= value < 1

  # Request execution
  inference_mode: sync                 # sync execution or async prefetch
  inference_horizon: null              # actions consumed per chunk; null = full chunk
  inference_delay_steps: null          # async latency in action steps; < horizon
~~~

With `num_frames: 33`, each generated action chunk contains 32 actions. `inference_horizon` only limits how many are consumed before the next generation.

Optimization fields are in the same deploy YAML:

~~~yaml
optimization:
  # Decode and cache
  decode_video: false                  # skip VAE decode when only actions are needed
  dit_cache:
    enabled: true                      # reuse similar video velocity predictions
    cosine_threshold: 0.99              # similarity threshold for a cache hit
    max_skips: 3                        # maximum consecutive skipped video forwards

  # Compile architecture-specific paths
  compile:
    enabled: true                      # compile fixed-shape paths
    self_attn: {torch_mode: default, dynamic: false}
    cross_attn: {torch_mode: default, dynamic: false}
    idm: {torch_mode: reduce-overhead, dynamic: false}
    tri_system: {torch_mode: reduce-overhead, dynamic: false}

  # Reuse text features
  prompt_embed_cache: {enabled: true, maxsize: 32}
~~~

Compilation can add first-request warm-up latency. Source code: `scripts/train.py`, `openwam/train/openwam_trainer.py`, `scripts/deploy.py`, and `openwam/deploy/`.
