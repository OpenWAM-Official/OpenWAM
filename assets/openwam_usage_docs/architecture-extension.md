# Extending OpenWAM architectures and backbones

OpenWAM keeps orchestration in an architecture and component behavior in backbones. The dependency direction is one way:

~~~text
architecture  ──calls──>  video / action / VLM backbones
~~~

A backbone must not import or call an architecture. An architecture selects backbones through registries, composes their public contracts, and owns the forward pass.

Source locations:

- Architectures and shared lifecycle: [openwam/model/architectures/](../../openwam/model/architectures/)
- Video backbones: [openwam/model/video_backbone/](../../openwam/model/video_backbone/)
- Visual encoders and S-VAE: [openwam/model/video_backbone/encoder/](../../openwam/model/video_backbone/encoder/)
- Action backbones: [openwam/model/action_backbone/](../../openwam/model/action_backbone/)
- VLM backbones: [openwam/model/vlm_backbone/](../../openwam/model/vlm_backbone/)
- Hydra model groups: [configs/model/](../../configs/model/)

The current registry keys are:

~~~text
# Video backbone
wan22_ti2v_5b
wan21_vace_1_3b
wan21_i2v_14b_480p
cosmos_predict25_2b
cosmos3_edge

# Visual encoder
wan22_vae
dinov3
vjepa21
flux2_vae

# VLM backbone
qwen3_vl_2b

# Architecture
single_system_vanilla
single_system_moe
dual_system_cross_attn
dual_system_self_attn
dual_system_idm
tri_system_joint_self_attn
~~~

Architecture keys are resolved from `architecture.framework` and `architecture.variant`. A new registration should declare its current support status and fail clearly when a required capability is unavailable.

## Add a video DiT backbone

Subclass `VideoBackbone` in [openwam/model/video_backbone/base.py](../../openwam/model/video_backbone/base.py), implement the execution lifecycle, and register the class in [openwam/model/video_backbone/__init__.py](../../openwam/model/video_backbone/__init__.py).

~~~python
from openwam.model.video_backbone import VideoBackbone, BlockLoopState, register_video_backbone

@register_video_backbone('my_video')
class MyVideoBackbone(VideoBackbone):
    # Construction and geometry
    @classmethod
    def from_pretrained(cls, source, **kwargs): ...
    @property
    def dim(self) -> int: ...
    @property
    def num_layers(self) -> int: ...
    @property
    def num_heads(self) -> int: ...
    @property
    def head_dim(self) -> int: ...
    @property
    def scheduler(self): ...

    # Training and block loop
    def preprocess_input_for_train(self, *, frames=None, text=None, **kwargs) -> dict: ...
    def prepare(self, **pipeline_inputs) -> BlockLoopState: ...
    def run_block(self, block_id, state): ...
    def finalize(self, state): ...

    # Deployment input path
    def preprocess_input_for_inference(self, **kwargs) -> dict: ...
~~~

The execution order is `prepare → run_block → finalize`. `prepare` returns a `BlockLoopState` containing hidden tokens, timestep data, RoPE, context, and grid metadata. `finalize` returns video noise with shape `(B, C, T, H, W)`. Preserve gradient-checkpointing data in the state.

Expose structural metadata used by token geometry and masks:

~~~python
# openwam/model/video_backbone/<my_video>.py
self._dit_patch_size = (1, 2, 2)  # temporal, height, width
self._temporal_compression = 4
self._causal_temporal = True
self._shift_video = 5.0            # scheduler shift, when supported
~~~

Implement only the capability hooks that the backbone supports:

~~~text
# MoT
pre_attn_at_layer / post_attn_at_layer

# IDM
merge_idm_video_branches / split_idm_video_branches

# SingleSystem
inject_shared_tokens / extract_shared_tokens

# Masks and deployment
build_video_to_video_mask
decode_video / save_deploy_assets
~~~

Add a matching Hydra group under [configs/model/video_backbone/](../../configs/model/video_backbone/):

~~~yaml
# configs/model/video_backbone/my_video.yaml
name: my_video                         # registry key
model_path: /path/to/video_backbone_checkpoint  # weights or bundle root
~~~

Use `build_video_backbone('my_video', cfg)` to instantiate it. Keep architecture code on the public backbone contract and do not reach into implementation details such as `backbone.dit`.

## Add a visual encoder and optional S-VAE

A visual encoder supplies latents when `model.video_backbone.from_scratch` is true. Implement and register `VideoEncoder` in [openwam/model/video_backbone/encoder/base.py](../../openwam/model/video_backbone/encoder/base.py):

~~~python
from openwam.model.video_backbone.encoder import (
    VideoEncoder,
    VideoEncoderProperties,
    register_video_encoder,
)

@register_video_encoder('my_encoder')
class MyEncoder(VideoEncoder):
    @property
    def properties(self) -> VideoEncoderProperties: ...

    # Input and latent paths
    def preprocess_video(self, frames): ...       # -> (B, 3, T, H, W)
    def batch_encode(self, video): ...            # -> (B, z_dim, T', H', W')

    @classmethod
    def from_pretrained(cls, model_path, **kwargs): ...
~~~

`properties` is the source of truth for latent width, compression, causality, pixel decode support, and DiT patch geometry. If `pixel_decode` is false, `decode_video` and `generate(decode_video=true)` must raise a clear error.

~~~yaml
# configs/model/video_backbone/encoder/my_encoder.yaml
name: my_encoder                         # encoder registry key
model_path: /path/to/visual_encoder      # encoder weights
svae_path: /path/to/svae_checkpoint      # optional reducer weights
svae_target_dim: 48                      # reducer output width
~~~

Select it from the video group:

~~~yaml
# configs/model/video_backbone/<selected_group>.yaml
from_scratch: true
encoder:
  name: my_encoder
~~~

For an S-VAE reducer, use [openwam/model/video_backbone/encoder/svae/reducer.py](../../openwam/model/video_backbone/encoder/svae/reducer.py), route `batch_encode` through `reducer.reduce`, and check `svae_target_dim` against the reducer output width. S-VAE inference is frozen and uses deterministic `encode_mean`. Train a reducer with [scripts/svae_train/](../../scripts/svae_train/), then set `svae_path` to its output.

A changed latent channel count requires `from_scratch: true`. Encoders with non-weight assets must implement `save_deploy_assets` and `from_skeleton` so deployment can reconstruct them from checkpoint-local sidecars.

## Add a VLM backbone

VLM backbones provide frozen features to TriSystem. Subclass `VlmBackbone` in [openwam/model/vlm_backbone/base.py](../../openwam/model/vlm_backbone/base.py), register it in [openwam/model/vlm_backbone/](../../openwam/model/vlm_backbone/), and expose `hidden_size`.

~~~python
from openwam.model.vlm_backbone import VlmBackbone, register_vlm_backbone

@register_vlm_backbone('my_vlm')
class MyVLM(VlmBackbone):
    @property
    def hidden_size(self) -> int: ...

    # Input preparation and feature extraction
    def prepare_vlm_inputs(self, prompts, images) -> dict: ...
    def batch_vlm_inputs(self, vlm_inputs): ...
    def extract_features(self, vlm_inputs): ...  # (B, L, hidden_size)
~~~

Select it through a Hydra group without editing TriSystem:

~~~yaml
# configs/model/vlm_backbone/my_vlm.yaml
name: my_vlm                         # Hydra group and registry key
checkpoint_path: /path/to/vlm_checkpoint
~~~

Keep the VLM frozen through the model-level `freeze` list. Qwen3-VL uses `vlm_backbone.vlm_model`. Implement `save_deploy_assets` when a processor or tokenizer must travel with the checkpoint. The architecture projects `hidden_size` into its trainable `understanding_expert`.

## Add an action backbone

The base contracts are in [openwam/model/action_backbone/base.py](../../openwam/model/action_backbone/base.py):

~~~text
# SingleSystem
SharedActionBackbone: encode(noisy_actions, timestep), encode_state(proprio), and inherited decode head

# DualSystem and TriSystem
ActionDiTBackbone: num_layers, num_heads, head_dim, bridge forward,
                  prepare_state, pre_attn_at_layer, post_attn_at_layer,
                  extract_prediction
~~~

There is no separate action-backbone registry. Each architecture constructs its action implementation directly. Keep the constructor compatible with the matching group under [configs/model/action_backbone/](../../configs/model/action_backbone/).

~~~yaml
# configs/model/action_backbone/separate_action_dit.yaml
dim: 1024                         # ActionDiT hidden width
ffn_dim: 4096                     # feed-forward width
shift_action: 5.0                 # action scheduler shift
# num_layers, video_dim, num_heads, and attn_head_dim resolve from video geometry

---
# configs/model/action_backbone/shared_action_backbone.yaml
action_decoder_hidden_dim: 1024   # shared action decoder width
expert_ffn_dim: 4096              # MoE expert width
~~~

`ActionDiT` expects one `bridge_layers` entry per action block and an even positive `attn_head_dim`. MoT requires video and action layer/head geometry to match. Keep actions and predictions shaped `(B, T_action, action_dim)` and preserve scheduler and `shift_action` behavior through the base interface.

## Add an architecture

Subclass `BaseWAMArchitecture` in [openwam/model/architectures/base.py](../../openwam/model/architectures/base.py) and register a canonical name in the relevant [openwam/model/architectures/](../../openwam/model/architectures/) package:

~~~python
from openwam.model.architectures import BaseWAMArchitecture, register_architecture

@register_architecture(
    'my_architecture',
    framework='my_framework',
    variant='default',
    status='experimental',
    note='Describe current limitations.',
)
class MyArchitecture(BaseWAMArchitecture):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.action_backbone = ...

    def forward(self, noisy_actions, action_timestep, *, timestep=None, **inputs):
        ...  # -> (video_noise, action_noise_or_none)
~~~

The forward method returns `(video_noise_pred, action_noise_pred_or_none)`. Use public backbone methods, set architecture-owned trainable modules explicitly, and fail clearly when a required capability is absent. Import the implementation from the family `__init__.py` so the decorator runs.

Add a model YAML with `architecture.framework` and `architecture.variant`. [openwam/model/architectures/registry.py](../../openwam/model/architectures/registry.py) maps that pair to the canonical name. Keep mask rules in the architecture. A backbone should expose geometry and mask builders as methods and must not inspect architecture configuration or import an architecture module.

For runtime settings, see [Training and deployment](train-and-deploy.md). For dataset and client adapters, see [Benchmark integration](benchmark-integration.md).
