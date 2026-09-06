# Integrating a new benchmark

Benchmark integration has two adapters:

1. A dataloader emits OpenWAM sample dictionaries for training.
2. A client speaks the policy server WebSocket protocol and converts returned actions to the benchmark controller.

The benchmark owns action ordering, units, signs, ranges, and gripper meaning. Keep those definitions in the dataloader and client adapter.

## 1. Build a dataloader

### Choose a base and register it

For a custom source, subclass `BaseDataset` in [openwam/dataloader/bases/dataset.py](../../openwam/dataloader/bases/dataset.py). For LeRobot v3 data, subclass `LeRobotV3Reader` in [openwam/dataloader/bases/lerobot_v3_reader.py](../../openwam/dataloader/bases/lerobot_v3_reader.py), which provides parquet windows, video decoding, prompts, statistics, masks, and multiview composition.

Override only the source hooks that differ, such as `_action_20d`, `_proprio_20d`, `_resolve_cameras`, `_resolve_prompt`, and `_load_stats`. Register the class in [openwam/dataloader/registry.py](../../openwam/dataloader/registry.py):

~~~python
from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.registry import register_dataset

@register_dataset('my_benchmark')
class MyBenchmarkDataset(BaseDataset):
    @classmethod
    def from_config(cls, config, split='train'):
        return cls(dataset_dir=config.dataset_dir, split=split)

    def __len__(self): ...
    def __getitem__(self, index): ...
~~~

Import the module before calling `build_dataset(config)`. Every registered class must implement `from_config`; built-in readers are imported by the registry.

### Return the canonical sample

The trainer uses `collate_fn=list`, so one item remains one dictionary. Return the fields below:

~~~text
# Video and prompt
video             list[PIL.Image], length = num_video_frames
first_frame_image list[PIL.Image], normally [video[0]]
vace_video        None or list[PIL.Image]
prompt            str

# Action and state
# action width D must match architecture.action_dim
# state width D_state must match architecture.state_dim
action            float32 Tensor, shape (num_frames - 1, D)
action_mask       bool Tensor, shape (num_frames - 1, D)
proprio           float32 Tensor, shape (1, D_state)
proprio_mask      bool Tensor, shape (1, D_state)

# Video validity
video_mask        bool Tensor, shape (num_video_frames,)
~~~

For video-only data, emit zero actions with an all-false `action_mask`; zero values alone do not disable loss. Repeat the final real frame only when a source window is short and mark padded time steps false. A missing head frame should raise; an auxiliary-camera failure may be black-filled. Keep camera composition and frame stride identical between training and deployment.

### Define an action space

An action may contain joints, EEF pose, base motion, or any combination. The final action width must be consistent across readers that are mixed together; `MixtureDataset` enforces this in strict mode.

For schema-free scatter/gather, use [openwam/dataloader/utils/unify_action.py](../../openwam/dataloader/utils/unify_action.py):

~~~python
from openwam.dataloader.utils.unify_action import (
    UNIFY_DIM,
    map_to_unify,
    parse_unify_spec,
    unmap_from_unify,
)

# Build the destination map
dst = parse_unify_spec(config.unify_action_map, unify_dim=UNIFY_DIM)
unified, dim_mask = map_to_unify(raw_action, dst, unify_dim=UNIFY_DIM)
raw_again = unmap_from_unify(unified, dst)
~~~

A mapping must cover raw dimensions exactly, use every destination once, and stay within `[0, UNIFY_DIM)`. The built-in unified width is `UNIFY_DIM=80`; a non-unified reader may expose another model width. The helper assigns no slot semantics, so document physical meanings in the reader.

### Configure frames, transforms, splits, and statistics

Add a YAML file under [configs/dataloader/](../../configs/dataloader/):

~~~yaml
# configs/dataloader/my_benchmark.yaml
# Dataset and sampling
type: my_benchmark
dataset_dir: /path/to/dataset
num_frames: 33
video_stride: 4
window_stride: 1

# Image layout
height: 384
width: 320
multiview: false

# Action preprocessing
normalize_mode: min-max              # min-max, z-score, quantile, or null
~~~

Use [openwam/dataloader/transforms/builder.py](../../openwam/dataloader/transforms/builder.py) for rotation conversion, normalization, and video augmentation. Compute statistics from the training split and store them under `meta/`. A deployable reader writes `meta/normalization_stats.npy`; deployment gathers unified slots back to raw dimensions before unnormalizing. Keep the same statistics artifact with the checkpoint.

Inspect one sample through the registry:

~~~python
from omegaconf import OmegaConf
from openwam.dataloader.registry import build_dataset

config = OmegaConf.load('configs/dataloader/my_benchmark.yaml')
dataset = build_dataset(config)
sample = dataset[0]
print(len(dataset), sample['action'].shape, sample['video'])
~~~

## 2. Adapt the benchmark client

The wire protocol is implemented by `PolicyServer` in [openwam/deploy/server.py](../../openwam/deploy/server.py) and mirrored by `WSPolicyClient` in [benchmarks/utils/transport.py](../../benchmarks/utils/transport.py). The server decodes and composes images, forwards the prompt, normalizes state, and denormalizes actions. Existing adapters are under [benchmarks/](../../benchmarks/).

### Messages and lifecycle

Keep one persistent WebSocket connection. Send one JSON observation per control step:

~~~json
{
  "type": "obs",
  "images": {
    "head_camera": "<base64_png>",
    "left_wrist_camera": null,
    "right_wrist_camera": null
  },
  "prompt": "<model_prompt>",
  "state": [0.0, 0.0]
}
~~~

`head_camera` is required. Wrist fields may be omitted or null; multiview servers black-fill missing slots and single-view servers ignore them. Encode RGB PNG with `encode_numpy_b64` or `encode_path_b64` from [benchmarks/utils/client.py](../../benchmarks/utils/client.py). Do not resize unless the training reader pre-resized a tile; reproduce that layout with `resize_for_lshape_slot`.

The response is a flat action array in physical units:

~~~json
{
  "type": "action",
  "action": [0.1, -0.2],
  "step": 4,
  "latency_ms": 12.3
}
~~~

Reset at the start of every episode and after a hard control failure:

~~~json
{"type": "reset"}
~~~

~~~json
{"type": "reset_ack"}
~~~

Reset clears buffered chunks and in-flight asynchronous inference; it does not reload weights. `ping` returns a `pong` response. Malformed messages return an error object. `predict()` retries once after a dropped socket; use `predict_once()` when retrying an ambiguous action is unsafe.

Synchronous and asynchronous execution use the same wire messages. The server may serve buffered action chunks without rerunning denoising on every step.

### Convert actions at the boundary

Read the returned list in the exact order documented by the reader. Apply only benchmark-specific conversion, such as rot6d-to-quaternion or EEF-to-OSC delta; the server response is already in physical units. Preserve the training gripper sign and threshold. Existing NumPy converters are in [benchmarks/utils/action_conversion.py](../../benchmarks/utils/action_conversion.py).

Add an adapter under `benchmarks/<name>/`, document its payload and lifecycle, and keep action conversion next to the client boundary so training and deployment conventions remain visible together.
