# OpenWAM

## TODO

- [ ] **DeepSpeed ZeRO-3 support**
  - [ ] Resume-from-checkpoint under ZeRO-3 — `OpenWAMTrainer.load_checkpoint`
        currently raises when `zero_stage >= 3`. After `accelerator.prepare()`
        every parameter is sharded across ranks, so loading a flat
        safetensors needs `deepspeed.zero.GatheredParameters` (or a
        pre-`prepare()` load hook). See `openwam/train/openwam_trainer.py`.
- [ ] **LeRobot Dataset Combination**

## What is OpenWAM

OpenWAM is an open-source framework for **World-Action Models (WAMs)**: video-diffusion policies that jointly model future visual dynamics and robot actions.

The repository is organized around the `openwam/` package and currently supports:

- Hydra-based training and deployment entrypoints
- WAM-specific action/video scheduling and receding-horizon execution
- RoboTwin dataset adapter with multi-task, multi-view support
- Two WAM architecture families: dual-system and shared backbone variants
- A policy server for robot deployment workflows

![Architecture](assets_repo/arch.png)

## What OpenWAM Focuses On

OpenWAM is not a VLA clone. Its core direction is to use a video world model as the control backbone.

- Backbone: Wan-family video diffusion models
- Action modeling: flow-matched action generation coupled to video denoising
- Strengths: temporal coherence, world-model-style rollout, flexible denoising schedules
- Primary use cases: joint video-action generation, action-only rollout, robot deployment

## Repository Layout

```text
OpenWAM/
├── openwam/
│   ├── dataloader/    # Dataset adapters (RoboTwin), transforms, registry
│   ├── model/         # WAM architecture families, action backbone, video backbone
│   │   ├── action_backbone/   # ActionDiT, MoE DiT, shared components
│   │   └── video_backbone/    # Vendored video pipeline (WanVideoPipeline, VAE, DiT)
│   ├── train/         # OpenWAMTrainer, flow-match loss, checkpointing, optimizer utils
│   ├── deploy/        # Policy server, model loader, joint/mock inference engines, scheduler
│   └── utils/         # Shared utilities
├── scripts/           # Entrypoints: train.sh, deploy.sh, inference tests
├── configs/           # Hydra configs for model, dataloader, training_strategy, accelerate
├── tests/             # Unit tests
├── docs/
├── assets_repo/       # Architecture diagrams
├── references/        # Reference implementations (FastWAM)
└── benchmarks/
    └── robotwin/      # RoboTwin eval client, single_eval.sh / multi_eval.sh scripts
```

## Support Status

### Architectures

| Architecture | Status | Description |
|---|---|---|
| `dual_system` | Supported | Dual-system family with `joint_cross_attn` / `joint_self_attn` variants |
| `shared_backbone` | Supported | Shared-backbone family with `vanilla` / `moe` variants |

### Benchmarks and Evaluation

| Benchmark | Status | Notes |
|---|---|---|
| RoboTwin eval | Supported | All 50 tasks; see `benchmarks/robotwin/` |
| SimplerEnv eval | Planned | Requires external environment setup |
| LIBERO eval | Planned | Requires external environment setup |
| RoboCasa eval | Planned | Requires external environment setup |
| Calvin eval | Planned | Requires external environment setup |
| BEHAVIOR-1K eval | Planned | Requires external environment setup |

## Installation

### Base installation

```bash
conda create -n openwam python=3.12

conda activate openwam
```

We recommend using PyTorch 2.7.1 with CUDA 12.8 （others may also work）:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install OpenWAM:

```bash
pip install -e .
```

### Optional extras

The `pyproject.toml` exposes a few optional dependency sets. Pick the ones you need:

```bash
pip install -e ".[sana]"          # SANA video backbone (pulls timm; see docs/sana_vendor.md)
pip install -e ".[dev]"           # pytest + ruff (needed for `make test` / `make lint`)
pip install -e ".[npu]"           # Huawei Ascend NPU (x86_64)
pip install -e ".[npu_aarch64]"   # Huawei Ascend NPU (aarch64)
```

Extras combine — e.g. `pip install -e ".[sana,dev]"` for SANA + tests.

Tip: **do not** run `pip install --reinstall` (or `uv pip install --reinstall`) on a
single package after the base install — both resolvers will re-pin the entire
dependency graph and silently swap your `torch==2.7.1+cu128` wheel for a
different CUDA build. If you need to repair one package (e.g. the timm
`version.py` missing-file bug from upstream wheels), use `--no-deps`:

```bash
pip install --no-deps --force-reinstall 'timm>=1.0.20,<1.1'
```

### Scale-oriented training controls

The training configs support separate optimizer knobs for the action
branch, the video backbone, and LoRA adapters:

- `training.action_lr`
- `training.video_lr`
- `training.lora_lr`

## Quick Start

### 0. Data Preparation

**Download the video backbone (Wan2.2-TI2V-5B):**

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir /path/to/Wan2.2-TI2V-5B
```

**Download the RoboTwin dataset:**

Full dataset or individual task zips can be downloaded from the HuggingFace Hub. For example, to download a single task:

```bash
huggingface-cli download TianxingChen/RoboTwin2.0 \
  dataset/adjust_bottle/aloha-agilex_clean_50.zip \
  --repo-type dataset \
  --local-dir /path/to/robotwin_2_0
```

After downloading, unzip the task files:

```bash
cd /path/to/robotwin_2_0/dataset/adjust_bottle
unzip aloha-agilex_clean_50.zip
```

### 1. Training

Training uses Hydra config composition from `configs/`. Quick sanity check after data preparation:

```bash
bash scripts/train.sh \
  dataloader.dataset_dir=/path/to/robotwin_2_0/dataset \
  dataloader.task_name=adjust_bottle \
  dataloader.variant=clean_50 \
  model.video_backbone.model_path=/path/to/Wan2.2-TI2V-5B \
  training.debug=true \
  training.batch_size=1 \
  training.output_path=/path/to/your/output_dir
```

This runs a short debug training (20 steps) on a single task to verify the full pipeline works end-to-end.

For full training, remove the `training.debug=true` override and adjust other config as needed:

```bash
bash scripts/train.sh \
  dataloader.dataset_dir=/path/to/robotwin_2_0/dataset \
  model.video_backbone.model_path=/path/to/Wan2.2-TI2V-5B
```

The main config is `configs/train.yaml`. All fields can be overridden via Hydra CLI as shown above.

Training strategy presets in `configs/training_strategy/`:

- `joint.yaml` — joint video + action training
- `video_only.yaml` — video-only training (action head frozen)

Architecture configs in `configs/model/`:

- `dual_system.yaml`     # dual_system family; choose `variant: joint_self_attn|joint_cross_attn|idm`
- `shared_backbone.yaml` # shared_backbone family; choose `variant: vanilla|moe`
- `tri_system.yaml`      # tri_system family; `variant: joint_self_attn`

Video backbone is configured inline in each framework yaml under the `video_backbone:` block (default: `wan22_ti2v_5b`). Switch backbones by either editing the yaml or overriding the inline fields on CLI:

```bash
bash scripts/train.sh model=dual_system \
    model.video_backbone.name=wan21_vace_1_3b \
    model.video_backbone.model_path=/path/to/Wan2.1-VACE-1.3B
```

Tested backbones: `wan22_ti2v_5b` (default), `wan21_vace_1_3b`, `wan21_i2v_14b_480p`. The latter two are listed (commented out) in each framework yaml's `video_backbone:` block — uncomment in place to switch without retyping the model_path. ActionDiT geometry (`num_heads`, `attn_head_dim`, `video_dim`, `num_dit_layers`) is auto-resolved from the loaded backbone — no need to mirror it in the framework yaml. The ActionDiT's own `num_layers` then follows `bridge_layers` / `bridge_interval` (default `bridge_interval=1` makes it equal to the backbone layer count).

**name vs model_path are NOT auto-coupled.** All Wan variants share one adapter class, so `video_backbone.name` only drives registry dispatch; the loaded weights are decided entirely by `video_backbone.model_path`. CLI overrides must update BOTH fields together — overriding only `name` silently loads whatever `model_path` still points at. `build_training_pipeline` runs a soft normalize-and-match cross-check on `(name, model_path)` and logs a WARNING on mismatch (it does not abort, so intentional name/path ablations are still allowed).

Shared-backbone notes:

- `configs/model/shared_backbone.yaml` enables `architecture.use_proprioception: true` by default. Training and generation with this config require a proprioceptive state tensor from `sample["proprio"]` or the deployment observation state, with last dimension matching `architecture.state_dim`.
- Shared-backbone state tokens are part of the shared DiT sequence. In joint attention mode, video tokens can attend to state tokens, so proprioception conditions both action prediction and video denoising steps.
- `architecture.action_decoder_hidden_dim` is explicitly set to `1024` in the default config. If the field is omitted or set to `null`, shared vanilla and MoE backbones also resolve it to the code default `1024`.
- This PR-era shared-backbone state-token config is not strict-load compatible with older shared-backbone checkpoints that used no state encoder, `action_decoder_hidden_dim = video_dim`, or `modality_tmod_bias`. Treat those checkpoints as requiring a matching old config/code path or an explicit manual state-dict migration.

Accelerate/DeepSpeed configs in `configs/accelerate/`:

- `deepspeed_zero1.yaml`
- `deepspeed_zero2.yaml`
- `deepspeed_zero3.yaml`

### 2. Deployment

Deploy a trained checkpoint as a policy server:

```bash
bash scripts/deploy.sh /path/to/checkpoint_dir
```

This reads `configs/deploy.yaml` for base settings (device, ports, inference parameters) and the `config.yaml` saved inside the checkpoint directory for model architecture. The latest `checkpoint_step_*.safetensors` file in the directory is loaded automatically; use `--ckpt-name` to pin a specific file.

#### Self-contained checkpoints

New checkpoints are intended to be deployable from the checkpoint directory alone. The copy happens at checkpoint creation time during training, not when `scripts/deploy.py` starts; deploy is read-only with respect to checkpoint artifacts. During training, rank 0 saves:

- `config.yaml` — the full training config, including video-backbone component specs when `model.video_backbone.model_path` is readable.
- `checkpoint_step_*.safetensors` — full model weights.
- `action_stats.npy` — action normalization stats required by the current deploy loader. When `dataloader.normalize_mode` is enabled, deploy uses these stats to normalize incoming proprioceptive state and unnormalize returned actions.
- `tokenizer/google/umt5-xxl/` — copied automatically from `<model.video_backbone.model_path>/google/umt5-xxl` when available, so deploy does not need the original Wan directory just to load the tokenizer.

Deploy's video-backbone source resolution is:

1. component specs embedded in `config.yaml` (preferred), with tokenizer loaded from `<ckpt_dir>/tokenizer/google/umt5-xxl/`;
2. `model.video_backbone.model_path` from the saved config, if it is still accessible.

`video_backbone_manifest.json` is no longer a deploy fallback. A checkpoint that only carries the old manifest, without embedded `model.video_backbone.components` and without an accessible `model_path`, is not deployable by the current loader. Fully portable deployment should keep the embedded component specs plus the copied `tokenizer/` directory alongside the checkpoint.

#### Configuration

`configs/deploy.yaml` is the central configuration file for deployment. Key sections:

```yaml
checkpoint_path: /path/to/checkpoint_dir  # used when --ckpt-dir is not passed
device: cuda:0
server:
  host: "0.0.0.0"
  ws_port: 8850
  http_port: 8848

inference:
  denoise_steps: 10      # denoising steps (FastWAM-Joint deploy default)
  schedule_type: sync    # sync | video_leading | cascade | action_only | decoupled_flash | decoupled_asymmetric
  shift: 5.0

optimization:
  decode_video: false    # false = action-only mode (skip VAE decode, faster)
  schedule:
    type: null           # optional deploy-time override for inference.schedule_type
    action_steps: 4      # action denoising steps for decoupled schedules
  dit_cache:
    enabled: false       # skip video DiT recompute when velocity prediction is stable
    cosine_threshold: 0.99
    max_skips: 3
  compile:
    mode: auto           # auto | none
    self_attn:
      torch_mode: reduce-overhead
      dynamic: false     # dual_system_self_attn MoT-loop fast path
    cross_attn:
      torch_mode: reduce-overhead
      dynamic: false     # action-side fast path for dual_system_cross_attn
  prompt_embed_cache:
    maxsize: 32          # LRU cache for prompt -> text embeddings
```

Compile mode is the single public selector. `auto` reads the loaded checkpoint's
architecture config: `joint_self_attn` selects `self_attn`,
`joint_cross_attn` selects `cross_attn`, and unsupported architectures run
eager. The per-section `enabled` field is an optional architecture-level kill
switch. Legacy public modes are intentionally removed in v0.2; migrate
`default` to `auto` or `none`.

`scripts/deploy.py` / `scripts/deploy.sh` expose a small set of CLI overrides
for the fields that are commonly changed per launch:

```bash
bash scripts/deploy.sh /path/to/checkpoint_dir \
  --device cuda:1 \
  --host 0.0.0.0 \
  --ws-port 9000 \
  --http-port 9001 \
  --denoise-steps 10 \
  --schedule-type sync \
  --shift 5.0 \
  --compile-mode auto \
  --ckpt-name checkpoint_step_10000.safetensors
```

These flags map to:

- `--ckpt-dir` / positional checkpoint path → checkpoint directory (`checkpoint_path` is used only when `--ckpt-dir` is absent)
- `--ckpt-name` → specific `checkpoint_step_*.safetensors` filename
- `--device` → `device`
- `--host` / `--ws-port` / `--http-port` → `server.*`
- `--denoise-steps` / `--schedule-type` / `--shift` → `inference.*`
- `--compile-mode` → `optimization.compile.mode` (`auto` or `none`)
- `--async-mode` -> `optimization.async_inference.mode` (`none` or `vanilla`)
- `--async-execution-horizon` / `--async-inference-delay-steps` require
  async mode `vanilla` in either CLI or yaml

Architecture-specific compile paths are selected automatically from the
checkpoint config when `mode: auto`.
On dual-system architectures, the first real request may include lazy
`torch.compile` warmup latency; use `--compile-mode none` when startup latency
is more important than steady-state throughput.

All of these overrides are optional; yaml values are used when a flag is not
provided. Other deploy settings, including `optimization.decode_video`,
`optimization.schedule.*`, `optimization.dit_cache.*`,
`optimization.compile.*`, and `optimization.prompt_embed_cache.*`, are read
from `configs/deploy.yaml` in the `scripts/deploy.py` path. To change them,
edit the yaml (or use the package entrypoint's OmegaConf dotlist overrides).

`scripts/deploy.py` and the package entrypoint (`openwam-serve` / `python -m openwam.deploy.policy_server`) both load checkpoints through the same package-native `load_from_checkpoint_dir` path. The package entrypoint defaults to `configs/deploy.yaml`, merges deploy overrides on top of the saved training config, and backfills `inference.height`, `inference.width`, and `inference.num_frames` from the checkpoint's dataloader config when they are not set explicitly.

#### Mock mode (no GPU or model weights required)

`MockInferenceEngine` implements the same interface as `JointInferenceEngine` but returns random Gaussian actions after a configurable simulated latency, making it suitable for integration testing, client benchmarking, and CI environments without a GPU.

```bash
# Start a mock server (no checkpoint needed)
python scripts/deploy.py --mock --mock-action-dim 20 --mock-latency-ms 2000
```

Mock-mode options:

| Flag | Default | Description |
|---|---|---|
| `--mock` | — | Enable mock mode (skips model loading) |
| `--mock-action-dim N` | `20` | Dimensionality of the returned action vector |
| `--mock-latency-ms T` | `2000` | Simulated inference latency in milliseconds |

Once the mock server is running, all normal client scripts work against it without modification:

```bash
python scripts/inference_single_test.py --test
python scripts/inference_continuous_test.py --steps 128
```

Server endpoints:

- HTTP `POST /predict` — send 3-camera `images` dict, base prompt, and optional raw `state`; receive action in the checkpoint's deploy scale. For normalized checkpoints this is already unnormalized back to physical units.
- HTTP `POST /reset` — reset policy state between episodes
- HTTP `GET /health` — health check
- HTTP `GET /info` — model info and policy runtime config

See [benchmarks/README.md](benchmarks/README.md) for the full client payload contract.

#### Debug mode (capture server-side requests)

Pass `--debug` to `scripts/deploy.py` (or the `deploy.sh` wrapper) to save the post-preprocessing image and per-step metadata under `--debug-dir` (default `./server_debug`):

```bash
python scripts/deploy.py --ckpt-dir /path/to/ckpt_dir \
    --debug --debug-dir ./server_debug
```

Each request writes `server_debug/ep0000/step_0001/{image_processed.jpg, meta.json}` style directories. `image_processed.jpg` is the exact post-preprocessing image the model saw (single-view crop/resize or multi-view composition), and `meta.json` records the wrapped prompt, state, action, latency, server step, and episode index.

### 3. Testing the Server

Client always sends the same 3-camera payload (head required, wrists optional). Server reads the checkpoint's `config.yaml` and dispatches to single- or multi-view preprocessing automatically. The bundled test scripts also send a zero raw `state` vector by default (`--state-dim 20`); pass real proprioception with `--state ...` or `--state-file state.json`, or use `--no-state` only for checkpoints that do not use proprioceptive conditioning. See [benchmarks/README.md](benchmarks/README.md) for the full client integration guide.

**Single inference test** — verify the server returns a valid action:

```bash
# Smoke test with 3 random images (no files needed)
python scripts/inference_single_test.py --test

# With real images
python scripts/inference_single_test.py \
  --server http://127.0.0.1:8848 \
  --head-camera /path/to/head.jpg \
  --left-wrist-camera /path/to/left.jpg \
  --right-wrist-camera /path/to/right.jpg \
  --prompt "pick up the bottle"

# Head-only (wrists sent as null — server black-fills if multi-view)
python scripts/inference_single_test.py \
  --head-camera /path/to/head.jpg \
  --prompt "pick up the bottle"
```

**Continuous inference test** — simulate a real robot control loop:

```bash
python scripts/inference_continuous_test.py --steps 128
```

This simulates 128 control steps, showing how the server handles action chunking internally: the first call triggers full inference (slow, generates an entire action chunk), subsequent calls pop cached actions from the buffer (fast, <10ms), and re-inference is triggered when the buffer is exhausted.

### 4. Benchmarks Support

Evaluation adapters live under `benchmarks/`. The normal single-task and multi-task scripts connect to an **already-running** OpenWAM policy server via HTTP — no model weights are needed on the evaluator machine.

For large RoboTwin runs, `benchmarks/robotwin/dlc_parallel_eval.sh` is the DLC/multi-node entrypoint: every node starts local OpenWAM policy servers, waits for `/health`, and runs RoboTwin clients against a shared filesystem queue. Rank 0 initializes `<log_dir>/.queue.txt`, `summary.tsv`, and `run.env`; workers use directory locks (`.queue.lock.d`, `summary.lock.d`) so this works on shared filesystems where `flock` may be unreliable. At the end, rank 0 verifies the summary row count and unique `task/mode` count match the expected total. Add `--dry-run` to skip servers/simulators and only test whether DLC nodes can automatically claim and distribute tasks from the shared queue.

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/robotwin/bin/python \
ROBOTWIN_RUN_ID=run1 \
bash benchmarks/robotwin/dlc_parallel_eval.sh \
  -m all -n openwam -d /path/to/ckpt_dir \
  --denoise-steps 10 \
  all
```

Export DLC results with:

```bash
python benchmarks/robotwin/export_results_csv.py /path/to/log_dir --strict
```

The exporter reads `summary.tsv` when present, parses nested per-task logs recursively, strips ANSI color codes before matching `Success rate`, and validates duplicate/missing/extra `task/mode` rows against `run.env` in strict mode.

| Benchmark | Status | Notes |
|---|---|---|
| [RoboTwin Benchmark](benchmarks/robotwin/README.md) | Supported | All 50 RoboTwin 2.0 tasks; single-task, multi-task, DLC multi-node, and CSV export scripts |
| SimplerEnv | Planned | Requires external environment setup |
| LIBERO | Planned | Requires external environment setup |
| RoboCasa | Planned | Requires external environment setup |
| Calvin | Planned | Requires external environment setup |
| BEHAVIOR-1K | Planned | Requires external environment setup |

## Config System

OpenWAM uses Hydra composition rooted at `configs/train.yaml`.

Key config groups:

- `configs/model/` — architecture type, action backbone, video backbone
- `configs/dataloader/` — dataset adapters (RoboTwin)
- `configs/training_strategy/` — training presets (joint, video_only)
- `configs/accelerate/` — distributed training (DeepSpeed ZeRO stages)
- `configs/deploy.yaml` — policy server defaults (`checkpoint_path`, `device`, `server`, `inference`, `optimization`)

Checkpoint outputs include:

- `checkpoint_step_*.safetensors` — full model weights
- `config.yaml` — complete training config snapshot, including video-backbone component specs when available
- `action_stats.npy` — action normalization stats required by deploy; active normalization uses them for raw-state normalization and physical-unit actions
- `tokenizer/google/umt5-xxl/` — tokenizer copied automatically from the Wan model directory for self-contained deployment

## Core Features

- Joint video-action denoising with configurable schedules (`sync`, `video_leading`, `cascade`, `action_only`, `decoupled_flash`, `decoupled_asymmetric`)
- Receding-horizon execution with temporal ensembling
- Two WAM architecture families: dual-system and shared backbone variants
- Package-native model loading for inference and serving (no dependency on training infrastructure at deploy time)
- Proprioceptive conditioning module for robot state input
- RoboTwin benchmark adapter (see `benchmarks/robotwin/`)
- WebSocket + HTTP policy server with unified 3-camera client contract; server handles all preprocessing and prompt wrapping from the checkpoint's saved config
- Mock inference engine for GPU-free integration testing and CI

## Development

Run the core test suite:

```bash
make test
```

Full validation (compile check + tests):

```bash
make check
```

Lint and format:

```bash
make lint      # check for issues
make format    # auto-fix formatting
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR guidelines.

## Acknowledgements

OpenWAM builds on ideas and components from:

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [StarVLA](https://github.com/starVLA/starVLA)

## License

OpenWAM is released under the MIT License. See `LICENSE`.

## Citation

If you use OpenWAM, please cite the repository directly:

```bibtex
@misc{openwam2026,
  title        = {OpenWAM: A Modular Open-Source Library for Systematic WAM Training, Inference and Deployment},
  author       = {OpenWAM Contributors},
  year         = {2026},
  url          = {https://github.com/KraHsu/OpenWAM},
  howpublished = {GitHub repository}
}
```
