# OpenWAM

## TODO

- [ ] **DeepSpeed ZeRO-3 support**
- [ ] **Video-Backbone Architecture Re-Built**
- [ ] **LeRobot Dataset Combination**
- [ ] **RoboTwin2 Benchmark Support**

## What is OpenWAM

OpenWAM is an open-source framework for **World-Action Models (WAMs)**: video-diffusion policies that jointly model future visual dynamics and robot actions.

The repository is organized around the `openwam/` package and currently supports:

- Hydra-based training and deployment entrypoints
- WAM-specific action/video scheduling and receding-horizon execution
- RoboTwin dataset adapter with multi-task, multi-view support
- Three WAM architectures: dual-system, MoE expert, shared backbone
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
│   ├── model/         # WAM architectures (DualSystem, MoE, SharedBackbone), ActionDiT
│   │   ├── action_model/      # ActionDiT, MoE expert, components, proprioceptive encoder
│   │   └── video_backbone/    # Vendored video pipeline (WanVideoPipeline, VAE, DiT)
│   ├── train/         # OpenWAMTrainer, flow-match loss, checkpointing, optimizer utils
│   ├── deployment/    # Policy server, model loader, joint/mock inference engines, scheduler
│   └── utils/         # Shared utilities
├── scripts/           # Entrypoints: train.sh, deploy.sh, inference tests
├── configs/           # Hydra configs for model, dataloader, training_strategy, accelerate
├── tests/             # Unit tests
├── assets_repo/       # Architecture diagrams
├── references/        # Reference implementations (FastWAM)
└── benchmarks/
    └── robotwin/      # RoboTwin eval client, single_eval.sh / multi_eval.sh scripts
```

## Support Status

### Architectures

| Architecture | Status | Description |
|---|---|---|
| `dual_system` | Supported | Separate ActionDiT with cross-attention or joint self-attention bridge |
| `moe_expert` | Supported | Shared attention + expert FFN within video DiT (BAGEL/MoT-inspired) |
| `shared_backbone` | Supported | Action tokens processed by video DiT directly (DreamZero-style) |

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

We recommend using PyTorch 2.7.1 with CUDA 12.8 （others may also work）:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install OpenWAM:

```bash
pip install -e .
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

- `dual_system.yaml`
- `moe_expert.yaml`
- `shared_backbone.yaml`

Accelerate/DeepSpeed configs in `configs/accelerate/`:

- `deepspeed_zero1.yaml`
- `deepspeed_zero2.yaml`
- `deepspeed_zero3.yaml`

### 2. Deployment

Deploy a trained checkpoint as a policy server:

```bash
bash scripts/deploy.sh /path/to/checkpoint_dir
```

This reads `configs/deployment.yaml` for base settings (device, ports, inference parameters) and the `config.yaml` saved inside the checkpoint directory for model architecture. The latest checkpoint in the directory is loaded automatically.

#### Configuration

`configs/deployment.yaml` is the central configuration file for deployment. Key sections:

```yaml
deployment:
  checkpoint_path: /path/to/checkpoint_dir  # used when --ckpt-dir is not passed
  device: cuda:0
  server:
    host: "0.0.0.0"
    ws_port: 8850
    http_port: 8848

inference:
  denoise_steps: 20      # denoising steps
  schedule_type: sync    # sync | cascade | decoupled_flash | decoupled_asymmetric
  cfg_scale: 1.0         # 1.0 = CFG disabled (recommended for robotics)
  shift: 5.0

deploy:
  decode_video: false    # false = action-only mode (skip VAE decode, faster)
  compile:
    enabled: true        # torch.compile ActionDiT (~30s one-time JIT warmup)
    video_dit: true      # torch.compile Video DiT blocks — default on; pay ~3-5 min
                         # CUDA Graph capture on first inference. Set false for
                         # short runs where warmup > per-step saving.
    vae: false           # torch.compile VAE decoder (only effective when tiled=false)
```

CLI flags override the yaml values for their respective fields:

```bash
bash scripts/deploy.sh /path/to/checkpoint_dir \
  --device cuda:1 \
  --ws-port 9000 \
  --http-port 9001 \
  --denoise-steps 10 \
  --ckpt-name checkpoint_step_10000.safetensors
```

All inference overrides (`--denoise-steps`, `--schedule-type`, `--cfg-scale`, `--shift`) are optional; the yaml values are used when they are not provided.

#### Mock mode (no GPU or model weights required)

`MockInferenceEngine` implements the same interface as `JointInferenceEngine` but returns random Gaussian actions immediately, making it suitable for integration testing, client benchmarking, and CI environments without a GPU.

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
python scripts/inference_continuous_test.py --steps 100
```

Server endpoints:

- HTTP `POST /predict` — send 3-camera `images` dict + base prompt, receive action (already denormalized to physical units)
- HTTP `POST /reset` — reset policy state between episodes
- HTTP `GET /health` — health check
- HTTP `GET /info` — model info and config

See [benchmarks/README.md](benchmarks/README.md) for the full client payload contract.

#### Debug mode (capture server-side requests)

Pass `--debug` to `scripts/deploy.py` (or the `deploy.sh` wrapper) to save the post-preprocessing image and per-step metadata under `--debug-dir` (default `./server_debug`):

```bash
python scripts/deploy.py --ckpt-dir /path/to/ckpt_dir \
    --debug --debug-dir ./server_debug
```

Each request writes `server_debug/ep<N>/step_<N>/{image_processed.jpg, meta.json}` — useful when debugging client/server contract issues.

### 3. Testing the Server

Client always sends the same 3-camera payload (head required, wrists optional). Server reads the checkpoint's `config.yaml` and dispatches to single- or multi-view preprocessing automatically. See [benchmarks/README.md](benchmarks/README.md) for the full client integration guide.

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
python scripts/inference_continuous_test.py --steps 100
```

This simulates 100 control steps, showing how the server handles action chunking internally: the first call triggers full inference (slow, generates an entire action chunk), subsequent calls pop cached actions from the buffer (fast, <10ms), and re-inference is triggered when the buffer is exhausted.

### 4. Benchmarks Support

Evaluation adapters live under `benchmarks/`. Each adapter connects to an **already-running** OpenWAM policy server via HTTP — no model weights are needed on the evaluator machine.

| Benchmark | Status | Notes |
|---|---|---|
| [RoboTwin Benchmark](benchmarks/robotwin/README.md) | Supported | All 50 RoboTwin 2.0 tasks; single-task & multi-task eval scripts |
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
- `configs/deployment.yaml` — policy server defaults

Checkpoint outputs include:

- `checkpoint_step_*.safetensors` — full model weights
- `config.yaml` — complete training config snapshot (self-contained for deployment)

## Core Features

- Joint video-action denoising with configurable schedules (sync, cascade, video-leading, action-only, decoupled)
- Receding-horizon execution with temporal ensembling
- Three WAM architectures: dual-system, MoE expert, shared backbone
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
