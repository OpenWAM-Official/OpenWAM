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
│   ├── deployment/    # Policy server, model loader, joint inference engine, scheduler
│   └── utils/         # Shared utilities
├── scripts/           # Entrypoints: train.sh, deploy.sh, inference tests
├── configs/           # Hydra configs for model, dataloader, training_strategy, accelerate
├── tests/             # Unit tests
├── assets_repo/       # Architecture diagrams
├── references/        # Reference implementations (FastWAM)
└── benchmarks/        # Benchmark adapters (WIP)
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
| RoboTwin eval | Supported | Multi-task, multi-view, multi-variant |
| SimplerEnv eval | Planned | Requires external environment setup |
| LIBERO eval | Planned | Requires external environment setup |
| RoboCasa eval | Planned | Requires external environment setup |
| Calvin eval | Planned | Requires external environment setup |
| BEHAVIOR-1K eval | Planned | Requires external environment setup |

### Datasets

| Dataset | Status | Notes |
|---|---|---|
| RoboTwin | Supported | Multi-task, multi-view, multi-variant |

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

This reads the `config.yaml` saved alongside checkpoints and starts an HTTP policy server. The latest checkpoint in the directory is loaded automatically.

Options:

```bash
bash scripts/deploy.sh /path/to/checkpoint_dir \
  --host 0.0.0.0 \
  --http-port 8766 \
  --ckpt-name checkpoint_step_10000.safetensors
```

Server endpoints:

- HTTP `POST /predict` — send image + prompt, receive action
- HTTP `POST /reset` — reset policy state
- HTTP `GET /health` — health check

### 3. Testing the Server

**Single inference test** — verify the server returns a valid action:

```bash
# Smoke test with a random image
python scripts/inference_single_test.py --test

# With a real image
python scripts/inference_single_test.py \
  --server http://127.0.0.1:8766 \
  --image /path/to/frame.jpg \
  --prompt "pick up the bottle"
```

**Continuous inference test** — simulate a real robot control loop:

```bash
python scripts/inference_continuous_test.py --steps 100
```

This simulates 100 control steps, showing how the server handles action chunking internally: the first call triggers full inference (slow, generates an entire action chunk), subsequent calls pop cached actions from the buffer (fast, <10ms), and re-inference is triggered when the buffer is exhausted.

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

- Joint video-action denoising with configurable schedules
- Receding-horizon execution with temporal ensembling
- Three WAM architectures: dual-system, MoE expert, shared backbone
- Package-native model loading for inference, evaluation, and serving
- MixtureDataset for multi-dataset co-training
- Embodiment-aware action conversion for cross-robot use
- Proprioceptive conditioning module for robot state input
- Evaluator registry with 7 benchmark adapters
- WebSocket + HTTP policy server for deployment workflows

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
