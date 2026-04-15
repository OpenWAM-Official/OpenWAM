# OpenWAM

## TODO

- [ ] **DeepSpeed ZeRO-3 support**
- [ ] **Video-Backbone Architecture Re-Built**
- [ ] **LeRobot Dataset Combination**
- [ ] **RoboTwin2 Benchmark Support**

## What is OpenWAM

OpenWAM is an open-source framework for **World-Action Models (WAMs)**: video-diffusion policies that jointly model future visual dynamics and robot actions.

The repository is organized around the `open_wam/` package and currently supports:

- Hydra-based training, inference, and evaluation entrypoints
- WAM-specific action/video scheduling and receding-horizon execution
- multi-dataset training utilities and embodiment-aware action conversion
- benchmark adapters for RoboTwin, SimplerEnv, LIBERO, RoboCasa, Calvin, and BEHAVIOR-1K
- a policy server for robot deployment workflows

![Architecture](assets_repo/arch.png)

## What OpenWAM Focuses On

OpenWAM is not a VLA clone. Its core direction is to use a video world model as the control backbone.

- Backbone: Wan-family video diffusion models
- Action modeling: flow-matched action generation coupled to video denoising
- Strengths: temporal coherence, world-model-style rollout, flexible denoising schedules
- Primary use cases: joint video-action generation, action-only rollout, embodied evaluation, robot serving

## Repository Layout

```text
OpenWAM/
├── open_wam/
│   ├── data/          # Dataset adapters, action stats, embodiment abstraction
│   ├── models/        # WAM architectures, ActionDiT, MoE, conditioning modules
│   ├── training/      # NativeTrainer, loss modules, callbacks
│   ├── inference/     # Joint inference engine, schedule utilities, FlowMatchScheduler
│   ├── evaluation/    # Evaluator registry, benchmark adapters
│   └── serving/       # Policy server for deployment
├── scripts/           # Hydra entrypoints: train / infer / eval
├── configs/           # Hydra configs for model, data, training, eval, deploy
├── tests/             # Unit tests for core OpenWAM functionality
├── assets_repo/            # Architecture and scheduling diagrams
└── third_party/       # Vendored video pipeline (WanVideoPipeline)
```

## Support Status

### Architectures

| Architecture | Status | Description |
|---|---|---|
| `dual_system` | Supported | Separate ActionDiT with cross-attention or joint self-attention bridge |
| `moe_expert` | Supported | Shared attention + expert FFN within video DiT (BAGEL/MoT-inspired) |
| `shared_backbone` | Supported | Action tokens processed by video DiT directly (DreamZero-style) |

### Benchmarks and Deployment

| Capability | Status | Notes |
|---|---|---|
| RoboTwin offline eval | Supported | Primary documented evaluation path |
| RoboTwin online eval | Supported | Environment-dependent |
| SimplerEnv eval | Supported | Requires external environment setup |
| LIBERO eval | Supported | Requires external environment setup |
| RoboCasa eval | Supported | Requires external environment setup |
| Calvin eval | Supported | Requires external environment setup |
| BEHAVIOR-1K eval | Supported | Requires external environment setup |
| Policy server | Supported | WebSocket + HTTP deployment |

## Installation

### Base installation

```bash
pip install -e .
```

### Optional serving dependencies

The policy server can be installed via the optional `serving` extra:

```bash
pip install -e ".[serving]"
```

### Scale-oriented training controls

The training configs support separate optimizer knobs for the action
branch, the video backbone, and LoRA adapters:

- `training.action_lr`
- `training.video_lr`
- `training.lora_lr`

For large backbones, start from the dedicated preset:

```bash
python scripts/train.py training=large_backbone model/backbone=ti2v_5b
```

This preset enables gradient checkpointing, initializes the model on CPU,
and uses more conservative video-backbone learning rates for 5B-class runs.

## Quick Start

### 1. Training

Default training uses Hydra config composition from `configs/`.

```bash
python scripts/train.py \
  data.dataset_dir=/path/to/robotwin_2_0/dataset
```

Useful overrides:

```bash
python scripts/train.py \
  training=joint \
  model/backbone=vace_1_3b \
  data=robotwin_multitask \
  data.dataset_dir=/path/to/robotwin_2_0/dataset \
  data.robot=arx-x5 \
  data.variant=clean_50
```

Other common training presets:

- `training=video_only`
- `training=action_finetune`
- `training=decoupled`
- `model/backbone=ti2v_5b`
- `data=robotwin`
- `data=mixture`

### 2. Inference

```bash
python scripts/infer.py \
  inference=sync \
  inference.prompt="robot picks up the bottle" \
  inference.seed=42
```

Available inference schedules in `configs/inference/`:

- `sync` — synchronized video + action denoising
- `action_only` — action denoising only (no video generation)
- `video_leading` — video denoises ahead of action
- `cascade` — sequential video then action
- `decoupled_flash` — 1-4 step action inference (DreamZero-Flash)
- `decoupled_asymmetric` — asymmetric video/action step counts

Programmatic schedule utilities are available in `open_wam.inference.schedule`.

### 3. Evaluation

Offline RoboTwin evaluation:

```bash
python scripts/eval.py \
  eval=robotwin_offline \
  eval.ckpt_path=/path/to/checkpoint.safetensors
```

Online RoboTwin evaluation:

```bash
python scripts/eval.py \
  eval=robotwin_online \
  eval.ckpt_path=/path/to/checkpoint.safetensors
```

Additional benchmark configs:

- `eval=simpler_env`
- `eval=libero`
- `eval=robocasa`
- `eval=calvin`
- `eval=behavior`

These benchmarks require their own simulator/environment dependencies.

### 4. Deployment

The deployment module lives in `open_wam.serving.policy_server` and the default deployment config is `configs/deploy/server.yaml`.

Recommended startup path:

```bash
openwam-serve \
  --ckpt-path /path/to/checkpoint.safetensors \
  --config configs/config.yaml \
  --host 0.0.0.0 \
  --ws-port 8765 \
  --http-port 8766
```

Equivalent module invocation:

```bash
python -m open_wam.serving.policy_server \
  --ckpt-path /path/to/checkpoint.safetensors
```

Server endpoints:

- WebSocket: `ws://HOST:WS_PORT`
- HTTP `POST /predict`
- HTTP `POST /reset`
- HTTP `GET /health`
- HTTP `GET /info`

Minimal HTTP client example:

```bash
python scripts/policy_client.py \
  --server http://127.0.0.1:8766 \
  --image /path/to/frame.jpg \
  --prompt "pick up the bottle"
```

## Config System

OpenWAM uses Hydra composition rooted at `configs/config.yaml`.

Default stack:

- model: `action_dit_small`
- backbone: `vace_1_3b`
- data: `robotwin_multitask`
- training: `joint`
- inference: `sync`
- eval: `robotwin_offline`

Important config groups:

- `configs/model/` — ActionDiT size, architecture, backbone
- `configs/data/` — dataset adapters (robotwin, droid, oxe, bridge_v2, mixture)
- `configs/training/` — training presets (joint, decoupled, action_finetune, etc.)
- `configs/inference/` — denoising schedules
- `configs/eval/` — benchmark configurations
- `configs/deploy/` — policy server deployment

Run traceability:

- Training writes run artifacts under `OUTPUT_PATH/run_artifacts/<RUN_ID>/`
- Saved files include: `resolved_config.yaml`, `resolved_config.json`, `flat_args.json`, `run_metadata.json`

## Core Features

- Joint video-action denoising with configurable schedules
- Receding-horizon execution with temporal ensembling
- Three WAM architectures: dual-system, MoE expert, shared backbone
- Package-native model loading for inference, evaluation, and serving
- `MixtureDataset` for multi-dataset co-training
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
