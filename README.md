# OpenWAM

<!--
## What is OpenWAM

OpenWAM is an open-source framework for **World-Action Models (WAMs)**: video-diffusion policies that jointly model future visual dynamics and robot actions.

The repository is organized around the `openwam/` package and currently supports:

- Hydra-based training and deployment entrypoints
- WAM-specific action/video scheduling and receding-horizon execution
- RoboTwin dataset adapter with multi-task, multi-view support
- Two WAM architecture families: dual-system and single system variants
- A policy server for robot deployment workflows

## What OpenWAM Focuses On

OpenWAM is not a VLA clone. Its core direction is to use a video world model as the control backbone.

- Backbone: Wan-family video diffusion models
- Action modeling: flow-matched action generation coupled to video denoising
- Strengths: temporal coherence, world-model-style rollout, flexible denoising schedules
- Primary use cases: joint video-action generation, action-only rollout, robot deployment
-->

## Repository Layout

```text
OpenWAM/
├── openwam/
│   ├── dataloader/    # Dataset adapters (RoboTwin), transforms, processors, registry
│   ├── model/
│   │   ├── architectures/    # WAM families: dual_system, single_system, tri_system
│   │   ├── action_backbone/  # ActionBackbone ABCs, separate ActionDiT, shared action backbone,
│   │   │                     #   latent action encoder/decoder, scheduler
│   │   ├── video_backbone/   # VideoBackbone ABC, Wan backbones, encoder/ (VAE / DINOv3 / V-JEPA 2.1)
│   │   └── vlm_backbone/     # VlmBackbone ABC, Qwen3-VL backbone
│   ├── train/         # OpenWAMTrainer, flow-match loss, checkpointing, optimizer utils
│   └── deploy/        # Policy server, model loader, inference engine, executors, optimizations
├── scripts/           # Entrypoints: train.sh, deploy.sh, inference tests, SVAE / LAPA tooling
├── configs/           # Hydra configs for model, dataloader, training, deploy
├── tests/             # Unit tests
├── benchmarks/
│   ├── robotwin/      # RoboTwin eval client, single / multi / DLC-parallel eval scripts
│   ├── libero/        # LIBERO WebSocket eval client
│   ├── robocasa365/   # RoboCasa365 native-action eval client
│   ├── robocasa_gr1/  # RoboCasa GR1 tabletop eval client
│   └── vlabench/      # VLABench eval client, single / multi-GPU track sweeps
├── assets/            # Base-model checkpoints (created by the download script; git-ignored)
└── third_party/       # Vendored externals (Cosmos-Predict2.5 submodule)
```

## Support Status

### Architectures

| Architecture | Variant | Description |
|---|---|---|
| `single_system` | `vanilla` | Single shared DiT carries video + action + state tokens in one sequence |
| `single_system` | `moe` | Shared DiT with mixture-of-experts FFN layers (expert FFN on the bridge layers) |
| `dual_system` | `joint_self_attn` | Separate ActionDiT + video DiT, fused per layer via one mixed self-attention (MoT driver). |
| `dual_system` | `joint_cross_attn` | Video DiT runs to completion → bridge features → ActionDiT runs once with cross-attention to them. Sub-variants via `detach_bridge`: `false` lets action gradients flow back into the video DiT, `true` blocks them (ActionDiT trains on detached video features) |
| `dual_system` | `idm` | Inverse-dynamics-style teacher-forcing training + two-stage inference; Wan, Cosmos-Predict2.5 and Cosmos3-Edge |
| `tri_system` | `joint_self_attn` | Adds a frozen VLM understanding expert to the joint self-attention sequence (`[video + action + understanding]`) |

All architectures are selected via `configs/model/<framework>.yaml` with `architecture.variant`. The video backbone is composed from the Hydra `video_backbone` group (default `wan22_ti2v_5b`).

### Benchmarks and Evaluation

| Benchmark | Status | Notes |
|---|---|---|
| RoboTwin eval | Supported | All 50 tasks; see `benchmarks/robotwin/` |
| SimplerEnv eval | Planned | Requires external environment setup |
| LIBERO eval | Supported | See `benchmarks/libero/` |
| RoboCasa365 eval | Supported | Native state19/action15 contract; see `benchmarks/robocasa365/` |
| RoboCasa GR1 eval | Supported | GR1 tabletop tasks; see `benchmarks/robocasa_gr1/` |
| VLABench eval | Supported | 10 primitive tasks across 6 evaluation tracks; see `benchmarks/vlabench/` |
| Calvin eval | Planned | Requires external environment setup |

## Installation

Create an environment with **conda**:

```bash
# Requires Python >= 3.10
conda create -n openwam python=3.10
conda activate openwam
```

or with **venv**:

```bash
# Requires Python >= 3.10 (check with `python3 --version`)
python3 -m venv .venv
source .venv/bin/activate
```

We recommend using PyTorch 2.7.1 with CUDA 12.8 (others may also work):

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install OpenWAM:

```bash
pip install -e .
```

<details>
<summary><b>Cosmos-Predict2.5 Extras (Optional)</b> — needed only for experiments with the <code>cosmos_predict25</code> video backbone</summary>

With your environment activated:

```bash
git submodule update --init third_party/cosmos-predict2.5
bash scripts/install_cosmos_predict25.sh
```

The script installs the upstream cosmos packages into the active environment and compiles `transformer-engine` (CUDA toolkit with `nvcc` required), then automatically restores the package versions OpenWAM pins.

</details>

## Assets Preparation

### 1. Download the Video Backbone

Run the interactive downloader to fetch the checkpoint you need. It saves the
weights under `assets/video_backbone_ckpt/` (or a directory you choose) and
points the matching config under `configs/model/video_backbone/` at the
download automatically:

```bash
python scripts/download_assets/download_video_backbone.py
```

Available models: `Wan2.2-TI2V-5B`, `Wan2.1-VACE-1.3B`, `Wan2.1-I2V-14B-480P`,
`Cosmos-Predict2.5-2B` (HuggingFace only; also fetches `Cosmos-Reason1-7B` as
its text encoder), and `Cosmos3-Edge` — each available from HuggingFace or
ModelScope.

<details>
<summary>Example session: downloading Wan2.2-TI2V-5B from HuggingFace</summary>

```text
$ python scripts/download_assets/download_video_backbone.py
OpenWAM video-backbone checkpoint downloader

Storage location
  Default: /path/to/OpenWAM/assets/video_backbone_ckpt
Storage path (press Enter for the default):            # press Enter
Created default directory /path/to/OpenWAM/assets/video_backbone_ckpt

Select the model to download
  (1) Wan2.2-TI2V-5B
  (2) Wan2.1-VACE-1.3B
  (3) Wan2.1-I2V-14B-480P
  (4) Cosmos-Predict2.5-2B
  (5) Cosmos3-Edge
Model number: 1                                        # type 1

Select the download source
  (1) huggingface
  (2) modelscope
Source number: 1                                       # type 1

Wan2.2-TI2V-5B needs about 34.2 GB under /path/to/OpenWAM/assets/video_backbone_ckpt.
Start the download? [Y/n]                              # press Enter to confirm
Downloading Wan-AI/Wan2.2-TI2V-5B -> /path/to/OpenWAM/assets/video_backbone_ckpt/Wan2.2-TI2V-5B
Fetching 23 files: 100%|██████████████████| 23/23 [12:41<00:00, 33.1s/it]

Done. Wan2.2-TI2V-5B is saved under:
  /path/to/OpenWAM/assets/video_backbone_ckpt/Wan2.2-TI2V-5B
Updated configs/model/video_backbone/wan22_ti2v_5b.yaml: model_path -> /path/to/OpenWAM/assets/video_backbone_ckpt/Wan2.2-TI2V-5B
```

Interrupted or partial downloads resume automatically on the next run.

</details>

### 2. Download the Benchmark Data

Run the interactive downloader to fetch the benchmark you need. It saves the
data under `assets/benchmark_data/<benchmark>/` (or a directory you choose),
verifies the in-dataset normalization stats (computing them on the spot when
the source ships none), and points `dataset_dir` in the matching config under
`configs/dataloader/` at the download automatically:

```bash
python scripts/download_assets/download_benchmark_data.py
```

Available benchmarks: `RoboTwin2.0`, `RoboDojo`, `RoboDojo-Real`, `LIBERO`,
`VLABench`, `EBench`, `RoboCasa365`, `RoboCasa_GR1`. RoboTwin2.0 comes from
the official upstream zips (`aloha-agilex` embodiment) and is unpacked — with
the archives cleaned up — automatically.

### 3. Download the VLM Backbone (Optional)

Only the `tri_system` architecture consumes a VLM backbone. The downloader
saves the weights under `assets/vlm_backbone_ckpt/` and updates
`configs/model/vlm_backbone/` accordingly:

```bash
python scripts/download_assets/download_vlm_backbone.py
```

### 4. Download the Visual Encoders (Optional)

Only needed for video-backbone variants that plug in an external visual
encoder (`configs/model/video_backbone/encoder/`). The downloader saves the
weights under `assets/visual_encoder_ckpt/` and updates the encoder configs
accordingly:

```bash
python scripts/download_assets/download_visual_encoder.py
```

### 5. Download Released OpenWAM Checkpoints (Optional)

Unlike steps 1-4, which fetch the components for training your own model,
this downloader fetches a **finished OpenWAM checkpoint** from our public
[collections](https://huggingface.co/OpenWAM) — the OpenWAM-Alpha releases or
the OpenWAM-Study ablations.

If you just want to test or finetune from a
released checkpoint, you can skip the component downloads above entirely:
every checkpoint directory is self-contained and deploys as-is, or serves as
a finetuning start by setting `training.finetune_ckpt_path` in
`configs/train.yaml` to the downloaded path (finetuning still needs the
benchmark data from step 2):

```bash
python scripts/download_assets/download_openwam_checkpoints.py
```

Checkpoints are saved under `assets/openwam_ckpt/openwam_alpha/` or
`assets/openwam_ckpt/openwam_study/<type>/` (or a directory you choose); no
config is rewritten. Deploy one directly with
`bash scripts/deploy.sh <download_dir>`.

## Quick Start

### 1. Training

Training uses Hydra composition rooted at `configs/train.yaml`; all fields are overridable on the CLI. Quick debug run (20 steps, single task, full pipeline end-to-end):

```bash
bash scripts/train.sh \
  dataloader.dataset_dir=/path/to/robotwin_2_0/dataset \
  dataloader.task_name=adjust_bottle \
  dataloader.variant=clean_50 \
  training.debug=true \
  training.batch_size=1 \
  training.output_path=/path/to/output_dir
```

Drop `training.debug=true` for a full run. Loss weights (`lambda_video` / `lambda_action`) live in `configs/train.yaml`; each architecture's frozen pretrained components are its `configs/model/*.yaml` top-level `freeze:` list.

**Architecture** is picked via `model=<framework>` (`dual_system` | `single_system` | `tri_system`) and `model.architecture.variant` (see the [Architectures](#architectures) table).

**Video backbone** is a Hydra group composed under each framework yaml (default `wan22_ti2v_5b`). Switch via `model/video_backbone=`:

```bash
bash scripts/train.sh model=dual_system \
    model/video_backbone=wan21_vace_1_3b
```

Available groups: `wan22_ti2v_5b` (Wan2.2-TI2V-5B, default), `wan21_vace_1_3b` (Wan2.1-VACE-1.3B), `wan21_i2v_14b_480p` (Wan2.1-I2V-14B-480P), `cosmos_predict25`, `cosmos3_edge` (Cosmos3-Edge 4B). Each group ships its own `model_path`; override `model.video_backbone.model_path=` only to point at a different weights dir. ActionDiT geometry (`num_heads`, `head_dim`, `video_dim`, `num_layers`) is auto-resolved from the loaded backbone — no need to mirror it in the yaml; ActionDiT depth then follows `bridge_layers` / `bridge_interval`.

> **Wan:** `video_backbone.name` only drives registry dispatch — the loaded weights are decided entirely by `video_backbone.model_path`. Override **both** together; the builder logs a WARNING (not an error) on a mismatched `(name, model_path)`.
>
> **Cosmos-Predict2.5:** requires the optional cosmos extras — see [Installation](#installation). `name` is validated (only `cosmos_predict25_2b` today; others raise), and the weights are located by `model_path` (bundle root) **plus** `model_variant` (e.g. `base/post-trained`) — so for cosmos both `model_path` and `model_variant` are load-bearing, not `name`. The action-side `text_dim` auto-derives from the backbone (1024), so no manual override is needed.
>
> **Cosmos3-Edge:** `name` is validated (only `cosmos3_edge`); weights load from the diffusers-style bundle at `model_path` (`transformer/` + `vae/` + `text_tokenizer/`, modeling code vendored under `cosmos3/_vendor/`). No external text encoder — the bundled tokenizer + the frozen und text stream encode prompts inline, and `text_dim` auto-derives (2048), so `joint_cross_attn` needs no action_backbone overrides. Supported variants: `joint_cross_attn`, `joint_self_attn`, `idm`, and `single_system`/{`vanilla`,`moe`} (`tri_system` is rejected — its driver does not widen the joint mask for the und prefix K/V). Launch with `bash scripts/train.sh model=dual_system model/video_backbone=cosmos3_edge`.

Distributed-training settings, including mixed precision and the DeepSpeed ZeRO stage, live under `training` in `configs/train.yaml` and can be overridden through Hydra CLI arguments.

### 2. Deployment

Deploy a trained checkpoint as a WebSocket policy server:

```bash
bash scripts/deploy.sh /path/to/checkpoint_dir
```

To serve one checkpoint from several GPUs at once, set `NUM_GPUS` — GPU `i` gets port `PORT_BASE + i` (default base 8848) and logs under `logs/deploy_gpu*.log`; Ctrl+C stops the whole fleet (`PORT_BASE`, `GPU_START`, `LOG_DIR` are also overridable):

```bash
NUM_GPUS=8 bash scripts/deploy.sh /path/to/checkpoint_dir
```

This reads `configs/deploy.yaml` for base settings and the `config.yaml` saved inside the checkpoint for model architecture. The latest `checkpoint_step_*.safetensors` is loaded automatically; use `--ckpt-name` to pin one.

`scripts/deploy.py` and the package entrypoint (`openwam-serve` / `python -m openwam.deploy.server`) both load via the same `load_from_checkpoint_dir` path, merging deploy overrides on top of the saved training config.

#### Self-contained checkpoints

Checkpoints are deployable from their directory alone. During training, rank 0 saves:

- `checkpoint_step_*.safetensors` — full model weights.
- `config.yaml` — full training config, including video-backbone component specs when `model.video_backbone.model_path` was readable.
- `normalization_stats.npy` — action normalization stats; when `dataloader.normalize_mode` is enabled, deploy uses them to normalize incoming state and unnormalize returned actions.
- `tokenizer/google/umt5-xxl/` — copied from the Wan directory so deploy needs no access to the original model path.

Deploy resolves the video backbone from the embedded component specs first (tokenizer from `<ckpt_dir>/tokenizer/`), falling back to `model.video_backbone.model_path` if still accessible. A checkpoint with neither is not deployable.

#### Configuration

`configs/deploy.yaml` is the central config. Every `inference.*` field has a same-name CLI override:

```yaml
device: cuda:0
server: { host: "0.0.0.0", port: 8848 }

inference:
  denoise_steps: 10             # denoising steps
  denoise_mode: sync            # denoising trajectory: sync | async
  lead_modality: video          # async denoising only: action | video
  variance_shift_alpha: 1.0     # async denoising only: lead curve shift, >= 1
  linear_offset: 0.0            # async denoising only: lag delay, 0 <= value < 1
  inference_mode: sync          # inference executor: sync | async
  inference_horizon: null       # both executors: actions per chunk; null = full generated chunk
  inference_delay_steps: null   # async executor only: expected latency in action steps

optimization:
  decode_video: false     # false = actions-only (skip VAE decode, faster)
  dit_cache: { enabled: false, cosine_threshold: 0.99, max_skips: 3 }
  compile: { enabled: true }
  prompt_embed_cache: { maxsize: 32 }
```

`denoise_mode` selects the video/action trajectory within one denoising pass. `linear_offset` is an inference-time lag delay. `inference_mode` independently selects the sync or background-prefetch executor, while `inference_horizon` bounds the number of actions consumed from each generated chunk in either mode. Passing `--denoise-mode sync` or `--inference-mode sync` resets that axis's async-only fields to their defaults, so an async-tuned deploy yaml runs as the sync baseline without unsetting each field; supplying an async-only flag with a nontrivial value alongside `sync` is still an error.

Compile paths are selected from the checkpoint architecture. On dual-system architectures the first request may carry `torch.compile` warmup latency; use `--compile-enabled false` to run eager.

Common per-launch CLI overrides:

```bash
bash scripts/deploy.sh /path/to/checkpoint_dir \
  --device cuda:1 --port 9000 \
  --denoise-steps 10 --denoise-mode sync \
  --compile-enabled false \
  --ckpt-name checkpoint_step_10000.safetensors
```

All flags are optional; yaml values apply when a flag is absent. Optimization settings (`decode_video`, `dit_cache.*`, `compile.*`, `prompt_embed_cache.*`) are yaml-only — edit `configs/deploy.yaml` to change them.

#### WebSocket messages

- `{"type": "obs", ...}` — send 3-camera `images` dict, the prompt (forwarded to the model verbatim; wrap per your checkpoint's template), optional raw `state`; receive an action in the checkpoint's deploy scale (unnormalized to physical units for normalized checkpoints).
- `{"type": "reset"}` — reset policy state between episodes.
- `{"type": "ping"}` — liveness check; server replies `{"type": "pong"}`.

See [benchmarks/README.md](benchmarks/README.md) for the full client payload contract.

### 3. Testing the Server

The client always sends a 3-camera payload (head required, wrists optional); the server reads the checkpoint's `config.yaml` and dispatches to single- or multi-view preprocessing. Test scripts send a zero `state` vector by default (`--state-dim 20`); pass real proprioception with `--state` / `--state-file`, or `--no-state` for checkpoints without proprioceptive conditioning.

```bash
# Smoke test with 3 random images (no files needed)
python scripts/inference_test/inference_single_test.py --test

# With real images
python scripts/inference_test/inference_single_test.py \
  --server ws://127.0.0.1:8848 \
  --head-camera /path/to/head.jpg \
  --left-wrist-camera /path/to/left.jpg \
  --right-wrist-camera /path/to/right.jpg \
  --prompt "pick up the bottle"
```

The server chunks actions internally: the first call runs full inference (slow), subsequent calls pop cached actions (<10ms), and re-inference triggers when the buffer empties.

### 4. Benchmarks

Evaluation adapters live under `benchmarks/`. Eval scripts connect to an **already-running** policy server over WebSocket — no model weights are needed on the evaluator machine. See [benchmarks/robotwin/README.md](benchmarks/robotwin/README.md) for single-task, multi-task, DLC multi-node, and CSV-export usage.

For large multi-node RoboTwin runs, `benchmarks/robotwin/dlc_parallel_eval.sh` claims tasks from a shared-filesystem queue:

```bash
ROBOTWIN_PATH=/path/to/RoboTwin \
ROBOTWIN_PYTHON=/path/to/robotwin/bin/python \
ROBOTWIN_RUN_ID=run1 \
bash benchmarks/robotwin/dlc_parallel_eval.sh \
  -m all -n openwam -d /path/to/ckpt_dir --denoise-steps 10 all
```

Benchmark support status is listed under [Support Status](#benchmarks-and-evaluation) above.

## Development

### Dev setup

On top of the base installation, install the dev toolchain and (optionally) the
pre-commit hooks:

```bash
pip install -e '.[dev]'

# Optional but recommended: ruff runs automatically on each commit
pre-commit install
```

### Common commands

```bash
make test      # run the core test suite
make lint      # check code quality with ruff
make format    # auto-format code
make check     # compile check + tests
make all       # lint + tests (full validation)
```

### Before submitting a PR

1. Run `make all` and make sure it passes.
2. Add tests for new functionality under `tests/`.
3. Update `README.md` if you changed user-visible behavior.
4. Keep commits focused: one logical change per commit.

## Acknowledgements

OpenWAM builds on ideas and components from:

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [StarVLA](https://github.com/starVLA/starVLA)

## License

OpenWAM is released under the [Apache License 2.0](LICENSE).

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
