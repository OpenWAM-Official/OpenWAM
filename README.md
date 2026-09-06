<p align="center">
  <img src="assets/repo_images/openwam_title.png" alt="OpenWAM" width="60%">
</p>

<p align="center"><strong>An Open, Modular Exploration Towards Systematic World–Action Model Pretraining</strong></p>

<p align="center">
  <a href="https://openwam-official.github.io/"><img src="https://img.shields.io/badge/Project%20Page-OPENWAM--OFFICIAL.GITHUB.IO-blue?style=for-the-badge&amp;logo=github" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXXX-red?style=for-the-badge&amp;logo=arxiv" alt="Technical Report (pending)">
  <a href="https://huggingface.co/OpenWAM"><img src="https://img.shields.io/badge/HuggingFace-Model%20%26%20Data-orange?style=for-the-badge&amp;logo=huggingface" alt="Model &amp; Data on Hugging Face"></a>
</p>

<p align="center">
  <img src="assets/repo_images/teaser.jpg" alt="OpenWAM teaser">
</p>

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

<!--
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
├── scripts/           # Entrypoints: train.sh, deploy.sh, inference tooling, SVAE / LAPA tooling
├── configs/           # Hydra configs for model, dataloader, training, deploy
├── benchmarks/
│   ├── robotwin/      # RoboTwin eval client, single / multi eval scripts
│   ├── libero/        # LIBERO WebSocket eval client
│   ├── libero-plus/   # LIBERO-plus perturbation-suite eval client
│   ├── robocasa365/   # RoboCasa365 native-action eval client
│   ├── robocasa_gr1/  # RoboCasa GR1 tabletop eval client
│   ├── vlabench/      # VLABench eval client, single / multi-GPU track sweeps
│   ├── ebench/        # EBench (GenManip) eval bridge
│   └── robodojo/      # RoboDojo: training in OpenWAM, evaluation via XPolicyLab
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
| LIBERO-plus eval | Supported | Perturbation-robustness suite over LIBERO; see `benchmarks/libero-plus/` |
| RoboCasa365 eval | Supported | Native state19/action15 contract; see `benchmarks/robocasa365/` |
| RoboCasa GR1 eval | Supported | GR1 tabletop tasks; see `benchmarks/robocasa_gr1/` |
| VLABench eval | Supported | 10 primitive tasks across 6 evaluation tracks; see `benchmarks/vlabench/` |
| EBench eval | Supported | GenManip generalist tasks; see `benchmarks/ebench/` |
| RoboDojo eval | External | Trains in OpenWAM (sim + real); evaluates via [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) |
| Calvin eval | Planned | Requires external environment setup |
-->

## News

**[2026/09/06]** 🔥OpenWAM Codebase Release！

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
<summary><b>Cosmos-Predict2.5 Extras (Optional)</b> — needed only for experiments with the <code>cosmos_predict25_2b</code> video backbone</summary>

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

If you just want to use or finetune from a
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

Quick Start provides a minimal end-to-end example: prepare the assets, train
or fine-tune a policy, deploy its checkpoint, and run a first inference check.

The example uses the DualSystem JointSelfAttention architecture, the
Wan2.2-TI2V-5B video backbone, and the Mutual attention mask.

| Component | Selection | Configuration |
|---|---|---|
| Architecture | dual_system / joint_self_attn | [configs/model/dual_system.yaml](configs/model/dual_system.yaml) |
| Video backbone | wan22_ti2v_5b | [configs/model/video_backbone/wan22_ti2v_5b.yaml](configs/model/video_backbone/wan22_ti2v_5b.yaml) |
| Attention mask | mutual | model.architecture.attention_mask_mode |
| Dataset | libero | [configs/dataloader/libero.yaml](configs/dataloader/libero.yaml) |

> **Resource recommendation:** We recommend 8 GPUs with 80 GB VRAM each for
> training. This configuration supports normal training for all architectures
> using Wan2.2-5B or smaller video backbones. More GPUs are better when
> available and can further improve training throughput.

### From Scratch Training

1. Download LIBERO and let the downloader update its dataloader configuration:

   ~~~bash
   python scripts/download_assets/download_benchmark_data.py
   ~~~

   Select LIBERO in the interactive menu.

2. Download Wan2.2-TI2V-5B:

   ~~~bash
   python scripts/download_assets/download_video_backbone.py
   ~~~

   Select Wan2.2-TI2V-5B and the desired model source.

3. Start a debug run with the complete model selection:

   ~~~bash
   bash scripts/train.sh \
     dataloader=libero \
     model=dual_system \
     model/video_backbone=wan22_ti2v_5b \
     model.architecture.variant=joint_self_attn \
     model.architecture.attention_mask_mode=mutual \
     training.debug=true
   ~~~

   training.debug=true runs 20 steps, saves at steps 10 and 20, and uses a
   constant learning rate. Check the run output, then set
   training.debug=false for normal training. Training defaults and CLI
   overrides are defined in [configs/train.yaml](configs/train.yaml).
   Debug outputs use training.output_path, whose default is
   outputs/openwam_checkpoints.

4. Deploy the debug checkpoint and inspect one input-output cycle:

   ~~~bash
   bash scripts/deploy.sh <debug_ckpt_dir_path>
   ~~~

   Keep the server running, then open another terminal and run the two
   inference helpers. Deployment enables compile by default, so the first
   inference may take longer while compilation warms up; later requests are
   typically faster:

   ~~~bash
   python scripts/inference_test/inference_single_test.py \
     --server ws://127.0.0.1:8848 --test --state-dim 10

   python scripts/inference_test/inference_continuous_test.py \
     --server ws://127.0.0.1:8848 --test --state-dim 10 -n 3
   ~~~

   The single-request helper checks ping, one prediction, and reset. The
   continuous helper sends repeated predictions over one connection and
   reports the returned action dimension and latency. Stop the deployment
   process after the checks.

### OpenWAM-α Fine-Tuning

1. Download LIBERO as shown above.

2. Download the OpenWAM-Alpha foundation checkpoint:

   ~~~bash
   python scripts/download_assets/download_openwam_checkpoints.py
   ~~~

   Select OpenWAM_Alpha and OpenWAM-Alpha-Pretrain-Foundation-Model.
   Keep the resulting directory as <foundation_ckpt_dir_path>.

3. Start fine-tuning from that directory:

   ~~~bash
   bash scripts/train.sh \
     dataloader=libero \
     training.finetune_ckpt_path=<foundation_ckpt_dir_path>
   ~~~

   The default model configuration already matches the required setup above.
   You can set the same field in [configs/train.yaml](configs/train.yaml)
   instead of passing it on the command line. num_frames=33 and
   video_stride=4 in [configs/dataloader/libero.yaml](configs/dataloader/libero.yaml)
   produce the 32-step action horizon expected by the sampler; no separate
   action_chunk override is needed.

4. Deploy the resulting checkpoint directory:

   ~~~bash
   bash scripts/deploy.sh <ckpt_dir_path>
   ~~~

   Install and run the LIBERO client according to the
   [LIBERO evaluation guide](benchmarks/libero/README.md).


## OpenWAM Usage Guidance

OpenWAM is configured through composable Hydra YAML files. Select an architecture,
backbone, dataloader, and runtime behavior by changing configuration values or
overriding them on the command line. The task guides below are the maintained
entry points for using and extending the repository:

| Guide | Use it when you want to… |
|---|---|
| [Training and deployment](assets/openwam_usage_docs/train-and-deploy.md) | choose a model/dataloader, prepare assets, train, fine-tune, resume, or deploy a policy |
| [Architecture extension](assets/openwam_usage_docs/architecture-extension.md) | add a video, visual, VLM, action backbone, or WAM architecture |
| [Benchmark integration](assets/openwam_usage_docs/benchmark-integration.md) | add a dataset reader and connect a benchmark client to the WebSocket protocol |
| [OpenWAM-α fine-tuning](assets/openwam_usage_docs/openwam-alpha-finetuning.md) | align a downstream action space and fine-tune the released foundation checkpoint |

Installation and Assets Preparation above cover environment setup and model or
dataset downloads. Benchmark-specific environment and evaluation details remain
in the benchmarks directory.

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
make lint      # check code quality with ruff
make format    # auto-format code
make check     # compile check
make all       # lint and compile check
```

### Before submitting a PR

1. Run make all and resolve any failures.
2. Update README.md if you changed user-visible behavior.
3. Keep commits focused: one logical change per commit.

<!--
## Acknowledgements

OpenWAM builds on ideas and components from:

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [StarVLA](https://github.com/starVLA/starVLA)
-->

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
