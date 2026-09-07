# VLABench Evaluation

Two processes: the **OpenWAM policy server** (this repo's env, holds the model) and the **VLABench client** (its own env). They talk over WebSocket ([wire protocol](../README.md)), so the two environments never interfere.

Commands below assume conda at `/path/to/miniconda3`, the OpenWAM checkout at
`/path/to/OpenWAM-Official`, and the VLABench checkout at `/path/to/VLABench` —
substitute your actual paths.

## 1. Environment Setup

VLABench pins `numpy==1.25.0` / `mujoco==3.2.2` / `dm_control==1.0.22` — a separate env is mandatory:

```bash
/path/to/miniconda3/bin/conda create -n vlabench python=3.10 -y
conda activate vlabench
git clone https://github.com/OpenMOSS/VLABench.git /path/to/VLABench && cd /path/to/VLABench
# Evaluation-only install (recommended for OpenWAM evaluation):
pip install \
  'numpy==1.25.0' 'scipy==1.14.0' 'mujoco==3.2.2' 'dm_control==1.0.22' \
  'opencv-python<4.12' 'gym==0.26.2' 'gymnasium==0.29.1' \
  'mediapy==1.2.0' 'h5py==3.11.0' 'imageio>=2.34' \
  'Pillow>=10' 'PyYAML>=6' 'websockets>=16,<17' 'open3d==0.18.0' \
  colorlog colorama openai 'rtree==1.2.0' networkx gdown
pip install -e src/rrt-algorithms --no-deps --no-build-isolation
pip install -e . --no-deps
python scripts/download_assets.py            # ~17 GB of scene/object assets

# Optional full training/data-conversion install:
# pip install -r requirements.txt
```

The evaluation-only path intentionally does not install the pinned LeRobot
commit from `requirements.txt`. That old commit currently declares
`pyav>=12.0.5`, but there is no usable `pyav` distribution for this setup, so
installing the entire file fails during dependency resolution. LeRobot is only
needed for dataset conversion/training and is not imported by the OpenWAM
VLABench evaluator. If training or conversion is required, install LeRobot in
a separate environment and resolve its `pyav` dependency there.

For headless Linux machines, the MuJoCo EGL runtime must also be available. With
conda this can be installed without root access:

```bash
conda install -n vlabench -c conda-forge libegl mesalib libglvnd -y
# The evaluator imports torch at module import time; CPU Torch is sufficient
# because policy inference runs in the separate OpenWAM server environment.
conda install -n vlabench -c pytorch -c conda-forge pytorch=2.5.* cpuonly -y
```

After installing, return to the OpenWAM checkout before running its scripts:

```bash
cd /path/to/OpenWAM-Official
```

Verify (`env` needs no GPU/checkpoint/server):

```bash
VLABENCH_PATH=/path/to/VLABench VLABENCH_PYTHON=/path/to/miniconda3/envs/vlabench/bin/python \
bash benchmarks/vlabench/run_smoke.sh env select_fruit
```

<details>
<summary><b>Constraints (details)</b></summary>

- Pin the VLABench commit for reproducible scores. The evaluator imports VLABench
  environments/tasks/robots directly (the WebSocket wire adapter itself is
  benchmark-agnostic), so an unpinned `main` can change task behavior. Caveats:
  `track_5` seeded episodes reproduce only per-commit, and a few tasks may fail to
  instantiate on current `main`.
- OS packages for EGL: `libglu1-mesa`, `libgl1-mesa-dri`, `libgl1-mesa-glx`. `MUJOCO_GL=egl` is exported by the launch scripts.
- `VLABENCH_PATH` is required by every command below; `VLABENCH_PYTHON` defaults to `python` on PATH.
- `run_smoke.sh loop select_fruit` runs a full closed loop against an in-process mock (append `8 127.0.0.1:8848` to use a real server).

</details>

## 2. Start the Policy Server

The evaluated checkpoint is
[`OpenWAM/OpenWAM-Alpha-Sim-VLABench`](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Sim-VLABench),
from the [OpenWAM-Alpha collection](https://huggingface.co/collections/OpenWAM/openwam-alpha):

```bash
python scripts/download_assets/download_openwam_checkpoints.py
# menu: OpenWAM_Alpha → OpenWAM-Alpha-Sim-VLABench
```

Run from the repo root — the checkpoint lands in `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-VLABench`. Then:

```bash
bash scripts/deploy.sh assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-VLABench
```

To serve a VLABench checkpoint you fine-tuned yourself (see
[§5](#5-fine-tuning-on-vlabench-data)), activate the OpenWAM environment and
pass its directory instead (the loader uses the checkpoint's embedded component
specifications):

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate openwam
cd /path/to/OpenWAM-Official
bash scripts/deploy.sh /path/to/your/checkpoint_dir \
  --device cuda:0 --port 8848 --compile-enabled false
```

The deploy directory must contain `config.yaml` and at least one
`checkpoint_step_*.safetensors`. Module *weights* (DiT / VAE / text encoder) are
rebuilt from the config's embedded `components` specs, so the original
`model.video_backbone.model_path` need not exist on this host. Two non-weight
artifacts are still read from disk whenever the saved config declares them:

- `normalization_stats.npy` — required whenever `dataloader.normalize_mode` is
  set; the loader raises `FileNotFoundError` without it.
- The tokenizer dir named by `model.video_backbone.tokenizer.subdir` (Wan:
  `tokenizer/google/umt5-xxl`; Cosmos backbones additionally need `reason1/`).
  It is resolved under the checkpoint dir first and only then under
  `model_path` — and since `model_path` usually still names the training host,
  keep these files inside the checkpoint dir.

Check the checkpoint's `config.yaml` for `dataloader.normalize_mode` and
`model.video_backbone.tokenizer.subdir` to see which of the two apply.

Do not point this benchmark at a checkpoint trained for a different one: a
RoboTwin checkpoint, for instance, declares `dataloader.type: robotwin` and
`attention_mask_mode: joint`, which is incompatible with this checkout's
VLABench deployment contract.

WebSocket port 8848 by default (`--port` to change). For the multi-GPU sweep below, start one server per port instead (loop shown there).

## 3. Run the Evaluation

Single task or whole track — args are `<task|all> [track] [n_episodes] [port] [host]`:

```bash
VLABENCH_PATH=/path/to/VLABench VLABENCH_PYTHON=/path/to/miniconda3/envs/vlabench/bin/python \
bash benchmarks/vlabench/single_eval.sh select_fruit track_1_in_distribution 50 8848

VLABENCH_PATH=/path/to/VLABench VLABENCH_PYTHON=/path/to/miniconda3/envs/vlabench/bin/python \
bash benchmarks/vlabench/single_eval.sh all track_2_cross_category 50 8848
```

Multi-GPU sweep — one server per port first, then `multi_eval.sh [tracks|all] [tasks|all] [n_episodes] [ports]` (parallelism = number of ports):

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup bash scripts/deploy.sh \
    assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-VLABench --device cuda:0 --port $((8880+i)) &
done
VLABENCH_PATH=/path/to/VLABench VLABENCH_PYTHON=/path/to/miniconda3/envs/vlabench/bin/python \
bash benchmarks/vlabench/multi_eval.sh all all 50 8880,8881,8882,8883
```

**One policy server per concurrent simulator** — hard correctness rule: two simulators sharing one port silently corrupt each other's action chunks.

<details>
<summary><b>Notes & troubleshooting</b></summary>

- Tracks: `track_1_in_distribution`, `track_2_cross_category`, `track_3_common_sense`, `track_4_semantic_instruction`, `track_5_cross_task`, `track_6_unseen_texture`.
- `track_5` ships no episode config: give `single_eval.sh` an explicit task (not `all`), or let `multi_eval.sh` use `VLABENCH_TRACK5_TASKS` / its default 10-task held-out split.
- Expected, not bugs: `Failed to converge after 99 steps` (IK giving up; frequent for undertrained ckpts), ~1% `PhysicsError mjWARN_BADQACC` (upstream MuJoCo), `get_intention_score raised KeyError` note (NaN fallback; success/progress survive).
- Budget 30-60 min per task per process (RGB+depth+segmentation every step, plus IK).
- Camera frames are pre-resized client-side to the training reader's L-shape
  slot sizes — head `320x256`, wrists `160x128`, Pillow LANCZOS — via
  `resize_for_lshape_slot`, so the server's same-size paste is geometrically a
  no-op. VLABench renders every camera at `480x480`, so without this the eval
  images would not match the resize the training data went through. Slot-to-
  camera indices are in `openwam2vlabench_interface.py`; the wire image
  contract is in [`benchmarks/README.md`](../README.md).
- `policy_config.yml` is preconfigured for the released checkpoint; use `POLICY_CONFIG_PATH` for a custom copy.
- Results: `benchmarks/vlabench/_eval_out/` (override `VLABENCH_SAVE_DIR`) —
  OpenWAM writes per-track `evaluation_result.json`, while VLABench writes
  per-task `detail_info.json`; `multi_eval.sh` merges into
  `combined_results.json` with logs under `_eval_out/logs/`. The requested
  episode budget is not necessarily the number of valid records: the evaluator
  catches failed episodes, so inspect logs and `detail_info.json` before quoting
  a score.

</details>

## 4. Results

Scores from the OpenWAM paper. **Bold** = best, <u>underline</u> = second best; Type distinguishes WAM vs VLA. Column groups: ID = in-distribution, Cat = cross-category, CS = common sense, Ins = semantic instruction, Tex = unseen texture; SR / PS / IS = success rate / progress score / intention score.

| Method | Type | ID SR | ID PS | ID IS | Cat SR | Cat PS | Cat IS | CS SR | CS PS | CS IS | Ins SR | Ins PS | Ins IS | Tex SR | Tex PS | Tex IS | Avg SR | Avg PS | Avg IS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| π₀ | VLA | 47.0 | 62.7 | 67.8 | 21.2 | 33.6 | 44.0 | 29.1 | 43.0 | 54.9 | 17.3 | 38.7 | 58.0 | 32.2 | 42.5 | 50.6 | 29.4 | 44.1 | 55.0 |
| LoHo-Manip | VLA | 54.0 | - | - | 23.0 | - | - | 36.0 | - | - | 42.0 | - | - | 39.0 | - | - | 39.0 | - | - |
| ACoT-VLA | VLA | - | 66.1 | 79.8 | - | 38.9 | <u>54.1</u> | - | 37.8 | 52.3 | - | 39.6 | 56.8 | - | 54.6 | 74.6 | - | 47.4 | 63.5 |
| π₀.₅ | VLA | 65.4 | 77.8 | 80.4 | 38.2 | 49.7 | 52.0 | 43.9 | 57.3 | <u>60.0</u> | 48.2 | 64.2 | 67.0 | 44.9 | 62.3 | 65.0 | 48.1 | 62.3 | 64.9 |
| ERVLA | VLA | 69.7 | 81.1 | <u>84.2</u> | <u>47.0</u> | <u>61.0</u> | **66.4** | 44.0 | 55.0 | 57.2 | <u>58.0</u> | <u>70.2</u> | <u>73.8</u> | 47.4 | 62.3 | 70.6 | 53.2 | 65.9 | <u>70.4</u> |
| Xiaomi-Robotics-1 | VLA | 75.6 | 85.0 | 79.8 | **53.0** | **66.6** | **66.4** | 48.4 | 58.3 | 58.2 | 55.8 | 66.8 | 70.2 | **62.6** | **74.9** | <u>74.8</u> | **59.1** | **70.3** | 69.9 |
| Bridge-WA | WAM | <u>78.0</u> | <u>85.8</u> | **85.0** | 23.0 | 28.8 | 39.0 | <u>51.1</u> | <u>64.4</u> | **74.2** | **67.0** | **80.3** | **82.0** | 45.0 | 60.3 | **76.0** | 52.8 | 64.0 | **71.2** |
| **OpenWAM-α** | WAM | **83.4** | **87.9** | 75.2 | 38.1 | 45.9 | 45.9 | **58.0** | **64.8** | 57.9 | 53.8 | 64.5 | 66.5 | <u>61.4</u> | <u>72.9</u> | 71.6 | <u>58.9</u> | <u>67.2</u> | 63.5 |

## 5. Fine-tuning on VLABench Data

Only needed to train your own checkpoint — evaluating the released one needs
nothing from this section. The general training flow (model selection, debug
run, multi-node launch) is in the root README under
[Quick Start](../../README.md#quick-start); what follows is the
VLABench-specific part. Run it in the **OpenWAM** environment, not the VLABench
one.

### Dataset

```bash
python scripts/download_assets/download_benchmark_data.py
# menu: VLABench   (~13.2 GB)
```

The downloader is the supported path and does three things a manual
`hf download` would not: it pulls
[`OpenWAM/VLABench`](https://huggingface.co/datasets/OpenWAM/VLABench) — a
mirror of `VLABench/vlabench_primitive_ft_lerobot_video` repacked into 16 large
tars, because the upstream repo's ~19k small files download at request-latency
rather than bandwidth — unpacks them, rewrites `dataset_dir` in
[`configs/dataloader/vlabench.yaml`](../../configs/dataloader/vlabench.yaml) to
the download location, and makes sure the normalization statistics exist.

Do not use `VLABench/vlabench_primitive_ft_dataset`: it redirects to
`VLABench/raw_primitive_datasets`, the pre-conversion tarballs, which this
reader cannot read.

Normalization statistics live at
`<dataset_dir>/meta/vlabench_normalization_stats.npy` and are **auto-built on
first use** whenever `dataloader.normalize_mode` is set — rank 0 scans the
corpus, the other ranks wait. To build them ahead of time (or to inspect the
table without writing it):

```bash
python -m openwam.dataloader.utils.stats_computation.vlabench_stats_computation \
  --dataset-dir /path/to/vlabench_data          # add --dry-run to only print
```

The rot6d dimensions are pinned to identity so normalization is a pass-through
on the rotation representation; the reader warns loudly if it loads a stats file
that predates that pin.

### Fine-tune

`vlabench.yaml` sets `unify_action_map: ["0-9"]`, so the 10 raw EEF dimensions
land on slots 0-9 of the unified 80-D action space — dimensions the foundation
model already pretrained.

```bash
python scripts/download_assets/download_openwam_checkpoints.py
# menu: OpenWAM_Alpha → OpenWAM-Alpha-Pretrain-Foundation-Model

bash scripts/train.sh \
  dataloader=vlabench \
  training.finetune_ckpt_path=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
  training.num_epochs=null training.max_steps=6000 \
  training.output_path=/path/to/output
```

`train.sh` uses all visible GPUs; narrow it with `NPROC_PER_NODE` or
`CUDA_VISIBLE_DEVICES`. Defaults and every overridable field are in
[`configs/train.yaml`](../../configs/train.yaml) — `training.batch_size` is
per-GPU (default 16). Add `training.debug=true` for a 20-step shakeout run
before committing to a full one.

Training writes `config.yaml` and copies `normalization_stats.npy` into the
checkpoint dir on its own, which covers the first of the two non-weight
artifacts [§2](#2-start-the-policy-server) needs. The **tokenizer dir is not
copied** — if the serving host cannot resolve the config's
`model.video_backbone.model_path` (usually a path on the training machine),
copy `tokenizer/google/umt5-xxl` into the checkpoint dir before deploying.
