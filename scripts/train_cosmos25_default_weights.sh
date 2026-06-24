#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Cosmos-Predict2.5 training launcher — DEFAULT pretrained weights.
#
# dual_system / joint_self_attn, live Reason1 text encoder, RoboTwin2.0.
# Uses the released Cosmos-Predict2.5-2B `base/post-trained` checkpoint (NOT the
# AgiBot-pretrained `robot/multiview-agibot` variant).
#
# Modeled on the reference repo's
# sandbox/phase53_formal/train_H_multiview_formal.sh, but adapted to THIS
# codebase, where:
#   - the backbone config group is `model/video_backbone` (not `model/backbone`),
#   - the raw context width is `model.architecture.text_dim` (the action backbone
#     no longer owns it), and
#   - the freeze list lives in `configs/model/dual_system.yaml` with Wan-style
#     paths, so Cosmos's `_vae_inner` / `_reason1_inner` must be
#     pinned explicitly (the default Wan paths silently no-op on Cosmos).
#
# Launch (single node, auto-detects GPUs):
#   bash scripts/train_cosmos25_default_weights.sh
# Quick smoke (20 steps + a checkpoint, tiny batch):
#   bash scripts/train_cosmos25_default_weights.sh training.debug=true training.batch_size=1
# Multi-node: the cloud scheduler injects NNODES/NODE_RANK/MASTER_ADDR; just run this.
# Extra hydra overrides pass through, e.g.:
#   bash scripts/train_cosmos25_default_weights.sh training.video_lr=5e-5
#
set -euo pipefail

# Repo root (this script lives in <repo>/scripts/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# This repo has no in-tree .venv; reuse a venv that already carries the GPU
# stack (torch+cu128 / cosmos_predict2 / transformer_engine / deepspeed /
# accelerate). Running from REPO_ROOT makes `openwam` resolve to THIS code.
# Override with: VENV_BIN=/path/to/venv/bin bash <this script>
VENV_BIN="${VENV_BIN:-/path/to/openwam/.venv/bin}"
export PATH="${VENV_BIN}:${PATH}"

# Assets (host-local; override via env if your paths differ).
COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-/path/to/assets/Cosmos-Predict2.5-2B}"
REASON1_PATH="${REASON1_PATH:-/path/to/assets/Cosmos-Reason1-7B}"
DATASET_DIR="${DATASET_DIR:-/path/to/RoboTwin2.0/dataset}"
OUTPUT_PATH="${OUTPUT_PATH:-/path/to/checkpoints/cosmos25_base_posttrained_formal}"

# wandb run name (override via env: RUN_NAME=my_run bash <this script>).
RUN_NAME="${RUN_NAME:-cosmos25_base_posttrained_formal}"

bash scripts/train.sh \
  model/video_backbone=cosmos25 \
  model.video_backbone.model_path="${COSMOS_MODEL_PATH}" \
  model.video_backbone.model_variant=base/post-trained \
  model.video_backbone.text_encoder=reason1_live \
  model.video_backbone.text_encoder_path="${REASON1_PATH}" \
  '++model.architecture.text_dim=1024' \
  'model.freeze=[video_backbone._vae_inner,video_backbone._reason1_inner]' \
  dataloader.dataset_dir="${DATASET_DIR}" \
  dataloader.variant=both \
  dataloader.task_name=null \
  dataloader.val_ratio=0.0 \
  training.video_lr=1e-4 \
  training.batch_size=24 \
  training.gradient_accumulation_steps=1 \
  training.num_epochs=5 \
  training.save_steps=2000 \
  training.keep_last_k_ckpts=2 \
  training.dataset_num_workers=4 \
  training.output_path="${OUTPUT_PATH}" \
  project.wandb.run_name="${RUN_NAME}" \
  "$@"
