#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# OpenWAM S-VAE Trainer — torchrun data-parallel launcher (1 GPU → N nodes)
#
# The S-VAE reducer is small and fully replicated, so this is plain DDP via
# HuggingFace Accelerate (no DeepSpeed) — mirrors scripts/train.sh so the same
# topology env vars apply. Standalone Hydra config
# (configs/model/video_backbone/encoder/svae/train.yaml) — no model/dataloader chain.
#
# Required:
#   train.features_dir=<dir written by collect_svae_features.py>
#
# ── Single node (auto-detect GPUs) ──
#   bash scripts/svae_train/train_svae.sh train.features_dir=/path/<encoder>_<ts>
#   NPROC_PER_NODE=4 bash scripts/svae_train/train_svae.sh train.features_dir=...
#
# ── Multi-node (env vars set by the cloud scheduler; 2 nodes shown) ──
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=192.0.2.1 bash scripts/svae_train/train_svae.sh train.features_dir=...
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=192.0.2.1 bash scripts/svae_train/train_svae.sh train.features_dir=...
#
# Single-GPU without torchrun also works (Accelerate runs single-process):
#   CUDA_VISIBLE_DEVICES=0 python scripts/svae_train/train_svae.py train.features_dir=...
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "${OPENWAM_VERBOSE_NCCL:-0}" == "1" ]]; then
    export NCCL_DEBUG=INFO
else
    export NCCL_DEBUG=WARN
fi

# ── GPU / Node topology ──
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
# `wc -l` prints "0" (non-empty) on a node without GPUs / nvidia-smi, which the
# `:-` default does NOT catch; clamp numerically so torchrun gets >=1.
[ "${NPROC_PER_NODE}" -ge 1 ] 2>/dev/null || NPROC_PER_NODE=1
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# Port shifted from train.sh (29500) / collect (29502) so an S-VAE training run
# can coexist with a main-training or collection run on the same node.
MASTER_PORT="${MASTER_PORT:-29503}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  OpenWAM S-VAE Trainer (DDP)                         ║"
echo "║  Nodes: ${NNODES}  GPUs/node: ${NPROC_PER_NODE}  Rank: ${NODE_RANK}            ║"
echo "║  Master: ${MASTER_ADDR}:${MASTER_PORT}                   ║"
echo "╚══════════════════════════════════════════════════════╝"

torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    scripts/svae_train/train_svae.py \
    "$@"
