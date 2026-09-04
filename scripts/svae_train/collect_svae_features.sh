#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# OpenWAM S-VAE Offline Feature Collection — torchrun multi-node launcher
#
# Mirrors ``scripts/train.sh`` so training overrides paste in verbatim. The
# wrapped entry point reads the same Hydra default chain as train.yaml (via
# ``configs/model/video_backbone/encoder/svae/collect.yaml``) plus an ``svae_collect`` block.
#
# Required CLI overrides (external encoders only). Select the encoder via these
# FIELD overrides — the ``model/video_backbone/encoder=<name>`` group-select idiom does NOT
# resolve from this nested entry config (Hydra searches the group relative to
# the config's own dir), and the collector refuses the default wan22_vae anyway:
#   model.video_backbone.encoder.name=vjepa21
#   model.video_backbone.encoder.model_path=<weights dir>
# (``model.video_backbone.from_scratch`` is NOT needed here — the collector
# builds only the encoder, not the full architecture, so that flag is inert.)
#
# The collector traverses the FULL dataset once (no sampling knob); features are
# written in chunked ``features_rank{r}_part{p}.pt`` shards. Typical knobs:
#   training.batch_size=4                 # clips/step (inherited from train.yaml)
#   svae_collect.output_dir=/path         # results root; per-run subdir auto-created
#
# ── Single-node (auto-detect GPUs) ──
#   bash scripts/svae_train/collect_svae_features.sh model.video_backbone.encoder.name=vjepa21 \
#       model.video_backbone.encoder.model_path=/path
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "${OPENWAM_VERBOSE_NCCL:-0}" == "1" ]]; then
    export NCCL_DEBUG=INFO
else
    export NCCL_DEBUG=WARN
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
# `wc -l` prints the string "0" (non-empty) on a CPU-only node or when
# nvidia-smi is missing, which the `:-` default above does NOT replace — that
# would pass `--nproc_per_node 0` to torchrun and fail immediately. Clamp
# numerically to >=1 (the Python collector has a gloo CPU fallback, so a
# 0-GPU node is a reachable launch path). The `2>/dev/null ||` also recovers
# from a non-numeric value (the `[` test errors, then we default to 1).
[ "${NPROC_PER_NODE}" -ge 1 ] 2>/dev/null || NPROC_PER_NODE=1
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# Port shifted from train.sh (29500) / pca_stats.sh (29501) so a collect run can
# coexist with a training or PCA run on the same node.
MASTER_PORT="${MASTER_PORT:-29502}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  OpenWAM S-VAE Offline Feature Collection            ║"
echo "║  Nodes: ${NNODES}  GPUs/node: ${NPROC_PER_NODE}  Rank: ${NODE_RANK}            ║"
echo "║  Master: ${MASTER_ADDR}:${MASTER_PORT}                   ║"
echo "╚══════════════════════════════════════════════════════╝"

torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    scripts/svae_train/collect_svae_features.py \
    "$@"
