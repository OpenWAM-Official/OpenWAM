#!/usr/bin/env bash
# OpenWAM LAPA-DINOv3 latent-action pretraining recipe.
#
# This is the lightweight/open-source launcher. It assumes model assets are
# already present locally and does not perform Hugging Face staging, /dev/shm
# staging, or company-cluster adaptation.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${OPENWAM_VERBOSE_NCCL:-0}" == "1" ]]; then
    export NCCL_DEBUG=INFO
else
    export NCCL_DEBUG=WARN
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

export OPENWAM_LAPA_MODEL_DIR="${OPENWAM_LAPA_MODEL_DIR:-${PWD}/models/LAPA-DINOv3/LAPA-DINOv3}"
export OPENWAM_LAPA_DINOV3_MODEL_DIR="${OPENWAM_LAPA_DINOV3_MODEL_DIR:-${PWD}/models/dinov3-vitl16-pretrain-lvd1689m}"

check_lapa_assets() {
    local ckpt="${OPENWAM_LAPA_MODEL_DIR}/laq_dinov3.pt"
    if [[ ! -f "${ckpt}" ]]; then
        echo "[lapa] LAPA-DINOv3 checkpoint not found: ${ckpt}" >&2
        exit 1
    fi
    if head -c 48 "${ckpt}" | grep -q "version https://git-lfs.github.com/spec/v1"; then
        echo "[lapa] LAPA-DINOv3 checkpoint is still a Git LFS pointer: ${ckpt}" >&2
        echo "[lapa] Refresh it with: git -C models/LAPA-DINOv3 lfs pull" >&2
        exit 1
    fi
    if [[ ! -d "${OPENWAM_LAPA_DINOV3_MODEL_DIR}" ]]; then
        echo "[lapa] DINOv3 model dir not found: ${OPENWAM_LAPA_DINOV3_MODEL_DIR}" >&2
        echo "[lapa] LAPA-DINOv3 requires a 1024-dim DINOv3 backbone, e.g. facebook/dinov3-vitl16-pretrain-lvd1689m" >&2
        exit 1
    fi
}

check_lapa_assets

echo "╔══════════════════════════════════════════════════════╗"
echo "║  OpenWAM LAPA-DINOv3 Latent-Action Pretraining      ║"
echo "║  Nodes: ${NNODES}  GPUs/node: ${NPROC_PER_NODE}  Rank: ${NODE_RANK}            ║"
echo "║  Master: ${MASTER_ADDR}:${MASTER_PORT}                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "[lapa] LAPA model dir: ${OPENWAM_LAPA_MODEL_DIR}"
echo "[lapa] DINOv3 model dir: ${OPENWAM_LAPA_DINOV3_MODEL_DIR}"

torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    scripts/train.py \
    model.action_backbone.type=latent \
    model.architecture.action_dim=1024 \
    model.architecture.use_proprioception=false \
    "$@"
