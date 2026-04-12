#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_MACHINES="${NNODES:-${WORLD_SIZE:-1}}"
MACHINE_RANK="${NODE_RANK:-${RANK:-0}}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE=configs/deepspeed/zero2.json
export ACCELERATE_MIXED_PRECISION=bf16

torchrun \
    --nnodes "${NUM_MACHINES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${MACHINE_RANK}" \
    --master_addr "${MAIN_PROCESS_IP}" \
    --master_port "${MAIN_PROCESS_PORT}" \
    scripts/train.py \
    "$@"
