#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_MACHINES="${NNODES:-${WORLD_SIZE:-1}}"
MACHINE_RANK="${NODE_RANK:-${RANK:-0}}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"
TOTAL_GPUS=$(( NPROC_PER_NODE * NUM_MACHINES ))

accelerate launch \
    --config_file configs/accelerate/deepspeed_zero2.yaml \
    --num_machines "${NUM_MACHINES}" \
    --num_processes "${TOTAL_GPUS}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MAIN_PROCESS_IP}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    scripts/train.py \
    "$@"
