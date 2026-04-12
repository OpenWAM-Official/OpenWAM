#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_MACHINES="${NNODES:-${WORLD_SIZE:-1}}"
MACHINE_RANK="${NODE_RANK:-${RANK:-0}}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"
TOTAL_GPUS=$(( NPROC_PER_NODE * NUM_MACHINES ))

unset WORLD_SIZE RANK LOCAL_RANK

ACCELERATE_CONFIG=$(mktemp /tmp/accelerate_ds_XXXXXX.yaml)
trap "rm -f ${ACCELERATE_CONFIG}" EXIT

cat > "${ACCELERATE_CONFIG}" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: ${NUM_MACHINES}
num_processes: ${TOTAL_GPUS}
machine_rank: ${MACHINE_RANK}
main_process_ip: ${MAIN_PROCESS_IP}
main_process_port: ${MAIN_PROCESS_PORT}
mixed_precision: bf16

deepspeed_config:
  zero_stage: 2
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero3_save_16bit_model: false
EOF

accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    scripts/train.py \
    "$@"
