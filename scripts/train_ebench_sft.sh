#!/usr/bin/env bash
# EBench pretrain-SFT entrypoint.
#
# Required:
#   FINETUNE_CKPT=/path/to/openwam_80d_pretrain/checkpoint_step_*.safetensors
#
# Optional examples:
#   EBENCH_BUCKETS='[simple_pnp/task1,teleop_tasks/peg_in_hole]' bash scripts/train_ebench_sft.sh
#   BATCH_SIZE=2 LR=1e-5 NPROC_PER_NODE=8 bash scripts/train_ebench_sft.sh training.max_steps=1000
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${FINETUNE_CKPT:-}" ]]; then
    echo "FINETUNE_CKPT is required, e.g. FINETUNE_CKPT=/path/to/checkpoint_step_x.safetensors" >&2
    exit 2
fi

EBENCH_DATASET_DIR="${EBENCH_DATASET_DIR:-/path/to/data_lake/EBench-Dataset}"
WAN22_PATH="${WAN22_PATH:-/path/to/Wan2.2-TI2V-5B}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/train_runs/openwam_ebench_sft}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

if [[ -z "${EBENCH_STATS_PATH:-}" ]]; then
    if [[ -n "${EBENCH_BUCKETS:-}" ]]; then
        EBENCH_STATS_PATH="${OUTPUT_DIR}/ebench_stats.npy"
    else
        EBENCH_STATS_PATH="${EBENCH_DATASET_DIR}/meta/ebench_stats.npy"
    fi
fi

overrides=(
    dataloader=ebench
    dataloader.dataset_dir="${EBENCH_DATASET_DIR}"
    dataloader.normalization_stats_path="${EBENCH_STATS_PATH}"
    dataloader.action_mode=ebench
    dataloader.unify_action=true
    'dataloader.unify_action_map=["0-9","34-43","68-70"]'
    model=dual_system
    model.architecture.action_dim=80
    model.architecture.state_dim=80
    model.video_backbone.model_path="${WAN22_PATH}"
    training.finetune_ckpt_path="${FINETUNE_CKPT}"
    training.output_path="${OUTPUT_DIR}"
    training.learning_rate="${LR}"
    training.batch_size="${BATCH_SIZE}"
    training.gradient_accumulation_steps="${GRAD_ACCUM}"
)

if [[ -n "${EBENCH_BUCKETS:-}" ]]; then
    overrides+=(
        dataloader.groups=null
        "dataloader.buckets=${EBENCH_BUCKETS}"
    )
fi

exec bash scripts/train.sh "${overrides[@]}" "$@"
