#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${FINETUNE_CKPT_PATH:-}" && -n "${RESUME_CKPT_PATH:-}" ]]; then
  echo "FINETUNE_CKPT_PATH and RESUME_CKPT_PATH are mutually exclusive" >&2
  exit 2
fi

overrides=(dataloader=wuji_real_task)
add_override() {
  if [[ -n "${2:-}" ]]; then
    overrides+=("$1=$2")
  fi
}

add_override training.finetune_ckpt_path "${FINETUNE_CKPT_PATH:-}"
add_override training.resume_ckpt_path "${RESUME_CKPT_PATH:-}"
add_override training.output_path "${OUTPUT_PATH:-}"
add_override training.batch_size "${BATCH_SIZE:-}"
add_override training.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-}"
add_override training.num_epochs "${NUM_EPOCHS:-}"
add_override training.max_steps "${MAX_STEPS:-}"
add_override training.learning_rate "${LEARNING_RATE:-}"
add_override training.weight_decay "${WEIGHT_DECAY:-}"
add_override training.action_lr "${ACTION_LR:-}"
add_override training.video_lr "${VIDEO_LR:-}"
add_override training.warmup_ratio "${WARMUP_RATIO:-}"
add_override training.lr_min_ratio "${LR_MIN_RATIO:-}"
add_override training.mixed_precision "${MIXED_PRECISION:-}"
add_override training.zero_stage "${ZERO_STAGE:-}"
add_override training.use_gradient_checkpointing "${USE_GRADIENT_CHECKPOINTING:-}"
add_override training.use_gradient_checkpointing_offload "${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-}"
add_override training.initialize_model_on_cpu "${INITIALIZE_MODEL_ON_CPU:-}"
add_override training.offload_optimizer_device "${OFFLOAD_OPTIMIZER_DEVICE:-}"
add_override training.dataset_num_workers "${DATASET_NUM_WORKERS:-}"
add_override training.save_steps "${SAVE_STEPS:-}"
add_override training.save_full_states_for_resume "${SAVE_FULL_STATES_FOR_RESUME:-}"
add_override training.keep_last_k_ckpts "${KEEP_LAST_K_CKPTS:-}"
add_override training.debug "${DEBUG:-}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
bash scripts/train.sh "${overrides[@]}" "$@"
