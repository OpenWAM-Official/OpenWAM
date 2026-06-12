#!/usr/bin/env bash
# Evaluate OpenWAM on a list of RoboTwin tasks against an already-running server.
#
# Tasks run sequentially; each call to single_eval.sh starts fresh (the client
# resets server state at the beginning of every episode via ModelClient.reset()).
#
# Usage:
#   bash multi_eval.sh -m <mode> -n <name> -d <ckpt_dir> [options] <tasks...>
#
# Required:
#   -m, --mode       demo_clean | demo_randomized
#   -n, --name       label for log directory naming
#   -d, --ckpt-dir   checkpoint directory (for result file naming; server uses it independently)
#
# Tasks (positional, after flags):
#   one or more task names   adjust_bottle open_laptop ...
#   "all"                    all 50 RoboTwin 2.0 tasks
#   path to a task-list file (one task per line, # comments supported)
#
# Options:
#       --host       OpenWAM server host    (default: 127.0.0.1, env: ROBOTWIN_POLICY_HOST)
#       --port       OpenWAM WebSocket port (default: 8848,      env: ROBOTWIN_PORT)
#   -g, --gpu        CUDA device for RoboTwin simulator (default: 0)
#   -h, --help
#
# Required env vars:
#   ROBOTWIN_PATH    — path to the RoboTwin repository
#   ROBOTWIN_PYTHON  — Python for RoboTwin (or set ROBOTWIN_ENV conda env name)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOTWIN_ALL_TASKS=(
    adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
    click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block
    handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad
    move_playingcard_away move_stapler_pad open_laptop open_microwave
    pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
    place_bread_basket place_bread_skillet place_burger_fries place_can_basket
    place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup
    place_fan place_mouse_pad place_object_basket place_object_scale
    place_object_stand place_phone_stand place_shoe press_stapler
    put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object
    shake_bottle_horizontally shake_bottle stack_blocks_three stack_blocks_two
    stack_bowls_three stack_bowls_two stamp_seal turn_switch
)

usage() {
    cat >&2 <<'EOF'
Usage:
  bash multi_eval.sh -m <mode> -n <name> -d <ckpt_dir> [options] <tasks...>

Required:
  -m, --mode       demo_clean | demo_randomized
  -n, --name       label for log directory naming
  -d, --ckpt-dir   OpenWAM checkpoint directory

Tasks (positional):
  task names, "all", or a task-list file (one per line)

Options:
      --host       server host           (default: 127.0.0.1)
      --port       server WebSocket port (default: 8848)
  -g, --gpu        CUDA device for RoboTwin simulator (default: 0)
  -h, --help

Examples:
  bash multi_eval.sh -m demo_clean -n run1 -d /ckpt/openwam adjust_bottle open_laptop
  bash multi_eval.sh -m demo_clean -n run1 -d /ckpt/openwam all
  bash multi_eval.sh -m demo_randomized -n run1 -d /ckpt/openwam --port 8768 all
EOF
}

trim() {
    local v="$1"
    v="${v#"${v%%[![:space:]]*}"}"; v="${v%"${v##*[![:space:]]}"}"
    printf '%s\n' "${v}"
}

resolve_tasks() {
    local -a raw=("$@") out=() parts=()
    if (( ${#raw[@]} == 1 )) && [[ -f "${raw[0]}" ]]; then
        local line
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="$(trim "${line%%#*}")"; [[ -n "${line}" ]] && out+=("${line}")
        done < "${raw[0]}"
    else
        local inp task
        for inp in "${raw[@]}"; do
            if [[ "${inp}" == "all" ]]; then out+=("${ROBOTWIN_ALL_TASKS[@]}"); continue; fi
            IFS=',' read -ra parts <<< "${inp}"
            for task in "${parts[@]}"; do
                task="$(trim "${task}")"; [[ -n "${task}" ]] && out+=("${task}")
            done
        done
    fi
    (( ${#out[@]} > 0 )) || { echo "[ERROR] No tasks resolved." >&2; return 1; }
    printf '%s\n' "${out[@]}"
}

find_conda_python() {
    local env="$1"
    local -a bases=(
        "${CONDA_EXE:+$(dirname "$(dirname "${CONDA_EXE}")")/envs}"
        "${CONDA_PREFIX:+$(dirname "${CONDA_PREFIX}")}"
        "${HOME}/miniconda3/envs" "${HOME}/anaconda3/envs"
        "${HOME}/miniforge3/envs" "${HOME}/mambaforge/envs"
        "/opt/conda/envs"
    )
    local b
    for b in "${bases[@]}"; do
        [[ -x "${b}/${env}/bin/python" ]] && { printf '%s\n' "${b}/${env}/bin/python"; return 0; }
    done
    echo "[ERROR] Cannot find Python for conda env '${env}'. Set ROBOTWIN_PYTHON explicitly." >&2
    return 1
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

TASK_CONFIG="" POLICY_NAME="" CKPT_DIR=""
SERVER_HOST="${ROBOTWIN_POLICY_HOST:-127.0.0.1}"
PORT="${ROBOTWIN_PORT:-8848}"
GPU_ID="0"

while (( $# > 0 )); do
    case "$1" in
        -m|--mode)       TASK_CONFIG="$2"; shift 2 ;;
        -n|--name)       POLICY_NAME="$2"; shift 2 ;;
        -d|--ckpt-dir)   CKPT_DIR="$2";    shift 2 ;;
        --host)          SERVER_HOST="$2"; shift 2 ;;
        --port)          PORT="$2";        shift 2 ;;
        -g|--gpu)        GPU_ID="$2";      shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        -*)              echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
        *)               break ;;
    esac
done

[[ -z "${TASK_CONFIG}" || -z "${POLICY_NAME}" || -z "${CKPT_DIR}" ]] && {
    echo "[ERROR] Missing required flags: -m, -n, -d" >&2; usage; exit 1; }
[[ "${TASK_CONFIG}" != "demo_clean" && "${TASK_CONFIG}" != "demo_randomized" ]] && {
    echo "[ERROR] Invalid mode: ${TASK_CONFIG}" >&2; exit 1; }
[[ ! -d "${CKPT_DIR}" ]] && { echo "[ERROR] ckpt_dir not found: ${CKPT_DIR}" >&2; exit 1; }
(( $# > 0 )) || { echo "[ERROR] No tasks specified." >&2; usage; exit 1; }

# Resolve ROBOTWIN_PYTHON
if [[ -z "${ROBOTWIN_PYTHON:-}" ]]; then
    ROBOTWIN_PYTHON="$(find_conda_python "${ROBOTWIN_ENV:-robotwin}")"
fi
export ROBOTWIN_PYTHON

mapfile -t TASKS < <(resolve_tasks "$@")

ckpt_label="$(basename "${CKPT_DIR}")"
timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROBOTWIN_LOG_ROOT:-${CKPT_DIR}/robotwin_eval_logs/${POLICY_NAME}_${TASK_CONFIG}_${ckpt_label}_${timestamp}}"
mkdir -p "${LOG_DIR}"

echo "[INFO] mode=${TASK_CONFIG}  name=${POLICY_NAME}"
echo "[INFO] server=ws://${SERVER_HOST}:${PORT}  gpu=${GPU_ID}"
echo "[INFO] ckpt_dir=${CKPT_DIR}"
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] tasks (${#TASKS[@]}): ${TASKS[*]}"

# ---------------------------------------------------------------------------
# Run tasks sequentially
# ---------------------------------------------------------------------------

FAILED_TASKS=()

for task_name in "${TASKS[@]}"; do
    log_file="${LOG_DIR}/${task_name/\//_}_${TASK_CONFIG}.log"
    echo "[INFO] Starting task=${task_name}"

    # Pipe to tee so the full output is saved; capture single_eval.sh's own exit code
    # via PIPESTATUS[0] rather than grep's exit code.  grep is display-only and
    # must not affect the success/failure decision.
    ROBOTWIN_PORT="${PORT}" ROBOTWIN_POLICY_HOST="${SERVER_HOST}" \
    bash "${SCRIPT_DIR}/single_eval.sh" \
        "${task_name}" "${TASK_CONFIG}" "${POLICY_NAME}" \
        "${GPU_ID}" \
        "${PORT}" "${SERVER_HOST}" \
        2>&1 | tee "${log_file}" || true
    eval_exit="${PIPESTATUS[0]}"

    grep --color=never "Success rate" "${log_file}" \
        | sed "s/^/[RESULT] ${task_name}: /" || true

    if [[ "${eval_exit}" -eq 0 ]]; then
        echo "[INFO] Finished task=${task_name}"
    else
        FAILED_TASKS+=("${task_name}")
        echo "[ERROR] task=${task_name} failed (exit ${eval_exit}). See ${log_file}" >&2
    fi
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "[ERROR] Failed tasks: ${FAILED_TASKS[*]}" >&2
    echo "[ERROR] Logs: ${LOG_DIR}" >&2
    exit 1
fi

echo "[INFO] All tasks finished. Logs: ${LOG_DIR}"
