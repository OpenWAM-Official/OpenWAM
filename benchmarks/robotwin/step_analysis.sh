#!/usr/bin/env bash
# Per-task step-count analysis runner.
#
# Runs every RoboTwin task under both demo_clean and demo_randomized, 5
# episodes each, against an already-running OpenWAM server, and writes
# per-episode {task, mode, episode, steps, success} records to a JSONL file.
# Pass the JSONL through analyze_steps.py to get the summary table.
#
# No checkpoint is touched here — the OpenWAM server owns the model. This
# script only orchestrates RoboTwin-side episodes.
#
# Usage:
#   bash step_analysis.sh [options] [tasks...]
#
# Options:
#       --host           OpenWAM host       (default: 127.0.0.1)
#       --http-port      OpenWAM HTTP port  (default: 8848)
#   -g, --gpu            CUDA device        (default: 0)
#   -s, --seed           eval seed          (default: 0)
#   -n, --name           run label          (default: step_analysis)
#   -o, --output-dir     output root        (default: ./step_analysis_results/<name>_<ts>)
#       --test-num       episodes per task×mode (default: 5)
#       --modes          comma list of modes (default: demo_clean,demo_randomized)
#       --skip-analyze   don't run analyze_steps.py at the end
#   -h, --help
#
# Positional tasks: defaults to all 50 RoboTwin 2.0 tasks. Accepts task names,
# "all", or a path to a newline-separated task list file.
#
# Required env:
#   ROBOTWIN_PATH    — path to the RoboTwin repo root
#   ROBOTWIN_PYTHON  — Python for RoboTwin (or set ROBOTWIN_ENV conda name)
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
    sed -n '1,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
}

trim() {
    local v="$1"
    v="${v#"${v%%[![:space:]]*}"}"; v="${v%"${v##*[![:space:]]}"}"
    printf '%s\n' "${v}"
}

resolve_tasks() {
    local -a raw=("$@") out=() parts=()
    if (( ${#raw[@]} == 0 )); then
        out=("${ROBOTWIN_ALL_TASKS[@]}")
    elif (( ${#raw[@]} == 1 )) && [[ -f "${raw[0]}" ]]; then
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

SERVER_HOST="${ROBOTWIN_POLICY_HOST:-127.0.0.1}"
HTTP_PORT="${ROBOTWIN_HTTP_PORT:-8848}"
GPU_ID="0"
EVAL_SEED="0"
RUN_NAME="step_analysis"
OUTPUT_DIR=""
TEST_NUM="5"
MODES_CSV="demo_clean,demo_randomized"
SKIP_ANALYZE="0"

while (( $# > 0 )); do
    case "$1" in
        --host)          SERVER_HOST="$2";  shift 2 ;;
        --http-port)     HTTP_PORT="$2";    shift 2 ;;
        -g|--gpu)        GPU_ID="$2";       shift 2 ;;
        -s|--seed)       EVAL_SEED="$2";    shift 2 ;;
        -n|--name)       RUN_NAME="$2";     shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2";   shift 2 ;;
        --test-num)      TEST_NUM="$2";     shift 2 ;;
        --modes)         MODES_CSV="$2";    shift 2 ;;
        --skip-analyze)  SKIP_ANALYZE="1";  shift 1 ;;
        -h|--help)       usage; exit 0 ;;
        -*)              echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
        *)               break ;;
    esac
done

if [[ -z "${ROBOTWIN_PYTHON:-}" ]]; then
    ROBOTWIN_PYTHON="$(find_conda_python "${ROBOTWIN_ENV:-robotwin}")"
fi
export ROBOTWIN_PYTHON

mapfile -t TASKS < <(resolve_tasks "$@")
IFS=',' read -ra MODES <<< "${MODES_CSV}"

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="./step_analysis_results/${RUN_NAME}_${timestamp}"
fi
mkdir -p "${OUTPUT_DIR}/logs"
# Resolve to absolute — single_eval.sh does `cd "${ROBOTWIN_PATH}"` before
# launching Python, so any relative path baked into env vars (JSONL, policy
# config) would otherwise be interpreted relative to the RoboTwin repo root.
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

JSONL_PATH="${OUTPUT_DIR}/steps.jsonl"
SUMMARY_PATH="${OUTPUT_DIR}/summary.csv"
: > "${JSONL_PATH}"  # start clean

# Turn debug=false in a per-run policy config so 500 rollouts don't flood disk
# with ~150k debug frames. Users who want debug output can point
# POLICY_CONFIG_PATH elsewhere before invoking this script.
RUN_POLICY_CONFIG="${OUTPUT_DIR}/policy_config.yml"
sed -e 's/^debug:.*/debug: false/' "${SCRIPT_DIR}/policy_config.yml" > "${RUN_POLICY_CONFIG}"
export POLICY_CONFIG_PATH="${RUN_POLICY_CONFIG}"

echo "[INFO] run_name=${RUN_NAME}  seed=${EVAL_SEED}  test_num=${TEST_NUM}"
echo "[INFO] server=http://${SERVER_HOST}:${HTTP_PORT}  gpu=${GPU_ID}"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] modes=(${MODES[*]})  tasks=${#TASKS[@]}"

export ROBOTWIN_EVAL_SCRIPT="${SCRIPT_DIR}/eval_policy_steps.py"
export ROBOTWIN_TEST_NUM="${TEST_NUM}"
# Without this Python block-buffers stdout when piped into ``tee`` below, so
# the slow SAPIEN / curobo import + Sapien_TEST() stretch looks like a hang.
export PYTHONUNBUFFERED=1

FAILED=()

for mode in "${MODES[@]}"; do
    mode="$(trim "${mode}")"
    [[ -z "${mode}" ]] && continue
    for task_name in "${TASKS[@]}"; do
        log_file="${OUTPUT_DIR}/logs/${mode}_${task_name/\//_}.log"
        echo "[INFO] Starting mode=${mode}  task=${task_name}"

        # single_eval.sh signature (6 args): <task_name> <task_config>
        # <ckpt_setting> <gpu_id> [http_port] [host]. No seed slot —
        # single_eval.sh hardcodes seed=0. Comments MUST stay above the
        # env-var prefix: a '\'-continuation followed by a '#' line joins
        # them into one "FOO=bar # comment" logical line, which bash
        # treats as a shell-local assignment (not exported), so child
        # processes lose OPENWAM_STEP_LOG_PATH and steps.jsonl stays empty.
        OPENWAM_STEP_LOG_PATH="${JSONL_PATH}" \
        OPENWAM_STEP_LOG_TASK="${task_name}" \
        OPENWAM_STEP_LOG_MODE="${mode}" \
        ROBOTWIN_HTTP_PORT="${HTTP_PORT}" \
        ROBOTWIN_POLICY_HOST="${SERVER_HOST}" \
        bash "${SCRIPT_DIR}/single_eval.sh" \
            "${task_name}" "${mode}" "${RUN_NAME}" \
            "${GPU_ID}" \
            "${HTTP_PORT}" "${SERVER_HOST}" \
            2>&1 | tee "${log_file}" || true
        eval_exit="${PIPESTATUS[0]}"

        if [[ "${eval_exit}" -ne 0 ]]; then
            FAILED+=("${mode}/${task_name}")
            echo "[WARN] mode=${mode} task=${task_name} exited ${eval_exit}. See ${log_file}" >&2
        fi
    done
done

echo "[INFO] Finished all runs. JSONL: ${JSONL_PATH}"
if (( ${#FAILED[@]} > 0 )); then
    echo "[WARN] Failures (${#FAILED[@]}): ${FAILED[*]}" >&2
fi

if [[ "${SKIP_ANALYZE}" == "0" ]]; then
    python3 "${SCRIPT_DIR}/analyze_steps.py" "${JSONL_PATH}" \
        --output "${SUMMARY_PATH}" \
        --markdown "${OUTPUT_DIR}/summary.md" || \
        echo "[WARN] analyze_steps.py failed; JSONL still available at ${JSONL_PATH}" >&2
    echo "[INFO] Summary: ${SUMMARY_PATH}"
fi
