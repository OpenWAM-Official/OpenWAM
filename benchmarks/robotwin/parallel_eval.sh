#!/usr/bin/env bash
# Parallel RoboTwin evaluation across N OpenWAM servers with dynamic task distribution.
#
# Pairs with scripts/deploy_multi.sh: worker i -> port = PORT_BASE + i.
# Tasks are pulled from a shared queue guarded by flock, so faster workers pick
# up remaining tasks automatically — no static partitioning or idle workers.
#
# Usage:
#   bash parallel_eval.sh -m <mode> -n <name> [options] <tasks...>
#
# Required:
#   -m, --mode           demo_clean | demo_randomized | all
#   -n, --name           label for log directory naming
#
# Tasks (positional, after flags): same as multi_eval.sh.
#
# Options:
#   -w, --num-workers    number of parallel workers (default: 8)
#       --host           server host (default: 127.0.0.1)
#       --port           base WebSocket port; worker i uses port+i (default: 8848)
#       --gpu-start      first simulator GPU index (default: 0)
#   -h, --help
#
# Environment:
#   ROBOTWIN_PATH        path to the RoboTwin repository (required)
#   ROBOTWIN_PYTHON      python for RoboTwin (or set ROBOTWIN_ENV conda env name)
#   SIM_GPU_STRIDE       stride between consecutive simulator GPUs (default: 1)
#
# Ctrl+C terminates all workers.
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
  bash parallel_eval.sh -m <mode> -n <name> [options] <tasks...>

Required:
  -m, --mode           demo_clean | demo_randomized | all
  -n, --name           label for log directory naming

Tasks (positional): task names, "all", or a task-list file (one per line).

Options:
  -w, --num-workers    parallel workers (default: 8)
      --host           server host (default: 127.0.0.1)
      --port           base WebSocket port; worker i uses port+i (default: 8848)
      --gpu-start      first simulator GPU index (default: 0)
  -h, --help

Logs are written to ./robotwin_eval_logs/<name>_<mode>_parallel_<ts>/ by default;
override with ROBOTWIN_LOG_ROOT.

Example:
  # Launch 8 servers first (on cuda:0..7, ws 8848..8855):
  bash scripts/deploy_multi.sh /ckpt/openwam
  # Then run the full task list in parallel, dynamically distributed:
  bash benchmarks/robotwin/parallel_eval.sh -m demo_clean -n run1 all
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

TASK_CONFIG="" POLICY_NAME=""
NUM_WORKERS=8
SERVER_HOST="${ROBOTWIN_POLICY_HOST:-127.0.0.1}"
PORT_BASE="${ROBOTWIN_PORT:-8848}"
GPU_START=0
SIM_GPU_STRIDE="${SIM_GPU_STRIDE:-1}"

while (( $# > 0 )); do
    case "$1" in
        -m|--mode)          TASK_CONFIG="$2";     shift 2 ;;
        -n|--name)          POLICY_NAME="$2";     shift 2 ;;
        -w|--num-workers)   NUM_WORKERS="$2";     shift 2 ;;
        --host)             SERVER_HOST="$2";     shift 2 ;;
        --port)             PORT_BASE="$2";       shift 2 ;;
        --gpu-start)        GPU_START="$2";       shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        -*)                 echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
        *)                  break ;;
    esac
done

[[ -z "${TASK_CONFIG}" || -z "${POLICY_NAME}" ]] && {
    echo "[ERROR] Missing required flags: -m, -n" >&2; usage; exit 1; }
[[ "${TASK_CONFIG}" != "demo_clean" && "${TASK_CONFIG}" != "demo_randomized" && "${TASK_CONFIG}" != "all" ]] && {
    echo "[ERROR] Invalid mode: ${TASK_CONFIG}" >&2; exit 1; }

if [[ "${TASK_CONFIG}" == "all" ]]; then
    MODES=(demo_clean demo_randomized)
else
    MODES=("${TASK_CONFIG}")
fi
(( NUM_WORKERS > 0 )) || { echo "[ERROR] --num-workers must be > 0" >&2; exit 1; }
(( $# > 0 )) || { echo "[ERROR] No tasks specified." >&2; usage; exit 1; }

if [[ -z "${ROBOTWIN_PYTHON:-}" ]]; then
    ROBOTWIN_PYTHON="$(find_conda_python "${ROBOTWIN_ENV:-robotwin}")"
fi
export ROBOTWIN_PYTHON

mapfile -t TASKS < <(resolve_tasks "$@")

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROBOTWIN_LOG_ROOT:-./robotwin_eval_logs/${POLICY_NAME}_${TASK_CONFIG}_parallel_${timestamp}}"
mkdir -p "${LOG_DIR}"

# Shared work queue: workers atomically pop "task|mode" pairs from here via flock.
QUEUE_FILE="${LOG_DIR}/.queue.txt"
LOCK_FILE="${LOG_DIR}/.queue.lock"
: > "${QUEUE_FILE}"
for task in "${TASKS[@]}"; do
    for mode in "${MODES[@]}"; do
        printf '%s|%s\n' "${task}" "${mode}" >> "${QUEUE_FILE}"
    done
done
: > "${LOCK_FILE}"

TOTAL_JOBS=$(( ${#TASKS[@]} * ${#MODES[@]} ))

echo "[INFO] mode=${TASK_CONFIG}  name=${POLICY_NAME}"
echo "[INFO] workers=${NUM_WORKERS}  host=${SERVER_HOST}  port_base=${PORT_BASE}"
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] tasks (${#TASKS[@]}): ${TASKS[*]}"
echo "[INFO] modes (${#MODES[@]}): ${MODES[*]}  total_jobs=${TOTAL_JOBS}"
echo ""

# ---------------------------------------------------------------------------
# Worker: pop task from queue, run it, repeat until queue is empty.
# ---------------------------------------------------------------------------

run_worker() {
    local worker_idx="$1"
    local sim_gpu=$((GPU_START + worker_idx * SIM_GPU_STRIDE))
    local port=$((PORT_BASE + worker_idx))
    local worker_dir="${LOG_DIR}/worker${worker_idx}"
    local worker_log="${worker_dir}/worker.log"
    local finished_file="${worker_dir}/finished.txt"
    local failed_file="${worker_dir}/failed.txt"

    mkdir -p "${worker_dir}"
    : > "${finished_file}"
    : > "${failed_file}"

    local tag="[worker${worker_idx}@gpu${sim_gpu}:${port}]"
    echo "${tag} started" | tee -a "${worker_log}"

    while :; do
        # Atomically pop one "task|mode" item from the shared queue.
        # fd 200 must be opened INSIDE the $() — otherwise the redirect applies
        # to the enclosing shell and flock sees a closed descriptor, which we
        # verified allows two workers to grab the same task.
        local item task mode
        item=$({
            flock -x 200
            head -n1 "${QUEUE_FILE}" || true
            tail -n +2 "${QUEUE_FILE}" > "${QUEUE_FILE}.tmp" 2>/dev/null || true
            mv -f "${QUEUE_FILE}.tmp" "${QUEUE_FILE}" 2>/dev/null || true
        } 200>"${LOCK_FILE}")

        [[ -z "${item}" ]] && break
        task="${item%%|*}"
        mode="${item#*|}"

        local task_log="${LOG_DIR}/${task/\//_}_${mode}.log"
        echo "${tag} starting task=${task} mode=${mode}" | tee -a "${worker_log}"

        ROBOTWIN_PORT="${port}" ROBOTWIN_POLICY_HOST="${SERVER_HOST}" \
        bash "${SCRIPT_DIR}/single_eval.sh" \
            "${task}" "${mode}" "${POLICY_NAME}" \
            "${sim_gpu}" \
            "${port}" "${SERVER_HOST}" \
            >"${task_log}" 2>&1 \
            && eval_exit=0 || eval_exit=$?

        grep --color=never "Success rate" "${task_log}" \
            | sed "s|^|[RESULT] ${tag} ${task} (${mode}): |" || true

        if (( eval_exit == 0 )); then
            echo "${task}|${mode}" >> "${finished_file}"
            echo "${tag} finished task=${task} mode=${mode}" | tee -a "${worker_log}"
        else
            echo "${task}|${mode}" >> "${failed_file}"
            echo "${tag} FAILED task=${task} mode=${mode} (exit ${eval_exit}). See ${task_log}" \
                | tee -a "${worker_log}" >&2
        fi
    done

    echo "${tag} queue empty, exiting" | tee -a "${worker_log}"
}

# ---------------------------------------------------------------------------
# Launch workers, wait, then aggregate.
# ---------------------------------------------------------------------------

pids=()

# Recursively signal a process and all its descendants. Descendants must be
# killed first — otherwise they get reparented to init once the parent shell
# dies and keep running (this was the Ctrl+C leak: single_eval.sh and the
# python simulator were outliving the worker subshell).
kill_tree() {
    local pid=$1 sig=${2:-TERM}
    [[ -z "$pid" ]] && return
    local child
    while read -r child; do
        [[ -n "$child" ]] && kill_tree "$child" "$sig"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
    kill -"$sig" "$pid" 2>/dev/null || true
}

cleanup() {
    trap - INT TERM
    echo ""
    echo "[INFO] Interrupt received. Stopping ${#pids[@]} workers..." >&2
    for pid in "${pids[@]}"; do kill_tree "$pid" TERM; done
    # Give children a moment to exit cleanly, then force-kill stragglers.
    local deadline=$((SECONDS + 5)) still_alive=1
    while (( SECONDS < deadline )); do
        still_alive=0
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && { still_alive=1; break; }
        done
        (( still_alive )) || break
        sleep 0.2
    done
    if (( still_alive )); then
        echo "[INFO] Escalating to SIGKILL for survivors..." >&2
        for pid in "${pids[@]}"; do kill_tree "$pid" KILL; done
    fi
    wait 2>/dev/null || true
    echo "[INFO] Workers stopped." >&2
    exit 130
}
trap cleanup INT TERM

for ((i = 0; i < NUM_WORKERS; i++)); do
    run_worker "$i" &
    pids+=($!)
done

echo "[INFO] Launched ${NUM_WORKERS} workers. PIDs: ${pids[*]}"
echo "[INFO] Tail worker logs:  tail -f ${LOG_DIR}/worker*/worker.log"
echo ""

wait

# Aggregate results
FINISHED_TASKS=()
FAILED_TASKS=()
for ((i = 0; i < NUM_WORKERS; i++)); do
    finished_file="${LOG_DIR}/worker${i}/finished.txt"
    failed_file="${LOG_DIR}/worker${i}/failed.txt"
    [[ -s "${finished_file}" ]] && mapfile -t -O "${#FINISHED_TASKS[@]}" FINISHED_TASKS < "${finished_file}"
    [[ -s "${failed_file}" ]]   && mapfile -t -O "${#FAILED_TASKS[@]}"   FAILED_TASKS   < "${failed_file}"
done

echo ""
echo "[SUMMARY] finished=${#FINISHED_TASKS[@]}  failed=${#FAILED_TASKS[@]}  total=${TOTAL_JOBS}"
echo "[SUMMARY] logs=${LOG_DIR}"

if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "[ERROR] Failed tasks: ${FAILED_TASKS[*]}" >&2
    exit 1
fi

echo "[INFO] All tasks finished successfully."
