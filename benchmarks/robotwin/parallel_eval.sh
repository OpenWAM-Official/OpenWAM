#!/usr/bin/env bash
# Parallel RoboTwin evaluation across N already-running OpenWAM servers, with
# EPISODE-LEVEL dynamic scheduling via a central dispatcher.
#
# Pairs with scripts/deploy_multi.sh: worker i -> port = PORT_BASE + i.
#
# Unlike the old whole-task queue, the schedulable unit here is a single
# episode. Each GPU slot runs a supervisor loop that keeps launching
# episode_worker processes; each worker claims one (task,mode) from the
# dispatcher, boots its env once, and streams that task's episodes. When no
# unstarted task remains, idle slots join an in-progress task (spawn a duplicate
# env) to drain its tail in parallel — so GPUs stop idling at the end of a run.
#
# Usage:
#   bash parallel_eval.sh -m <mode> -n <name> [options] <tasks...>
#
# Required:
#   -m, --mode           demo_clean | demo_randomized | all
#   -n, --name           label for log directory naming
#
# Tasks (positional, after flags): task names, "all", or a task-list file.
#
# Options:
#   -w, --num-workers          parallel GPU slots (default: 8)
#       --host                 server host (default: 127.0.0.1)
#       --port                 base WebSocket port; slot i uses port+i (default: 8848)
#       --gpu-start            first simulator GPU index (default: 0)
#       --test-num             episodes per (task,mode) (default: 100)
#       --seed                 base seed (st_seed = 100000*(1+seed)) (default: 0)
#       --min-remaining-for-dup N   don't spawn a new env for a task with fewer
#                                   remaining episodes than N (default: 8)
#       --no-dup               strict: exactly one env per task, no duplication
#       --dispatch-port        dispatcher TCP port (default: 8790)
#       --http-port            dispatcher live-status HTTP port (0=off, default: 0)
#   -h, --help
#
# Environment:
#   ROBOTWIN_PATH        path to the RoboTwin repository (required)
#   ROBOTWIN_PYTHON      python for RoboTwin (or set ROBOTWIN_ENV conda env name)
#   DISPATCHER_PYTHON    python for the dispatcher (stdlib only; default: ROBOTWIN_PYTHON)
#   SIM_GPU_STRIDE       stride between consecutive simulator GPUs (default: 1)
#
# Ctrl+C terminates the dispatcher and all workers.
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
    sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
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
TEST_NUM="${ROBOTWIN_TEST_NUM:-100}"   # env-overridable (legacy flow); --test-num still wins
BASE_SEED=0
MIN_REMAINING_FOR_DUP=8
NO_DUP=0
DISPATCH_PORT=8790
HTTP_PORT=0
DRY_RUN=0
DRY_RUN_SLEEP_SEC="${DRY_RUN_SLEEP_SEC:-0.2}"

while (( $# > 0 )); do
    case "$1" in
        -m|--mode)                 TASK_CONFIG="$2";            shift 2 ;;
        -n|--name)                 POLICY_NAME="$2";            shift 2 ;;
        -w|--num-workers)          NUM_WORKERS="$2";            shift 2 ;;
        --host)                    SERVER_HOST="$2";            shift 2 ;;
        --port)                    PORT_BASE="$2";              shift 2 ;;
        --gpu-start)               GPU_START="$2";              shift 2 ;;
        --test-num)                TEST_NUM="$2";               shift 2 ;;
        --seed)                    BASE_SEED="$2";              shift 2 ;;
        --min-remaining-for-dup)   MIN_REMAINING_FOR_DUP="$2";  shift 2 ;;
        --no-dup)                  NO_DUP=1;                    shift ;;
        --dispatch-port)           DISPATCH_PORT="$2";          shift 2 ;;
        --http-port)               HTTP_PORT="$2";              shift 2 ;;
        --dry-run|--dryrun)        DRY_RUN=1;                   shift ;;
        -h|--help)                 usage; exit 0 ;;
        -*)                        echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
        *)                         break ;;
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

if (( ! DRY_RUN )) && [[ -z "${ROBOTWIN_PYTHON:-}" ]]; then
    ROBOTWIN_PYTHON="$(find_conda_python "${ROBOTWIN_ENV:-robotwin}")"
fi
export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-}"
WORKER_PYTHON="${ROBOTWIN_PYTHON:-python3}"
DISPATCHER_PYTHON="${DISPATCHER_PYTHON:-${ROBOTWIN_PYTHON:-python3}}"

mapfile -t TASKS < <(resolve_tasks "$@")

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROBOTWIN_LOG_ROOT:-./robotwin_eval_logs/${POLICY_NAME}_${TASK_CONFIG}_parallel_${timestamp}}"
mkdir -p "${LOG_DIR}"

ADDR_FILE="${LOG_DIR}/.dispatcher_addr"
RESULTS_FILE="${LOG_DIR}/results.jsonl"
SUMMARY_FILE="${LOG_DIR}/summary.tsv"
STATE_FILE="${LOG_DIR}/state.json"
DONE_FILE="${LOG_DIR}/.done"
DISPATCHER_LOG="${LOG_DIR}/dispatcher.log"
rm -f "${ADDR_FILE}" "${DONE_FILE}"

TOTAL_JOBS=$(( ${#TASKS[@]} * ${#MODES[@]} ))
echo "[INFO] mode=${TASK_CONFIG} name=${POLICY_NAME} test_num=${TEST_NUM}"
echo "[INFO] workers=${NUM_WORKERS} host=${SERVER_HOST} port_base=${PORT_BASE} dup_theta=${MIN_REMAINING_FOR_DUP} no_dup=${NO_DUP}"
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] tasks (${#TASKS[@]}): ${TASKS[*]}"
echo "[INFO] modes (${#MODES[@]}): ${MODES[*]} total_jobs=${TOTAL_JOBS}"

# ---------------------------------------------------------------------------
# Start the dispatcher
# ---------------------------------------------------------------------------

dispatcher_args=(
    "${SCRIPT_DIR}/dispatcher.py"
    --host 127.0.0.1 --port "${DISPATCH_PORT}"
    --tasks "${TASKS[@]}"
    --modes "${MODES[@]}"
    --test-num "${TEST_NUM}"
    --seed "${BASE_SEED}"
    --min-remaining-for-dup "${MIN_REMAINING_FOR_DUP}"
    --num-slots "${NUM_WORKERS}"
    --results "${RESULTS_FILE}"
    --summary "${SUMMARY_FILE}"
    --state-file "${STATE_FILE}"
    --addr-file "${ADDR_FILE}"
    --done-file "${DONE_FILE}"
)
(( NO_DUP )) && dispatcher_args+=(--no-dup)
(( HTTP_PORT > 0 )) && dispatcher_args+=(--http-port "${HTTP_PORT}")
# Optional watchdog tuning via env (see README): stall/worker/idle timeouts + give-up cap.
[[ -n "${STALL_TIMEOUT:-}" ]]      && dispatcher_args+=(--stall-timeout "${STALL_TIMEOUT}")
[[ -n "${WORKER_TIMEOUT:-}" ]]     && dispatcher_args+=(--worker-timeout "${WORKER_TIMEOUT}")
[[ -n "${IDLE_GRACE:-}" ]]         && dispatcher_args+=(--idle-grace "${IDLE_GRACE}")
[[ -n "${MAX_ATTEMPT_FACTOR:-}" ]] && dispatcher_args+=(--max-attempt-factor "${MAX_ATTEMPT_FACTOR}")

"${DISPATCHER_PYTHON}" "${dispatcher_args[@]}" >"${DISPATCHER_LOG}" 2>&1 &
DISPATCHER_PID=$!

# Wait for the dispatcher to publish its address.
for _ in $(seq 1 60); do
    [[ -s "${ADDR_FILE}" ]] && break
    kill -0 "${DISPATCHER_PID}" 2>/dev/null || { echo "[ERROR] dispatcher died on startup; see ${DISPATCHER_LOG}" >&2; exit 1; }
    sleep 0.5
done
[[ -s "${ADDR_FILE}" ]] || { echo "[ERROR] dispatcher did not become ready; see ${DISPATCHER_LOG}" >&2; kill "${DISPATCHER_PID}" 2>/dev/null || true; exit 1; }
DISPATCHER_ADDR="$(tr -d '[:space:]' < "${ADDR_FILE}")"
echo "[INFO] dispatcher at ${DISPATCHER_ADDR} (pid ${DISPATCHER_PID}); log ${DISPATCHER_LOG}"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

pids=()

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
    set +e  # teardown must not be aborted mid-way by set -e
    echo "" >&2
    echo "[INFO] Interrupt received. Stopping workers + dispatcher..." >&2
    for pid in "${pids[@]}"; do kill_tree "$pid" TERM; done
    kill_tree "${DISPATCHER_PID}" TERM
    local deadline=$((SECONDS + 5))
    while (( SECONDS < deadline )); do
        local alive=0
        for pid in "${pids[@]}" "${DISPATCHER_PID}"; do
            kill -0 "$pid" 2>/dev/null && { alive=1; break; }
        done
        (( alive )) || break
        sleep 0.2
    done
    for pid in "${pids[@]}" "${DISPATCHER_PID}"; do kill_tree "$pid" KILL; done
    wait 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

# ---------------------------------------------------------------------------
# Slot supervisor: relaunch episode_worker until the dispatcher says "exit".
# ---------------------------------------------------------------------------

run_supervisor() {
    local slot_idx="$1"
    local sim_gpu=$((GPU_START + slot_idx * SIM_GPU_STRIDE))
    local port=$((PORT_BASE + slot_idx))
    local worker_dir="${LOG_DIR}/worker${slot_idx}"
    local worker_log="${worker_dir}/worker.log"
    mkdir -p "${worker_dir}"

    local tag="[worker${slot_idx}@gpu${sim_gpu}:${port}]"
    echo "${tag} supervisor started" | tee -a "${worker_log}"

    local attempt=0
    while :; do
        attempt=$((attempt + 1))
        local run_log="${worker_dir}/run$(printf '%03d' "${attempt}").log"
        if (( DRY_RUN )); then
            "${WORKER_PYTHON}" "${SCRIPT_DIR}/episode_worker.py" --dry-run \
                --dispatcher "${DISPATCHER_ADDR}" --node 0 --worker "${slot_idx}" --gpu "${sim_gpu}" \
                --dry-run-sleep "${DRY_RUN_SLEEP_SEC}" >"${run_log}" 2>&1 && rc=0 || rc=$?
        else
            ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}" \
            bash "${SCRIPT_DIR}/episode_eval.sh" \
                "${DISPATCHER_ADDR}" 0 "${slot_idx}" "${sim_gpu}" "${port}" "${SERVER_HOST}" "${POLICY_NAME}" "${BASE_SEED}" \
                >"${run_log}" 2>&1 && rc=0 || rc=$?
            grep --color=never "Success rate" "${run_log}" \
                | sed "s|^|[RESULT] ${tag} |" || true
        fi

        case "${rc}" in
            0) : ;;  # job drained -> claim another task
            3) echo "${tag} no more work; supervisor exiting" | tee -a "${worker_log}"; break ;;
            *) echo "${tag} worker error (rc=${rc}); see ${run_log}; supervisor exiting" | tee -a "${worker_log}" >&2; break ;;
        esac
    done
}

for ((i = 0; i < NUM_WORKERS; i++)); do
    run_supervisor "$i" &
    pids+=($!)
done

echo "[INFO] Launched ${NUM_WORKERS} slot supervisors. PIDs: ${pids[*]}"
echo "[INFO] Tail:  tail -f ${LOG_DIR}/worker*/worker.log   |   dispatcher: tail -f ${DISPATCHER_LOG}"

# Reaper: when the dispatcher signals it is finished (complete OR stall/abort),
# kill any straggler/wedged supervisors so `wait` below can never hang on a
# hung sim process.
supervisor_pids=("${pids[@]}")
( trap - INT TERM
  while [[ ! -f "${DONE_FILE}" ]]; do sleep 2; done
  for pid in "${supervisor_pids[@]}"; do kill_tree "$pid" TERM; done ) &
REAPER_PID=$!

# Wait for all slots to finish claiming.
for pid in "${pids[@]}"; do wait "$pid" || true; done
kill_tree "${REAPER_PID}" TERM 2>/dev/null || true

# Dispatcher exits once every job hits its target; give it a moment.
wait "${DISPATCHER_PID}" 2>/dev/null || true
trap - INT TERM

echo ""
if [[ -f "${SUMMARY_FILE}" ]]; then
    echo "[SUMMARY] ${SUMMARY_FILE}"
    column -t -s $'\t' < "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"
fi
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] Export CSV:  ${DISPATCHER_PYTHON} ${SCRIPT_DIR}/export_results_csv.py ${LOG_DIR}"
