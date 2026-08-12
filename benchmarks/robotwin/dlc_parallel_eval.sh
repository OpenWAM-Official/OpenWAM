#!/usr/bin/env bash
# DLC multi-node RoboTwin evaluation with EPISODE-LEVEL dynamic scheduling.
#
# Each DLC worker runs this same script. Rank 0 starts a single central
# dispatcher (bound on 0.0.0.0) and publishes its address to the shared log
# directory; every node starts local OpenWAM policy servers and local RoboTwin
# "slot supervisors". Each supervisor keeps launching episode_worker processes;
# each worker claims one (task,mode) from the dispatcher, boots its env once,
# and streams that task's episodes. When no unstarted task remains, idle slots
# across all nodes join an in-progress task (spawn a duplicate env) to drain its
# tail in parallel — so GPUs stop idling at the end of a run.
#
# This replaces the previous whole-task per-job `queue/pending` + atomic-`mv`
# claim protocol: scheduling, seed dedup, and result aggregation now all live in
# the dispatcher. The shared filesystem is used only for logs, the dispatcher
# address, results.jsonl, and summary.tsv.
#
# Usage:
#   bash benchmarks/robotwin/dlc_parallel_eval.sh -m <mode> -n <name> -d <ckpt_dir> [options] <tasks...>
#
# Required:
#   -m, --mode       demo_clean | demo_randomized | all
#   -n, --name       label for log directory naming
#   -d, --ckpt-dir   OpenWAM checkpoint directory used by local policy servers
#
# Tasks (positional): task names, "all", or a task-list file (one per line).
#
# DLC / cluster environment:
#   MLP_WORKER_NUM, MLP_ROLE_INDEX are preferred when present.
#   Falls back to NNODES/NODE_RANK, then WORLD_SIZE/RANK.
#
# Key environment overrides:
#   ROBOTWIN_PATH        path to the RoboTwin repository (required unless --dry-run)
#   ROBOTWIN_PYTHON      Python for RoboTwin (or set ROBOTWIN_ENV)
#   ROBOTWIN_RUN_ID      shared run id; default: latest
#   ROBOTWIN_LOG_ROOT    shared log root; default: <ckpt_dir>/robotwin_eval_logs
#   ROBOTWIN_TEST_NUM    episodes per (task,mode); default: 100 (--test-num overrides)
#   SERVER_PYTHON        Python used to launch local policy servers; default: python
#   DISPATCHER_PYTHON    Python for the dispatcher (stdlib only); default: SERVER_PYTHON
#   DISPATCHER_ADVERTISE_HOST  rank0 host advertised to other nodes (default: MASTER_ADDR / hostname -i)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; }

trim() {
    local v="$1"; v="${v#"${v%%[![:space:]]*}"}"; v="${v%"${v##*[![:space:]]}"}"
    printf '%s\n' "${v}"
}

is_safe_task_name() { [[ "$1" =~ ^[A-Za-z0-9_]+$ ]]; }

validate_task_names() {
    local task
    for task in "$@"; do
        is_safe_task_name "${task}" || { echo "[ERROR] Invalid task name '${task}'." >&2; return 1; }
    done
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

detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local count; count="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
        [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 )) && { printf '%s\n' "${count}"; return; }
    fi
    printf '1\n'
}

kill_tree() {
    local pid=$1 sig=${2:-TERM}
    [[ -z "${pid}" ]] && return
    local child
    while read -r child; do [[ -n "${child}" ]] && kill_tree "${child}" "${sig}"; done < <(pgrep -P "${pid}" 2>/dev/null || true)
    kill -"${sig}" "${pid}" 2>/dev/null || true
}

health_check() { timeout 2 bash -c ">/dev/tcp/$1/$2" 2>/dev/null; }

wait_for_server() {
    local host="$1" port="$2" log_file="$3" timeout_sec="$4"
    local deadline=$((SECONDS + timeout_sec))
    until health_check "${host}" "${port}"; do
        (( SECONDS >= deadline )) && { echo "[ERROR] server not ready ws://${host}:${port}; see ${log_file}" >&2; return 1; }
        sleep 2
    done
}

advertise_host() {
    if [[ -n "${DISPATCHER_ADVERTISE_HOST:-}" ]]; then printf '%s\n' "${DISPATCHER_ADVERTISE_HOST}"; return; fi
    if [[ -n "${MASTER_ADDR:-}" ]]; then printf '%s\n' "${MASTER_ADDR}"; return; fi
    local ip; ip="$(hostname -i 2>/dev/null | awk '{print $1}')"
    printf '%s\n' "${ip:-127.0.0.1}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

TASK_CONFIG="" POLICY_NAME="" CKPT_DIR=""
NUM_WORKERS="${NUM_WORKERS:-$(detect_gpu_count)}"
GPU_START="${GPU_START:-0}"
SIM_GPU_STRIDE="${SIM_GPU_STRIDE:-1}"
PORT_BASE="${PORT_BASE:-8848}"
SERVER_PYTHON="${SERVER_PYTHON:-python}"
SERVER_SCRIPT="${SERVER_SCRIPT:-${REPO_ROOT}/scripts/deploy.py}"
SERVER_BIND_HOST="${SERVER_BIND_HOST:-127.0.0.1}"
SERVER_CLIENT_HOST="${SERVER_CLIENT_HOST:-127.0.0.1}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-900}"
QUEUE_READY_TIMEOUT_SEC="${QUEUE_READY_TIMEOUT_SEC:-600}"
TEST_NUM="${ROBOTWIN_TEST_NUM:-100}"   # env-overridable (legacy single/whole-task flow); --test-num still wins
BASE_SEED=0
MIN_REMAINING_FOR_DUP=8
NO_DUP=0
APPEND_RESULTS=0
DISPATCH_PORT=8790
HTTP_PORT=0
DRY_RUN_SLEEP_SEC="${DRY_RUN_SLEEP_SEC:-0.5}"
FRESH_RUN=0
DRY_RUN=0
DEPLOY_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        -m|--mode)               TASK_CONFIG="$2"; shift 2 ;;
        -n|--name)               POLICY_NAME="$2"; shift 2 ;;
        -d|--ckpt-dir)           CKPT_DIR="$2"; shift 2 ;;
        -w|--num-workers)        NUM_WORKERS="$2"; shift 2 ;;
        --gpu-start)             GPU_START="$2"; shift 2 ;;
        --port)                  PORT_BASE="$2"; shift 2 ;;
        --server-python)         SERVER_PYTHON="$2"; shift 2 ;;
        --server-script)         SERVER_SCRIPT="$2"; shift 2 ;;
        --bind-host)             SERVER_BIND_HOST="$2"; shift 2 ;;
        --client-host)           SERVER_CLIENT_HOST="$2"; shift 2 ;;
        --ckpt-name)             DEPLOY_ARGS+=(--ckpt-name "$2"); shift 2 ;;
        --denoise-steps)         DEPLOY_ARGS+=(--denoise-steps "$2"); shift 2 ;;
        --denoise-mode)          DEPLOY_ARGS+=(--denoise-mode "$2"); shift 2 ;;
        --lead-modality)         DEPLOY_ARGS+=(--lead-modality "$2"); shift 2 ;;
        --variance-shift-alpha)  DEPLOY_ARGS+=(--variance-shift-alpha "$2"); shift 2 ;;
        --linear-offset)         DEPLOY_ARGS+=(--linear-offset "$2"); shift 2 ;;
        --inference-mode)        DEPLOY_ARGS+=(--inference-mode "$2"); shift 2 ;;
        --inference-horizon)     DEPLOY_ARGS+=(--inference-horizon "$2"); shift 2 ;;
        --inference-delay-steps) DEPLOY_ARGS+=(--inference-delay-steps "$2"); shift 2 ;;
        --test-num)              TEST_NUM="$2"; shift 2 ;;
        --seed)                  BASE_SEED="$2"; shift 2 ;;
        --min-remaining-for-dup) MIN_REMAINING_FOR_DUP="$2"; shift 2 ;;
        --no-dup)                NO_DUP=1; shift ;;
        --append-results)        APPEND_RESULTS=1; shift ;;
        --dispatch-port)         DISPATCH_PORT="$2"; shift 2 ;;
        --http-port)             HTTP_PORT="$2"; shift 2 ;;
        --dry-run|--dryrun)      DRY_RUN=1; shift ;;
        --fresh)                 FRESH_RUN=1; shift ;;
        -h|--help)               usage; exit 0 ;;
        -*)                      echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
        *)                       break ;;
    esac
done

[[ -z "${TASK_CONFIG}" || -z "${POLICY_NAME}" || -z "${CKPT_DIR}" ]] && {
    echo "[ERROR] Missing required flags: -m, -n, -d" >&2; usage; exit 1; }
[[ "${TASK_CONFIG}" != "demo_clean" && "${TASK_CONFIG}" != "demo_randomized" && "${TASK_CONFIG}" != "all" ]] && {
    echo "[ERROR] Invalid mode: ${TASK_CONFIG}" >&2; exit 1; }
(( NUM_WORKERS > 0 )) || { echo "[ERROR] --num-workers must be > 0" >&2; exit 1; }
(( $# > 0 )) || { echo "[ERROR] No tasks specified." >&2; usage; exit 1; }
if (( ! DRY_RUN )) && [[ ! -d "${CKPT_DIR}" ]]; then echo "[ERROR] ckpt_dir not found: ${CKPT_DIR}" >&2; exit 1; fi
if (( ! DRY_RUN )) && [[ ! -f "${SERVER_SCRIPT}" ]]; then echo "[ERROR] server script not found: ${SERVER_SCRIPT}" >&2; exit 1; fi

# Worker python: RoboTwin env for real runs; any python3 for dry-run.
if (( ! DRY_RUN )); then
    [[ -z "${ROBOTWIN_PYTHON:-}" ]] && ROBOTWIN_PYTHON="$(find_conda_python "${ROBOTWIN_ENV:-robotwin}")"
    export ROBOTWIN_PYTHON
    ROBOTWIN_PATH="${ROBOTWIN_PATH:?ROBOTWIN_PATH must be set to the RoboTwin repository root}"
    [[ -d "${ROBOTWIN_PATH}" ]] || { echo "[ERROR] ROBOTWIN_PATH not found: ${ROBOTWIN_PATH}" >&2; exit 1; }
fi
WORKER_PYTHON="${ROBOTWIN_PYTHON:-python3}"
DISPATCHER_PYTHON="${DISPATCHER_PYTHON:-${SERVER_PYTHON:-python3}}"

NNODES="${MLP_WORKER_NUM:-${NNODES:-${WORLD_SIZE:-1}}}"
NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-${RANK:-0}}}"
RUN_ID="${ROBOTWIN_RUN_ID:-latest}"

if [[ "${TASK_CONFIG}" == "all" ]]; then MODES=(demo_clean demo_randomized); else MODES=("${TASK_CONFIG}"); fi
mapfile -t TASKS < <(resolve_tasks "$@")
validate_task_names "${TASKS[@]}"
TOTAL_JOBS=$(( ${#TASKS[@]} * ${#MODES[@]} ))

LOG_ROOT="${ROBOTWIN_LOG_ROOT:-${CKPT_DIR}/robotwin_eval_logs}"
LOG_DIR="${LOG_ROOT}/${POLICY_NAME}_${TASK_CONFIG}_dlc_${RUN_ID}"
NODE_DIR="${LOG_DIR}/node${NODE_RANK}"
SERVER_LOG_DIR="${NODE_DIR}/servers"
ADDR_FILE="${LOG_DIR}/.dispatcher_addr"
RESULTS_FILE="${LOG_DIR}/results.jsonl"
SUMMARY_FILE="${LOG_DIR}/summary.tsv"
STATE_FILE="${LOG_DIR}/state.json"
DONE_FILE="${LOG_DIR}/.done"
DISPATCHER_LOG="${LOG_DIR}/dispatcher.log"

server_pids=()
worker_pids=()
DISPATCHER_PID=""
REAPER_PID=""

kill_group() {
    for pid in "${worker_pids[@]}"; do kill_tree "${pid}" "$1"; done
    for pid in "${server_pids[@]}"; do kill_tree "${pid}" "$1"; done
    [[ -n "${DISPATCHER_PID}" ]] && kill_tree "${DISPATCHER_PID}" "$1"
    # Include the .done reaper — otherwise it outlives cleanup and the final
    # `wait` can block on it until the dispatcher writes .done.
    [[ -n "${REAPER_PID}" ]] && kill_tree "${REAPER_PID}" "$1"
    return 0
}

cleanup() {
    local code=$?
    trap - EXIT INT TERM
    set +e  # teardown must never be aborted mid-way by set -e (e.g. a kill/[[ ]] returning 1)
    echo "" >&2
    echo "[node${NODE_RANK}] stopping workers/servers/dispatcher..." >&2
    kill_group TERM
    local deadline=$((SECONDS + 8))
    while (( SECONDS < deadline )); do
        local alive=0
        for pid in "${worker_pids[@]}" "${server_pids[@]}" ${DISPATCHER_PID:+$DISPATCHER_PID} ${REAPER_PID:+$REAPER_PID}; do
            [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && { alive=1; break; }
        done
        (( alive )) || break; sleep 0.2
    done
    kill_group KILL
    wait 2>/dev/null || true
    exit "${code}"
}
trap cleanup EXIT INT TERM

mkdir -p "${NODE_DIR}" "${SERVER_LOG_DIR}"

# ---------------------------------------------------------------------------
# Rank 0 starts the dispatcher and publishes its address; others wait for it.
# ---------------------------------------------------------------------------

if [[ "${NODE_RANK}" == "0" ]]; then
    if (( FRESH_RUN )); then
        rm -f "${ADDR_FILE}" "${RESULTS_FILE}" "${SUMMARY_FILE}" "${STATE_FILE}" "${DONE_FILE}" "${LOG_DIR}/run.env"
    elif [[ -f "${ADDR_FILE}" ]]; then
        echo "[rank0] ERROR: stale run metadata at ${LOG_DIR} (${ADDR_FILE})." >&2
        echo "[rank0] Use ROBOTWIN_RUN_ID=<new_id> or pass --fresh." >&2
        exit 1
    fi
    mkdir -p "${LOG_DIR}"
    {
        echo "run_id=${RUN_ID}"; echo "policy_name=${POLICY_NAME}"; echo "mode=${TASK_CONFIG}"
        echo "ckpt_dir=${CKPT_DIR}"; echo "test_num=${TEST_NUM}"; echo "seed=${BASE_SEED}"
        echo "nnodes=${NNODES}"; echo "num_workers_per_node=${NUM_WORKERS}"
        echo "total_jobs=${TOTAL_JOBS}"; echo "min_remaining_for_dup=${MIN_REMAINING_FOR_DUP}"
        echo "no_dup=${NO_DUP}"; echo "dry_run=${DRY_RUN}"; echo "scheduler=episode-level-dispatcher"
        printf 'tasks=%s\n' "${TASKS[*]}"
    } > "${LOG_DIR}/run.env"

    ADV_HOST="$(advertise_host)"
    dispatcher_args=(
        "${SCRIPT_DIR}/dispatcher.py" --host 0.0.0.0 --port "${DISPATCH_PORT}"
        --advertise-host "${ADV_HOST}"
        --tasks "${TASKS[@]}" --modes "${MODES[@]}"
        --test-num "${TEST_NUM}" --seed "${BASE_SEED}"
        --min-remaining-for-dup "${MIN_REMAINING_FOR_DUP}"
        --num-slots "$(( NNODES * NUM_WORKERS ))"
        --results "${RESULTS_FILE}" --summary "${SUMMARY_FILE}"
        --state-file "${STATE_FILE}" --addr-file "${ADDR_FILE}"
        --done-file "${DONE_FILE}"
    )
    (( NO_DUP )) && dispatcher_args+=(--no-dup)
    (( APPEND_RESULTS )) && dispatcher_args+=(--append-results)
    (( HTTP_PORT > 0 )) && dispatcher_args+=(--http-port "${HTTP_PORT}")
    # Optional watchdog tuning via env (see README): stall/worker/idle timeouts + give-up cap.
    [[ -n "${STALL_TIMEOUT:-}" ]]      && dispatcher_args+=(--stall-timeout "${STALL_TIMEOUT}")
    [[ -n "${WORKER_TIMEOUT:-}" ]]     && dispatcher_args+=(--worker-timeout "${WORKER_TIMEOUT}")
    [[ -n "${IDLE_GRACE:-}" ]]         && dispatcher_args+=(--idle-grace "${IDLE_GRACE}")
    [[ -n "${MAX_ATTEMPT_FACTOR:-}" ]] && dispatcher_args+=(--max-attempt-factor "${MAX_ATTEMPT_FACTOR}")

    "${DISPATCHER_PYTHON}" "${dispatcher_args[@]}" >"${DISPATCHER_LOG}" 2>&1 &
    DISPATCHER_PID=$!
    echo "[rank0] dispatcher pid ${DISPATCHER_PID} advertise=${ADV_HOST}:${DISPATCH_PORT}; log ${DISPATCHER_LOG}"
fi

echo "[node${NODE_RANK}] waiting for dispatcher address ${ADDR_FILE} (timeout ${QUEUE_READY_TIMEOUT_SEC}s)"
deadline=$((SECONDS + QUEUE_READY_TIMEOUT_SEC))
while [[ ! -s "${ADDR_FILE}" ]]; do
    if [[ "${NODE_RANK}" == "0" && -n "${DISPATCHER_PID}" ]] && ! kill -0 "${DISPATCHER_PID}" 2>/dev/null; then
        echo "[rank0] dispatcher died on startup; see ${DISPATCHER_LOG}" >&2; exit 1
    fi
    (( SECONDS >= deadline )) && { echo "[node${NODE_RANK}] timeout waiting for dispatcher" >&2; exit 1; }
    sleep 1
done
DISPATCHER_ADDR="$(tr -d '[:space:]' < "${ADDR_FILE}")"
echo "[node${NODE_RANK}] dispatcher at ${DISPATCHER_ADDR}"

cat <<BANNER
╔══════════════════════════════════════════════════════╗
║  OpenWAM RoboTwin DLC Eval (episode-level)
║  Nodes: ${NNODES}  Rank: ${NODE_RANK}  Workers/node: ${NUM_WORKERS}
║  Mode: ${TASK_CONFIG}  Jobs: ${TOTAL_JOBS}  test_num: ${TEST_NUM}
║  Dispatcher: ${DISPATCHER_ADDR}  Dry-run: ${DRY_RUN}
║  LOGS: ${LOG_DIR}
╚══════════════════════════════════════════════════════╝
BANNER

# ---------------------------------------------------------------------------
# Local policy servers (skipped in dry-run).
# ---------------------------------------------------------------------------

if (( ! DRY_RUN )); then
    echo "[node${NODE_RANK}] starting ${NUM_WORKERS} local policy servers"
    for ((i = 0; i < NUM_WORKERS; i++)); do
        gpu=$((GPU_START + i * SIM_GPU_STRIDE)); port=$((PORT_BASE + i))
        server_log="${SERVER_LOG_DIR}/server_worker${i}_gpu${gpu}.log"
        CUDA_VISIBLE_DEVICES="${gpu}" "${SERVER_PYTHON}" "${SERVER_SCRIPT}" \
            --ckpt-dir "${CKPT_DIR}" --device "cuda:0" \
            --host "${SERVER_BIND_HOST}" --port "${port}" \
            "${DEPLOY_ARGS[@]}" > "${server_log}" 2>&1 &
        server_pids+=($!)
    done
    for ((i = 0; i < NUM_WORKERS; i++)); do
        port=$((PORT_BASE + i)); gpu=$((GPU_START + i * SIM_GPU_STRIDE))
        wait_for_server "${SERVER_CLIENT_HOST}" "${port}" "${SERVER_LOG_DIR}/server_worker${i}_gpu${gpu}.log" "${SERVER_READY_TIMEOUT_SEC}"
    done
    echo "[node${NODE_RANK}] all local servers healthy"
fi

# ---------------------------------------------------------------------------
# Slot supervisors: relaunch episode_worker until the dispatcher says "exit".
# ---------------------------------------------------------------------------

run_supervisor() {
    # Backgrounded (`&`) subshells inherit the parent's EXIT trap; without this
    # each finishing supervisor would re-run cleanup() — tearing down the whole
    # group and polluting the exit code. Signals are still handled by the main
    # shell's trap, which kill_tree's these children.
    trap - EXIT INT TERM
    local slot_idx="$1"
    local sim_gpu=$((GPU_START + slot_idx * SIM_GPU_STRIDE))
    local port=$((PORT_BASE + slot_idx))
    local worker_dir="${NODE_DIR}/worker${slot_idx}"
    local worker_log="${worker_dir}/worker.log"
    mkdir -p "${worker_dir}"
    local tag="[node${NODE_RANK}/worker${slot_idx}@gpu${sim_gpu}:${port}]"
    echo "${tag} supervisor started" | tee -a "${worker_log}"

    local attempt=0 rc=0
    while :; do
        attempt=$((attempt + 1))
        local run_log="${worker_dir}/run$(printf '%03d' "${attempt}").log"
        if (( DRY_RUN )); then
            "${WORKER_PYTHON}" "${SCRIPT_DIR}/episode_worker.py" --dry-run \
                --dispatcher "${DISPATCHER_ADDR}" --node "${NODE_RANK}" --worker "${slot_idx}" --gpu "${sim_gpu}" \
                --dry-run-sleep "${DRY_RUN_SLEEP_SEC}" >"${run_log}" 2>&1 && rc=0 || rc=$?
        else
            bash "${SCRIPT_DIR}/episode_eval.sh" \
                "${DISPATCHER_ADDR}" "${NODE_RANK}" "${slot_idx}" "${sim_gpu}" "${port}" "${SERVER_CLIENT_HOST}" "${POLICY_NAME}" "${BASE_SEED}" \
                >"${run_log}" 2>&1 && rc=0 || rc=$?
            grep --color=never "Success rate" "${run_log}" | sed "s|^|[RESULT] ${tag} |" || true
        fi
        case "${rc}" in
            0) : ;;
            3) echo "${tag} no more work; supervisor exiting" | tee -a "${worker_log}"; break ;;
            *) echo "${tag} worker error (rc=${rc}); see ${run_log}; supervisor exiting" | tee -a "${worker_log}" >&2; break ;;
        esac
    done
}

for ((i = 0; i < NUM_WORKERS; i++)); do
    run_supervisor "${i}" &
    worker_pids+=($!)
done
echo "[node${NODE_RANK}] launched ${NUM_WORKERS} slot supervisors: ${worker_pids[*]}"

# Reaper: once rank0's dispatcher signals done (complete OR stall/abort) it
# touches the shared DONE_FILE; every node then tears down its local (possibly
# wedged) workers so no node's `wait` hangs on a stuck sim process.
reaper_supervisors=("${worker_pids[@]}")
( trap - EXIT INT TERM
  while [[ ! -f "${DONE_FILE}" ]]; do sleep 2; done
  for pid in "${reaper_supervisors[@]}"; do kill_tree "${pid}" TERM; done ) &
REAPER_PID=$!

for pid in "${worker_pids[@]}"; do wait "${pid}" || true; done
worker_pids=()
kill_tree "${REAPER_PID}" TERM 2>/dev/null || true
REAPER_PID=""
echo "[node${NODE_RANK}] local supervisors finished"

# ---------------------------------------------------------------------------
# Rank 0 waits for the dispatcher to complete, then checks + prints the summary.
# ---------------------------------------------------------------------------

if [[ "${NODE_RANK}" == "0" && -n "${DISPATCHER_PID}" ]]; then
    wait "${DISPATCHER_PID}" 2>/dev/null || true
    DISPATCHER_PID=""
    if [[ -f "${SUMMARY_FILE}" ]]; then
        finished="$(awk -F '\t' 'NR>1 && $7=="ok" {c++} END {print c+0}' "${SUMMARY_FILE}")"
        echo "[SUMMARY] finished_jobs=${finished}/${TOTAL_JOBS}"
        column -t -s $'\t' < "${SUMMARY_FILE}" || cat "${SUMMARY_FILE}"
        echo "[SUMMARY] logs=${LOG_DIR}"
        echo "[INFO] Export CSV:  ${DISPATCHER_PYTHON} ${SCRIPT_DIR}/export_results_csv.py ${LOG_DIR}"
        if (( finished != TOTAL_JOBS )); then
            echo "[ERROR] incomplete: ${finished}/${TOTAL_JOBS} jobs reached target" >&2
            exit 1
        fi
    else
        echo "[ERROR] no summary.tsv produced; see ${DISPATCHER_LOG}" >&2
        exit 1
    fi
fi

echo "[node${NODE_RANK}] done"
