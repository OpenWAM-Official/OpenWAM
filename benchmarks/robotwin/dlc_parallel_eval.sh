#!/usr/bin/env bash
# DLC multi-node RoboTwin evaluation.
#
# Each DLC worker runs this same script. Rank 0 creates one shared file per
# task/mode job; every node starts local OpenWAM policy servers and local
# RoboTwin clients. Clients across all nodes atomically claim jobs by moving a
# pending job file into the claimed directory, so faster nodes keep taking work
# until the whole run is finished without editing a shared queue file.
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
# Important environment overrides:
#   ROBOTWIN_PATH        path to the RoboTwin repository (required)
#   ROBOTWIN_PYTHON      Python for RoboTwin (or set ROBOTWIN_ENV)
#   ROBOTWIN_RUN_ID      shared run id; default: latest
#   ROBOTWIN_LOG_ROOT    shared log root; default: <ckpt_dir>/robotwin_eval_logs
#   NUM_WORKERS          local servers/clients per node; default: GPU count
#   GPU_START            first local GPU index; default: 0
#   SIM_GPU_STRIDE       stride between worker GPUs; default: 1
#   PORT_BASE            local WebSocket port base; default: 8848
#   SERVER_PYTHON        Python used to launch local policy servers; default: python
#   SERVER_SCRIPT        Python script used to launch local policy servers; default: <repo>/scripts/deploy.py
#   SERVER_BIND_HOST     server bind host; default: 127.0.0.1
#   SERVER_CLIENT_HOST   host passed to local RoboTwin clients; default: 127.0.0.1
#   DRY_RUN_SLEEP_SEC    seconds to sleep for each claimed dry-run job; default: 1
#   DRY_RUN_BARRIER_TIMEOUT_SEC  dry-run node rendezvous timeout; default: QUEUE_READY_TIMEOUT_SEC
#
# Example:
#   ROBOTWIN_PATH=/path/to ROBOTWIN_RUN_ID=ckpt_100k \
#   bash benchmarks/robotwin/dlc_parallel_eval.sh \
#       -m all -n openwam -d /path/to/openwam all
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

usage() {
    cat >&2 <<'EOF'
Usage:
  bash dlc_parallel_eval.sh -m <mode> -n <name> -d <ckpt_dir> [options] <tasks...>

Required:
  -m, --mode       demo_clean | demo_randomized | all
  -n, --name       label for log directory naming
  -d, --ckpt-dir   OpenWAM checkpoint directory

Tasks (positional): task names, "all", or a task-list file (one per line).

Options:
  -w, --num-workers    local servers/clients per node (default: GPU count)
      --gpu-start      first local GPU index (default: 0)
      --port           local WebSocket port base (default: 8848)
      --server-python  Python used to launch local policy servers (default: python)
      --server-script  Python script used to launch local policy servers
      --bind-host      OpenWAM server bind host (default: 127.0.0.1)
      --client-host    host used by local RoboTwin clients (default: 127.0.0.1)
      --ckpt-name      checkpoint filename passed to scripts/deploy.py
      --denoise-steps  denoising step count passed to scripts/deploy.py
      --schedule-type  schedule type passed to scripts/deploy.py
      --async-mode     async inference mode passed to scripts/deploy.py
      --async-execution-horizon N
                       async execution horizon passed to scripts/deploy.py
      --async-inference-delay-steps N
                       async inference delay passed to scripts/deploy.py
      --shift          flow-matching shift passed to scripts/deploy.py
      --dry-run        skip servers/eval and only test shared-queue assignment
      --fresh          remove this run's stale queue/sentinel/log metadata first
  -h, --help

Common DLC invocation:
  ROBOTWIN_PATH=/path/to/RoboTwin ROBOTWIN_RUN_ID=my_run \
  bash benchmarks/robotwin/dlc_parallel_eval.sh -m all -n openwam -d /ckpt/openwam all
EOF
}

trim() {
    local v="$1"
    v="${v#"${v%%[![:space:]]*}"}"
    v="${v%"${v##*[![:space:]]}"}"
    printf '%s\n' "${v}"
}

is_safe_task_name() {
    [[ "$1" =~ ^[A-Za-z0-9_]+$ ]]
}

validate_task_names() {
    local task
    for task in "$@"; do
        if ! is_safe_task_name "${task}"; then
            echo "[ERROR] Invalid task name '${task}'. Expected [A-Za-z0-9_]+." >&2
            return 1
        fi
    done
}

resolve_tasks() {
    local -a raw=("$@") out=() parts=()
    if (( ${#raw[@]} == 1 )) && [[ -f "${raw[0]}" ]]; then
        local line
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="$(trim "${line%%#*}")"
            [[ -n "${line}" ]] && out+=("${line}")
        done < "${raw[0]}"
    else
        local inp task
        for inp in "${raw[@]}"; do
            if [[ "${inp}" == "all" ]]; then
                out+=("${ROBOTWIN_ALL_TASKS[@]}")
                continue
            fi
            IFS=',' read -ra parts <<< "${inp}"
            for task in "${parts[@]}"; do
                task="$(trim "${task}")"
                [[ -n "${task}" ]] && out+=("${task}")
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
        local count
        count="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
        if [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 )); then
            printf '%s\n' "${count}"
            return
        fi
    fi
    printf '1\n'
}

kill_tree() {
    local pid=$1 sig=${2:-TERM}
    [[ -z "${pid}" ]] && return
    local child
    while read -r child; do
        [[ -n "${child}" ]] && kill_tree "${child}" "${sig}"
    done < <(pgrep -P "${pid}" 2>/dev/null || true)
    kill -"${sig}" "${pid}" 2>/dev/null || true
}

# Readiness probe: the server binds its WebSocket port only after the model is
# loaded (deploy.py builds the engine, then run() opens the listener), so a
# successful TCP connect means it is ready. Bash /dev/tcp keeps each poll cheap
# — no per-attempt Python interpreter startup.
health_check() {
    local host="$1" port="$2"
    timeout 2 bash -c ">/dev/tcp/${host}/${port}" 2>/dev/null
}

wait_for_server() {
    local host="$1" port="$2" log_file="$3" timeout_sec="$4"
    local deadline=$((SECONDS + timeout_sec))
    until health_check "${host}" "${port}"; do
        if (( SECONDS >= deadline )); then
            echo "[ERROR] Server did not become ready: ws://${host}:${port}" >&2
            echo "[ERROR] See ${log_file}" >&2
            return 1
        fi
        sleep 2
    done
}

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
DRY_RUN_SLEEP_SEC="${DRY_RUN_SLEEP_SEC:-1}"
DRY_RUN_BARRIER_TIMEOUT_SEC="${DRY_RUN_BARRIER_TIMEOUT_SEC:-${QUEUE_READY_TIMEOUT_SEC}}"
FRESH_RUN=0
DRY_RUN=0

DEPLOY_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        -m|--mode)          TASK_CONFIG="$2"; shift 2 ;;
        -n|--name)          POLICY_NAME="$2"; shift 2 ;;
        -d|--ckpt-dir)      CKPT_DIR="$2"; shift 2 ;;
        -w|--num-workers)   NUM_WORKERS="$2"; shift 2 ;;
        --gpu-start)        GPU_START="$2"; shift 2 ;;
        --port)             PORT_BASE="$2"; shift 2 ;;
        --server-python)    SERVER_PYTHON="$2"; shift 2 ;;
        --server-script)    SERVER_SCRIPT="$2"; shift 2 ;;
        --bind-host)        SERVER_BIND_HOST="$2"; shift 2 ;;
        --client-host)      SERVER_CLIENT_HOST="$2"; shift 2 ;;
        --ckpt-name)        DEPLOY_ARGS+=(--ckpt-name "$2"); shift 2 ;;
        --denoise-steps)    DEPLOY_ARGS+=(--denoise-steps "$2"); shift 2 ;;
        --schedule-type)    DEPLOY_ARGS+=(--schedule-type "$2"); shift 2 ;;
        --async-mode)       DEPLOY_ARGS+=(--async-mode "$2"); shift 2 ;;
        --async-execution-horizon) DEPLOY_ARGS+=(--async-execution-horizon "$2"); shift 2 ;;
        --async-inference-delay-steps) DEPLOY_ARGS+=(--async-inference-delay-steps "$2"); shift 2 ;;
        --shift)            DEPLOY_ARGS+=(--shift "$2"); shift 2 ;;
        --dry-run|--dryrun) DRY_RUN=1; shift ;;
        --fresh)            FRESH_RUN=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        -*)                 echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
        *)                  break ;;
    esac
done

[[ -z "${TASK_CONFIG}" || -z "${POLICY_NAME}" || -z "${CKPT_DIR}" ]] && {
    echo "[ERROR] Missing required flags: -m, -n, -d" >&2; usage; exit 1; }
[[ "${TASK_CONFIG}" != "demo_clean" && "${TASK_CONFIG}" != "demo_randomized" && "${TASK_CONFIG}" != "all" ]] && {
    echo "[ERROR] Invalid mode: ${TASK_CONFIG}" >&2; exit 1; }
if (( DRY_RUN )) && ! [[ "${DRY_RUN_SLEEP_SEC}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[ERROR] DRY_RUN_SLEEP_SEC must be a non-negative number: ${DRY_RUN_SLEEP_SEC}" >&2
    exit 1
fi
if (( DRY_RUN )) && ! [[ "${DRY_RUN_BARRIER_TIMEOUT_SEC}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] DRY_RUN_BARRIER_TIMEOUT_SEC must be a non-negative integer: ${DRY_RUN_BARRIER_TIMEOUT_SEC}" >&2
    exit 1
fi
if (( ! DRY_RUN )) && [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[ERROR] ckpt_dir not found: ${CKPT_DIR}" >&2
    exit 1
fi
if (( ! DRY_RUN )); then
    [[ -f "${SERVER_SCRIPT}" ]] || { echo "[ERROR] server script not found: ${SERVER_SCRIPT}" >&2; exit 1; }
fi
(( NUM_WORKERS > 0 )) || { echo "[ERROR] --num-workers must be > 0" >&2; exit 1; }
(( $# > 0 )) || { echo "[ERROR] No tasks specified." >&2; usage; exit 1; }

if (( ! DRY_RUN )); then
    if [[ -z "${ROBOTWIN_PYTHON:-}" ]]; then
        ROBOTWIN_PYTHON="$(find_conda_python "${ROBOTWIN_ENV:-robotwin}")"
    fi
    export ROBOTWIN_PYTHON

    ROBOTWIN_PATH="${ROBOTWIN_PATH:?ROBOTWIN_PATH must be set to the RoboTwin repository root}"
    [[ -d "${ROBOTWIN_PATH}" ]] || { echo "[ERROR] ROBOTWIN_PATH not found: ${ROBOTWIN_PATH}" >&2; exit 1; }
fi

NNODES="${MLP_WORKER_NUM:-${NNODES:-${WORLD_SIZE:-1}}}"
NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-${RANK:-0}}}"
RUN_ID="${ROBOTWIN_RUN_ID:-latest}"

if [[ "${TASK_CONFIG}" == "all" ]]; then
    MODES=(demo_clean demo_randomized)
else
    MODES=("${TASK_CONFIG}")
fi
mapfile -t TASKS < <(resolve_tasks "$@")
validate_task_names "${TASKS[@]}"
TOTAL_JOBS=$(( ${#TASKS[@]} * ${#MODES[@]} ))

LOG_ROOT="${ROBOTWIN_LOG_ROOT:-${CKPT_DIR}/robotwin_eval_logs}"
LOG_DIR="${LOG_ROOT}/${POLICY_NAME}_${TASK_CONFIG}_dlc_${RUN_ID}"
NODE_DIR="${LOG_DIR}/node${NODE_RANK}"
QUEUE_FILE="${LOG_DIR}/.queue.txt"
QUEUE_DIR="${LOG_DIR}/queue"
QUEUE_PENDING_DIR="${QUEUE_DIR}/pending"
QUEUE_CLAIMED_DIR="${QUEUE_DIR}/claimed"
READY_FILE="${LOG_DIR}/.queue_ready"
SUMMARY_FILE="${LOG_DIR}/summary.tsv"
SUMMARY_LOCK_DIR="${LOG_DIR}/summary.lock.d"
DRY_RUN_NODE_READY_DIR="${LOG_DIR}/dryrun_node_ready"
DRY_RUN_START_FILE="${LOG_DIR}/.dryrun_start"
DRY_RUN_WORKER_READY_DIR="${LOG_DIR}/dryrun_worker_ready"
DRY_RUN_WORKER_START_FILE="${LOG_DIR}/.dryrun_workers_start"
SERVER_LOG_DIR="${NODE_DIR}/servers"

server_pids=()
worker_pids=()

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    echo ""
    echo "[node${NODE_RANK}] stopping workers and servers..." >&2
    for pid in "${worker_pids[@]}"; do kill_tree "${pid}" TERM; done
    for pid in "${server_pids[@]}"; do kill_tree "${pid}" TERM; done

    local deadline=$((SECONDS + 8)) still_alive=1
    while (( SECONDS < deadline )); do
        still_alive=0
        for pid in "${worker_pids[@]}" "${server_pids[@]}"; do
            [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && { still_alive=1; break; }
        done
        (( still_alive )) || break
        sleep 0.2
    done
    if (( still_alive )); then
        echo "[node${NODE_RANK}] escalating to SIGKILL for survivors..." >&2
        for pid in "${worker_pids[@]}" "${server_pids[@]}"; do kill_tree "${pid}" KILL; done
    fi
    wait 2>/dev/null || true
    echo "[node${NODE_RANK}] cleanup complete." >&2
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

mkdir -p "${NODE_DIR}" "${SERVER_LOG_DIR}"

if [[ "${NODE_RANK}" == "0" ]]; then
    if (( FRESH_RUN )); then
        rm -f "${QUEUE_FILE}" "${READY_FILE}" "${SUMMARY_FILE}"
        rm -f "${DRY_RUN_START_FILE}" "${DRY_RUN_WORKER_START_FILE}"
        rm -rf "${QUEUE_DIR}" "${LOG_DIR}/.queue.lock.d" "${SUMMARY_LOCK_DIR}" \
            "${DRY_RUN_NODE_READY_DIR}" "${DRY_RUN_WORKER_READY_DIR}"
        find "${LOG_DIR}" -maxdepth 1 -name '.node*_done' -type f -delete 2>/dev/null || true
    elif [[ -f "${READY_FILE}" ]]; then
        echo "[rank0] ERROR: stale run metadata exists at ${LOG_DIR}" >&2
        echo "[rank0] Use ROBOTWIN_RUN_ID=<new_id> or pass --fresh." >&2
        exit 1
    fi

    mkdir -p "${LOG_DIR}"
    : > "${QUEUE_FILE}"
    rm -rf "${QUEUE_DIR}"
    mkdir -p "${QUEUE_PENDING_DIR}" "${QUEUE_CLAIMED_DIR}"
    job_id=0
    for task in "${TASKS[@]}"; do
        for mode in "${MODES[@]}"; do
            printf '%s|%s\n' "${task}" "${mode}" >> "${QUEUE_FILE}"
            job_file="${QUEUE_PENDING_DIR}/$(printf '%06d' "${job_id}")_${task}_${mode}.job"
            printf 'task=%s\nmode=%s\n' "${task}" "${mode}" > "${job_file}"
            job_id=$((job_id + 1))
        done
    done
    printf 'task\tmode\tnode\tworker\tstatus\texit_code\tlog\n' > "${SUMMARY_FILE}"
    {
        echo "run_id=${RUN_ID}"
        echo "policy_name=${POLICY_NAME}"
        echo "mode=${TASK_CONFIG}"
        echo "ckpt_dir=${CKPT_DIR}"
        echo "server_python=${SERVER_PYTHON}"
        echo "server_script=${SERVER_SCRIPT}"
        echo "nnodes=${NNODES}"
        echo "num_workers_per_node=${NUM_WORKERS}"
        echo "total_jobs=${TOTAL_JOBS}"
        echo "queue_backend=per-job-mv"
        echo "dry_run=${DRY_RUN}"
        if (( DRY_RUN )); then
            echo "dry_run_sleep_sec=${DRY_RUN_SLEEP_SEC}"
            echo "dry_run_barrier_timeout_sec=${DRY_RUN_BARRIER_TIMEOUT_SEC}"
        fi
        printf 'tasks=%s\n' "${TASKS[*]}"
    } > "${LOG_DIR}/run.env"
    touch "${READY_FILE}"
    echo "[rank0] queue initialized: ${TOTAL_JOBS} per-job files at ${QUEUE_PENDING_DIR}"
else
    echo "[rank${NODE_RANK}] waiting for queue ${READY_FILE} (timeout ${QUEUE_READY_TIMEOUT_SEC}s)"
    deadline=$((SECONDS + QUEUE_READY_TIMEOUT_SEC))
    while [[ ! -f "${READY_FILE}" ]]; do
        if (( SECONDS >= deadline )); then
            echo "[rank${NODE_RANK}] queue timeout; aborting" >&2
            exit 1
        fi
        sleep 1
    done
fi

if (( DRY_RUN )); then
    mkdir -p "${DRY_RUN_NODE_READY_DIR}" "${DRY_RUN_WORKER_READY_DIR}"
    touch "${DRY_RUN_NODE_READY_DIR}/node${NODE_RANK}"
    if [[ "${NODE_RANK}" == "0" ]]; then
        echo "[rank0] dry-run waiting for ${NNODES} node(s) before task claiming"
        deadline=$((SECONDS + DRY_RUN_BARRIER_TIMEOUT_SEC))
        for ((rank = 0; rank < NNODES; rank++)); do
            ready_node="${DRY_RUN_NODE_READY_DIR}/node${rank}"
            while [[ ! -f "${ready_node}" ]]; do
                if (( SECONDS >= deadline )); then
                    echo "[rank0] dry-run timeout waiting for ${ready_node}" >&2
                    exit 1
                fi
                sleep 1
            done
        done
        touch "${DRY_RUN_START_FILE}"
    else
        echo "[rank${NODE_RANK}] dry-run waiting for rank0 start signal"
        deadline=$((SECONDS + DRY_RUN_BARRIER_TIMEOUT_SEC))
        while [[ ! -f "${DRY_RUN_START_FILE}" ]]; do
            if (( SECONDS >= deadline )); then
                echo "[rank${NODE_RANK}] dry-run start timeout; aborting" >&2
                exit 1
            fi
            sleep 1
        done
    fi
fi

cat <<BANNER
╔══════════════════════════════════════════════════════╗
║  OpenWAM RoboTwin DLC Eval                          ║
║  Nodes: ${NNODES}  Rank: ${NODE_RANK}  Workers/node: ${NUM_WORKERS}
║  Mode: ${TASK_CONFIG}  Jobs: ${TOTAL_JOBS}
║  Dry-run: ${DRY_RUN}
║  CKPT: ${CKPT_DIR}
║  Server: ${SERVER_PYTHON} ${SERVER_SCRIPT}
║  LOGS: ${LOG_DIR}
╚══════════════════════════════════════════════════════╝
BANNER

if (( DRY_RUN )); then
    echo "[node${NODE_RANK}] dry-run: skipping local policy server startup and RoboTwin eval"
else
    echo "[node${NODE_RANK}] starting ${NUM_WORKERS} local policy servers"
    for ((i = 0; i < NUM_WORKERS; i++)); do
        gpu=$((GPU_START + i * SIM_GPU_STRIDE))
        port=$((PORT_BASE + i))
        server_log="${SERVER_LOG_DIR}/server_worker${i}_gpu${gpu}.log"

        echo "[node${NODE_RANK}] server worker${i}: gpu=${gpu} ws=${port}"
        # Pin each worker to exactly one physical GPU via CUDA_VISIBLE_DEVICES.
        # `--device cuda:0` then refers to that single visible device. Without
        # this, `torch.cuda.current_device()` defaults to 0 in every worker and
        # any default-cuda allocation (e.g. lazy buffer init that bypasses the
        # explicit --device plumbing) lands on physical GPU 0 regardless of
        # what we passed — which caused the cuda:0 vs cuda:N mismatch crash at
        # `block.adaln_modulation_self_attn` on workers 1-7.
        CUDA_VISIBLE_DEVICES="${gpu}" "${SERVER_PYTHON}" "${SERVER_SCRIPT}" \
            --ckpt-dir "${CKPT_DIR}" \
            --device "cuda:0" \
            --host "${SERVER_BIND_HOST}" \
            --port "${port}" \
            "${DEPLOY_ARGS[@]}" \
            > "${server_log}" 2>&1 &
        server_pids+=($!)
    done

    for ((i = 0; i < NUM_WORKERS; i++)); do
        gpu=$((GPU_START + i * SIM_GPU_STRIDE))
        port=$((PORT_BASE + i))
        server_log="${SERVER_LOG_DIR}/server_worker${i}_gpu${gpu}.log"
        wait_for_server "${SERVER_CLIENT_HOST}" "${port}" \
            "${server_log}" "${SERVER_READY_TIMEOUT_SEC}"
    done
    echo "[node${NODE_RANK}] all local servers are healthy"
fi

with_lock_dir() {
    local lock_dir="$1"
    local command="$2"
    local result_file="$3"
    local acquired=0
    local rc=0
    while (( ! acquired )); do
        if mkdir "${lock_dir}" 2>/dev/null; then
            acquired=1
        else
            sleep 0.05
        fi
    done
    "${command}" "${result_file}" || rc=$?
    rmdir "${lock_dir}" 2>/dev/null || true
    return "${rc}"
}

claim_queue_item() {
    local result_file="$1"
    local worker_idx="$2"
    local job_file claimed_file task mode
    : > "${result_file}"

    # Each job is a standalone file. `mv` within the shared queue directory is
    # the atomic claim operation, so workers never edit the same queue file.
    shopt -s nullglob
    for job_file in "${QUEUE_PENDING_DIR}"/*.job; do
        claimed_file="${QUEUE_CLAIMED_DIR}/$(basename "${job_file}").node${NODE_RANK}.worker${worker_idx}"
        if mv "${job_file}" "${claimed_file}" 2>/dev/null; then
            task="$(awk -F= '$1 == "task" {print $2; exit}' "${claimed_file}")"
            mode="$(awk -F= '$1 == "mode" {print $2; exit}' "${claimed_file}")"
            if [[ -z "${task}" || -z "${mode}" ]]; then
                echo "[ERROR] malformed claimed job file: ${claimed_file}" >&2
                return 13
            fi
            printf '%s|%s|%s' "${task}" "${mode}" "${claimed_file}" > "${result_file}"
            shopt -u nullglob
            return 0
        fi
    done
    shopt -u nullglob
    return 0
}

append_summary_row() {
    local row_file="$1"
    cat "${row_file}" >> "${SUMMARY_FILE}"
}

run_worker() {
    local worker_idx="$1"
    local sim_gpu=$((GPU_START + worker_idx * SIM_GPU_STRIDE))
    local port=$((PORT_BASE + worker_idx))
    local worker_dir="${NODE_DIR}/worker${worker_idx}"
    local worker_log="${worker_dir}/worker.log"
    local finished_file="${worker_dir}/finished.txt"
    local failed_file="${worker_dir}/failed.txt"

    mkdir -p "${worker_dir}"
    : > "${finished_file}"
    : > "${failed_file}"

    local tag="[node${NODE_RANK}/worker${worker_idx}@gpu${sim_gpu}:${port}]"
    echo "${tag} started" | tee -a "${worker_log}"

    if (( DRY_RUN )); then
        touch "${DRY_RUN_WORKER_READY_DIR}/node${NODE_RANK}_worker${worker_idx}"
        echo "${tag} dry-run ready for synchronized task claiming" | tee -a "${worker_log}"
        local dryrun_deadline=$((SECONDS + DRY_RUN_BARRIER_TIMEOUT_SEC))
        while [[ ! -f "${DRY_RUN_WORKER_START_FILE}" ]]; do
            if (( SECONDS >= dryrun_deadline )); then
                echo "${tag} dry-run worker start timeout" | tee -a "${worker_log}" >&2
                return 1
            fi
            sleep 0.2
        done
    fi

    while :; do
        local item task mode claimed_job_file eval_exit task_log tmp_item_file tmp_summary_file
        tmp_item_file="${worker_dir}/.queue_pop.tmp"
        claim_queue_item "${tmp_item_file}" "${worker_idx}" || exit $?
        item="$(cat "${tmp_item_file}")"

        [[ -z "${item}" ]] && break
        IFS='|' read -r task mode claimed_job_file <<< "${item}"
        if ! is_safe_task_name "${task}"; then
            echo "${tag} invalid task name from queue: ${task}" | tee -a "${worker_log}" >&2
            return 1
        fi
        if [[ "${mode}" != "demo_clean" && "${mode}" != "demo_randomized" ]]; then
            echo "${tag} invalid mode from queue: ${mode}" | tee -a "${worker_log}" >&2
            return 1
        fi
        task_log="${worker_dir}/${task}_${mode}.log"

        echo "${tag} claimed $(basename "${claimed_job_file}") task=${task} mode=${mode}" | tee -a "${worker_log}"
        if (( DRY_RUN )); then
            {
                echo "dry_run=1"
                echo "task=${task}"
                echo "mode=${mode}"
                echo "node=${NODE_RANK}"
                echo "worker=${worker_idx}"
                echo "sim_gpu=${sim_gpu}"
                echo "port=${port}"
                echo "claimed_job_file=${claimed_job_file}"
                echo "sleep_sec=${DRY_RUN_SLEEP_SEC}"
                echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            } > "${task_log}"
            echo "${tag} dry-run simulating task=${task} mode=${mode} for ${DRY_RUN_SLEEP_SEC}s" \
                | tee -a "${worker_log}"
            if sleep "${DRY_RUN_SLEEP_SEC}"; then
                eval_exit=0
                {
                    echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                    echo "assignment_status=ok"
                } >> "${task_log}"
            else
                eval_exit=$?
                echo "assignment_status=failed" >> "${task_log}"
            fi
        else
            ROBOTWIN_PORT="${port}" ROBOTWIN_POLICY_HOST="${SERVER_CLIENT_HOST}" \
            bash "${SCRIPT_DIR}/single_eval.sh" \
                "${task}" "${mode}" "${POLICY_NAME}" \
                "${sim_gpu}" \
                "${port}" "${SERVER_CLIENT_HOST}" \
                > "${task_log}" 2>&1 \
                && eval_exit=0 || eval_exit=$?

            grep --color=never "Success rate" "${task_log}" \
                | sed "s|^|[RESULT] ${tag} ${task} (${mode}): |" || true
        fi

        tmp_summary_file="${worker_dir}/.summary_row.tmp"
        if (( eval_exit == 0 )); then
            printf '%s\t%s\t%s\t%s\tok\t0\t%s\n' \
                "${task}" "${mode}" "${NODE_RANK}" "${worker_idx}" "${task_log}" > "${tmp_summary_file}"
        else
            printf '%s\t%s\t%s\t%s\tfailed\t%s\t%s\n' \
                "${task}" "${mode}" "${NODE_RANK}" "${worker_idx}" "${eval_exit}" "${task_log}" > "${tmp_summary_file}"
        fi
        with_lock_dir "${SUMMARY_LOCK_DIR}" append_summary_row "${tmp_summary_file}"

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

for ((i = 0; i < NUM_WORKERS; i++)); do
    run_worker "${i}" &
    worker_pids+=($!)
done

echo "[node${NODE_RANK}] launched client workers: ${worker_pids[*]}"
if (( DRY_RUN )) && [[ "${NODE_RANK}" == "0" ]]; then
    expected_worker_ready=$((NNODES * NUM_WORKERS))
    echo "[rank0] dry-run waiting for ${expected_worker_ready} worker(s) before task claiming"
    deadline=$((SECONDS + DRY_RUN_BARRIER_TIMEOUT_SEC))
    while :; do
        ready_count="$(find "${DRY_RUN_WORKER_READY_DIR}" -maxdepth 1 -name 'node*_worker*' -type f 2>/dev/null | awk 'END {print NR}')"
        if (( ready_count >= expected_worker_ready )); then
            break
        fi
        if (( SECONDS >= deadline )); then
            echo "[rank0] dry-run timeout waiting for workers: ready=${ready_count}, expected=${expected_worker_ready}" >&2
            exit 1
        fi
        sleep 0.2
    done
    touch "${DRY_RUN_WORKER_START_FILE}"
fi
wait "${worker_pids[@]}"
worker_pids=()

touch "${LOG_DIR}/.node${NODE_RANK}_done"
echo "[node${NODE_RANK}] local workers finished"

if [[ "${NODE_RANK}" == "0" ]]; then
    echo "[rank0] waiting for all node done sentinels"
    deadline=$((SECONDS + ${ALL_NODES_DONE_TIMEOUT_SEC:-86400}))
    for ((rank = 0; rank < NNODES; rank++)); do
        done_file="${LOG_DIR}/.node${rank}_done"
        while [[ ! -f "${done_file}" ]]; do
            if (( SECONDS >= deadline )); then
                echo "[rank0] timeout waiting for ${done_file}" >&2
                exit 1
            fi
            sleep 2
        done
    done

    failed_count="$(awk -F '\t' 'NR > 1 && $5 != "ok" {c++} END {print c+0}' "${SUMMARY_FILE}")"
    finished_count="$(awk -F '\t' 'NR > 1 && $5 == "ok" {c++} END {print c+0}' "${SUMMARY_FILE}")"
    unique_count="$(awk -F '\t' 'NR > 1 {seen[$1 FS $2]=1} END {print length(seen)+0}' "${SUMMARY_FILE}")"
    echo "[SUMMARY] finished=${finished_count} failed=${failed_count} unique=${unique_count} total=${TOTAL_JOBS}"
    echo "[SUMMARY] logs=${LOG_DIR}"
    if (( finished_count + failed_count != TOTAL_JOBS )); then
        echo "[ERROR] Summary row count mismatch: expected ${TOTAL_JOBS}, got $((finished_count + failed_count))" >&2
        exit 1
    fi
    if (( unique_count != TOTAL_JOBS )); then
        echo "[ERROR] Summary contains duplicate or missing task/mode entries: unique=${unique_count}, expected=${TOTAL_JOBS}" >&2
        exit 1
    fi
    if (( failed_count > 0 )); then
        echo "[ERROR] Failed jobs are recorded in ${SUMMARY_FILE}" >&2
        exit 1
    fi
    if (( DRY_RUN )); then
        echo "[DRYRUN] shared-queue assignment validated: ${finished_count}/${TOTAL_JOBS} jobs claimed exactly once"
    fi
fi

echo "[node${NODE_RANK}] done"
