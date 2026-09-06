#!/usr/bin/env bash
# End-to-end RoboCasa365 launcher: start one OpenWAM server, wait until it is
# ready, start the isolated RoboCasa client, evaluate, then stop the server.
#
# Usage:
#   bash benchmarks/robocasa365/run_eval.sh CKPT_DIR CKPT_NAME [TASK|target|TASK_FILE]
#
# Important environment overrides:
#   SERVER_PYTHON=/path/to/openwam/python
#   ROBOCASA365_PYTHON=/path/to/robocasa365/python
#   SERVER_DEVICE=cuda:0  PORT=8848  SPLIT=pretrain  OUTPUT_DIR=/path/to/results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -ge 2 ]]; then
    CKPT_DIR="$1"
    CKPT_NAME="$2"
    TARGET="${3:-${ROBOCASA365_EVAL_TARGET:-OpenDrawer}}"
else
    CKPT_DIR="${CKPT_DIR:-}"
    CKPT_NAME="${CKPT_NAME:-}"
    TARGET="${ROBOCASA365_EVAL_TARGET:-OpenDrawer}"
fi
[[ -n "${CKPT_DIR}" && -n "${CKPT_NAME}" ]] || {
    echo "Usage: bash benchmarks/robocasa365/run_eval.sh CKPT_DIR CKPT_NAME [TASK|target|TASK_FILE]" >&2
    exit 2
}

SERVER_PYTHON="${SERVER_PYTHON:-python}"
command -v setsid >/dev/null || {
    echo "[ERROR] Required command not found: setsid" >&2
    exit 1
}
SERVER_DEVICE="${SERVER_DEVICE:-cuda:0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8848}"
SPLIT="${SPLIT:-pretrain}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1200}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/robocasa365/${RUN_TAG}}"
mkdir -p "${OUTPUT_DIR}"

server_pid=""
cleanup() {
    trap - EXIT INT TERM
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        echo "[robocasa365] stopping server pid=${server_pid}"
        kill -TERM -- "-${server_pid}" 2>/dev/null || kill -TERM "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[robocasa365] checkpoint=${CKPT_DIR}/${CKPT_NAME} target=${TARGET} split=${SPLIT}"
setsid "${SERVER_PYTHON}" "${REPO_ROOT}/scripts/deploy.py" \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-name "${CKPT_NAME}" \
    --device "${SERVER_DEVICE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    >"${OUTPUT_DIR}/server.log" 2>&1 &
server_pid=$!

deadline=$((SECONDS + SERVER_START_TIMEOUT))
while ! "${SERVER_PYTHON}" -c \
    'import socket,sys; s=socket.create_connection((sys.argv[1], int(sys.argv[2])), 1); s.close()' \
    "${HOST}" "${PORT}" >/dev/null 2>&1; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "[ERROR] OpenWAM server exited before becoming ready; see ${OUTPUT_DIR}/server.log" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "[ERROR] Timed out waiting for ws://${HOST}:${PORT}; see ${OUTPUT_DIR}/server.log" >&2
        exit 1
    fi
    sleep 2
done
echo "[robocasa365] server ready at ws://${HOST}:${PORT}"

if [[ "${TARGET}" == "all" || "${TARGET}" == "target" || -f "${TARGET}" ]]; then
    ROBOCASA365_POLICY_HOST="${HOST}" ROBOCASA365_PORT="${PORT}" \
        bash "${SCRIPT_DIR}/multi_eval.sh" \
        --split "${SPLIT}" --host "${HOST}" --port "${PORT}" \
        --out "${OUTPUT_DIR}/tasks" "${TARGET}" \
        2>&1 | tee "${OUTPUT_DIR}/client.log"
else
    ROBOCASA365_POLICY_HOST="${HOST}" ROBOCASA365_PORT="${PORT}" \
        bash "${SCRIPT_DIR}/single_eval.sh" "${TARGET}" "${SPLIT}" "${PORT}" "${HOST}" \
        2>&1 | tee "${OUTPUT_DIR}/client.log"
fi
