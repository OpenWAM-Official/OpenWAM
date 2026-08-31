#!/usr/bin/env bash
# Run one canonical native-action LIBERO task against an already-running OpenWAM server.
#
# Usage:
#   bash benchmarks/libero/single_eval.sh libero_spatial 0 8848 127.0.0.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_PATH="${LIBERO_PATH:-/path/to/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/miniconda3/envs/libero/bin/python}"
SUITE="${1:-libero_spatial}"
TASK_ID="${2:-0}"
PORT="${3:-${LIBERO_PORT:-8848}}"
HOST="${4:-${LIBERO_POLICY_HOST:-127.0.0.1}}"

[[ -d "${LIBERO_PATH}" ]] || { echo "[ERROR] LIBERO repo not found: ${LIBERO_PATH}" >&2; exit 1; }
if [[ "${LIBERO_PYTHON}" == */* ]]; then
    [[ -x "${LIBERO_PYTHON}" ]] || { echo "[ERROR] Python not executable: ${LIBERO_PYTHON}" >&2; exit 1; }
else
    LIBERO_PYTHON="$(command -v "${LIBERO_PYTHON}")" || {
        echo "[ERROR] Python command not found" >&2
        exit 1
    }
fi

export LIBERO_PATH
export PYTHONPATH="${LIBERO_PATH}:${SCRIPT_DIR}:${SCRIPT_DIR}/../..:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

CONFIG="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"
[[ -f "${CONFIG}" ]] || { echo "[ERROR] policy config not found: ${CONFIG}" >&2; exit 1; }

echo "suite  : ${SUITE}"
echo "task_id: ${TASK_ID}"
echo "server : ws://${HOST}:${PORT}"
echo "python : ${LIBERO_PYTHON}"

PYTHONUNBUFFERED=1 "${LIBERO_PYTHON}" "${SCRIPT_DIR}/single_eval.py" \
    --config "${CONFIG}" \
    --suite "${SUITE}" \
    --task-id "${TASK_ID}" \
    --host "${HOST}" \
    --port "${PORT}"
