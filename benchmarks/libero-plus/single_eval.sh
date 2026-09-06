#!/usr/bin/env bash
# Run one canonical native-action LIBERO-plus task against an already-running OpenWAM server.
#
# Usage:
#   bash benchmarks/libero-plus/single_eval.sh libero_spatial 0 8848 127.0.0.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_PLUS_PATH="${LIBERO_PLUS_PATH:-/path/to/LIBERO-plus}"
LIBERO_PLUS_PYTHON="${LIBERO_PLUS_PYTHON:-/path/to/miniconda3/envs/libero-plus/bin/python}"
SUITE="${1:-libero_spatial}"
TASK_ID="${2:-0}"
PORT="${3:-${LIBERO_PLUS_PORT:-8848}}"
HOST="${4:-${LIBERO_PLUS_POLICY_HOST:-127.0.0.1}}"

[[ -d "${LIBERO_PLUS_PATH}" ]] || { echo "[ERROR] LIBERO-plus repo not found: ${LIBERO_PLUS_PATH}" >&2; exit 1; }
if [[ "${LIBERO_PLUS_PYTHON}" == */* ]]; then
    [[ -x "${LIBERO_PLUS_PYTHON}" ]] || { echo "[ERROR] Python not executable: ${LIBERO_PLUS_PYTHON}" >&2; exit 1; }
else
    LIBERO_PLUS_PYTHON="$(command -v "${LIBERO_PLUS_PYTHON}")" || {
        echo "[ERROR] Python command not found" >&2
        exit 1
    }
fi

export LIBERO_PLUS_PATH
export PYTHONPATH="${LIBERO_PLUS_PATH}:${SCRIPT_DIR}:${SCRIPT_DIR}/../..:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

CONFIG="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"
[[ -f "${CONFIG}" ]] || { echo "[ERROR] policy config not found: ${CONFIG}" >&2; exit 1; }

echo "suite  : ${SUITE}"
echo "task_id: ${TASK_ID}"
echo "server : ws://${HOST}:${PORT}"
echo "python : ${LIBERO_PLUS_PYTHON}"

PYTHONUNBUFFERED=1 "${LIBERO_PLUS_PYTHON}" "${SCRIPT_DIR}/single_eval.py" \
    --config "${CONFIG}" \
    --suite "${SUITE}" \
    --task-id "${TASK_ID}" \
    --host "${HOST}" \
    --port "${PORT}"
