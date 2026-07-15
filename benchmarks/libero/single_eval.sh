#!/usr/bin/env bash
# Run one LIBERO task against an already-running OpenWAM server.
#
# Usage:
#   LIBERO_PATH=/path/to/LIBERO LIBERO_PYTHON=/path/to/env/bin/python \
#     bash benchmarks/libero/single_eval.sh libero_spatial 0 8848 127.0.0.1
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
suite="${1:-libero_spatial}"
task_id="${2:-0}"
port="${3:-${LIBERO_PORT:-8848}}"
host="${4:-${LIBERO_POLICY_HOST:-127.0.0.1}}"

: "${LIBERO_PATH:?LIBERO_PATH must point to the ordinary LIBERO repo}"
python_bin="${LIBERO_PYTHON:-python}"
repo_root="${LIBERO_PATH}"

if [[ "${python_bin}" == */* ]]; then
    [[ -x "${python_bin}" ]] || {
        echo "[ERROR] Python not executable: ${python_bin}" >&2
        exit 1
    }
else
    python_command="${python_bin}"
    python_bin="$(command -v "${python_command}")" || {
        echo "[ERROR] Python command not found: ${python_command}" >&2
        exit 1
    }
fi

policy_config="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"
[[ -f "${policy_config}" ]] || { echo "[ERROR] policy config not found: ${policy_config}" >&2; exit 1; }
[[ -d "${repo_root}" ]] || { echo "[ERROR] LIBERO repo not found: ${repo_root}" >&2; exit 1; }

export PYTHONPATH="${repo_root}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
echo "suite  : ${suite}"
echo "task_id: ${task_id}"
echo "server : ws://${host}:${port}"
echo "python : ${python_bin}"

PYTHONUNBUFFERED=1 "${python_bin}" "${SCRIPT_DIR}/single_eval.py" \
    --config "${policy_config}" \
    --suite "${suite}" \
    --task-id "${task_id}" \
    --host "${host}" \
    --port "${port}"
