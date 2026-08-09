#!/usr/bin/env bash
# Run one VLABench task (or a whole track) against an already-running OpenWAM server.
#
# Usage:
#   VLABENCH_PATH=/path/to/VLABench VLABENCH_PYTHON=/path/to/env/bin/python \
#     bash benchmarks/vlabench/single_eval.sh <task> [track] [n_episodes] [port] [host]
#
# Examples:
#   bash benchmarks/vlabench/single_eval.sh select_fruit
#   bash benchmarks/vlabench/single_eval.sh select_fruit track_1_in_distribution 50 8848
#   bash benchmarks/vlabench/single_eval.sh all track_2_cross_category 50 8848 127.0.0.1
#
# `all` as the task runs every task in the track.
#
# VLABench pins numpy 1.25 / mujoco 3.2.2 / dm_control 1.0.22, so it needs its
# own environment — point VLABENCH_PYTHON at it. Only numpy, Pillow and
# websockets are needed on top of a working VLABench install.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

task="${1:-select_fruit}"
track="${2:-track_1_in_distribution}"
n_episodes="${3:-${VLABENCH_N_EPISODES:-50}}"
port="${4:-${VLABENCH_PORT:-8848}}"
host="${5:-${VLABENCH_POLICY_HOST:-127.0.0.1}}"

: "${VLABENCH_PATH:?VLABENCH_PATH must point to the VLABench repo}"
python_bin="${VLABENCH_PYTHON:-python}"

if [[ "${python_bin}" == */* ]]; then
    [[ -x "${python_bin}" ]] || { echo "[ERROR] Python not executable: ${python_bin}" >&2; exit 1; }
else
    python_bin="$(command -v "${python_bin}")" || {
        echo "[ERROR] Python command not found: ${VLABENCH_PYTHON:-python}" >&2
        exit 1
    }
fi

policy_config="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"
[[ -f "${policy_config}" ]] || { echo "[ERROR] policy config not found: ${policy_config}" >&2; exit 1; }
[[ -d "${VLABENCH_PATH}" ]] || { echo "[ERROR] VLABench repo not found: ${VLABENCH_PATH}" >&2; exit 1; }

export PYTHONPATH="${VLABENCH_PATH}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export VLABENCH_ROOT="${VLABENCH_ROOT:-${VLABENCH_PATH}/VLABench}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

save_dir="${VLABENCH_SAVE_DIR:-${SCRIPT_DIR}/_eval_out}"

echo "task     : ${task}"
echo "track    : ${track}"
echo "episodes : ${n_episodes}"
echo "server   : ws://${host}:${port}"
echo "python   : ${python_bin}"
echo "root     : ${VLABENCH_ROOT}"

task_args=()
if [[ "${task}" != "all" ]]; then
    task_args=(--tasks "${task}")
fi

PYTHONUNBUFFERED=1 "${python_bin}" "${SCRIPT_DIR}/single_eval.py" \
    --config "${policy_config}" \
    --eval-track "${track}" \
    "${task_args[@]}" \
    --n-episodes "${n_episodes}" \
    --save-dir "${save_dir}" \
    --host "${host}" \
    --port "${port}"
