#!/usr/bin/env bash
# Smoke launcher for the VLABench client. No GPU, checkpoint or server needed.
#
# Usage:
#   VLABENCH_PATH=/path/to/VLABench VLABENCH_PYTHON=/path/to/env/bin/python \
#     bash benchmarks/vlabench/run_smoke.sh [mode] [task] [max_steps] [server]
#
# Modes:
#   env   - load one task, assert the observation contract (cameras, ee_state,
#           robot base, instruction) the adapter depends on
#   loop  - env + full closed loop against an in-process mock OpenWAM server
#           that echoes proprio back as the action (hold-still policy)
#
# server (4th arg, or VLABENCH_SMOKE_SERVER): loop mode only. host:port of a REAL
# OpenWAM server to drive instead of the mock — exercises the wire action width,
# server-side denormalization and latency. Empty (default) uses the mock.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="${1:-${VLABENCH_SMOKE_MODE:-env}}"
task="${2:-${VLABENCH_SMOKE_TASK:-select_fruit}}"
max_steps="${3:-8}"
server="${4:-${VLABENCH_SMOKE_SERVER:-}}"

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
[[ -d "${VLABENCH_PATH}" ]] || { echo "[ERROR] VLABench repo not found: ${VLABENCH_PATH}" >&2; exit 1; }

export PYTHONPATH="${VLABENCH_PATH}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export VLABENCH_ROOT="${VLABENCH_ROOT:-${VLABENCH_PATH}/VLABench}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

smoke_args=(--mode "${mode}" --task "${task}" --max-steps "${max_steps}")
if [[ -n "${server}" ]]; then
    smoke_args+=(--server "${server}")
fi

PYTHONUNBUFFERED=1 "${python_bin}" "${SCRIPT_DIR}/smoke_vlabench.py" "${smoke_args[@]}"
