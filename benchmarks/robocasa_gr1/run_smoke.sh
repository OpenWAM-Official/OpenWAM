#!/usr/bin/env bash
# Smoke-check a RoboCasa GR1 tabletop installation.
#
# Usage:
#   ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks \
#   ROBOCASA_GR1_PYTHON=/path/to/python \
#   bash benchmarks/robocasa_gr1/run_smoke.sh [import|task|env|dataset]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${1:-${ROBOCASA_GR1_SMOKE_MODE:-import}}"
PYTHON_BIN="${ROBOCASA_GR1_PYTHON:-python}"

: "${ROBOCASA_GR1_PATH:?ROBOCASA_GR1_PATH must point to the robocasa-gr1-tabletop-tasks repo}"
[[ -d "${ROBOCASA_GR1_PATH}" ]] || {
    echo "[ERROR] ROBOCASA_GR1_PATH not found: ${ROBOCASA_GR1_PATH}" >&2
    exit 1
}

export PYTHONPATH="${ROBOCASA_GR1_PATH}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${ROBOCASA_GR1_SMOKE_GPU:-0}}"

echo "[robocasa-gr1-smoke] mode=${MODE} python=${PYTHON_BIN}"
echo "[robocasa-gr1-smoke] repo=${ROBOCASA_GR1_PATH}"

args=(--mode "${MODE}" --env-id "${ROBOCASA_GR1_ENV_ID:-gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env}")
if [[ "${MODE}" == "env" ]]; then
    args+=(--steps "${ROBOCASA_GR1_SMOKE_STEPS:-1}")
    if [[ "${ROBOCASA_GR1_ENABLE_RENDER:-1}" == "1" ]]; then
        export MUJOCO_GL="${MUJOCO_GL:-${ROBOCASA_GR1_RENDER_BACKEND:-egl}}"
        if [[ "${MUJOCO_GL}" == "egl" ]]; then
            export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
        fi
        export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${ROBOCASA_GR1_SMOKE_GPU:-0}}"
        echo "[robocasa-gr1-smoke] render=1 MUJOCO_GL=${MUJOCO_GL} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-<unset>} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
        args+=(--enable-render)
    else
        echo "[robocasa-gr1-smoke] render=0"
    fi
fi
if [[ "${MODE}" == "dataset" ]]; then
    : "${ROBOCASA_GR1_DATASET:?ROBOCASA_GR1_DATASET must point to an HDF5 dataset file}"
    args+=(--dataset "${ROBOCASA_GR1_DATASET}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_robocasa_gr1.py" "${args[@]}"
