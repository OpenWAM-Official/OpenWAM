#!/usr/bin/env bash
# Smoke launcher for LIBERO-plus.
#
# Usage:
#   bash benchmarks/libero-plus/run_smoke.sh import
#   bash benchmarks/libero-plus/run_smoke.sh task
#   bash benchmarks/libero-plus/run_smoke.sh env
#
# Modes:
#   import  - verify package import and generated LIBERO_CONFIG_PATH
#   task    - import + retrieve one benchmark task and its BDDL file
#   env     - task + instantiate OffScreenRenderEnv; requires assets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LIBERO_PLUS_PATH="/path/to/LIBERO-plus"
DEFAULT_LIBERO_PLUS_PYTHON="/path/to/miniconda3/envs/libero-plus/bin/python"

MODE="${1:-${LIBERO_SMOKE_MODE:-import}}"
EXTERNAL_REPO="${LIBERO_PLUS_PATH:-${DEFAULT_LIBERO_PLUS_PATH}}"
export LIBERO_PLUS_PATH="${EXTERNAL_REPO}"
export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG_ROOT:-${HOME}/.libero-openwam-plus}"
PYTHON_BIN="${LIBERO_PLUS_PYTHON:-${DEFAULT_LIBERO_PLUS_PYTHON}}"

case "${MODE}" in
    import|task|env) ;;
    *)
        echo "[ERROR] Unknown smoke mode '${MODE}'. Use import | task | env." >&2
        exit 1
        ;;
esac

if [[ "${PYTHON_BIN}" == */* ]]; then
    [[ -x "${PYTHON_BIN}" ]] || {
        echo "[ERROR] Python not executable: ${PYTHON_BIN}" >&2
        exit 1
    }
else
    PYTHON_COMMAND="${PYTHON_BIN}"
    PYTHON_BIN="$(command -v "${PYTHON_COMMAND}")" || {
        echo "[ERROR] Python command not found: ${PYTHON_COMMAND}" >&2
        exit 1
    }
fi
[[ -d "${EXTERNAL_REPO}" ]] || {
    echo "[ERROR] LIBERO-plus repository not found: ${EXTERNAL_REPO}" >&2
    exit 1
}

export PYTHONPATH="${EXTERNAL_REPO}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${LIBERO_SMOKE_GPU:-0}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${LIBERO_SMOKE_GPU:-0}}"

echo "[libero-plus-smoke] mode=${MODE} python=${PYTHON_BIN}"
echo "[libero-plus-smoke] repo=${EXTERNAL_REPO}"
echo "[libero-plus-smoke] config=${LIBERO_CONFIG_PATH}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_libero.py" \
    --mode "${MODE}" \
    --suite "${LIBERO_SMOKE_SUITE:-libero_spatial}" \
    --task-id "${LIBERO_SMOKE_TASK_ID:-0}" \
    --camera-size "${LIBERO_SMOKE_CAMERA_SIZE:-128}" \
    --steps "${LIBERO_SMOKE_STEPS:-1}"
