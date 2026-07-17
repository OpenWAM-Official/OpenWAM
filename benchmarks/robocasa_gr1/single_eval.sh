#!/usr/bin/env bash
# Run one RoboCasa GR1 tabletop task against an already-running OpenWAM server.
#
# Usage:
#   ROBOCASA_GR1_PATH=/path/to/robocasa-gr1-tabletop-tasks \
#   ROBOCASA_GR1_PYTHON=/path/to/python \
#   bash benchmarks/robocasa_gr1/single_eval.sh [env_id] [port] [host]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${ROBOCASA_GR1_PATH:?ROBOCASA_GR1_PATH must point to the robocasa-gr1-tabletop-tasks repo}"
[[ -d "${ROBOCASA_GR1_PATH}" ]] || {
    echo "[ERROR] ROBOCASA_GR1_PATH not found: ${ROBOCASA_GR1_PATH}" >&2
    exit 1
}

env_id="${1:-${ROBOCASA_GR1_ENV_ID:-gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env}}"
port="${2:-${ROBOCASA_GR1_PORT:-8848}}"
host="${3:-${ROBOCASA_GR1_POLICY_HOST:-127.0.0.1}}"
python_bin="${ROBOCASA_GR1_PYTHON:-python}"
policy_config="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"

[[ -f "${policy_config}" ]] || {
    echo "[ERROR] policy config not found: ${policy_config}" >&2
    exit 1
}
if ! [[ "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "[ERROR] Invalid port '${port}'. Expected 1..65535." >&2
    exit 1
fi
if [[ "${env_id}" != gr1_unified/* ]]; then
    echo "[ERROR] Expected a gr1_unified/* env_id, got: ${env_id}" >&2
    exit 1
fi

runtime_config="$(mktemp "${TMPDIR:-/tmp}/openwam_robocasa_gr1.XXXXXX.yml")"
trap 'rm -f "${runtime_config}"' EXIT

sed \
    -e "s/^host:.*/host: \"${host}\"/" \
    -e "s/^port:.*/port: ${port}/" \
    -e "s|^env_id:.*|env_id: \"${env_id}\"|" \
    "${policy_config}" > "${runtime_config}"

export PYTHONPATH="${ROBOCASA_GR1_PATH}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-${ROBOCASA_GR1_RENDER_BACKEND:-egl}}"
if [[ "${MUJOCO_GL}" == "egl" ]]; then
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${ROBOCASA_GR1_GPU:-0}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${ROBOCASA_GR1_GPU:-0}}"

echo "env_id : ${env_id}"
echo "server : ws://${host}:${port}"
echo "python : ${python_bin}"
echo "repo   : ${ROBOCASA_GR1_PATH}"
echo "render : MUJOCO_GL=${MUJOCO_GL} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-<unset>} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"

PYTHONUNBUFFERED=1 "${python_bin}" "${SCRIPT_DIR}/single_eval.py" \
    --config "${runtime_config}" \
    --env-id "${env_id}" \
    --host "${host}" \
    --port "${port}"
