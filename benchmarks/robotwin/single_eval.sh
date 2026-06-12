#!/usr/bin/env bash
# Run a single RoboTwin task evaluation against an already-running OpenWAM server.
#
# Usage:
#   bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [port] [host]
#
# Args:
#   task_name    — RoboTwin task (e.g. adjust_bottle)
#   task_config  — demo_clean | demo_randomized
#   ckpt_setting — label used in result filenames (e.g. openwam)
#   gpu_id       — CUDA device for the RoboTwin simulator process
#   port         — OpenWAM WebSocket port (default: 8848, env: ROBOTWIN_PORT)
#   host         — OpenWAM server host (default: 127.0.0.1, env: ROBOTWIN_POLICY_HOST)
#
# Required env vars:
#   ROBOTWIN_PATH    — path to the RoboTwin repository
#   ROBOTWIN_PYTHON  — Python interpreter for the RoboTwin env
# Optional env vars:
#   ROBOTWIN_TEST_NUM — cap RoboTwin eval episodes for smoke runs (default: upstream 100)
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [port] [host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOTWIN_PATH="${ROBOTWIN_PATH:?ROBOTWIN_PATH must be set to the RoboTwin repository root}"
[[ -d "${ROBOTWIN_PATH}" ]] || { echo "[ERROR] ROBOTWIN_PATH not found: ${ROBOTWIN_PATH}" >&2; exit 1; }

robotwin_eval_script="${SCRIPT_DIR}/eval_policy_wrapper.py"
[[ -f "${robotwin_eval_script}" ]] || { echo "[ERROR] eval wrapper not found: ${robotwin_eval_script}" >&2; exit 1; }

task_name="$1"
task_config="$2"
ckpt_setting="${3:-openwam}"
gpu_id="${4:-0}"
port="${5:-${ROBOTWIN_PORT:-8848}}"
host="${6:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
seed="0"

robotwin_python="${ROBOTWIN_PYTHON:-python}"
policy_config_template="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"

[[ -f "${policy_config_template}" ]] || {
    echo "[ERROR] policy_config.yml not found: ${policy_config_template}" >&2; exit 1; }

if ! [[ "${task_name}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "[ERROR] Invalid task_name '${task_name}'. Expected [A-Za-z0-9_]+." >&2
    exit 1
fi
if [[ "${task_config}" != "demo_clean" && "${task_config}" != "demo_randomized" ]]; then
    echo "[ERROR] Invalid task_config '${task_config}'." >&2
    exit 1
fi
if ! [[ "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "[ERROR] Invalid port '${port}'. Expected 1..65535." >&2
    exit 1
fi
if ! [[ "${host}" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
    echo "[ERROR] Invalid host '${host}'. Expected hostname/IP characters only." >&2
    exit 1
fi

maybe_configure_sapien_egl() {
    [[ -n "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" || -n "${__EGL_VENDOR_LIBRARY_DIRS:-}" ]] && return 0

    local egl_json
    egl_json="$(dirname "$(dirname "${robotwin_python}")")/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    [[ -n "${egl_json}" && -f "${egl_json}" ]] || return 0

    # SAPIEN's import-time EGL probe crashes on some cluster images because it
    # blindly lists /usr/share/glvnd/egl_vendor.d when that directory is absent.
    export __EGL_VENDOR_LIBRARY_FILENAMES="${egl_json}"
    echo "[INFO] SAPIEN EGL ICD: ${__EGL_VENDOR_LIBRARY_FILENAMES}"
}

# Inject runtime host and port into a temp config
runtime_config="$(mktemp "${TMPDIR:-/tmp}/openwam_policy_config.XXXXXX.yml")"
trap 'rm -f "${runtime_config}"' EXIT

sed \
    -e "s/^host:.*/host: \"${host}\"/" \
    -e "s/^port:.*/port: ${port}/" \
    "${policy_config_template}" > "${runtime_config}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
# PYTHONPATH: RoboTwin modules + this directory (for openwam2robotwin_interface.py)
export PYTHONPATH="${ROBOTWIN_PATH}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
maybe_configure_sapien_egl

cd "${ROBOTWIN_PATH}"

echo "task_name    : ${task_name}"
echo "task_config  : ${task_config}"
echo "ckpt_setting : ${ckpt_setting}"
echo "server       : ws://${host}:${port}"
echo "gpu          : ${gpu_id}"
echo "seed         : ${seed}"

PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
"${robotwin_python}" "${robotwin_eval_script}" \
    --config    "${runtime_config}" \
    --overrides \
    --task_name        "${task_name}" \
    --task_config      "${task_config}" \
    --ckpt_setting     "${ckpt_setting}" \
    --seed             "${seed}" \
    --policy_name      "openwam2robotwin_interface"
