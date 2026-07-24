#!/usr/bin/env bash
# Env-setup shim for one episode-level RoboTwin worker process.
#
# Mirrors single_eval.sh's environment setup (SAPIEN EGL, PYTHONPATH,
# CUDA_VISIBLE_DEVICES, MPLCONFIGDIR) but instead of running RoboTwin's
# whole-task eval it execs episode_worker.py, which claims ONE (task,mode) from
# the dispatcher and streams its episodes. The dispatcher decides the task, so
# this script does not take a task name.
#
# Usage:
#   bash episode_eval.sh <dispatcher host:port> <node> <worker> <gpu> <server_port> <server_host> <ckpt_setting> [seed]
#
# Exit code is episode_worker.py's: 0 = job drained (relaunch me),
# 3 = no more work (stop this slot), other = error.
#
# Required env vars:
#   ROBOTWIN_PATH    — path to the RoboTwin repository
#   ROBOTWIN_PYTHON  — Python interpreter for the RoboTwin env
set -euo pipefail

if [[ $# -lt 7 ]]; then
    echo "Usage: bash episode_eval.sh <disp host:port> <node> <worker> <gpu> <server_port> <server_host> <ckpt_setting> [seed]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOTWIN_PATH="${ROBOTWIN_PATH:?ROBOTWIN_PATH must be set to the RoboTwin repository root}"
[[ -d "${ROBOTWIN_PATH}" ]] || { echo "[ERROR] ROBOTWIN_PATH not found: ${ROBOTWIN_PATH}" >&2; exit 2; }

dispatcher="$1"
node="$2"
worker="$3"
gpu="$4"
server_port="$5"
server_host="$6"
ckpt_setting="$7"
seed="${8:-0}"

robotwin_python="${ROBOTWIN_PYTHON:-python}"
policy_config="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"
worker_script="${SCRIPT_DIR}/episode_worker.py"

[[ -f "${policy_config}" ]] || { echo "[ERROR] policy_config.yml not found: ${policy_config}" >&2; exit 2; }
[[ -f "${worker_script}" ]] || { echo "[ERROR] episode_worker.py not found: ${worker_script}" >&2; exit 2; }

maybe_configure_sapien_egl() {
    [[ -n "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" || -n "${__EGL_VENDOR_LIBRARY_DIRS:-}" ]] && return 0
    local egl_json
    egl_json="$(dirname "$(dirname "${robotwin_python}")")/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
    [[ -n "${egl_json}" && -f "${egl_json}" ]] || return 0
    export __EGL_VENDOR_LIBRARY_FILENAMES="${egl_json}"
}

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${ROBOTWIN_PATH}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
maybe_configure_sapien_egl

cd "${ROBOTWIN_PATH}"

PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
exec "${robotwin_python}" "${worker_script}" \
    --config       "${policy_config}" \
    --dispatcher   "${dispatcher}" \
    --node         "${node}" \
    --worker       "${worker}" \
    --gpu          "${gpu}" \
    --server-host  "${server_host}" \
    --server-port  "${server_port}" \
    --ckpt-setting "${ckpt_setting}" \
    --seed         "${seed}"
