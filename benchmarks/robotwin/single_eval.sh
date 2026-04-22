#!/usr/bin/env bash
# Run a single RoboTwin task evaluation against an already-running OpenWAM server.
#
# Usage:
#   bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [http_port] [host]
#
# Args:
#   task_name    — RoboTwin task (e.g. adjust_bottle)
#   task_config  — demo_clean | demo_randomized
#   ckpt_setting — label used in result filenames (e.g. openwam)
#   gpu_id       — CUDA device for the RoboTwin simulator process
#   http_port    — OpenWAM HTTP port  (default: 8848, env: ROBOTWIN_HTTP_PORT)
#   host         — OpenWAM server host (default: 127.0.0.1, env: ROBOTWIN_POLICY_HOST)
#
# Required env vars:
#   ROBOTWIN_PATH    — path to the RoboTwin repository
#   ROBOTWIN_PYTHON  — Python interpreter for the RoboTwin env
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [http_port] [host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOTWIN_PATH="${ROBOTWIN_PATH:?ROBOTWIN_PATH must be set to the RoboTwin repository root}"
[[ -d "${ROBOTWIN_PATH}" ]] || { echo "[ERROR] ROBOTWIN_PATH not found: ${ROBOTWIN_PATH}" >&2; exit 1; }

robotwin_eval_script="${ROBOTWIN_PATH}/script/eval_policy.py"
[[ -f "${robotwin_eval_script}" ]] || { echo "[ERROR] eval script not found: ${robotwin_eval_script}" >&2; exit 1; }

task_name="$1"
task_config="$2"
ckpt_setting="${3:-openwam}"
gpu_id="${4:-0}"
http_port="${5:-${ROBOTWIN_HTTP_PORT:-8848}}"
host="${6:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
seed="0"

robotwin_python="${ROBOTWIN_PYTHON:-python}"
policy_config_template="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"

[[ -f "${policy_config_template}" ]] || {
    echo "[ERROR] policy_config.yml not found: ${policy_config_template}" >&2; exit 1; }

# Inject runtime host and http_port into a temp config
runtime_config="$(mktemp "${TMPDIR:-/tmp}/openwam_policy_config.XXXXXX.yml")"
trap 'rm -f "${runtime_config}"' EXIT

sed \
    -e "s/^host:.*/host: \"${host}\"/" \
    -e "s/^http_port:.*/http_port: ${http_port}/" \
    "${policy_config_template}" > "${runtime_config}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
# PYTHONPATH: RoboTwin modules + this directory (for openwam2robotwin_interface.py)
export PYTHONPATH="${ROBOTWIN_PATH}:${SCRIPT_DIR}:${PYTHONPATH:-}"

cd "${ROBOTWIN_PATH}"

echo "task_name    : ${task_name}"
echo "task_config  : ${task_config}"
echo "ckpt_setting : ${ckpt_setting}"
echo "server       : http://${host}:${http_port}"
echo "gpu          : ${gpu_id}"
echo "seed         : ${seed}"

PYTHONWARNINGS=ignore::UserWarning \
"${robotwin_python}" "${robotwin_eval_script}" \
    --config    "${runtime_config}" \
    --overrides \
    --task_name        "${task_name}" \
    --task_config      "${task_config}" \
    --ckpt_setting     "${ckpt_setting}" \
    --seed             "${seed}" \
    --policy_name      "openwam2robotwin_interface"
