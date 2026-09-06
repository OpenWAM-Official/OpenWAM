#!/usr/bin/env bash
# Run one canonical RoboCasa365 native-action task against an OpenWAM policy server.
#
# Start the server first (in the OpenWAM env):
#   bash scripts/deploy.sh --ckpt-dir /path/to/robocasa365_ckpt --port 8848
#
# Then, inside the separate robocasa365 env:
#   ROBOCASA365_PYTHON=/path/to/robocasa365/env/bin/python \
#     bash benchmarks/robocasa365/single_eval.sh OpenDrawer pretrain 8848 127.0.0.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

task="${1:-OpenDrawer}"
split="${2:-pretrain}"
port="${3:-${ROBOCASA365_PORT:-8848}}"
host="${4:-${ROBOCASA365_POLICY_HOST:-127.0.0.1}}"

python_bin="${ROBOCASA365_PYTHON:-python}"
policy_config="${ROBOCASA365_POLICY_CONFIG:-${SCRIPT_DIR}/policy_config.yml}"
[[ -f "${policy_config}" ]] || { echo "[ERROR] policy config not found: ${policy_config}" >&2; exit 1; }

export PYTHONPATH="${REPO_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

echo "task   : ${task}"
echo "split  : ${split}"
echo "server : ws://${host}:${port}"
echo "python : ${python_bin}"

PYTHONUNBUFFERED=1 "${python_bin}" "${SCRIPT_DIR}/single_eval.py" \
    --config "${policy_config}" \
    --task "${task}" \
    --split "${split}" \
    --host "${host}" \
    --port "${port}"
