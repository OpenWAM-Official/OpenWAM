#!/usr/bin/env bash
# End-to-end LIBERO launcher: start managed OpenWAM server replica(s), start
# LIBERO client workers, evaluate, summarize, and stop the servers.
#
# Usage:
#   bash benchmarks/libero/run_eval.sh CKPT_DIR CKPT_NAME [run_all_suites.py options]
#
# Important environment overrides:
#   SERVER_PYTHON=/path/to/openwam/python
#   LIBERO_PYTHON=/path/to/libero/python
#   LIBERO_PATH=/path/to/LIBERO
#   GPUS=0,1  REPLICAS_PER_GPU=1  OUTPUT_DIR=/path/to/results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -ge 2 ]]; then
    CKPT_DIR="$1"
    CKPT_NAME="$2"
    shift 2
else
    CKPT_DIR="${CKPT_DIR:-}"
    CKPT_NAME="${CKPT_NAME:-}"
fi
[[ -n "${CKPT_DIR}" && -n "${CKPT_NAME}" ]] || {
    echo "Usage: bash benchmarks/libero/run_eval.sh CKPT_DIR CKPT_NAME [options]" >&2
    exit 2
}

CLIENT_PYTHON="${LIBERO_PYTHON:-/path/to/miniconda3/envs/libero/bin/python}"
CLIENT_REPO="${LIBERO_PATH:-/path/to/LIBERO}"
POLICY_CONFIG="${POLICY_CONFIG_PATH:-${SCRIPT_DIR}/policy_config.yml}"

SERVER_PYTHON="${SERVER_PYTHON:-python}"
if [[ "${SERVER_PYTHON}" != */* ]]; then
    SERVER_PYTHON="$(command -v "${SERVER_PYTHON}")" || {
        echo "[ERROR] SERVER_PYTHON command not found" >&2
        exit 1
    }
fi
if [[ "${CLIENT_PYTHON}" != */* ]]; then
    CLIENT_PYTHON="$(command -v "${CLIENT_PYTHON}")" || {
        echo "[ERROR] LIBERO client Python command not found" >&2
        exit 1
    }
fi
GPUS="${GPUS:-0}"
REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"
BASE_PORT="${BASE_PORT:-8920}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/libero/${RUN_TAG}}"
mkdir -p "${OUTPUT_DIR}"

echo "[libero] checkpoint=${CKPT_DIR}/${CKPT_NAME}"
echo "[libero] client_python=${CLIENT_PYTHON} client_repo=${CLIENT_REPO}"
echo "[libero] output=${OUTPUT_DIR}"

"${SERVER_PYTHON}" "${SCRIPT_DIR}/run_all_suites.py" \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-name "${CKPT_NAME}" \
    --server-python "${SERVER_PYTHON}" \
    --libero-python "${CLIENT_PYTHON}" \
    --libero-path "${CLIENT_REPO}" \
    --policy-config "${POLICY_CONFIG}" \
    --gpus "${GPUS}" \
    --replicas-per-gpu "${REPLICAS_PER_GPU}" \
    --base-port "${BASE_PORT}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/launcher.log"
