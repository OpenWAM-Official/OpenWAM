#!/usr/bin/env bash
# Run the complete 10-epoch LIBERO evaluation in the default client environment.
# Intended to run inside a persistent tmux session.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
INFERENCE_HORIZON="${INFERENCE_HORIZON:-10}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/libero/10epoch_horizon${INFERENCE_HORIZON}_${RUN_TAG}}"
SERVER_PYTHON="${SERVER_PYTHON:-/usr/bin/python3.12}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/path/to/miniconda3/envs/libero/bin/python}"
LIBERO_PATH="${LIBERO_PATH:-/path/to/LIBERO}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8920}"

mkdir -p "${RUN_ROOT}"

echo "[$(date --iso-8601=seconds)] starting LIBERO evaluation (MuJoCo 3.3.2)"
"${SERVER_PYTHON}" "${SCRIPT_DIR}/run_10epoch_all_suites.py" \
    --gpus "${GPUS}" \
    --base-port "${BASE_PORT}" \
    --libero-python "${LIBERO_PYTHON}" \
    --libero-path "${LIBERO_PATH}" \
    --inference-horizon "${INFERENCE_HORIZON}" \
    --output-dir "${RUN_ROOT}" \
    "$@" \
    2>&1 | tee "${RUN_ROOT}/launcher.log"
echo "[$(date --iso-8601=seconds)] finished LIBERO evaluation"
