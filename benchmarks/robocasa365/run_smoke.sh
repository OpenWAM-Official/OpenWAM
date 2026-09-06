#!/usr/bin/env bash
# Smoke launcher for the canonical RoboCasa365 benchmark.
#
# Usage (run inside the separate robocasa365 env):
#   ROBOCASA365_PYTHON=/path/to/robocasa365/env/bin/python \
#     bash benchmarks/robocasa365/run_smoke.sh import
#   ... run_smoke.sh env OpenDrawer
#   ... run_smoke.sh roundtrip            # needs a running OpenWAM server
#
# Modes:
#   import     - import robocasa + gym wrapper; confirm robocasa/<Task> registration
#   env        - gym.make + reset + step a zero action dict (sim plumbing, no policy)
#   roundtrip  - ping + predict a dummy obs against a running OpenWAM server
#
# Per-step obs/action inspection during a real eval: set debug: true in
# policy_config.yml (the adapter dumps ep{N}/step_{N}/ bundles).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="${1:-import}"
TASK="${2:-${ROBOCASA365_SMOKE_TASK:-OpenDrawer}}"
PYTHON_BIN="${ROBOCASA365_PYTHON:-python}"

case "${MODE}" in
    import|env|roundtrip) ;;
    *)
        echo "[ERROR] Unknown smoke mode '${MODE}'. Use import | env | roundtrip." >&2
        exit 1
        ;;
esac

export PYTHONPATH="${REPO_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${ROBOCASA365_SMOKE_GPU:-0}}"

echo "[robocasa365-smoke] mode=${MODE} task=${TASK} python=${PYTHON_BIN}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_robocasa365.py" \
    --mode "${MODE}" \
    --task "${TASK}" \
    --split "${ROBOCASA365_SMOKE_SPLIT:-pretrain}" \
    --steps "${ROBOCASA365_SMOKE_STEPS:-1}" \
    --host "${ROBOCASA365_POLICY_HOST:-127.0.0.1}" \
    --port "${ROBOCASA365_PORT:-8848}"
