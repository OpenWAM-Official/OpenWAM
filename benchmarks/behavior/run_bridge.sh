#!/usr/bin/env bash
# Start the BEHAVIOR-1K (OmniGibson) eval bridge.
#
# The bridge is the north-facing openpi-protocol WebSocket server that
# OmniGibson's `eval.py policy=websocket` connects to; it forwards to an
# already-running OpenWAM policy server (south, default 8848) and converts the
# unified 80-D action into the R1Pro 21-D IK controller vector.
#
# Usage:
#   bash run_bridge.sh [--port 8000] [--south-host 127.0.0.1] [--south-port 8848] \
#                      [--task-names task_names.json] [--default-prompt "..."] \
#                      [--no-send-state]
#
# Env:
#   BRIDGE_PYTHON  — interpreter with numpy + Pillow + websockets + msgpack
#                    (the OmniGibson env works; it does NOT need torch/openwam).
#
# Prereqs:
#   1. The OpenWAM server is already running on --south-host:--south-port
#      (e.g. `bash scripts/deploy.sh --ckpt-dir <ckpt> --port 8848`).
#   2. Generate task_names.json once from the installed OmniGibson:
#        python -c "from omnigibson.learning.utils.eval_utils import TASK_INDICES_TO_NAMES; \
#          import json; json.dump({int(k): v for k, v in TASK_INDICES_TO_NAMES.items()}, \
#          open('task_names.json', 'w'), indent=2)"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_PYTHON="${BRIDGE_PYTHON:-python}"

exec "${BRIDGE_PYTHON}" "${SCRIPT_DIR}/openwam2behavior_bridge.py" "$@"
