#!/usr/bin/env bash
# Run N EBench eval workers, each bridged to its own OpenWAM policy server.
#
# The OpenWAM action executor is stateful per episode, so worker i talks to
# south port SOUTH_PORT_BASE+i. Start one deploy server per worker first —
# scripts/deploy.sh does exactly this mapping (GPU i -> port PORT_BASE+i):
#   NUM_GPUS=$NUM_WORKERS PORT_BASE=8848 bash scripts/deploy.sh <ckpt_dir>
#
# Usage:
#   NUM_WORKERS=4 SOUTH_PORT_BASE=8848 EBENCH_PYTHON=/path/to/python \
#     bash benchmarks/ebench/multi_eval.sh --url http://127.0.0.1:8087 --run-id X \
#         [--ckpt-config <ckpt>/config.yaml]
set -euo pipefail
cd "$(dirname "$0")/../.."

NUM_WORKERS="${NUM_WORKERS:-1}"
SOUTH_PORT_BASE="${SOUTH_PORT_BASE:-8848}"
EBENCH_PYTHON="${EBENCH_PYTHON:-python}"

pids=()
# Ctrl+C / TERM must take the workers down too, or they keep holding GenManip worker slots.
trap 'kill "${pids[@]}" 2>/dev/null; exit 130' INT TERM
for ((i = 0; i < NUM_WORKERS; i++)); do
    "${EBENCH_PYTHON}" benchmarks/ebench/openwam2ebench_interface.py \
        --worker-id "$i" --south-port "$((SOUTH_PORT_BASE + i))" "$@" &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
exit "$status"
