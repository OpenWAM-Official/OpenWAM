#!/usr/bin/env bash
# Run N EBench workers; each needs its own stateful OpenWAM policy server.
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
# Clean up workers on interrupt.
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; exit 130' INT TERM
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
