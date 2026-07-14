#!/usr/bin/env bash
# Run N EBench eval workers, each bridged to its own OpenWAM policy server.
#
# The OpenWAM action executor is stateful per episode, so worker i talks to
# south port SOUTH_PORT_BASE+i. Start one deploy server per worker first, e.g.:
#   for i in $(seq 0 $((NUM_WORKERS-1))); do
#     CUDA_VISIBLE_DEVICES=$i bash scripts/deploy.sh --ckpt-dir <ckpt> --port $((8848+i)) &
#   done
#
# Usage:
#   NUM_WORKERS=4 SOUTH_PORT_BASE=8848 EBENCH_PYTHON=/path/to/python \
#     bash benchmarks/ebench/multi_eval.sh --url http://127.0.0.1:8087 --run-id X
set -euo pipefail
cd "$(dirname "$0")/../.."

NUM_WORKERS="${NUM_WORKERS:-1}"
SOUTH_PORT_BASE="${SOUTH_PORT_BASE:-8848}"
EBENCH_PYTHON="${EBENCH_PYTHON:-python}"

pids=()
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
