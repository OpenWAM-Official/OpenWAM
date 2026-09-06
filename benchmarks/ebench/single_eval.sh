#!/usr/bin/env bash
# Run one EBench eval worker bridged to one OpenWAM policy server.
#
# Prereqs:
#   1. OpenWAM server serving the EBench checkpoint:
#        bash scripts/deploy.sh --ckpt-dir <ckpt> --port 8848
#   2. A GenManip eval server (local ray_eval_server.py with a job submitted
#      via `gmp submit ebench/... --run_id X`), or an online endpoint from
#      `gmp online submit` (then set TOKEN and RUN_ID=task_id).
#   3. EBENCH_PYTHON = python of the env with genmanip-client installed.
#
# Usage:
#   EBENCH_PYTHON=/path/to/python bash benchmarks/ebench/single_eval.sh \
#       [--url http://127.0.0.1:8087] [--run-id X] [--token T] \
#       [--worker-id 0] [--south-port 8848]
set -euo pipefail
cd "$(dirname "$0")/../.."

EBENCH_PYTHON="${EBENCH_PYTHON:-python}"
exec "${EBENCH_PYTHON}" benchmarks/ebench/openwam2ebench_interface.py "$@"
