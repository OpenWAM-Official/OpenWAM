#!/usr/bin/env bash
# Run one EBench worker. Requires an OpenWAM policy server, a GenManip
# endpoint, and EBENCH_PYTHON with genmanip-client installed.
#
# Usage:
#   EBENCH_PYTHON=/path/to/python bash benchmarks/ebench/single_eval.sh \
#       [--url http://127.0.0.1:8087] [--run-id X] [--token T] \
#       [--worker-id 0] [--south-port 8848] [--ckpt-config <ckpt>/config.yaml]
set -euo pipefail
cd "$(dirname "$0")/../.."

EBENCH_PYTHON="${EBENCH_PYTHON:-python}"
exec "${EBENCH_PYTHON}" benchmarks/ebench/openwam2ebench_interface.py "$@"
