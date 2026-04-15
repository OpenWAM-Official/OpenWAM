#!/bin/bash
# Deploy OpenWAM policy server from a checkpoint directory.
#
# Usage:
#   bash scripts/deploy.sh /path/to/checkpoint_dir
#   bash scripts/deploy.sh /path/to/checkpoint_dir --device cuda:1 --ws-port 9000
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/deploy.py" --ckpt-dir "$@"
