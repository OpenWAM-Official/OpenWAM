#!/bin/bash
# Deploy OpenWAM policy server from a checkpoint directory.
#
# Usage:
#   bash scripts/deploy.sh /path/to/checkpoint_dir
#   bash scripts/deploy.sh /path/to/checkpoint_dir --device cuda:1 --port 9000
#   bash scripts/deploy.sh --ckpt-dir /path/to/checkpoint_dir --ckpt-name checkpoint_step_1000.safetensors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Backward-compat: if first arg is a path (not a flag), treat it as --ckpt-dir
if [[ $# -gt 0 && "$1" != -* ]]; then
    ckpt_dir="$1"
    shift
    python "$SCRIPT_DIR/deploy.py" --ckpt-dir "$ckpt_dir" "$@"
else
    python "$SCRIPT_DIR/deploy.py" "$@"
fi
