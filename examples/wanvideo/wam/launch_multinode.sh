#!/bin/bash
# =============================================================================
# Multi-node distributed training launcher for ReCamMaster VAM
#
# Wraps `accelerate launch` with multi-node configuration. Run this script on
# EVERY node with the appropriate --node_rank.
#
# Usage:
#   # Node 0 (master)
#   bash launch_multinode.sh --master_addr 192.0.2.1 --num_nodes 4 --node_rank 0 \
#       --training_script examples/wanvideo/wam/train_video_action.py \
#       -- [training args...]
#
#   # Node 1
#   bash launch_multinode.sh --master_addr 192.0.2.1 --num_nodes 4 --node_rank 1 \
#       --training_script examples/wanvideo/wam/train_video_action.py \
#       -- [training args...]
#
# The "--" separator divides launcher args from training script args.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MASTER_ADDR=""
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES=4
NODE_RANK=""
GPUS_PER_NODE=8
CONFIG_FILE=""
TRAINING_SCRIPT=""

# ---------------------------------------------------------------------------
# Parse launcher arguments (everything before "--")
# ---------------------------------------------------------------------------
TRAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --master_addr)   MASTER_ADDR="$2";      shift 2 ;;
        --master_port)   MASTER_PORT="$2";       shift 2 ;;
        --num_nodes)     NUM_NODES="$2";         shift 2 ;;
        --node_rank)     NODE_RANK="$2";         shift 2 ;;
        --gpus_per_node) GPUS_PER_NODE="$2";     shift 2 ;;
        --config_file)   CONFIG_FILE="$2";       shift 2 ;;
        --training_script) TRAINING_SCRIPT="$2"; shift 2 ;;
        --)              shift; TRAINING_ARGS=("$@"); break ;;
        *)
            echo "Unknown launcher arg: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$MASTER_ADDR" ]]; then
    echo "ERROR: --master_addr is required" >&2
    exit 1
fi
if [[ -z "$NODE_RANK" ]]; then
    echo "ERROR: --node_rank is required" >&2
    exit 1
fi
if [[ -z "$TRAINING_SCRIPT" ]]; then
    echo "ERROR: --training_script is required" >&2
    exit 1
fi

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# Default config file: multinode config in the same directory as this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "$CONFIG_FILE" ]]; then
    CONFIG_FILE="${SCRIPT_DIR}/accelerate_config_multinode.yaml"
fi

# ---------------------------------------------------------------------------
# Set working directory (same convention as existing training scripts)
# ---------------------------------------------------------------------------
WORK_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${WORK_DIR}"
export PYTHONPATH="${WORK_DIR}:${WORK_DIR}/examples/wanvideo/wam:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Network auto-detection: InfiniBand vs Ethernet
# ---------------------------------------------------------------------------
if command -v ibstat &>/dev/null && ibstat 2>/dev/null | grep -q "Active"; then
    echo "[launch_multinode] InfiniBand detected — enabling NCCL IB"
    export NCCL_IB_DISABLE=0
    export NCCL_NET_GDR_LEVEL=2
    # Pick the first active IB-associated network interface
    if command -v ibdev2netdev &>/dev/null; then
        IB_IFACE=$(ibdev2netdev 2>/dev/null | grep "Up" | awk '{print $5}' | head -1)
        if [[ -n "$IB_IFACE" ]]; then
            export NCCL_SOCKET_IFNAME="$IB_IFACE"
        fi
    fi
else
    echo "[launch_multinode] No active InfiniBand — falling back to Ethernet"
    export NCCL_IB_DISABLE=1
    # Try to auto-detect the primary ethernet interface
    ETH_IFACE=$(ip -o -4 route show default 2>/dev/null | awk '{print $5}' | head -1)
    if [[ -n "$ETH_IFACE" ]]; then
        export NCCL_SOCKET_IFNAME="$ETH_IFACE"
    else
        export NCCL_SOCKET_IFNAME=eth0
    fi
fi

# NCCL tuning
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[launch_multinode] Configuration:"
echo "  MASTER_ADDR     = ${MASTER_ADDR}"
echo "  MASTER_PORT     = ${MASTER_PORT}"
echo "  NUM_NODES       = ${NUM_NODES}"
echo "  NODE_RANK       = ${NODE_RANK}"
echo "  GPUS_PER_NODE   = ${GPUS_PER_NODE}"
echo "  TOTAL_GPUS      = ${TOTAL_GPUS}"
echo "  CONFIG_FILE     = ${CONFIG_FILE}"
echo "  TRAINING_SCRIPT = ${TRAINING_SCRIPT}"
echo "  NCCL_IB_DISABLE = ${NCCL_IB_DISABLE}"
echo "  NCCL_SOCKET_IFNAME = ${NCCL_SOCKET_IFNAME:-<not set>}"
echo "  WORK_DIR        = ${WORK_DIR}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Launch via accelerate
# CLI flags override values in the config YAML, so machine_rank / num_machines
# in the YAML serve only as defaults for single-node fallback.
# ---------------------------------------------------------------------------
accelerate launch \
    --config_file "${CONFIG_FILE}" \
    --num_machines "${NUM_NODES}" \
    --num_processes "${TOTAL_GPUS}" \
    --machine_rank "${NODE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    "${TRAINING_SCRIPT}" \
    "${TRAINING_ARGS[@]}"
