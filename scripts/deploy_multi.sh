#!/usr/bin/env bash
# Launch N OpenWAM policy servers across N GPUs with incrementing ports.
#
# Port mapping (default): GPU i -> port = 8848 + i.
# Logs: logs/deploy_gpu${i}.log (override with LOG_DIR).
#
# Usage:
#   bash scripts/deploy_multi.sh /path/to/checkpoint_dir
#   bash scripts/deploy_multi.sh /path/to/checkpoint_dir --denoise-steps 10
#   NUM_GPUS=4 PORT_BASE=9000 bash scripts/deploy_multi.sh /path/to/checkpoint_dir
#
# Environment overrides:
#   NUM_GPUS         Number of GPUs / servers to launch (default: 8)
#   PORT_BASE        Base WebSocket port (default: 8848)
#   LOG_DIR          Log directory (default: ./logs)
#   GPU_START        First GPU index (default: 0)
#
# Ctrl+C terminates all launched servers.
#
# Example: NUM_GPUS=8 PORT_BASE=9000 bash scripts/deploy_multi.sh <ckpt>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_GPUS="${NUM_GPUS:-8}"
PORT_BASE="${PORT_BASE:-8848}"
GPU_START="${GPU_START:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

mkdir -p "$LOG_DIR"

# Backward-compat with deploy.sh: first positional (non-flag) arg is ckpt_dir.
ckpt_args=()
if [[ $# -gt 0 && "$1" != -* ]]; then
    ckpt_args+=(--ckpt-dir "$1")
    shift
fi

pids=()

# Recursive tree-kill so grandchildren (e.g. torch dataloader workers spawned
# by deploy.py) don't survive as orphans when the parent python dies.
kill_tree() {
    local pid=$1 sig=${2:-TERM}
    [[ -z "$pid" ]] && return
    local child
    while read -r child; do
        [[ -n "$child" ]] && kill_tree "$child" "$sig"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
    kill -"$sig" "$pid" 2>/dev/null || true
}

cleanup() {
    trap - INT TERM
    echo ""
    echo "[deploy_multi] Shutting down ${#pids[@]} servers..."
    for pid in "${pids[@]}"; do kill_tree "$pid" TERM; done
    local deadline=$((SECONDS + 5)) still_alive=1
    while (( SECONDS < deadline )); do
        still_alive=0
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && { still_alive=1; break; }
        done
        (( still_alive )) || break
        sleep 0.2
    done
    if (( still_alive )); then
        echo "[deploy_multi] Escalating to SIGKILL for survivors..."
        for pid in "${pids[@]}"; do kill_tree "$pid" KILL; done
    fi
    wait 2>/dev/null || true
    echo "[deploy_multi] All servers stopped."
}
trap cleanup INT TERM

echo "[deploy_multi] Launching ${NUM_GPUS} servers (GPU ${GPU_START}..$((GPU_START + NUM_GPUS - 1)))"
echo "[deploy_multi] port: ${PORT_BASE}..$((PORT_BASE + NUM_GPUS - 1))"
echo "[deploy_multi] Logs: ${LOG_DIR}/deploy_gpu*.log"
echo ""

for ((i = 0; i < NUM_GPUS; i++)); do
    gpu=$((GPU_START + i))
    port=$((PORT_BASE + i))
    log_file="${LOG_DIR}/deploy_gpu${gpu}.log"

    echo "[deploy_multi] GPU ${gpu} -> ws=${port} log=${log_file}"

    python "$SCRIPT_DIR/deploy.py" \
        "${ckpt_args[@]}" \
        --device "cuda:${gpu}" \
        --port "$port" \
        "$@" \
        >"$log_file" 2>&1 &

    pids+=($!)
done

echo ""
echo "[deploy_multi] All servers launched. PIDs: ${pids[*]}"
echo "[deploy_multi] Tail logs with:  tail -f ${LOG_DIR}/deploy_gpu*.log"
echo "[deploy_multi] Press Ctrl+C to stop all servers."

wait
