#!/usr/bin/env bash
# Deploy OpenWAM policy server(s) from a checkpoint directory.
#
# Single server (default) runs in the foreground with logs on the terminal.
# NUM_GPUS > 1 launches one server per GPU with incrementing ports, logs per
# GPU under LOG_DIR, and Ctrl+C tears the whole fleet down.
#
# Usage:
#   bash scripts/deploy.sh /path/to/checkpoint_dir
#   bash scripts/deploy.sh /path/to/checkpoint_dir --device cuda:1 --port 9000
#   bash scripts/deploy.sh --ckpt-dir /path/to/checkpoint_dir --ckpt-name checkpoint_step_1000.safetensors
#   NUM_GPUS=8 bash scripts/deploy.sh /path/to/checkpoint_dir
#   NUM_GPUS=4 PORT_BASE=9000 bash scripts/deploy.sh /path/to/checkpoint_dir --denoise-steps 10
#
# Environment overrides (multi-server mode, NUM_GPUS > 1):
#   NUM_GPUS         Number of GPUs / servers to launch (default: 1)
#   PORT_BASE        Base WebSocket port; GPU i -> PORT_BASE + i (default: 8848)
#   GPU_START        First GPU index (default: 0)
#   LOG_DIR          Log directory (default: ./logs), files deploy_gpu${i}.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_GPUS="${NUM_GPUS:-1}"

# Backward-compat: if first arg is a path (not a flag), treat it as --ckpt-dir
ckpt_args=()
if [[ $# -gt 0 && "$1" != -* ]]; then
    ckpt_args+=(--ckpt-dir "$1")
    shift
fi

# ── Single server: foreground, terminal logs, plain signal semantics ─────────
if (( NUM_GPUS == 1 )); then
    exec python "$SCRIPT_DIR/deploy.py" "${ckpt_args[@]}" "$@"
fi

# ── Multi server: one process per GPU, incrementing ports, log files ─────────
PORT_BASE="${PORT_BASE:-8848}"
GPU_START="${GPU_START:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

mkdir -p "$LOG_DIR"

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
    echo "[deploy] Shutting down ${#pids[@]} servers..."
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
        echo "[deploy] Escalating to SIGKILL for survivors..."
        for pid in "${pids[@]}"; do kill_tree "$pid" KILL; done
    fi
    wait 2>/dev/null || true
    echo "[deploy] All servers stopped."
}
trap cleanup INT TERM

echo "[deploy] Launching ${NUM_GPUS} servers (GPU ${GPU_START}..$((GPU_START + NUM_GPUS - 1)))"
echo "[deploy] port: ${PORT_BASE}..$((PORT_BASE + NUM_GPUS - 1))"
echo "[deploy] Logs: ${LOG_DIR}/deploy_gpu*.log"
echo ""

for ((i = 0; i < NUM_GPUS; i++)); do
    gpu=$((GPU_START + i))
    port=$((PORT_BASE + i))
    log_file="${LOG_DIR}/deploy_gpu${gpu}.log"

    echo "[deploy] GPU ${gpu} -> ws=${port} log=${log_file}"

    python "$SCRIPT_DIR/deploy.py" \
        "${ckpt_args[@]}" \
        --device "cuda:${gpu}" \
        --port "$port" \
        "$@" \
        >"$log_file" 2>&1 &

    pids+=($!)
done

echo ""
echo "[deploy] All servers launched. PIDs: ${pids[*]}"
echo "[deploy] Tail logs with:  tail -f ${LOG_DIR}/deploy_gpu*.log"
echo "[deploy] Press Ctrl+C to stop all servers."

wait
