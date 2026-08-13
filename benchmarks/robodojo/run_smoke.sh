#!/usr/bin/env bash
# Progressive RoboDojo/OpenWAM smoke launcher.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CALLER_CWD="${PWD}"

resolve_from_caller() {
    if [[ "$1" = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "${CALLER_CWD}" "$1"
    fi
}

MODE="${1:-contract}"
if [[ $# -gt 0 ]]; then
    shift
fi

CONFIG="${ROBODOJO_POLICY_CONFIG:-${SCRIPT_DIR}/policy_config.yml}"
CONFIG="$(resolve_from_caller "${CONFIG}")"

if [[ -n "${ROBODOJO_PYTHON:-}" ]]; then
    PYTHON_CMD=("${ROBODOJO_PYTHON}")
else
    PYTHON_CMD=(conda run -n RoboDojo python)
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

ARGS=(
    -m benchmarks.robodojo.smoke_robodojo
    --mode "${MODE}"
    --config "${CONFIG}"
)
if [[ -v OPENWAM_HOST ]]; then
    ARGS+=(--host "${OPENWAM_HOST}")
fi
if [[ -v OPENWAM_PORT ]]; then
    ARGS+=(--port "${OPENWAM_PORT}")
fi
if [[ -v OPENWAM_TIMEOUT ]]; then
    ARGS+=(--timeout "${OPENWAM_TIMEOUT}")
fi
if [[ -n "${ROBODOJO_ROOT:-}" ]]; then
    ARGS+=(--robodojo-root "$(resolve_from_caller "${ROBODOJO_ROOT}")")
fi

case "${MODE}" in
    contract)
        if [[ -n "${ROBODOJO_CALIBRATION:-}" ]]; then
            ARGS+=(--calibration "$(resolve_from_caller "${ROBODOJO_CALIBRATION}")")
        elif [[ "${1:-}" == *.json ]]; then
            ARGS+=(--calibration "$(resolve_from_caller "$1")")
            shift
        fi
        if [[ -n "${1:-${ROBODOJO_DATASET_ROOT:-}}" ]]; then
            DATASET_ROOT="$(resolve_from_caller "${1:-${ROBODOJO_DATASET_ROOT}}")"
            ARGS+=(
                --dataset-root "${DATASET_ROOT}"
                --task "${2:-${ROBODOJO_SMOKE_TASK:-stack_blocks}}"
            )
        fi
        ;;
    ping)
        ;;
    debug)
        if [[ -n "${ROBODOJO_CALIBRATION:-}" ]]; then
            ARGS+=(--calibration "$(resolve_from_caller "${ROBODOJO_CALIBRATION}")")
        elif [[ "${1:-}" == *.json ]]; then
            ARGS+=(--calibration "$(resolve_from_caller "$1")")
            shift
        fi
        export EVAL_ENV_TYPE=debug
        ARGS+=(
            --steps "${ROBODOJO_DEBUG_STEPS:-1}"
            --task "${1:-${ROBODOJO_SMOKE_TASK:-stack_blocks}}"
        )
        ;;
    isaac)
        ARGS+=(
            --steps "${ROBODOJO_ISAAC_STEPS:-3}"
            --task "${1:-${ROBODOJO_SMOKE_TASK:-stack_blocks}}"
        )
        ;;
    *)
        echo "[ERROR] Unknown mode '${MODE}'. Use contract | ping | debug | isaac." >&2
        exit 2
        ;;
esac

echo "[robodojo-smoke] mode=${MODE} config=${CONFIG}"
if [[ "${MODE}" != "isaac" ]]; then
    exec "${PYTHON_CMD[@]}" "${ARGS[@]}"
fi

if [[ -z "${ROBODOJO_RUN_ID:-}" ]]; then
    export ROBODOJO_RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)-$$"
fi
MAX_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-10}"
RETRY_DELAY="${ROBODOJO_RETRY_DELAY_SECONDS:-5}"
if [[ ! "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ROBODOJO_MAX_BASH_RETRIES must be a non-negative integer." >&2
    exit 2
fi

retries_used=0
while :; do
    set +e
    "${PYTHON_CMD[@]}" "${ARGS[@]}"
    rc=$?
    set -e
    case "${rc}" in
        0)
            exit 0
            ;;
        99|134|139)
            if [[ "${retries_used}" -ge "${MAX_RETRIES}" ]]; then
                echo "[robodojo-smoke] retry cap reached (${retries_used}/${MAX_RETRIES} retries after initial launch, rc=${rc}, run_id=${ROBODOJO_RUN_ID})." >&2
                exit "${rc}"
            fi
            retries_used=$((retries_used + 1))
            echo "[robodojo-smoke] retrying Isaac after rc=${rc} (${retries_used}/${MAX_RETRIES} retries, run_id=${ROBODOJO_RUN_ID})." >&2
            sleep "${RETRY_DELAY}"
            ;;
        *)
            exit "${rc}"
            ;;
    esac
done
