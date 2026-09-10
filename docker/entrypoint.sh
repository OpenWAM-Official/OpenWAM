#!/usr/bin/env bash
set -euo pipefail

for cache_dir in "${XDG_CACHE_HOME:-/cache}" "${RUFF_CACHE_DIR:-/cache/ruff}" \
    "${CUDA_CACHE_PATH:-/cache/cuda}" "${HF_HOME:-/cache/huggingface}" \
    "${TORCH_HOME:-/cache/torch}" "${TORCH_EXTENSIONS_DIR:-/cache/torch_extensions}" \
    "${TORCHINDUCTOR_CACHE_DIR:-/cache/torchinductor}" "${TRITON_CACHE_DIR:-/cache/triton}" \
    "${WANDB_CACHE_DIR:-/cache/wandb}" "${WANDB_CONFIG_DIR:-/cache/wandb/config}"; do
    if ! mkdir -p "$cache_dir" || [[ ! -w "$cache_dir" ]]; then
        echo "OpenWAM: cache directory is not writable: $cache_dir (check bind-mount UID/GID)." >&2
        exit 1
    fi
done

# Usually handled by NVIDIA Container Toolkit. Opt in only when validating an
# older data-center driver that needs the image's forward-compatibility libraries.
if [[ "${OPENWAM_CUDA_COMPAT:-0}" == 1 ]]; then
    compat_dir="${CUDA_HOME:-/usr/local/cuda}/compat"
    if [[ ! -e "$compat_dir/libcuda.so.1" ]]; then
        echo "OpenWAM: CUDA compatibility libraries missing from $compat_dir." >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="$compat_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

case "${1:-serve}" in
    serve)
        if (( $# )); then shift; fi
        exec python /opt/openwam/scripts/deploy.py "$@"
        ;;
    train)
        shift
        # torch respects CUDA_VISIBLE_DEVICES; nvidia-smi may still list every
        # device exposed to the container. Do not launch ranks for hidden GPUs.
        visible_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')"
        export NPROC_PER_NODE="${NPROC_PER_NODE:-$visible_gpus}"
        if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || (( NPROC_PER_NODE > visible_gpus )); then
            echo "OpenWAM: NPROC_PER_NODE=$NPROC_PER_NODE, but torch sees $visible_gpus GPU(s). Check GPU access and CUDA_VISIBLE_DEVICES." >&2
            exit 1
        fi
        exec bash /opt/openwam/scripts/train.sh "$@"
        ;;
    *) exec "$@" ;;
esac
