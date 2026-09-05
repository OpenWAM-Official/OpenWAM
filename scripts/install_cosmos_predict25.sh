#!/usr/bin/env bash
# Install the Cosmos-Predict2.5 extra (cosmos-oss / cosmos-cuda / cosmos-predict2)
# from the upstream submodule under third_party/cosmos-predict2.5/, plus the
# transformer-engine[pytorch] wheel that the Cosmos DiT needs at import time.
#
# Why not a pip extra? cosmos-oss is not on PyPI and pip cannot resolve a
# relative file:// path in [project.optional-dependencies]. This script
# replaces what `pip install -e '.[cosmos_predict25]'` would do if it could.
#
# Prerequisites on the host:
#   - third_party/cosmos-predict2.5/ checked out (run `git submodule update
#     --init third_party/cosmos-predict2.5` after cloning).
#   - CUDA toolkit (nvcc) at /usr/local/cuda (CUDA 12.x).
#   - cuDNN headers — torch ships them under
#     <venv>/lib/python*/site-packages/nvidia/cudnn/include/, this script
#     points the compiler at that path.
#   - torch already installed in the active venv (we read its cuDNN bundle).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COSMOS_ROOT="${REPO_ROOT}/third_party/cosmos-predict2.5"

# Interpreter resolution: explicit $PYBIN > the environment ACTIVATED in the
# calling shell (venv $VIRTUAL_ENV, then non-base conda $CONDA_PREFIX) > conda
# env named `openwam` (probed via `conda env list`, then common conda roots —
# no activation needed) > system python3. In short: activate your env and run
# the script — it installs into that env.
resolve_pybin() {
    if [ -n "${PYBIN:-}" ]; then
        echo "${PYBIN}"
        return
    fi
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        echo "${VIRTUAL_ENV}/bin/python"
        return
    fi
    case "${CONDA_PREFIX:-}" in
        */envs/*)  # an explicitly activated conda env (base is skipped on purpose)
            if [ -x "${CONDA_PREFIX}/bin/python" ]; then
                echo "${CONDA_PREFIX}/bin/python"
                return
            fi
            ;;
    esac
    if command -v conda >/dev/null 2>&1; then
        local env_path
        env_path="$(conda env list 2>/dev/null | awk '$1 == "openwam" {print $NF}')"
        if [ -n "${env_path}" ] && [ -x "${env_path}/bin/python" ]; then
            echo "${env_path}/bin/python"
            return
        fi
    fi
    local root
    for root in "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/miniforge3" /opt/conda /opt/miniconda3; do
        if [ -x "${root}/envs/openwam/bin/python" ]; then
            echo "${root}/envs/openwam/bin/python"
            return
        fi
    done
    command -v python3 || true
}

PYBIN="$(resolve_pybin)"
if [ -z "${PYBIN}" ] || [ ! -x "${PYBIN}" ]; then
    echo "error: no usable python found (no \$PYBIN, no conda env 'openwam', no system python3); set PYBIN=/path/to/python." >&2
    exit 1
fi
echo "[install_cosmos_predict25] python:    ${PYBIN}"

if [ ! -f "${COSMOS_ROOT}/pyproject.toml" ]; then
    echo "error: cosmos-predict2.5 submodule not checked out at ${COSMOS_ROOT}." >&2
    echo "       Run: git submodule update --init third_party/cosmos-predict2.5" >&2
    exit 1
fi

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "error: nvcc not found at ${CUDA_HOME}/bin/nvcc. Install CUDA toolkit or set CUDA_HOME." >&2
    exit 1
fi

# Locate cuDNN headers bundled inside torch's nvidia-cudnn-cu12 wheel.
CUDNN_INC=$("${PYBIN}" -c "import importlib.util, os; spec=importlib.util.find_spec('nvidia.cudnn'); print(os.path.join(os.path.dirname(spec.origin), 'include') if spec and spec.origin else '')")
if [ -z "${CUDNN_INC}" ] || [ ! -f "${CUDNN_INC}/cudnn.h" ]; then
    echo "error: cuDNN headers not found via ${PYBIN} (nvidia-cudnn-cu12 missing)." >&2
    echo "       This python likely isn't the env with torch installed — activate it or set PYBIN=/path/to/env/bin/python." >&2
    exit 1
fi
CUDNN_LIB=$("${PYBIN}" -c "import importlib.util, os; spec=importlib.util.find_spec('nvidia.cudnn'); print(os.path.join(os.path.dirname(spec.origin), 'lib'))")

# Compute capability — restrict to the local GPU(s) to keep TE compile cheap.
# Falls back to a broad list if nvidia-smi is missing.
if command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -)"
else
    TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
fi

echo "[install_cosmos_predict25] venv:      ${PYBIN}"
echo "[install_cosmos_predict25] submodule: ${COSMOS_ROOT}"
echo "[install_cosmos_predict25] CUDA:      ${CUDA_HOME}"
echo "[install_cosmos_predict25] cuDNN:     ${CUDNN_INC}"
echo "[install_cosmos_predict25] SM list:   ${TORCH_CUDA_ARCH_LIST}"

# Packages the host repo pins that cosmos-oss would otherwise downgrade (its
# transformers==4.51.3 pin predates Qwen3-VL and would break the tri_system
# VLM backbone). Snapshot the versions now, restore them after the installs;
# runtime coexistence with cosmos_predict2 is verified by the smoke import.
PROTECTED_PKGS=(transformers tokenizers huggingface_hub diffusers)
pkg_ver() { { "${PYBIN}" -m pip show "$1" 2>/dev/null || true; } | awk '/^Version:/{print $2}'; }
declare -A PRE_VER
for p in "${PROTECTED_PKGS[@]}"; do PRE_VER[$p]="$(pkg_ver "$p")"; done

# 1) cosmos-cuda is a sentinel package that cosmos-oss.__init__ imports to
#    confirm a CUDA extra was installed — it carries no code, just version.
"${PYBIN}" -m pip install -e "${COSMOS_ROOT}/packages/cosmos-cuda"

# 2) cosmos-oss pulls the bulk of upstream deps (megatron-core, diffusers,
#    transformers, peft, multi-storage-client, ...). ~80 packages, ~10 min.
"${PYBIN}" -m pip install -e "${COSMOS_ROOT}/packages/cosmos-oss"

# 3) cosmos-predict2 itself pins `cosmos-oss==0.1.0` (the PyPI placeholder),
#    so we install it with --no-deps and rely on the already-installed 1.5.0.
"${PYBIN}" -m pip install --no-deps -e "${COSMOS_ROOT}"

# 4) Build deps for transformer-engine (TE-torch is sdist-only on PyPI;
#    it spawns nvcc+g++ against the cuDNN headers we located above).
"${PYBIN}" -m pip install pybind11

# 5) transformer-engine[pytorch]==2.7.0 — the TE build OpenWAM trains and
#    deploys the Cosmos DiT against (the DiT imports transformer_engine at
#    runtime). Coexists with the restored transformers stack. Set NVTE_FRAMEWORK
#    so TE skips JAX. The compile-time CPATH lets g++ see cudnn.h; LD path lets
#    the resulting .so find libcudnn at runtime.
CUDA_HOME="${CUDA_HOME}" \
PATH="${CUDA_HOME}/bin:$(dirname "${PYBIN}"):${PATH}" \
LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}" \
CPATH="${CUDNN_INC}:${CPATH:-}" \
CPLUS_INCLUDE_PATH="${CUDNN_INC}:${CPLUS_INCLUDE_PATH:-}" \
C_INCLUDE_PATH="${CUDNN_INC}:${C_INCLUDE_PATH:-}" \
NVTE_FRAMEWORK=pytorch \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    "${PYBIN}" -m pip install --no-build-isolation 'transformer-engine[pytorch]==2.7.0'

# 6) Restore the host repo's package pins that cosmos-oss moved. After this,
#    `pip check` reports cosmos-oss's stricter transformers pin as a metadata
#    conflict — expected and harmless; the smoke import below proves the
#    restored stack and cosmos_predict2 coexist at runtime.
RESTORE=()
for p in "${PROTECTED_PKGS[@]}"; do
    pre="${PRE_VER[$p]}"
    [ -n "${pre}" ] || continue
    post="$(pkg_ver "$p")"
    if [ "${post}" != "${pre}" ]; then
        RESTORE+=("${p}==${pre}")
    fi
done
if [ "${#RESTORE[@]}" -gt 0 ]; then
    echo "[install_cosmos_predict25] restoring host-repo pins moved by cosmos deps: ${RESTORE[*]}"
    "${PYBIN}" -m pip install "${RESTORE[@]}"
else
    echo "[install_cosmos_predict25] no protected packages were moved; nothing to restore."
fi

# 7) Smoke import: what build_cosmos_predict25_pipeline needs at runtime, plus
#    proof that the restored transformers stack still serves the VLM backbone.
"${PYBIN}" - <<'PY'
import cosmos_oss
import cosmos_predict2
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, Block
from transformer_engine.pytorch import RMSNorm  # noqa: F401
import transformers
print(f"OK cosmos_predict2 v{cosmos_predict2.__about__.__version__} from {cosmos_predict2.__file__}")
print(f"OK MiniTrainDIT, Block from {MiniTrainDIT.__module__}")
print(f"OK transformers {transformers.__version__} coexists")
try:
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    print("OK Qwen3-VL classes available (tri_system VLM backbone unaffected)")
except ImportError:
    print("WARNING: this transformers lacks Qwen3-VL; if you use tri_system, "
          "restore the host pin with `pip install -e .` from the repo root.")
PY

echo "[install_cosmos_predict25] done."
