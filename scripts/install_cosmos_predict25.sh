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
PYBIN="${PYBIN:-${REPO_ROOT}/.venv/bin/python}"
COSMOS_ROOT="${REPO_ROOT}/third_party/cosmos-predict2.5"

if [ ! -x "${PYBIN}" ]; then
    echo "error: ${PYBIN} not found; set PYBIN=/path/to/python or create ${REPO_ROOT}/.venv first." >&2
    exit 1
fi

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
CUDNN_INC=$("${PYBIN}" -c "import importlib.util, os; spec=importlib.util.find_spec('nvidia.cudnn'); print(os.path.join(os.path.dirname(spec.origin), 'include'))")
if [ ! -f "${CUDNN_INC}/cudnn.h" ]; then
    echo "error: cudnn.h not found at ${CUDNN_INC}. Install nvidia-cudnn-cu12 (torch dep) or system cuDNN-dev." >&2
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

# 5) transformer-engine[pytorch]==2.7.0 — pinned to the version cosmos-oss's
#    cu128_torch27 extra resolves to. Set NVTE_FRAMEWORK so TE skips JAX. The
#    compile-time CPATH lets g++ see cudnn.h; LD path lets the resulting .so
#    find libcudnn at runtime.
CUDA_HOME="${CUDA_HOME}" \
PATH="${CUDA_HOME}/bin:${REPO_ROOT}/.venv/bin:${PATH}" \
LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}" \
CPATH="${CUDNN_INC}:${CPATH:-}" \
CPLUS_INCLUDE_PATH="${CUDNN_INC}:${CPLUS_INCLUDE_PATH:-}" \
C_INCLUDE_PATH="${CUDNN_INC}:${C_INCLUDE_PATH:-}" \
NVTE_FRAMEWORK=pytorch \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    "${PYBIN}" -m pip install --no-build-isolation 'transformer-engine[pytorch]==2.7.0'

# 6) Smoke import: this is what build_cosmos_predict25_pipeline will need at runtime.
"${PYBIN}" - <<'PY'
import cosmos_oss
import cosmos_predict2
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import MiniTrainDIT, Block
from transformer_engine.pytorch import RMSNorm  # noqa: F401
print(f"OK cosmos_predict2 v{cosmos_predict2.__about__.__version__} from {cosmos_predict2.__file__}")
print(f"OK MiniTrainDIT, Block from {MiniTrainDIT.__module__}")
PY

echo "[install_cosmos_predict25] done."
