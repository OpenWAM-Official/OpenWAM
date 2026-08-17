#!/usr/bin/env bash
# Reproduce the ordinary LIBERO evaluation environment validated by OpenWAM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="/path/to/miniconda3/bin/conda"
ENV_PREFIX="${LIBERO_ENV_PREFIX:-/path/to/miniconda3/envs/libero}"
LIBERO_PATH="${LIBERO_PATH:-/path/to/LIBERO}"
LIBERO_REMOTE="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
LIBERO_PATCH="${SCRIPT_DIR}/patches/libero-pytorch-load.patch"

[[ -x "${CONDA_BIN}" ]] || {
    echo "[ERROR] Miniconda not found at ${CONDA_BIN}" >&2
    exit 1
}

if [[ ! -d "${LIBERO_PATH}/.git" ]]; then
    git clone "${LIBERO_REMOTE}" "${LIBERO_PATH}"
    git -C "${LIBERO_PATH}" checkout --detach "${LIBERO_COMMIT}"
fi

actual_commit="$(git -C "${LIBERO_PATH}" rev-parse HEAD)"
[[ "${actual_commit}" == "${LIBERO_COMMIT}" ]] || {
    echo "[ERROR] Expected LIBERO commit ${LIBERO_COMMIT}, found ${actual_commit}" >&2
    exit 1
}
git -C "${LIBERO_PATH}" remote set-url origin "${LIBERO_REMOTE}"

if git -C "${LIBERO_PATH}" apply --reverse --check "${LIBERO_PATCH}" >/dev/null 2>&1; then
    echo "[setup] LIBERO PyTorch compatibility patch already applied"
elif git -C "${LIBERO_PATH}" apply --check "${LIBERO_PATCH}" >/dev/null 2>&1; then
    git -C "${LIBERO_PATH}" apply "${LIBERO_PATCH}"
else
    echo "[ERROR] LIBERO checkout has incompatible local changes" >&2
    exit 1
fi

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    "${CONDA_BIN}" env update --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
else
    "${CONDA_BIN}" env create --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
fi

"${ENV_PREFIX}/bin/python" -m pip install --no-deps --editable "${LIBERO_PATH}"
# LIBERO's setup.py uses a namespace-style outer ``libero/`` directory that
# modern PEP 660 editable discovery leaves unmapped. Pin the checkout root on
# sys.path explicitly so ``import libero`` remains valid after a fresh install.
site_packages="$("${ENV_PREFIX}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${LIBERO_PATH}" > "${site_packages}/libero_source.pth"
"${ENV_PREFIX}/bin/python" - <<'PY'
import importlib.metadata as metadata
import libero
import mujoco

assert mujoco.__version__ == "3.3.2", mujoco.__version__
assert metadata.version("robosuite") == "1.4.0"
assert metadata.version("bddl") == "1.0.1"
assert any(path.endswith("/LIBERO/libero") for path in libero.__path__), list(libero.__path__)
print("LIBERO evaluation environment ready: MuJoCo 3.3.2")
PY
