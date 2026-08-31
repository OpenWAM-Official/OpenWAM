#!/usr/bin/env bash
# Reproduce the ordinary LIBERO evaluation environment validated by OpenWAM.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-/path/to/miniconda3/bin/conda}"
ENV_PREFIX="${LIBERO_ENV_PREFIX:-/path/to/miniconda3/envs/libero}"
LIBERO_PATH="${LIBERO_PATH:-/path/to/LIBERO}"
export LIBERO_PATH
LIBERO_REMOTE="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
LIBERO_PATCH="${SCRIPT_DIR}/patches/libero-pytorch-load.patch"
UPSTREAM_REQUIREMENTS="${LIBERO_PATH}/requirements.txt"

# Do not inherit machine-global pip indexes: the environment file declares the
# only extra index it needs (PyTorch CPU wheels).
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off

[[ -x "${CONDA_BIN}" ]] || {
    echo "[ERROR] Conda not found or not executable: ${CONDA_BIN}" >&2
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
    "${CONDA_BIN}" --no-plugins env update --solver classic \
        --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
else
    "${CONDA_BIN}" --no-plugins env create --solver classic \
        --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
fi

"${ENV_PREFIX}/bin/python" -m pip install --requirement "${UPSTREAM_REQUIREMENTS}"
"${ENV_PREFIX}/bin/python" -m pip install --no-deps --editable "${LIBERO_PATH}"
# LIBERO's setup.py uses a namespace-style outer ``libero/`` directory that
# modern PEP 660 editable discovery leaves unmapped. Pin the checkout root on
# sys.path explicitly so ``import libero`` remains valid after a fresh install.
site_packages="$("${ENV_PREFIX}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${LIBERO_PATH}" > "${site_packages}/libero_source.pth"
"${ENV_PREFIX}/bin/python" -m pip check
"${ENV_PREFIX}/bin/python" - <<'PY'
import importlib.metadata as metadata
import os
from pathlib import Path

import libero
import mujoco

assert mujoco.__version__ == "3.3.2", mujoco.__version__
expected = {
    "hydra-core": "1.2.0",
    "numpy": "1.22.4",
    "wandb": "0.13.1",
    "easydict": "1.9",
    "transformers": "4.21.1",
    "opencv-python": "4.6.0.66",
    "robomimic": "0.2.0",
    "einops": "0.4.1",
    "thop": "0.1.1-2209072238",
    "robosuite": "1.4.0",
    "bddl": "1.0.1",
    "future": "0.18.2",
    "matplotlib": "3.5.3",
    "cloudpickle": "2.1.0",
    "gym": "0.25.2",
}
for package, required in expected.items():
    actual = metadata.version(package)
    assert actual == required, f"{package}: expected {required}, found {actual}"
expected_package_root = (Path(os.environ["LIBERO_PATH"]).resolve() / "libero").resolve()
assert any(Path(path).resolve() == expected_package_root for path in libero.__path__), list(libero.__path__)
print("LIBERO evaluation environment ready with official requirements and MuJoCo 3.3.2")
PY
