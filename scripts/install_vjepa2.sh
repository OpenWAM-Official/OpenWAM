#!/usr/bin/env bash
# Initialize the third_party/vjepa2 submodule (branch ``vjepa2_1``) — the one
# piece of V-JEPA 2 / 2.1 setup that pip cannot do for you.
#
# The Python runtime dep (``timm``, which the submodule's vision_transformer.py
# imports as ``timm.models.layers.drop_path``) is declared in the ``[vjepa2]``
# extra, so ``pip install -e .[vjepa2]`` installs it — this script no longer needs
# a separate ``pip install``. What pip still cannot do is resolve the submodule via
# a relative file:// path in [project.optional-dependencies], so fetching the
# upstream code stays here.
#
# Prerequisites:
#   - git checked-out repo (the script does the submodule init for you)
#   - openwam installed with the vjepa2 extra (``pip install -e .[vjepa2]``) — brings timm + torch
#
# Usage:
#   PYBIN=/path/to/python bash scripts/install_vjepa2.sh
#   # or set up PYBIN to default to <repo>/.venv/bin/python first

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VJEPA_ROOT="${REPO_ROOT}/third_party/vjepa2"

# Pick a python interpreter in this order:
#   1. ``PYBIN`` from the environment (explicit override)
#   2. ``$VIRTUAL_ENV/bin/python`` (active virtualenv)
#   3. ``$CONDA_PREFIX/bin/python`` (active conda env)
#   4. ``${REPO_ROOT}/.venv/bin/python`` (project default venv)
#   5. ``python3`` / ``python`` on PATH
# This covers the conda case the original PYBIN fallback missed.
if [ -z "${PYBIN:-}" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        PYBIN="${VIRTUAL_ENV}/bin/python"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        PYBIN="${CONDA_PREFIX}/bin/python"
    elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
        PYBIN="${REPO_ROOT}/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYBIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYBIN="$(command -v python)"
    fi
fi

if [ -z "${PYBIN:-}" ] || [ ! -x "${PYBIN}" ]; then
    echo "error: no python interpreter found; pass PYBIN=/path/to/python explicitly." >&2
    exit 1
fi
echo "Using python: ${PYBIN}"

echo "[1/1] git submodule update --init --recursive third_party/vjepa2"
git -C "${REPO_ROOT}" submodule update --init --recursive third_party/vjepa2

if [ ! -d "${VJEPA_ROOT}/app/vjepa_2_1" ]; then
    echo "error: third_party/vjepa2/app/vjepa_2_1 missing after submodule init." >&2
    echo "       Check that the submodule points at the vjepa2_1 branch (see .gitmodules)." >&2
    exit 1
fi

# timm (the submodule's only extra Python dep) ships in the ``[vjepa2]`` extra, so
# there is no separate ``pip install`` step here. If the smoke check below fails
# with ModuleNotFoundError: timm, run ``pip install -e .[vjepa2]`` to install it.

echo
echo "✓ V-JEPA 2.1 ready. Smoke check:"
echo "    ${PYBIN} -c \"from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY; print('vjepa2_1' in _VIDEO_ENCODER_REGISTRY)\""
