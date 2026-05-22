#!/usr/bin/env bash
# Install the V-JEPA 2.1 extra:
#   1. Initialize the third_party/vjepa2 submodule (branch ``vjepa2_1``).
#   2. ``pip install timm`` — the only V-JEPA runtime dep openwam doesn't
#      already ship (vision_transformer.py imports timm.models.layers.drop_path).
#
# Why not a pip extra? Same reason as install_cosmos25.sh: pip cannot resolve
# the submodule via a relative file:// path in [project.optional-dependencies],
# and we don't want every user to pull V-JEPA-only deps just to install
# openwam. This script replaces what `pip install -e '.[vjepa2]'` would do
# if pip supported it.
#
# Prerequisites:
#   - git checked-out repo (the script does the submodule init for you)
#   - active venv / conda env with torch already installed
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

echo "[1/2] git submodule update --init --recursive third_party/vjepa2"
git -C "${REPO_ROOT}" submodule update --init --recursive third_party/vjepa2

if [ ! -d "${VJEPA_ROOT}/app/vjepa_2_1" ]; then
    echo "error: third_party/vjepa2/app/vjepa_2_1 missing after submodule init." >&2
    echo "       Check that the submodule points at the vjepa2_1 branch (see .gitmodules)." >&2
    exit 1
fi

echo "[2/2] pip install timm  (the only runtime dep openwam doesn't ship)"
"${PYBIN}" -m pip install --no-input timm

echo
echo "✓ V-JEPA 2.1 ready. Smoke check:"
echo "    ${PYBIN} -c \"from openwam.model.video_backbone.encoder import _VIDEO_ENCODER_REGISTRY; print('vjepa2_1' in _VIDEO_ENCODER_REGISTRY)\""
