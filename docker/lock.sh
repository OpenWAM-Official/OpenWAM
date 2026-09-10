#!/usr/bin/env bash
# Regenerate the Python 3.12 / Linux amd64 container dependency lock with uv.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv pip compile pyproject.toml docker/requirements.in \
    --extra dev --python-version 3.12 --python-platform x86_64-manylinux_2_39 \
    --torch-backend cu128 --constraint docker/constraints-cu128.txt \
    --output-file docker/requirements-cu128.txt \
    --custom-compile-command 'bash docker/lock.sh' "$@"
