#!/usr/bin/env bash
# Reproduce the isolated RoboCasa365 evaluation environment used by OpenWAM.
# The OpenWAM model server remains in the normal OpenWAM environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-/path/to/miniconda3/bin/conda}"
ENV_PREFIX="${ROBOCASA365_ENV_PREFIX:-/path/to/miniconda3/envs/robocasa365}"
ROBOCASA365_PATH="${ROBOCASA365_PATH:-/path/to/robocasa}"
ROBOSUITE_PATH="${ROBOSUITE_PATH:-/path/to/robosuite}"
ROBOCASA_REMOTE="https://github.com/robocasa/robocasa.git"
ROBOSUITE_REMOTE="https://github.com/ARISE-Initiative/robosuite.git"
# RoboCasa main reports package version 1.0.1 at this pinned revision. Version
# 1.0.1 contains the official 1.5x benchmark-horizon update used by single_eval.
ROBOCASA_COMMIT="a07e365c958c4216cd6bbd5f30b47f09a65c6f00"
# RoboCasa requires robosuite master; pin the validated master revision so a
# future upstream change cannot silently alter the evaluator.
ROBOSUITE_COMMIT="5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
DOWNLOAD_ASSETS="${ROBOCASA365_DOWNLOAD_ASSETS:-1}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off

[[ -x "${CONDA_BIN}" ]] || {
    echo "[ERROR] Conda not found or not executable: ${CONDA_BIN}" >&2
    exit 1
}
command -v git >/dev/null || {
    echo "[ERROR] Required command not found: git" >&2
    exit 1
}
case "${DOWNLOAD_ASSETS}" in
    0|1) ;;
    *)
        echo "[ERROR] ROBOCASA365_DOWNLOAD_ASSETS must be 0 or 1" >&2
        exit 1
        ;;
esac

checkout_pinned_repo() {
    local path="$1" remote="$2" commit="$3" label="$4"
    if [[ ! -d "${path}/.git" ]]; then
        git clone "${remote}" "${path}"
        git -C "${path}" checkout --detach "${commit}"
    fi
    local actual_commit
    actual_commit="$(git -C "${path}" rev-parse HEAD)"
    [[ "${actual_commit}" == "${commit}" ]] || {
        echo "[ERROR] Expected ${label} commit ${commit}, found ${actual_commit} at ${path}" >&2
        exit 1
    }
    git -C "${path}" remote set-url origin "${remote}"
}

checkout_pinned_repo "${ROBOSUITE_PATH}" "${ROBOSUITE_REMOTE}" "${ROBOSUITE_COMMIT}" robosuite
checkout_pinned_repo "${ROBOCASA365_PATH}" "${ROBOCASA_REMOTE}" "${ROBOCASA_COMMIT}" RoboCasa365

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    "${CONDA_BIN}" --no-plugins env update --solver classic \
        --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
else
    "${CONDA_BIN}" --no-plugins env create --solver classic \
        --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
fi

PYTHON_BIN="${ENV_PREFIX}/bin/python"
"${PYTHON_BIN}" -m pip install --editable "${ROBOSUITE_PATH}"
"${PYTHON_BIN}" -m pip install --editable "${ROBOCASA365_PATH}"
"${PYTHON_BIN}" -m robocasa.scripts.setup_macros
if [[ "${DOWNLOAD_ASSETS}" == "1" ]]; then
    echo "[setup] downloading/checking RoboCasa kitchen assets (about 10 GB)"
    "${PYTHON_BIN}" -m robocasa.scripts.download_kitchen_assets
else
    echo "[setup] asset download skipped; run the official downloader before env/eval"
fi

"${PYTHON_BIN}" -m pip check
"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata

import gymnasium as gym
import mujoco
import numpy
import robocasa  # noqa: F401
import robocasa.wrappers.gym_wrapper  # noqa: F401

assert metadata.version("robocasa") == "1.0.1", metadata.version("robocasa")
assert mujoco.__version__ == "3.3.1", mujoco.__version__
assert numpy.__version__ == "2.2.5", numpy.__version__
assert metadata.version("websockets") == "15.0.1", metadata.version("websockets")
ids = [key for key in gym.envs.registry if key.startswith("robocasa/")]
assert ids, "RoboCasa gym wrapper registered no robocasa/* environments"
print(f"RoboCasa365 evaluation environment ready; registered_envs={len(ids)}")
PY

echo "[setup] export ROBOCASA365_PYTHON=${PYTHON_BIN}"
