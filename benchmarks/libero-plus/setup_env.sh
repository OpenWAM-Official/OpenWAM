#!/usr/bin/env bash
# Reproduce the LIBERO-plus client environment without touching ordinary LIBERO.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-/path/to/miniconda3/bin/conda}"
ENV_PREFIX="${LIBERO_PLUS_ENV_PREFIX:-/path/to/miniconda3/envs/libero-plus}"
LIBERO_PLUS_PATH="${LIBERO_PLUS_PATH:-/path/to/LIBERO-plus}"
LIBERO_PLUS_REMOTE="https://github.com/sylvestf/LIBERO-plus.git"
LIBERO_PLUS_COMMIT="4976dc30028e805ff8094b55501d532c48fec182"
ASSET_URL="https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"
ASSET_SHA256="96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
ENV_FILE="${SCRIPT_DIR}/environment.yml"
LIBERO_PATCH="${SCRIPT_DIR}/patches/libero-plus-compatibility.patch"
UPSTREAM_REQUIREMENTS="${LIBERO_PLUS_PATH}/requirements.txt"
UPSTREAM_EXTRA_REQUIREMENTS="${LIBERO_PLUS_PATH}/extra_requirements.txt"
PACKAGE_ROOT="${LIBERO_PLUS_PATH}/libero/libero"
ASSET_ARCHIVE="${PACKAGE_ROOT}/assets.zip"
ASSET_ROOT="${PACKAGE_ROOT}/assets"
NESTED_ASSET_ROOT="${PACKAGE_ROOT}/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"

export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_PROGRESS_BAR=off

assets_complete() {
    local required
    for required in \
        articulated_objects new_objects scenes stable_hope_objects \
        stable_scanned_objects textures turbosquid_objects; do
        [[ -d "${ASSET_ROOT}/${required}" ]] || return 1
    done
    for required in serving_region.xml wall_frames.stl wall.xml; do
        [[ -f "${ASSET_ROOT}/${required}" ]] || return 1
    done
}

[[ -x "${CONDA_BIN}" ]] || {
    echo "[ERROR] Conda not found or not executable: ${CONDA_BIN}" >&2
    exit 1
}
for command in git curl unzip sha256sum; do
    command -v "${command}" >/dev/null || {
        echo "[ERROR] Required command not found: ${command}" >&2
        exit 1
    }
done

if [[ ! -d "${LIBERO_PLUS_PATH}/.git" ]]; then
    git clone "${LIBERO_PLUS_REMOTE}" "${LIBERO_PLUS_PATH}"
    git -C "${LIBERO_PLUS_PATH}" checkout --detach "${LIBERO_PLUS_COMMIT}"
fi

actual_commit="$(git -C "${LIBERO_PLUS_PATH}" rev-parse HEAD)"
[[ "${actual_commit}" == "${LIBERO_PLUS_COMMIT}" ]] || {
    echo "[ERROR] Expected LIBERO-plus commit ${LIBERO_PLUS_COMMIT}, found ${actual_commit}" >&2
    exit 1
}
git -C "${LIBERO_PLUS_PATH}" remote set-url origin "${LIBERO_PLUS_REMOTE}"

if git -C "${LIBERO_PLUS_PATH}" apply --reverse --check "${LIBERO_PATCH}" >/dev/null 2>&1; then
    echo "[setup] LIBERO-plus compatibility patch already applied"
elif git -C "${LIBERO_PLUS_PATH}" apply --check "${LIBERO_PATCH}" >/dev/null 2>&1; then
    git -C "${LIBERO_PLUS_PATH}" apply "${LIBERO_PATCH}"
else
    echo "[ERROR] LIBERO-plus checkout has incompatible local changes" >&2
    exit 1
fi

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
    "${CONDA_BIN}" --no-plugins env update --solver classic \
        --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
else
    "${CONDA_BIN}" --no-plugins env create --solver classic \
        --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
fi

"${ENV_PREFIX}/bin/python" -m pip install \
    --requirement "${UPSTREAM_REQUIREMENTS}" \
    --requirement "${UPSTREAM_EXTRA_REQUIREMENTS}"
"${ENV_PREFIX}/bin/python" -m pip uninstall --yes libero >/dev/null 2>&1 || true
"${ENV_PREFIX}/bin/python" -m pip install --no-deps --editable "${LIBERO_PLUS_PATH}"

if ! assets_complete; then
    curl --location --fail --retry 5 --retry-delay 5 --continue-at - \
        --output "${ASSET_ARCHIVE}" "${ASSET_URL}"
    echo "${ASSET_SHA256}  ${ASSET_ARCHIVE}" | sha256sum --check --status || {
        echo "[ERROR] LIBERO-plus asset checksum mismatch: ${ASSET_ARCHIVE}" >&2
        exit 1
    }
    unzip -q -o "${ASSET_ARCHIVE}" -d "${PACKAGE_ROOT}"
    if [[ -d "${NESTED_ASSET_ROOT}" && ! -e "${ASSET_ROOT}" ]]; then
        mv "${NESTED_ASSET_ROOT}" "${ASSET_ROOT}"
    fi
fi

assets_complete || {
    echo "[ERROR] LIBERO-plus assets are incomplete under ${ASSET_ROOT}" >&2
    exit 1
}

"${ENV_PREFIX}/bin/python" -m pip check
"${ENV_PREFIX}/bin/python" - <<'PY'
import importlib.metadata as metadata
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
metadata.version("usd-core")
assert metadata.version("wand") == "0.7.2"
assert metadata.version("scikit-image") == "0.19.3"
print(
    "LIBERO-plus evaluation environment ready with official requirements; "
    f"MuJoCo {mujoco.__version__}"
)
PY
