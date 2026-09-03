#!/usr/bin/env bash
# Download official RoboDojo HDF5 for the OpenWAM dataloader.
# Only Hugging Face / ModelScope HDF5 is supported (not LeRobot, depth, or real).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m"; exit 1; }

HF_REPO_ID="${HF_REPO_ID:-RoboDojo-Benchmark/RoboDojo}"
HF_REVISION="${HF_REVISION:-main}"
HF_REPO_URL="${HF_REPO_URL:-https://huggingface.co/datasets/${HF_REPO_ID}}"
MODELSCOPE_REPO_ID="${MODELSCOPE_REPO_ID:-RoboDojo-Benchmark/RoboDojo}"
MODELSCOPE_REVISION="${MODELSCOPE_REVISION:-master}"
MODELSCOPE_REPO_URL="${MODELSCOPE_REPO_URL:-https://modelscope.cn/datasets/${MODELSCOPE_REPO_ID}}"
MODELSCOPE_DATA_ROOT="${MODELSCOPE_DATA_ROOT:-data}"

SOURCE="${1:-}"
DATA_TYPE="hdf5"
DATA_SIZE="523GB"
DATA_DESCRIPTION="HDF5, formal <task>/arx_x5/data/episode_*.hdf5 layout"
DATA_DIR_NAME="RoboDojo"
DATA_ROOT="${ROBO_DOJO_DATA_ROOT:-${REPO_ROOT}/data}"

usage() {
  cat <<EOF
Download official RoboDojo HDF5 for OpenWAM.

Usage:
  bash benchmarks/robodojo/download_robodojo.sh <source>

Sources:
  huggingface
  modelscope

Examples:
  bash benchmarks/robodojo/download_robodojo.sh huggingface
  bash benchmarks/robodojo/download_robodojo.sh modelscope

The OpenWAM dataloader only reads HDF5 (~523GB):

  <data_root>/RoboDojo/<task>/arx_x5/data/episode_*.hdf5

Default <data_root> is <openwam>/data. After a successful download,
dataloader.dataset_dir should be <data_root>/RoboDojo.

LeRobot, depth, and real-robot dumps are not supported.

Environment overrides:
  HF_REPO_ID, HF_REPO_URL, HF_REVISION
  MODELSCOPE_REPO_ID, MODELSCOPE_REPO_URL, MODELSCOPE_REVISION
  MODELSCOPE_DATA_ROOT (default: data)
  ROBO_DOJO_DATA_ROOT
EOF
}

resolve_source() {
  case "${SOURCE,,}" in
    huggingface)
      SOURCE="huggingface"
      REPO_ID="${HF_REPO_ID}"
      REPO_URL="${HF_REPO_URL}"
      REPO_REVISION="${HF_REVISION}"
      REMOTE_DATA_ROOT="data"
      ;;
    modelscope)
      SOURCE="modelscope"
      REPO_ID="${MODELSCOPE_REPO_ID}"
      REPO_URL="${MODELSCOPE_REPO_URL}"
      REPO_REVISION="${MODELSCOPE_REVISION}"
      REMOTE_DATA_ROOT="${MODELSCOPE_DATA_ROOT#/}"
      REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT%/}"
      ;;
    *)
      error "Invalid source: ${SOURCE}. Expected 'huggingface' or 'modelscope'."
      ;;
  esac

  if [[ -n "${REMOTE_DATA_ROOT}" ]]; then
    REMOTE_DIR="${REMOTE_DATA_ROOT}/${DATA_DIR_NAME}"
  else
    REMOTE_DIR="${DATA_DIR_NAME}"
  fi
  TARGET_DIR="${DATA_ROOT}/${DATA_DIR_NAME}"
  DATA_CACHE_DIR="${REPO_ROOT}/.cache/robodojo_data_${SOURCE}_${DATA_TYPE}_repo"
}

check_download_tools() {
  if ! command -v git >/dev/null 2>&1; then
    error "git not found. Please install git first."
  fi
  if ! git lfs version >/dev/null 2>&1; then
    error "git-lfs not found. Please install git-lfs first."
  fi
}

data_ready() {
  [[ -d "${TARGET_DIR}" && -f "${TARGET_DIR}/.download_complete" ]]
}

clone_data_repo() {
  info "Cloning sparse data repo into '${DATA_CACHE_DIR}'..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse --branch "${REPO_REVISION}" \
    "${REPO_URL}" "${DATA_CACHE_DIR}"
}

archive_path() {
  local path="$1"
  local partial_path="${path}.partial.$(date +%Y%m%d_%H%M%S)"
  warn "Moving existing path to '${partial_path}'."
  mv "${path}" "${partial_path}"
}

download_data() {
  info "Repo root: ${REPO_ROOT}"
  info "Data target: ${TARGET_DIR}"
  info "Source: ${SOURCE}"
  info "Repository: ${REPO_ID} (revision=${REPO_REVISION})"
  info "Data format: ${DATA_TYPE} (${DATA_SIZE})"
  info "${DATA_DESCRIPTION}"

  if data_ready; then
    warn "'${TARGET_DIR}' already exists and is marked complete, skipping..."
    return 0
  fi

  mkdir -p "${DATA_ROOT}" "$(dirname "${DATA_CACHE_DIR}")"

  if [[ -e "${TARGET_DIR}" || -L "${TARGET_DIR}" ]]; then
    warn "'${TARGET_DIR}' exists but is not marked complete."
    archive_path "${TARGET_DIR}"
  fi

  if [[ ! -d "${DATA_CACHE_DIR}/.git" ]]; then
    clone_data_repo
  else
    if [[ -n "$(git -C "${DATA_CACHE_DIR}" config --get remote.origin.promisor || true)" ]]; then
      warn "Existing cache was created as a partial clone and may hit Hugging Face promisor fetch errors."
      archive_path "${DATA_CACHE_DIR}"
      clone_data_repo
    else
      info "Updating sparse data repo cache..."
      if ! GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_CACHE_DIR}" fetch --depth 1 origin "${REPO_REVISION}"; then
        warn "Failed to update existing data cache."
        archive_path "${DATA_CACHE_DIR}"
        clone_data_repo
      fi
    fi
  fi

  info "Configuring sparse checkout for ${REMOTE_DIR}/** (without downloading LFS objects)..."
  GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_CACHE_DIR}" sparse-checkout set "${REMOTE_DIR}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_CACHE_DIR}" \
    -c advice.detachedHead=false checkout --quiet --force --detach FETCH_HEAD 2>/dev/null || \
    GIT_LFS_SKIP_SMUDGE=1 git -C "${DATA_CACHE_DIR}" checkout --quiet --force "${REPO_REVISION}"

  if ! git -C "${DATA_CACHE_DIR}" cat-file -e "HEAD:${REMOTE_DIR}" 2>/dev/null; then
    error "Remote folder '${REMOTE_DIR}' is not available from source '${SOURCE}' in ${REPO_ID}."
  fi

  info "Pulling only ${REMOTE_DIR}/** LFS objects..."
  git -C "${DATA_CACHE_DIR}" lfs install --local >/dev/null
  git -C "${DATA_CACHE_DIR}" lfs pull --include="${REMOTE_DIR}/**" --exclude=""

  if [[ ! -d "${DATA_CACHE_DIR}/${REMOTE_DIR}" ]]; then
    error "Remote folder '${REMOTE_DIR}' was not found in ${REPO_ID}."
  fi

  ln -sfn "${DATA_CACHE_DIR}/${REMOTE_DIR}" "${TARGET_DIR}"
  cat > "${TARGET_DIR}/.download_complete" <<EOF
source=${SOURCE}
repo_id=${REPO_ID}
revision=${REPO_REVISION}
remote_dir=${REMOTE_DIR}
data_type=${DATA_TYPE}
data_dir_name=${DATA_DIR_NAME}
size=${DATA_SIZE}
EOF
}

verify_data() {
  if [[ ! -d "${TARGET_DIR}" ]]; then
    error "Expected '${TARGET_DIR}' after download, but it was not created."
  fi
  if [[ ! -f "${TARGET_DIR}/.download_complete" ]]; then
    error "Expected '${TARGET_DIR}/.download_complete' after download, but it was not created."
  fi

  local episode
  episode="$(find -L "${TARGET_DIR}" -mindepth 4 -maxdepth 4 \
    -path "*/arx_x5/data/episode_*.hdf5" -type f -print -quit)"
  if [[ -z "${episode}" ]]; then
    error "No formal episode at '${TARGET_DIR}/<task>/arx_x5/data/episode_*.hdf5'. Do not point dataloader.dataset_dir at the parent of RoboDojo."
  fi
  info "Formal layout check passed: ${episode}"
}

if [[ -z "${SOURCE}" || "${SOURCE}" == "-h" || "${SOURCE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -ne 1 ]]; then
  usage
  exit 1
fi

resolve_source
check_download_tools
download_data
verify_data

info "Data directory is ready: ${TARGET_DIR}"
info "Set dataloader.dataset_dir=${TARGET_DIR}"
