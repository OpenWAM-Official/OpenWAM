ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66
FROM ${CUDA_IMAGE} AS system-dependencies

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_SNAPSHOT=20260910T000000Z
LABEL io.openwam.ubuntu.snapshot="${UBUNTU_SNAPSHOT}"
# Resolve every apt dependency from one immutable Ubuntu archive. CUDA stays
# supplied by the digest-pinned base, with no rolling NVIDIA apt source enabled.
RUN [[ "${UBUNTU_SNAPSHOT}" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] \
    && rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources \
    && printf '%s\n' 'Types: deb' \
        "URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" \
        'Suites: noble noble-updates noble-security noble-backports' \
        'Components: main restricted universe multiverse' \
        'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
        > /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update --error-on=any && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential git ninja-build \
        ca-certificates ffmpeg libgl1 libglib2.0-0 libgomp1 libnss-wrapper \
    && rm -rf /var/lib/apt/lists/* \
    && install -d /usr/local/share/openwam \
    && dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' \
        | LC_ALL=C sort > /usr/local/share/openwam/system-packages.tsv

FROM system-dependencies AS dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}" \
    VIRTUAL_ENV=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_FALLBACK_INDEX_URL=
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
COPY docker/requirements-cu128.txt /opt/openwam-requirements.txt
COPY docker/pip_install.sh /usr/local/bin/openwam-pip-install
# Fetch only the three torch wheels from the dedicated index. Install their
# locked dependencies from PyPI before building DeepSpeed against that environment.
RUN --mount=type=cache,target=/root/.cache/pip \
    bash /usr/local/bin/openwam-pip-install \
        -c /opt/openwam-requirements.txt pip setuptools wheel packaging ninja uv \
    && python -m pip install --no-deps --index-url "${TORCH_INDEX_URL}" \
        -c /opt/openwam-requirements.txt torch torchvision torchaudio
RUN --mount=type=cache,target=/root/.cache/uv \
    indexes=(--default-index "${PIP_INDEX_URL}") \
    && if [[ -n "${PIP_FALLBACK_INDEX_URL}" ]]; then \
        indexes=(--index "${PIP_INDEX_URL}" --default-index "${PIP_FALLBACK_INDEX_URL}" \
            --index-strategy unsafe-first-match); \
    fi \
    && sed '/^deepspeed==/d' /opt/openwam-requirements.txt > /tmp/openwam-base-requirements.txt \
    && uv pip install --python /opt/venv/bin/python --link-mode=copy \
        --no-build-isolation "${indexes[@]}" \
        -r /tmp/openwam-base-requirements.txt \
    && DS_BUILD_OPS=0 uv pip install --python /opt/venv/bin/python --link-mode=copy --no-build-isolation \
        "${indexes[@]}" -r /opt/openwam-requirements.txt

FROM dependencies AS openwam
COPY docker/runtime_user.py /usr/local/lib/openwam/runtime_user.py
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="OpenWAM" \
      org.opencontainers.image.source="https://github.com/OpenWAM-Official/OpenWAM" \
      org.opencontainers.image.revision="${VCS_REF}" \
      io.openwam.bundle.schema="4" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /opt/openwam
COPY . .
# Editable installation deliberately preserves configs/ and scripts/ beside the package.
RUN python -m pip install --no-deps --no-build-isolation -e . \
    && python -m pip check \
    && install -d -m 1777 /cache /outputs \
    && ln -s /outputs /opt/openwam/outputs \
    && chmod +x docker/entrypoint.sh
# NCCL RAS and the NSS preload reproduced a shutdown SIGSEGV on H100.
# Keep user/group lookup; disable RAS monitoring until that interaction is fixed.
ENV NCCL_RAS_ENABLE=0 \
    XDG_CACHE_HOME=/cache \
    HOME=/cache/home \
    USER=openwam \
    LOGNAME=openwam \
    LD_PRELOAD=libnss_wrapper.so \
    NSS_WRAPPER_PASSWD=/tmp/openwam-nss/passwd \
    NSS_WRAPPER_GROUP=/tmp/openwam-nss/group \
    LD_LIBRARY_PATH="/tmp/openwam-runtime/cuda-compat:${LD_LIBRARY_PATH}" \
    RUFF_CACHE_DIR=/cache/ruff \
    CUDA_CACHE_PATH=/cache/cuda \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    TRITON_CACHE_DIR=/cache/triton \
    WANDB_DIR=/outputs \
    WANDB_CACHE_DIR=/cache/wandb \
    WANDB_CONFIG_DIR=/cache/wandb/config \
    WANDB_MODE=offline \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER 1000:1000
EXPOSE 8848
ENTRYPOINT ["python", "/usr/local/lib/openwam/runtime_user.py"]
CMD ["serve", "--help"]

# Optional, separate dependency boundary. Requires the initialized submodule.
# Upstream Cosmos has known metadata pin conflicts; see install_cosmos_predict25.sh.
FROM openwam AS cosmos
USER root
RUN LD_PRELOAD= PYBIN=/opt/venv/bin/python bash scripts/install_cosmos_predict25.sh
USER 1000:1000

# Plain `docker build` keeps the standard Wan environment as its default.
FROM openwam AS default
