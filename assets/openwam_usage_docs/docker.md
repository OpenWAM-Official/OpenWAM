# Docker and offline deployment

Run an OpenWAM policy, train a model, or edit the source in a reproducible CUDA
image. Docker includes Python 3.12, PyTorch 2.7.1 and CUDA 12.8; you do not need
to install the project's Python environment on the host. Benchmark simulators
use their own environments and connect to the policy's WebSocket endpoint.

| I want to… | Start here |
| --- | --- |
| Run a released policy | [Prepare a checkout](#prepare-a-checkout), then [serve a checkpoint](#serve-a-checkpoint) |
| Train or fine-tune | [Prepare a checkout](#prepare-a-checkout), then [train](#train) |
| Edit code and run tests | [Prepare a checkout](#prepare-a-checkout), then [develop from a checkout](#develop-from-a-checkout) |
| Deploy to a server without internet | [Offline delivery](#offline-delivery) |

[Networking and inference options](#networking-and-inference-options) ·
[Troubleshooting](#troubleshooting) · [Maintainer reference](#maintainer-reference)

## Before you start

| Machine / task | Requirements |
| --- | --- |
| All Docker hosts | Linux x86-64, Docker Engine with a working daemon, Docker Compose 2.30+ |
| Build images or develop from source | Git, Make, Python 3.8+ for host-side validation/export; internet for the initial build and downloads. A GPU is not required for building or CPU tests. |
| Serve or train | NVIDIA GPU, host NVIDIA driver and NVIDIA Container Toolkit; local weights and, for training, a dataset |
| Import an offline bundle | Python 3.8+; install Docker, Compose, driver and Toolkit before disconnecting the server |

Installation: [Docker Engine](https://docs.docker.com/engine/install/) and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Allow disk space for the CUDA image, compilation cache, weights and outputs.
The example policy download is about 25 GB; training checkpoints need much more.

**Where commands run:** `docker`, `docker compose` and `make docker-*` commands
run in a **host terminal**, from the checkout root or the offline bundle directory
specified below. The development container has no Docker CLI. Commands to type
after entering that container are explicitly marked **Inside the container**.

On each host, confirm Docker is accessible before continuing:

```bash
# Host terminal — any directory
docker version
docker compose version
```

`docker version` must show both Client and Server. If it cannot connect, fix the
host daemon/context or permissions first; see [troubleshooting](#troubleshooting).

## Prepare a checkout

Use a connected host for this section. If you already have a checkout, open a
host terminal at its root and skip the clone command. Offline bundle users
should start at [offline delivery](#offline-delivery).

```bash
# Connected host terminal — parent directory for your checkout
git clone https://github.com/OpenWAM-Official/OpenWAM.git
cd OpenWAM
```

Create the local configuration and writable directories:

```bash
# Host terminal — checkout root; keep an existing .env
test -f .env || cp docker/.env.example .env
mkdir -p .cache/docker outputs
id -u
id -g
```

Edit `.env` before running containers:

| Setting | What to enter |
| --- | --- |
| `OPENWAM_UID`, `OPENWAM_GID` | The numbers printed by `id -u` and `id -g`; create cache/output directories as this user |
| `OPENWAM_IMAGE` | Keep `openwam:cu128` for a first local build, or choose your own image tag |
| `OPENWAM_GPU_ID` | One available host GPU index for serving |
| `OPENWAM_TRAIN_GPUS` | Available GPU indices for training and GPU checks, e.g. `1,3`; choose these independently of the serving index |

Keep the default cache/output paths initially. Absolute host paths are supported
when using another disk; create those directories before starting a container.
On a GPU host, `nvidia-smi` lists GPUs and their memory usage.

Build and validate the image on the connected host:

```bash
# Connected host terminal — checkout root
make docker-build
make docker-check
make docker-integration-check PYTHON=python3
```

The first build downloads large CUDA/PyTorch dependencies. Checks should finish
with no dependency errors, passing CPU tests and an `OK` integration-test result.
They require neither a GPU nor model weights. Continue with your chosen task.

## Serve a checkpoint

This example runs the released RoboTwin end-effector policy. Use a GPU host
with the image built above or imported through [offline delivery](#offline-delivery).
It does not require a simulator for the first inference check.

### 1. Prepare the weights

If you already have a complete checkpoint, skip the download and set
`OPENWAM_CHECKPOINT_DIR` in `.env` to its absolute **host** directory. It must
contain `config.yaml`, `checkpoint_step_*.safetensors`, the tokenizer and
normalization assets expected by that config. Serving mounts it read-only.

Otherwise, download the example using the image's Python environment:

```bash
# Connected host terminal — checkout root, after the image/config setup
# No GPU is required for downloading.
docker compose -f compose.yaml -f compose.dev.yaml run --rm \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 dev \
  python scripts/download_assets/download_openwam_checkpoints.py \
  --family alpha --name OpenWAM-Alpha-Sim-RoboTwin-Full --yes
```

The downloader prints `Done` and the saved path. The default destination is
`assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full`, matching
`OPENWAM_CHECKPOINT_DIR` in the template. If downloading on a different machine,
copy that entire directory to the GPU host and update its `.env` path.

### 2. Start the server

```bash
# GPU host terminal — checkout root or configured offline bundle directory
docker compose up -d serve
docker compose ps serve
```

Startup can take several minutes while the model loads. Inspect recent logs:

```bash
# GPU host terminal — same directory
docker compose logs --tail=50 serve
```

For continuous logs, run `docker compose logs -f serve` separately. Press
**Ctrl+C** to stop following logs; the detached server keeps running.
The health status should eventually change from `starting` to `healthy`.

### 3. Check readiness and run one prediction

```bash
# GPU host terminal — same directory; the helper executes inside the server
docker compose exec -T serve python docker/healthcheck.py
```

A successful health check exits with code 0 and no output. It only pings the
protocol; it does not test model execution. Once it succeeds, run:

```bash
# GPU host terminal — same directory
docker compose exec -T serve python scripts/inference_test/inference_single_test.py \
  --test --state-dim 20
```

Expect successful Ping, Predict and Reset steps, `action dim=20`, then
`Smoke test passed.` The first prediction can be slow while compilation warms
up. This uses random test observations to check execution, not benchmark quality.
For a different checkpoint, use its raw state dimension instead of 20.

The default endpoint is `ws://127.0.0.1:8848` on the GPU host. For remote clients
or a different port, see [networking](#networking-and-inference-options).

### 4. Stop when finished

```bash
# GPU host terminal — same directory; run only when you want to stop services
docker compose down
```

Outputs, cache and downloaded weights remain on the host.

## Train

First complete [checkout preparation](#prepare-a-checkout). Choose enough free
GPUs in `OPENWAM_TRAIN_GPUS` for the model: the single-GPU default selects a
device, not a recommended training memory budget. The project recommends
8 × 80 GB GPUs for the full Wan2.2-5B training setup. A smaller debug batch still
needs space for model, gradients and optimizer states; see the resource notes
in [troubleshooting](#troubleshooting).

For a LIBERO debug run, prepare the dataset and Wan backbone. Skip either
command if those assets are already complete locally:

```bash
# Connected host terminal — checkout root; downloads need no GPU
docker compose -f compose.yaml -f compose.dev.yaml run --rm \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 dev \
  python scripts/download_assets/download_benchmark_data.py --name LIBERO --yes

docker compose -f compose.yaml -f compose.dev.yaml run --rm \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 dev \
  python scripts/download_assets/download_video_backbone.py \
  --name Wan2.2-TI2V-5B --source huggingface --yes
```

These write to the checkout's `assets/` and update source YAML files. Keep
`OPENWAM_ASSETS_DIR=./assets` for that tree, or set it to the absolute host path
where you copied the assets. Start training with explicit **container** paths:

```bash
# GPU host terminal — checkout root or configured offline bundle directory
docker compose run --rm train train \
  dataloader=libero training.debug=true training.batch_size=1 \
  dataloader.dataset_dir=/opt/openwam/assets/benchmark_data/libero \
  model.video_backbone.model_path=/opt/openwam/assets/video_backbone_ckpt/Wan2.2-TI2V-5B
```

The first `train` names the Compose service; the second selects the image's
training command. The debug run finishes at step 20 and uses a constant learning
rate. Relative training outputs persist under `OPENWAM_OUTPUT_DIR` on the host.

For fine-tuning, add `training.finetune_ckpt_path` pointing to a complete
checkpoint inside the mounted asset tree. For recovery, set
`training.save_full_states_for_resume=true` when saving, then pass
`training.resume_ckpt_path` to resume from the completed state; read
[checkpoint lifecycle](#checkpoint-lifecycle) before testing an interrupted run.

## Develop from a checkout

After [checkout preparation](#prepare-a-checkout), enter a CPU development shell:

```bash
# Host terminal — checkout root
docker compose -f compose.yaml -f compose.dev.yaml run --rm dev bash
```

```bash
# Inside the container — /opt/openwam
make all
# Edit the mounted source with your host editor, then rerun tests here.
exit
```

Source edits are visible immediately. Restart a running policy server to load
changed Python code. HOME, outputs and caches persist across container recreation.
Configure Git identity inside the container if needed; host Git settings are
not copied automatically. For GPU development, enter `run --rm train bash`
with the same two Compose files and the training GPU/asset settings above.

To keep the source overlay active, set
`COMPOSE_FILE=compose.yaml:compose.dev.yaml` in `.env`. With host networking use
`COMPOSE_FILE=compose.yaml:compose.host.yaml:compose.dev.yaml`. Add
`compose.worktree.yaml` as described in [linked worktrees](#linked-git-worktrees)
when using `git worktree`.

For dependency changes, edit `pyproject.toml` in the checkout. **Exit the
container**, then regenerate the lock and rebuild from the host:

```bash
# Connected host terminal — checkout root
# Use the existing image's uv to regenerate the lock in the writable checkout.
docker compose -f compose.yaml -f compose.dev.yaml run --rm dev bash docker/lock.sh
make docker-build docker-check
```

Recreate the development container to use the new image. Source-only edits do
not require rebuilding; rebuild before exporting a release so edits are included.

## Offline delivery

### On the connected build machine

Complete [checkout preparation](#prepare-a-checkout) and download the weights
or training assets you need. GPU validation runs on the destination GPU host.
Choose an image tag in `.env` before building; see the maintainer reference
for public release revision requirements.

Before transferring, replace `gpu-host` with your SSH destination and the
example paths with your storage locations. Create those destination directories
on the GPU host first. The commands below transfer the serving checkpoint;
for training, transfer the required asset tree instead.

```bash
# Connected build host terminal — checkout root
make docker-export PYTHON=python3
rsync -a --partial dist/openwam-offline/ gpu-host:/path/to/openwam-offline/
rsync -a --partial assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full/ \
  gpu-host:/path/to/checkpoints/OpenWAM-Alpha-Sim-RoboTwin-Full/
```

Each export requires a new destination; set `DOCKER_BUNDLE=dist/my-new-bundle`
for another export. The image archive excludes datasets, weights and run outputs.
If `OPENWAM_IMAGE` names a repository without a tag, such as `openwam`, export
selects `openwam:latest` and excludes other tags in that repository.

### On the offline GPU server

```bash
# Offline GPU host terminal — received bundle directory
cd /path/to/openwam-offline
python3 docker/offline.py load .
test -f .env || cp .env.example .env
mkdir -p .cache/docker outputs
id -u
id -g
```

Import should finish with `Loaded <image>`. Edit `.env`: use this server's
UID/GID, free GPU indices, and the absolute checkpoint path you transferred.
Keep the image tag supplied by the bundle; if retaining an old `.env`, update
its `OPENWAM_IMAGE` to the newly loaded tag in `.env.example`.
Then follow [start the server](#2-start-the-server) and the readiness/prediction
checks above. No repository checkout or host Python packages are needed here.

The bundle uses `docker.md` as the name of this self-contained guide. Defaults
prevent image pulls and keep Hugging Face and W&B offline. Complete local assets
are required. `python3 docker/offline.py verify .` checks the bundle without
loading it; use the importer shipped with the bundle.

## Networking and inference options

**GPU host terminal — checkout root or offline bundle directory.** In `.env`,
`OPENWAM_BIND_HOST` defaults to `127.0.0.1` and `OPENWAM_PORT` to `8848`.
Run `docker compose up -d serve` after changing settings to recreate the service.

| Network | Host endpoint | Server / health endpoint inside the container |
| --- | --- | --- |
| Default bridge | `OPENWAM_BIND_HOST:OPENWAM_PORT` | Port 8848; health uses `127.0.0.1:8848` |
| Optional host network | `OPENWAM_BIND_HOST:OPENWAM_PORT` | The same configured host and port |

For host networking on Linux, set
`COMPOSE_FILE=compose.yaml:compose.host.yaml` in `.env`. It removes published
port mappings. The built-in health/inference helpers follow the selected
endpoint automatically. A separate bridge-network client container uses
`ws://serve:8848`; Compose service-name DNS does not reach a host-networked server.
Remote clients need an SSH tunnel or a reachable host interface. For example:

```bash
# Client machine terminal — forward local port 8848 to the GPU host's loopback port
ssh -N -L 8848:127.0.0.1:8848 gpu-host
```

Use the configured remote port if it differs. Keep the tunnel running while
connecting the client to `ws://127.0.0.1:8848`.

### Change inference options

Compose replaces the entire command when overriding it. To change denoising
steps, save **one** of the following variants as `compose.inference.yaml` beside
`compose.yaml`, choosing the variant matching your network mode.

**Bridge network — `compose.inference.yaml`:**

```yaml
services:
  serve:
    command:
      - serve
      - --ckpt-dir
      - /checkpoint
      - --device
      - cuda:0
      - --host
      - "0.0.0.0"
      - --port
      - "8848"
      - --denoise-steps
      - "5"
```

Set `COMPOSE_FILE=compose.yaml:compose.inference.yaml` in `.env`.
The internal port remains 8848; `.env` still controls the published host port
and interface.

**Host network — `compose.inference.yaml`:**

```yaml
services:
  serve:
    command:
      - serve
      - --ckpt-dir
      - /checkpoint
      - --device
      - cuda:0
      - --host
      - "${OPENWAM_BIND_HOST:-127.0.0.1}"
      - --port
      - "${OPENWAM_PORT:-8848}"
      - --denoise-steps
      - "5"
```

Set `COMPOSE_FILE=compose.yaml:compose.host.yaml:compose.inference.yaml` in `.env`.
Compose reads these values from `.env`, so binding and health checks agree even
with a custom port. If also using development/worktree overlays, retain those
files in the list and append `compose.inference.yaml` last.

Apply from the GPU host terminal in that same directory:

```bash
# GPU host terminal — same checkout or bundle directory
docker compose config --quiet
docker compose up -d serve
```

Wait for model startup, then follow the
[readiness and prediction checks](#3-check-readiness-and-run-one-prediction).
See [Compose merge rules](https://docs.docker.com/compose/how-tos/multiple-compose-files/merge/)
for additional overrides.

## Troubleshooting

| Symptom | What to check next |
| --- | --- |
| Cannot connect to Docker daemon | On the host, check `docker context show`, `docker info` and whether the daemon is running. If only `sudo docker info` works, resolve user access to that same daemon. |
| Bind source path does not exist | Create the host directories selected in `.env`; confirm checkpoint/asset paths are host paths. |
| Cache or HOME is not writable | Match UID/GID to directory ownership. Use separate cache directories for different host users. |
| Image is missing | Run the build on a connected host or load the offline bundle. Check `docker compose config --images serve`; exported variables override `.env`. |
| GPU unavailable or compiler errors | Check driver/Toolkit setup, then run the [GPU check](#gpu-validation-without-weights); `nvidia-smi` alone is insufficient. |
| Server remains starting/unhealthy | Inspect `docker compose logs --tail=100 serve`; verify complete local weights and the network/port settings. Loading and the first compiled prediction can be slow. |
| Training runs out of memory | Reduce the training workload or use an appropriate distributed/offload configuration. A debug batch of 1 does not remove optimizer-state costs. |
| Multi-GPU startup hangs on the tested H100 setup | See the NVLS workaround in the GPU reference below; test it on your selected GPUs. |

The tested full RoboTwin model had about 6.02B trainable parameters. Two 80 GB
GPUs ran out of memory at the first Adam update; four passed that specific
configuration. Its resumable state used about 102.3 GiB plus 23.1 GiB of deployment
weights. Allow space for old and new checkpoint generations before pruning.
These measurements are not minimum requirements for every model or configuration.

## Maintainer reference

The details below are optional for the first run. They remain in this guide so
the offline copy has the same reference information.

<details>
<summary>Image selection, package indexes and CI</summary>

The default image is `openwam:cu128`. **`OPENWAM_IMAGE` is the shared image
setting** for Make and Compose: set it in `.env` or export it in your shell.
`make docker-image` prints the resolved value. For example, in the same shell:

```bash
# Connected host terminal — checkout root
export OPENWAM_IMAGE=openwam:my-revision-cu128
make docker-build docker-check
make docker-image
```

An exported value overrides `.env`. A Make command-line assignment applies only
to that Make invocation; use `.env` or `export` when subsequent Compose commands
must select the same image. The old `DOCKER_IMAGE` variable is rejected instead
of silently selecting a different image.

Make resolves the `serve` image from the effective Compose configuration,
including the automatic `compose.override.yaml` and files selected through
`COMPOSE_FILE` in `.env` or the environment. An explicit `image:` in an overlay
therefore also applies to Make's build, checks and export. Use the same
`COMPOSE_FILE` for subsequent commands; Make also accepts it as a command-line
assignment for that invocation. Compose's normal merge and precedence rules
apply. If overriding service images individually, keep `serve`, `train`, `dev`
and `gpu-check` on the intended image; Make's image-based checks target `serve`.

The CUDA base is pinned by digest and `docker/requirements-cu128.txt` pins all
Python dependencies. Build-time indexes can be selected through Make, for example
`make docker-build DOCKER_BUILD_ARGS='--build-arg PIP_INDEX_URL=https://pypi.org/simple'`.
`--build-arg TORCH_INDEX_URL=...` selects the torch wheel index. No package
installation occurs at startup.

If a public mirror has not synchronized every locked version, add
`--build-arg PIP_FALLBACK_INDEX_URL=https://pypi.org/simple`. This applies both
to bootstrap tools (`pip`, `setuptools`, `wheel`, `packaging`, `ninja`, `uv`) and
to the subsequent locked Python dependencies. Bootstrap pip searches both
indexes without source priority; the uv phase tries the selected mirror first.
All versions remain constrained by the lock. Use only trusted indexes.
The three CUDA torch wheels still use the separate `TORCH_INDEX_URL`.

To deliberately upgrade locked dependencies, use the host-side container
command in [development](#develop-from-a-checkout), adding `--upgrade` after
`docker/lock.sh`. Review the diff, then rebuild and repeat CPU/GPU checks. `docker/constraints-cu128.txt` holds the selected core versions.

The build context excludes downloaded weights, datasets, local environments,
credentials and run outputs. The runtime retains `configs/` beside `openwam/`
because the existing deployment and Hydra entrypoints resolve files there.

Docker CI builds and loads this actual CUDA image on every pull request, main
or `docker/**` branch push, and `v*` tag. It runs dependency checks, container CPU tests and Docker
integration tests (image selection, source/output mounts and an offline
round-trip). It uses the tested commit as the image revision. GPU/model checks
still need to run on a GPU host; a green CPU build does not establish that compatibility.

</details>

<details>
<summary>Multiple deployments and project names</summary>

Compose uses the directory basename as its project name. Different directory
names therefore have independent containers and networks. If two checkouts or
offline bundles have the same basename (for example, both are named `OpenWAM`),
set a different `COMPOSE_PROJECT_NAME` in each `.env`, such as
`openwam-stable` and `openwam-experiment`. Make and Compose both honor it; `-p`
overrides it for an individual Compose invocation. Select different serving
ports and available GPUs for concurrently running deployments as well.

When upgrading a deployment created with the former fixed project name, set
`COMPOSE_PROJECT_NAME=openwam` in that deployment's `.env` to keep managing its
existing containers. Choose independent names for additional deployments.

</details>

<details>
<summary>Health checks and checkpoint lifecycle</summary>

The health check sends only an application-level `ping`, never observations
or resets. Model loading and lazy compilation can be slow; the check allows a
10-minute startup period and retries to tolerate synchronous inference blocking
the WebSocket loop. A healthy ping confirms protocol availability; the first
successful observation confirms model execution. Health status does not
automatically restart the process. `docker compose down` stops it explicitly.

### Checkpoint lifecycle

Full resume requires a completed `accel_state_step_*/trainer_state.json`, written
after every rank finishes saving optimizer/model/RNG state. Stop the container
only after that marker appears when testing recovery. A normally completed run
removes resumable states and retains deployment weights; use
`training.finetune_ckpt_path` to start a new run from those weights. The debug
mode finishes at step 20, saves at step 10, and uses a constant learning rate.

Training assets are writable because some readers compute normalization stats
on first use. Prepare those stats before using read-only dataset mounts. The
download scripts also update source YAML files; run asset preparation in a
writable source checkout, then pass explicit container paths when training.
The image's source/config tree is not writable by the runtime user.

For multi-node training, use one container per node with a routable rendezvous
address and the launcher's `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and
`MASTER_PORT`. Network/RDMA topology remains site-specific; the supplied Compose
profile is for single-node training.

</details>

<details>
<summary>Linked worktrees and runtime users</summary>

### Linked Git worktrees

From a source checkout, add `compose.worktree.yaml` after the development
overlay when using `git worktree`. Choose a workspace directory containing the
main repository, its Git metadata and the linked worktrees, for example
`/home/me/openwam-workspace/main` and `/home/me/openwam-workspace/feature`:

```bash
# Run on the host from the feature worktree; complete checkout preparation first.
export OPENWAM_SOURCE_DIR="$(git rev-parse --show-toplevel)"
export OPENWAM_WORKSPACE_DIR=/home/me/openwam-workspace
export COMPOSE_FILE=compose.yaml:compose.dev.yaml:compose.worktree.yaml
docker compose run --rm dev bash
```

Both paths must be absolute. The selected workspace is mounted writable at the
same path inside the container, preserving Git's links in both directions.
The shell starts at the worktree's host path; `/opt/openwam` still aliases the
edited source for the existing entrypoints and Python installation. Outputs
and training assets use the same mounts at both paths. This supports Git in
`run` and `exec`, including edits and commits visible immediately on the host,
without rewriting `.git` files or setting a process-wide `GIT_DIR`.

Select a workspace containing only the repositories you intend to expose to
the development container. Repositories, separate Git directories or symlink
targets outside it need their own mounts preserving the host paths. These
checkout-only settings can also be saved in `.env`; with host networking use
`COMPOSE_FILE=compose.yaml:compose.host.yaml:compose.dev.yaml:compose.worktree.yaml`.
The offline bundle does not contain this development-only overlay; use it
from a source checkout on the development machine.

Inside the container, run `git status`, `git add` and `git commit` from the worktree directory.

### Runtime user

The container keeps the selected numeric UID/GID and presents it as the logical
user/group `openwam`, including for IDs absent from the image's `/etc/passwd`.
`HOME=/cache/home` is writable and persists with the cache mount, so global Git
configuration and shell settings survive container recreation. Both `run` and
`exec` support username lookup (`whoami`, Python `pwd`/`getpass`). This uses
`libnss-wrapper` and private per-container account files without modifying system
accounts or starting as root. Use separate cache directories for different host
users, and configure your Git identity inside the development container when
needed; host Git configuration is not copied automatically.

</details>

<details>
<summary>GPU validation and CUDA compatibility</summary>

### GPU validation without weights

On the GPU host, from the checkout root or offline bundle directory, set
`OPENWAM_IMAGE` and `OPENWAM_TRAIN_GPUS` in `.env`.
The check requires writable cache/output directories, but no checkpoint or dataset.
Run it with the same Compose configuration used for deployment:

```bash
# GPU host terminal — checkout root or offline bundle directory
docker compose run --rm -T gpu-check
```

This tests CUDA BF16, compiled forward/backward execution and the DeepSpeed
FusedAdam extension on every selected GPU. With multiple selected GPUs, it also
tests NCCL all-reduce. Temporary overrides work the same as for training:

```bash
# GPU host terminal — same directory; choose free GPUs
OPENWAM_TRAIN_GPUS=1,3 docker compose run --rm -T gpu-check
```

From a source checkout, `make docker-gpu-check` is an equivalent wrapper. The
check uses the configured image and `/cache` mount with networking disabled. Keep
caches separate across image revisions and GPU/driver changes. Follow these
checks with real checkpoint inference and a debug training run on prepared data.

On the tested H100 host (driver 550.54.14, NCCL 2.26.2), a four-GPU communicator
hung during initialization while the two-GPU check passed. The same four-GPU
all-reduce succeeded with NVLink SHARP disabled:

```bash
# GPU host terminal — same directory, with prepared assets at the paths below
docker compose run --rm -T -e NCCL_NVLS_ENABLE=0 gpu-check
docker compose run --rm -e NCCL_NVLS_ENABLE=0 train train \
  dataloader=robotwin training.debug=true training.batch_size=1 \
  training.finetune_ckpt_path=/opt/openwam/assets/checkpoint \
  dataloader.dataset_dir=/opt/openwam/assets/training-data/robotwin
```

Apply the same setting to initial training and resume if this issue reproduces
on your host. It disables NVLS collective offload, with a possible performance
cost; it does not disable GPU peer-to-peer communication. This is a host-specific
workaround, not a default for every H100 installation. See
[NVIDIA's NCCL NVLS setting](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/env.html#nccl-nvls-enable).

The image uses Ubuntu 24.04 userspace; the host does not need the same Ubuntu or
Python version. The host GPU driver is shared with containers. In particular,
older 535/550 data-center drivers need validation with the image's CUDA 12.8
libraries and `torch.compile`; successful `nvidia-smi` alone is not sufficient.
NVIDIA Container Toolkit normally handles CUDA forward compatibility. If it
does not, test `OPENWAM_CUDA_COMPAT=1` to select `/usr/local/cuda/compat`, then run
the GPU check in this section. This does not install or replace the host kernel driver.
See [NVIDIA's compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

</details>

<details>
<summary>Offline integrity and optional Cosmos builds</summary>

All configuration and the importer are extracted from the selected image by
immutable image ID, without starting a container process. The exporting checkout
cannot substitute newer files into an older release. The manifest binds the
source revision, image layers/runtime configuration and checksums of the delivered
files. `.env.example` is rendered to select that image tag. Do not move the tag
during export; the exporter rejects a tag that changes before completion.

`make docker-build` records the full Git commit and a `-dirty` suffix for local
changes. Use a clean, committed checkout and a revision tag for public releases;
the image identity also distinguishes local builds sharing a dirty revision.
Direct Docker builds must supply `--build-arg VCS_REF=<commit>` before export.
Images predating bundle schema 2 must be rebuilt. Existing schema 1 bundles can
still be loaded using their own bundled importer.

### Optional Cosmos-Predict2.5 image

```bash
# Connected host terminal — checkout root
git submodule update --init third_party/cosmos-predict2.5
docker build --platform linux/amd64 --target cosmos --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -f docker/Dockerfile -t openwam:cosmos-cu128 .
```

This separate target reuses `scripts/install_cosmos_predict25.sh`, including
its upstream dependency restoration and Transformer Engine compilation. It
requires extra build time and its own GPU/checkpoint validation; the standard
image does not include these optional packages. Upstream Cosmos metadata pins
conflict with the restored OpenWAM transformer versions as documented in that
installer, so the standard image's clean `pip check` does not apply to Cosmos.

</details>
