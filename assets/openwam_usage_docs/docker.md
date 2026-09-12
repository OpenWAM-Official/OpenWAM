# Docker

Run OpenWAM serving, training and source development with Python 3.12,
PyTorch 2.7.1 and CUDA 12.8 in one image. The host does not need the project's
Python environment. Benchmark simulators keep their own environments and
connect to the policy over WebSocket.

[Setup](#setup) · [Serve](#serve) · [Train](#train) · [Develop](#develop) ·
[Offline delivery](#offline-delivery) · [Troubleshooting](#troubleshooting) ·
[Maintaining the image](#maintaining-the-image)

Unless marked **Inside the container**, run commands in a **host terminal at
the checkout root**. For offline deployment, use the received bundle root.
Docker files live in `docker/`; the root `compose.yaml` preserves the normal
`docker compose` entrypoint. GitHub's workflow stays in `.github/workflows/`.

## Setup

Requirements:

- Linux x86-64, [Docker Engine](https://docs.docker.com/engine/install/) and
  Docker Compose 2.30+. `docker version` must report both Client and Server.
- For serving/training: NVIDIA GPUs, a host driver and
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- For building/developing: Git, Make and internet access. Host-side checks and
  export need Python 3.8+. Building and CPU tests need no GPU.
- Enough disk for the CUDA image, caches, weights and outputs. The example
  policy is about 25 GB; full training states can exceed 100 GiB.

On a connected host, clone the repository, or use your existing checkout:

```bash
git clone https://github.com/OpenWAM-Official/OpenWAM.git
cd OpenWAM
test -f .env || cp docker/.env.example .env
mkdir -p .cache/docker outputs
id -u
id -g
```

Edit `.env`:

| Setting | Value |
| --- | --- |
| `OPENWAM_UID`, `OPENWAM_GID` | Your `id -u` / `id -g`; create writable directories as this user |
| `OPENWAM_IMAGE` | `openwam:cu128` for a first build, or your own revision tag |
| `OPENWAM_GPU_ID` | One free host GPU index for serving |
| `OPENWAM_TRAIN_GPUS` | Free host GPU indices for training/checks, e.g. `1,3` |
| `OPENWAM_CHECKPOINT_DIR` | Host directory of a complete serving checkpoint |
| `OPENWAM_ASSETS_DIR` | Host asset tree for training; default `./assets` |
| `OPENWAM_CACHE_DIR`, `OPENWAM_OUTPUT_DIR` | Existing writable host directories; defaults work from the checkout root |

Paths are relative to the root `compose.yaml`; absolute host paths also work.
Use `nvidia-smi` to check available GPUs. A single selected GPU is a default
selection, not a training memory recommendation.

**Upgrading an existing checkout:** keep your root `.env`. In `COMPOSE_FILE`
and saved commands, replace `compose.dev.yaml`, `compose.host.yaml` and
`compose.worktree.yaml` with their `docker/` paths. Leave the first
`compose.yaml` unchanged. Host asset/cache/output paths stay the same.

```bash
make docker-build
make docker-check
make docker-integration-check PYTHON=python3
```

The initial build downloads large dependencies. Checks should report no pip
conflicts, passing CPU tests and an `OK` integration result. On the GPU host,
validate the selected GPUs/compiler/optimizer before loading model weights:

```bash
docker compose run --rm -T gpu-check
```

This checks BF16, compiled forward/backward, DeepSpeed FusedAdam and multi-GPU
NCCL all-reduce with networking disabled. It requires the cache/output mounts,
but no weights or dataset. Success requires the entire command to exit with
code 0: worker computation messages alone do not rule out a shutdown failure.
Follow it with real inference or training.

## Serve

Use a GPU host with the image built above or loaded from an offline bundle.
A complete checkpoint contains `config.yaml`, `checkpoint_step_*.safetensors`,
the tokenizer and normalization assets expected by its config. Set its **host**
path in `OPENWAM_CHECKPOINT_DIR`; serving mounts it read-only.

To download the example RoboTwin policy, run this from a **connected checkout**:

```bash
docker compose -f compose.yaml -f docker/compose.dev.yaml run --rm \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 dev \
  python scripts/download_assets/download_openwam_checkpoints.py \
  --family alpha --name OpenWAM-Alpha-Sim-RoboTwin-Full --yes
```

The default download path matches `.env.example`. On the GPU host:

```bash
docker compose up -d serve
docker compose ps serve
docker compose logs --tail=50 serve
```

Allow several minutes for loading. Once health changes to `healthy`, check
protocol readiness and run one prediction:

```bash
docker compose exec -T serve python /opt/openwam/docker/healthcheck.py
docker compose exec -T serve python scripts/inference_test/inference_single_test.py \
  --test --state-dim 20
```

The health command exits silently with code 0; it only sends a ping. The
prediction test should report `Smoke test passed.` and 20 action values.
The first prediction can be slow while compilation warms up. Random test
observations validate execution, not policy quality. Other checkpoints may
require a different raw state dimension.

The endpoint defaults to `ws://127.0.0.1:8848` on the GPU host. Stop services
with `docker compose down`; cache, outputs and weights remain on the host.

### Networking and inference options

Set `OPENWAM_BIND_HOST` and `OPENWAM_PORT` in `.env`, then run
`docker compose up -d serve` to apply changes. With the default bridge network,
the container still listens on port 8848; other containers on that network can
connect to `ws://serve:8848`.

For Linux host networking, set this in `.env`:

```dotenv
COMPOSE_FILE=compose.yaml:docker/compose.host.yaml
```

The server and health/inference helpers then use the configured host address
and port directly. Service-name DNS does not reach a host-networked server.
For remote access to the default loopback endpoint, run on the **client host**:

```bash
ssh -N -L 8848:127.0.0.1:8848 gpu-host
```

Keep the tunnel running and connect to `ws://127.0.0.1:8848`. Replace the SSH
destination and remote port to match your server.

To change inference arguments, create `compose.inference.yaml` beside the root
`compose.yaml`. Compose replaces the entire command, so retain required options:

```yaml
services:
  serve:
    command: [serve, --ckpt-dir, /checkpoint, --device, "cuda:0", --host, "0.0.0.0", --port, "8848", --denoise-steps, "5"]
```

Set `COMPOSE_FILE=compose.yaml:compose.inference.yaml` in `.env`. For host
networking, use `"${OPENWAM_BIND_HOST:-127.0.0.1}"` and
`"${OPENWAM_PORT:-8848}"` for the command's host/port and set
`COMPOSE_FILE=compose.yaml:docker/compose.host.yaml:compose.inference.yaml`.
Retain any development overlays and append your inference override last.
Run `docker compose config --quiet`, recreate the service, then repeat the
readiness/prediction checks.

## Train

Prepare the dataset and backbone from a **connected checkout**. Skip downloads
for assets already present:

```bash
docker compose -f compose.yaml -f docker/compose.dev.yaml run --rm \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 dev \
  python scripts/download_assets/download_benchmark_data.py --name LIBERO --yes

docker compose -f compose.yaml -f docker/compose.dev.yaml run --rm \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 dev \
  python scripts/download_assets/download_video_backbone.py \
  --name Wan2.2-TI2V-5B --source huggingface --yes
```

Downloads write `assets/` and update source YAML files. Set `OPENWAM_ASSETS_DIR`
to that host tree (default `./assets`), and pass explicit **container** paths:

```bash
docker compose run --rm train train \
  dataloader=libero training.debug=true training.num_epochs=null training.batch_size=1 \
  dataloader.dataset_dir=/opt/openwam/assets/benchmark_data/libero \
  model.video_backbone.model_path=/opt/openwam/assets/video_backbone_ckpt/Wan2.2-TI2V-5B
```

The first `train` selects the Compose service; the second selects its training
command. Debug mode ends at step 20 with a constant learning rate. Relative
outputs persist under `OPENWAM_OUTPUT_DIR`. Assets are writable because readers
may compute normalization statistics; the image's source/config tree is not.

The project recommends 8 × 80 GB GPUs for full Wan2.2-5B training. In one tested
RoboTwin configuration, two H100 80 GB GPUs ran out of memory at the first Adam
update, while four passed. Batch size 1 still needs optimizer-state memory.
See [validation scope and measured storage](docker-validation.md).

For fine-tuning, set `training.finetune_ckpt_path` to a checkpoint directory
under `/opt/openwam/assets` or `/outputs`. For full recovery, enable
`training.save_full_states_for_resume=true` during the initial run, then set
`training.resume_ckpt_path` to that run's **output directory**, for example
`/outputs/openwam_checkpoints/2026-09-10_12-00-00`. It must contain `config.yaml`
and completed `accel_state_step_*` subdirectories; the trainer selects the latest
usable state itself. Do not point this option directly at a state subdirectory.
Use container paths, retain the dataset mount and compatible distributed settings,
and set `training.finetune_ckpt_path=null` when resuming: the two modes are exclusive.

A state is complete only after `accel_state_step_*/trainer_state.json` appears.
Stop after this marker when testing recovery; stopping a container does not itself
save a checkpoint.
Normal completion removes resumable states and retains deployment weights.
Allow storage for old and new checkpoint generations before pruning.

Compose supplies single-node training. Multi-node/RDMA needs site-specific
networking and the launcher's `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`.

## Develop

Use a source checkout. Enter a CPU shell with the source overlay:

```bash
docker compose -f compose.yaml -f docker/compose.dev.yaml run --rm dev bash
```

**Inside the container**, at `/opt/openwam`:

```bash
make all
exit
```

For a GPU shell, use `run --rm train bash` with the same two Compose files.
Inside that shell, launch training with `bash docker/entrypoint.sh train`,
followed by the Hydra overrides from [Train](#train). This entrypoint sets
`NPROC_PER_NODE` from PyTorch's visible GPU count. Calling `scripts/train.sh`
directly without an explicit `NPROC_PER_NODE` can count all GPUs reported by
`nvidia-smi`, including those hidden by `OPENWAM_TRAIN_GPUS`/`CUDA_VISIBLE_DEVICES`.

To enable source mounts for subsequent commands, put
`COMPOSE_FILE=compose.yaml:docker/compose.dev.yaml` in `.env`. Include
`docker/compose.host.yaml` after the base file if using host networking.

The package is installed with `pip install -e .`: its small editable wheel
points to `/opt/openwam/openwam`, and the overlay mounts your checkout at
`/opt/openwam`. Source edits are visible to new Python processes immediately;
restart long-running servers to reload code. Source-only edits need no rebuild.
Without this overlay, containers use the source copied when the image was built.

Outputs are mounted at both `/outputs` and `/opt/openwam/outputs`, so mounting
the checkout does not hide saved results. `HOME=/cache/home` persists, and the
selected numeric UID/GID resolves to the logical user `openwam`, including under
`docker compose exec`. Configure Git identity inside the container if needed;
host Git settings are not copied. Use separate caches for different host users.
The container has no Docker CLI: build/export commands run on the host.

### Linked Git worktrees

Add the worktree overlay after the source overlay. From the **feature checkout
on the host**, select an absolute workspace path containing both the main
repository (including Git metadata) and the linked worktree:

```bash
export OPENWAM_SOURCE_DIR="$(git rev-parse --show-toplevel)"
export OPENWAM_WORKSPACE_DIR=/home/me/openwam-workspace
export COMPOSE_FILE=compose.yaml:docker/compose.dev.yaml:docker/compose.worktree.yaml
docker compose run --rm dev bash
```

The selected workspace is mounted writable at the same host path, preserving
Git links. The shell starts in the worktree; `/opt/openwam` aliases the same
source. Outputs/assets remain consistent at both paths. Only expose the
workspace you need; external Git directories or symlink targets need their
own mounts. These settings may also go in `.env`. The worktree overlay is
checkout-only and is not included in offline deployment bundles.

## Offline delivery

Install Docker, Compose, the driver, Toolkit and Python 3.8+ on the destination
before disconnecting it. No project Python packages or checkout are required
there. Build and download assets on the connected host first.

**Connected host, checkout root:**

```bash
make docker-export PYTHON=python3
rsync -a --partial dist/openwam-offline/ gpu-host:/path/to/openwam-offline/
rsync -a --partial assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-RoboTwin-Full/ \
  gpu-host:/path/to/checkpoints/OpenWAM-Alpha-Sim-RoboTwin-Full/
```

Replace the SSH destination/paths and create remote directories first. For
training, transfer the required asset tree instead. Weights, data and outputs
are excluded from the image archive. Each export needs a new destination;
use `DOCKER_BUNDLE=dist/another-bundle` for a subsequent export.

The bundle contains the image's standard Compose files and environment template.
Your host `.env` and custom overlays such as `compose.inference.yaml` are not
exported. Transfer any custom overlays separately, review their paths/settings
for the destination, and include them in its `COMPOSE_FILE`. Keep the bundled
standard files unchanged so integrity verification continues to work.

Deliver each new bundle to a fresh remote directory too. `verify` and `load`
reject an existing `.env`, alternative default Compose filenames and automatic
`*.override.yaml` / `*.override.yml` files (the `compose` and `docker-compose`
names). Move these files outside the bundle before importing; nothing is deleted
or overwritten for you. Create host configuration only after a successful load.
Relative data/cache/output paths now resolve from the new directory;
use absolute paths to retain existing storage. To keep managing the same running
deployment from a new directory, retain its existing `COMPOSE_PROJECT_NAME` as well.

**Offline GPU host, received bundle root:**

```bash
cd /path/to/openwam-offline
python3 docker/offline.py load . && cp docker/.env.example .env
mkdir -p .cache/docker outputs
id -u
id -g
```

After `Loaded <image>`, edit `.env` for this host's UID/GID, GPU indices and
absolute checkpoint/asset paths. Keep the bundled image tag; when reusing an
old `.env`, update `OPENWAM_IMAGE` from `docker/.env.example`. Follow [Serve](#serve)
or [Train](#train). Default settings prevent pulls and keep Hugging Face/W&B offline.

`python3 docker/offline.py verify .` checks integrity without loading the image.
Run it before creating `.env`; to recheck a configured bundle, temporarily move
`.env` and any automatic overrides outside it first. Verification covers the
delivered files and image, not your shell environment or subsequent host edits.
Before starting, review `docker compose config` and compare the selected image
with `docker/.env.example`; exported variables override `.env`. Include only
reviewed overlays in `COMPOSE_FILE` and update their paths when reusing settings.
Use the importer and instructions delivered with that bundle. Current bundles
use the checkout's paths, with runtime files in `docker/` and guides in
`assets/openwam_usage_docs/` (schema 4). The exporter and importer also support
schemas 2 and 3, preserving their original files and instructions. Schema 1
bundles require their original importer.

A bare repository reference exports only `:latest`. Digest/image-ID references
receive a portable `openwam-bundle:sha256-…` tag, recorded in `.env.example`;
`manifest.json` retains the original reference. This reusable tag also remains
on the build host. Configuration and importer are extracted from the selected
immutable image, so an older image keeps its own instructions. Checksums,
revision and image layers/runtime configuration are verified across save/load.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Cannot connect to daemon | Check host `docker context show` and `docker info`. If only `sudo docker info` works, resolve user access to that same daemon. |
| Missing bind source | Create directories selected in `.env`; paths are host paths. |
| Cache/HOME permission denied | Match UID/GID to directory ownership; keep caches separate between users. |
| Missing or wrong image | Build or import it; run `docker compose config --images serve`. Exported variables override `.env`. |
| Unhealthy server | Read `docker compose logs --tail=100 serve`; check local weights and ports. Health pings do not test model execution or automatically restart containers. |
| GPU unavailable/compile failure | Check driver/Toolkit and run `gpu-check`; `nvidia-smi` alone does not establish CUDA/compiler compatibility. |
| Training OOM | Review GPU count, model size and distributed/offload settings; debug mode does not remove optimizer-state costs. |
| Worktree Git fails | Ensure the selected workspace includes the main repository, Git metadata and worktree at their original absolute paths. |

The image disables [NCCL RAS monitoring](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ras-enable)
with `NCCL_RAS_ENABLE=0`: H100 testing reproduced a process-exit crash when RAS
and the image's NSS user/group lookup library were both enabled. Collective
communication remains enabled. Keep this default when using the supplied image;
see [the validation record](docker-validation.md) for the tested configuration.

On the historical H100 setup, four-GPU NCCL initialization hung until NVLS was
disabled. If a GPU check reproduces this, test:

```bash
docker compose run --rm -T -e NCCL_NVLS_ENABLE=0 gpu-check
```

If needed, pass the same `-e NCCL_NVLS_ENABLE=0` to both initial and resumed
`docker compose run … train train …` commands. This host-specific workaround
can affect performance; see [NVIDIA's NVLS reference](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/env.html#nccl-nvls-enable).

The host shares its NVIDIA driver with the container; it need not use Ubuntu
24.04 or the image's Python version. Toolkit normally handles CUDA forward
compatibility. If it does not, test `OPENWAM_CUDA_COMPAT=1` in `.env`, recreate
containers and rerun the GPU check. This selects `/usr/local/cuda/compat` for
both startup and subsequent `exec` processes; it does not replace the kernel
driver. Default `0` leaves selection to Toolkit. See
[NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
and the version-specific [validation record](docker-validation.md).

## Maintaining the image

The Make/build commands in this section require a source checkout on the host.
Offline deployment bundles have no Makefile or build context.

- `make docker-help` lists Docker targets. `OPENWAM_IMAGE` in `.env` or the
  shell is shared by Make and Compose. Make resolves the effective `serve`
  image, including automatic `compose.override.yaml` and `COMPOSE_FILE`.
  Keep service images aligned if overriding them individually. A Make command-line
  variable applies only to that invocation; exported variables override `.env`.
- Compose names projects after the root directory. Checkouts/bundles with the
  same basename need distinct `COMPOSE_PROJECT_NAME` values in `.env`; concurrent
  servers also need distinct ports/free GPUs. Existing deployments created with
  the former fixed name can retain `COMPOSE_PROJECT_NAME=openwam`.
- The CUDA base is pinned by digest; `requirements-cu128.txt` locks Python
  dependencies. Ubuntu packages resolve from the fixed `UBUNTU_SNAPSHOT` in
  `Dockerfile`, using the [Ubuntu snapshot service](https://snapshot.ubuntu.com/).
  The rolling NVIDIA apt source is disabled; CUDA comes from the pinned base.
  `Dockerfile.dockerignore` excludes downloaded weights/data, local environments,
  credentials and outputs while preserving source/configs together. Startup
  does not install dependencies.
- Choose trusted build indexes with `DOCKER_BUILD_ARGS='--build-arg PIP_INDEX_URL=…'`.
  Optional `PIP_FALLBACK_INDEX_URL` applies to bootstrap and locked dependencies.
  Bootstrap pip searches both indexes without priority; uv tries the primary
  first. The three CUDA torch wheels use the separate `TORCH_INDEX_URL`.
- Build public releases from a clean committed checkout with a revision tag.
  `make docker-build` records the commit plus `-dirty` for local changes;
  direct builds need `--build-arg VCS_REF=<commit>` for offline export.
- CI builds the actual CUDA image and runs pip, Ruff, CPU and Docker integration
  checks on PRs, main/`docker/**` pushes and `v*` tags. GPU/model checks need a
  GPU host; historical results are summarized in [docker-validation.md](docker-validation.md).

After editing dependencies in `pyproject.toml` or `docker/requirements.in`,
regenerate the lock from the **connected host**, review it, then rebuild:

```bash
make docker-lock
make docker-build docker-check
```

Use `make docker-lock DOCKER_LOCK_ARGS=--upgrade` for deliberate upgrades;
`docker/constraints-cu128.txt` controls core versions. Recreate containers and
repeat relevant GPU checks. Rebuild before exporting source edits, too.

Update `UBUNTU_SNAPSHOT` deliberately to receive Ubuntu fixes, then rebuild and
run `make docker-check docker-integration-check` plus relevant GPU checks before
publishing. To evaluate another snapshot without changing the default, use
`DOCKER_BUILD_ARGS='--build-arg UBUNTU_SNAPSHOT=YYYYMMDDTHHMMSSZ'` with a real UTC
timestamp. A missing/unavailable snapshot fails the build; there is no fallback
to current repositories.

Each image records its snapshot in the `io.openwam.ubuntu.snapshot` label and
its complete system package inventory (package, version, architecture) at
`/usr/local/share/openwam/system-packages.tsv`. On a host with the image loaded:

```bash
docker run --rm --pull=never --network none --entrypoint cat \
  "$(docker compose config --images serve)" \
  /usr/local/share/openwam/system-packages.tsv > system-packages.tsv
```

The inventory is carried inside offline image archives too. These pins stabilize
dependency selection; they do not promise byte-identical rebuilt image layers.
Keep the built image or offline bundle for exact redeployment, including beyond
the snapshot service's retention period.

### Optional Cosmos image

```bash
git submodule update --init third_party/cosmos-predict2.5
docker build --platform linux/amd64 --target cosmos --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -f docker/Dockerfile -t openwam:cosmos-cu128 .
```

This separate target reuses `scripts/install_cosmos_predict25.sh` and compiles
Transformer Engine. It needs its own GPU/checkpoint validation. Upstream Cosmos
metadata conflicts with restored OpenWAM transformer versions; the standard
image's clean `pip check` does not apply to this optional target.
