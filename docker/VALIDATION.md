# Docker validation scope

Historical results below apply to the named revisions/images. They are not a
claim that every subsequent commit, driver or model has passed GPU acceptance.
For setup and commands, use [README.md](README.md).

## Historical GPU acceptance — 2026-09-10

Revision: `9ef0cd98edb7899ddff1e38fbe7791005fd257ac` (standard Wan image).
Host: H100 80 GB GPUs, driver 550.54.14, NVIDIA Container Toolkit 1.18.1,
kernel 5.4.0. Image: Python 3.12, PyTorch 2.7.1+cu128, Accelerate 1.14.0,
DeepSpeed 0.18.9.

| Check | Evidence / result |
| --- | --- |
| Remote CUDA build and CPU checks | [Actions run 34466639632](https://github.com/KraHsu/OpenWAM-Official/actions/runs/34466639632): image built/loaded, pip/Ruff passed, 1,960 CPU tests and 4 Docker integration tests passed. |
| Cold build | [Actions run 34463896096](https://github.com/KraHsu/OpenWAM-Official/actions/runs/34463896096) passed before the rendezvous fix. |
| Offline transfer | Archive SHA256 and image layers/runtime config matched between Docker 29.8.0/containerd and Docker 28.1.1/overlay2. |
| GPU smoke | Two H100s: BF16, compiled forward/backward, FusedAdam and NCCL all-reduce passed without networking. |
| Serving | Original released RoboTwin checkpoint, one H100, host networking and loopback port 18848: scheduled health, helper and inference client passed; prediction returned 20 action values and reset succeeded. |
| Training/resume | Four H100s: steps 1–10, intentional container stop after completed state, new container resumed at 11 and finished step 20 with exit code 0. |

Training used four real RoboTwin simulation episodes from
`adjust_bottle/aloha-agilex_clean_50`: 583 timesteps / 579 windows. It trained
about 6.02B parameters using BF16, ZeRO-2, batch size 1 per GPU, gradient
checkpointing, CPU model initialization and no optimizer offload. Debug mode
used a constant learning rate; both phases set `NCCL_NVLS_ENABLE=0`.

The step-10 `trainer_state.json` completion marker recorded global/optimizer
step 10. The completed state included four optimizer shards and four RNG files,
totaling **102.3 GiB**. Resume produced exactly steps 11–20; the combined metrics
had no missing/duplicate steps and all loss/gradient norms were finite. Final
weights occupied **23.1 GiB** and differed from step 10. Normal completion removed
resumable states and retained deployment weights.

Limitations and operational findings:

- Two 80 GB GPUs ran out of memory at the first Adam update for this setup;
  four passed. This does not establish a universal minimum.
- Default four-GPU NCCL initialization hung; a minimal all-reduce reproduced it.
  `NCCL_NVLS_ENABLE=0` resolved it. This is a host-specific workaround.
- The intentionally stopped first container exited with code 1. Acceptance was
  based on the completed state and verified subsequent recovery, not a clean
  exit from that stop or an automatic save on shutdown.
- A resume INFO line was absent from stdout. The initial log-only harness failed
  that condition; independent checks verified actual state, steps and artifacts.
- No convergence/benchmark-quality claim, uninterrupted bitwise control run,
  changing-scheduler recovery, full-dataset training, multi-node/RDMA or Cosmos
  GPU validation was performed.

## Subsequent local regression checks

These were CPU/container workflow checks, not new H100 training runs:

| Image tag | Coverage added | Result |
| --- | --- | --- |
| `openwam:opensource-fixes` | Checkout isolation, arbitrary UID/GID and bootstrap index fallback | 1,960 CPU tests; 7 integration tests passed |
| `openwam:compose-worktree-fixes` | Effective Compose overrides and linked Git worktrees | 1,960 CPU tests; 9 integration tests passed |
| `openwam:digest-compat-fixes` | Portable digest bundles and CUDA compatibility under `exec` | 1,972 CPU tests; 11 integration tests passed |

The last image was built from the source later committed as `b27be2f` (its
revision label predates that commit and has a `-dirty` suffix). CUDA compatibility
checks covered library selection and process behavior; they did not replace a
GPU compiler/inference test. Earlier detailed reports remain in Git history;
local raw logs are not distributed as reproducible public test artifacts.

## Directory consolidation — 2026-09-10

Local image `openwam:docker-layout`, ID
`sha256:6b461eece665d7b0fb0b51e4ef29c40739bd5397078878840407b84f131a2e0f`,
was built from `b27be2f` plus the directory/bundle changes. Pip, Ruff and the
serve CLI passed; **1,976 CPU tests** and **12 Docker integration tests** passed.
Checks include root `.env`/relative mounts with relocated overlays and schema 3
offline delivery. A separate cold-load check used the original schema 2 importer
extracted from `openwam:digest-compat-fixes`, confirming old bundle paths still
work with the new exporter. Shell/workflow syntax and documentation links passed.
No new H100 training or remote CI run is claimed. This summary was added after
the validation build.

The follow-up review image `openwam:docker-review` passed **1,984 CPU tests**
and **12 integration tests**, including runs with exported custom port/asset
settings. Its ID is
`sha256:088aee706a114117d7a134072f19a81641293335e352ae5dcf6e5df337fd0e01`.
The review added malformed-manifest rejection and isolated test configuration;
it also corrected resume-directory and offline-upgrade instructions. These are
local checks, with no additional GPU acceptance run.

## Recheck a revision

From a checkout, select `OPENWAM_IMAGE` in `.env`, then:

```bash
make docker-build
make docker-check
make docker-integration-check PYTHON=python3
```

Integration tests exercise real Docker/Compose on a CPU host, including image
selection, checkout isolation, editable mounts, Git worktrees, runtime users,
CUDA library selection and offline delivery. On the destination GPU host, run
`docker compose run --rm -T gpu-check`, real checkpoint inference and a short
training/resume check on prepared data, following [README.md](README.md).
Record the tested commit/image, host driver/Toolkit, commands and results;
a successful CPU CI run alone does not establish GPU compatibility.
