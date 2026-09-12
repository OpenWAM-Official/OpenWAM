# Docker validation scope

Results apply to the named revision/image and environment. Use
[docker.md](docker.md) for setup and test commands. No Cosmos or LIBERO assets
were downloaded for the 2026-09-11 checks below.

## Current checks (2026-09-11)

The standard Wan image was built from `24d88ec6712f53e6c925d8ada7a50badbfbd6811`.
Its local image ID was
`sha256:4a7059711b90df4fd0c8ef757fa675d6061b48ebe91bfcbbaf1d5081ff3952d5`.

| Check | Result and scope |
| --- | --- |
| Build and CPU checks | Actual CUDA image built; pip, serve CLI and Ruff passed. **1,990 CPU tests** passed; **12 Docker integration tests** passed separately. |
| Additional CPU tests | All **38** non-GPU tests in `test_tri_system_smoke.py`, excluded by `docker-check`, passed separately. |
| Dependency locking | `make docker-lock` resolved 118 packages and reproduced the committed file exactly in an isolated checkout. `--upgrade` resolved/wrote successfully; three changed packages were installed in a disposable container and passed pip/Ruff and 1,990 CPU tests. The repository lock stayed unchanged; no full upgraded-image build was claimed. |
| Full offline delivery | The complete 9 GB compressed CUDA bundle was exported using schema 3, transferred to the offline H100 and loaded with its bundled importer. Checksums and image layers/config matched across Docker 29.8/containerd and 28.1.1/overlay2. |
| Runtime compatibility | Real GPU compiler/backward/FusedAdam checks passed via `docker exec` before and after restart with `OPENWAM_CUDA_COMPAT=1`. |
| GPU regressions | **7** lightweight GPU-related cases passed with the RAS adjustment below: five use real CUDA tensors and two use simulated dispatch inputs. These cover attention dispatch, train/deploy consistency and RoPE migration without downloaded checkpoints. |
| Serving | Existing RoboTwin checkpoint: container-local bridge health/prediction/reset passed. Host-to-bridge access timed out on this server. The documented host-network overlay passed real ping/prediction/reset from the development machine over an SSH tunnel, including `OPENWAM_SERVER_URL`; predictions contained 20 action values. |
| Training/resume | Four H100s and four existing real RoboTwin episodes: initial training used the source-mounted development shell entrypoint; a new image-only container restored completed step-10 state, ran steps 11–20 and exited 0. Both phases explicitly used `NCCL_RAS_ENABLE=0` and `NCCL_NVLS_ENABLE=0`. |

The H100 host used driver 550.54.14, Toolkit 1.18.1 and kernel 5.4.0.
Training used 583 timesteps / 579 windows, about 6.02B trainable parameters,
BF16, ZeRO-2, batch size 1 per GPU, gradient checkpointing, CPU initialization
and no optimizer offload. Four optimizer shards and four RNG files were
verified; full state was 102.3 GiB. Final weights were 23.1 GiB and differed
from step 10; losses and gradient norms were finite. Debug mode used constant LR.

**Logging limitation:** the first container completed unsaved step 11 before
receiving the stop signal. Resume correctly restarted at 11, but the append-only
CSV retained both step-11 records. The strict no-duplicate-log assertion failed;
an independent artifact verifier confirmed recovery while retaining this
limitation. This does not establish clean, deduplicated metrics after rollback.

## RAS shutdown fix found by GPU testing

With NSS user/group lookup preloaded, both GPU workers completed the smoke
calculations, then rank 0 exited with SIGSEGV. Disabling NVLS or enabling CUDA
compatibility alone did not fix it. Disabling either NSS preload or NCCL RAS
allowed a clean exit. The image now retains NSS and defaults `NCCL_RAS_ENABLE=0`.

The locally rebuilt fix image `openwam:24d88ec-ras-fix`, ID
`sha256:ab59032106f82f8f7df914298367fc0d6e5e38025a765f869c131460e7cc034b`,
passed pip/Ruff, 1,990 CPU tests and 12 Docker integration tests. The actual
rebuilt image was then transferred to H100 over an SSH-forwarded local registry;
its filesystem layers and runtime configuration matched the local build. With
no RAS override, the default two-GPU smoke completed BF16, compiled backward,
FusedAdam and NCCL all-reduce, and the launcher exited **0**. This verifies the
image default, including process shutdown. Earlier GPU regression and training
checks used an equivalent ENV-only image or an explicit RAS setting.

## CI and remaining scope

The original revision's [PR Docker CI](https://github.com/OpenWAM-Official/OpenWAM/actions/runs/34576398602)
passed. Its [push run](https://github.com/OpenWAM-Official/OpenWAM/actions/runs/34576395292)
first failed during Ubuntu index download with `Hash Sum mismatch`, before tests;
its second attempt passed. These CI runs predate the RAS fix.

Not validated: Cosmos image/GPU execution, LIBERO download-to-training workflow,
full-dataset or long-running training, convergence/benchmark quality, changing-LR
scheduler recovery, bitwise equality to uninterrupted training, multi-node/RDMA,
or a broader GPU/driver matrix. CPU CI cannot establish these properties.

The earlier `9ef0cd9` H100 acceptance and local review-image records remain in
Git history; [its original CUDA CI](https://github.com/KraHsu/OpenWAM-Official/actions/runs/34466639632)
also passed. Two H100s ran out of memory for the historical training setup;
four-GPU initialization required the host-specific NVLS workaround.
