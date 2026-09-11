# Docker validation scope

Results apply to the named revision or image, not every subsequent commit,
driver or model. Use [README.md](README.md) for setup and test commands.

## GPU acceptance

Tested on 2026-09-10 at revision
`9ef0cd98edb7899ddff1e38fbe7791005fd257ac`, using H100 80 GB GPUs,
driver 550.54.14, NVIDIA Container Toolkit 1.18.1 and kernel 5.4.0.
The standard Wan image used Python 3.12, PyTorch 2.7.1+cu128,
Accelerate 1.14.0 and DeepSpeed 0.18.9.

| Check | Result |
| --- | --- |
| Remote CUDA build | [Actions run 34466639632](https://github.com/KraHsu/OpenWAM-Official/actions/runs/34466639632): actual image built/loaded; pip/Ruff, 1,960 CPU tests and 4 integration tests passed. |
| Offline transfer | Archive checksum and image layers/config matched across Docker 29.8.0/containerd and Docker 28.1.1/overlay2. |
| GPU smoke | Two H100s: BF16, compiled forward/backward, FusedAdam and NCCL all-reduce passed without networking. |
| Serving | Released RoboTwin checkpoint, one H100, host networking on loopback port 18848: health, prediction (20 action values) and reset passed. |
| Training/resume | Four H100s: steps 1–10, stop after completed state, then a new container resumed at 11 and finished step 20 with exit code 0. |

Training used four real RoboTwin simulation episodes (583 timesteps / 579
windows), about 6.02B trainable parameters, BF16, ZeRO-2, batch size 1 per GPU,
gradient checkpointing, CPU model initialization and no optimizer offload.
Both phases used `NCCL_NVLS_ENABLE=0`. The step-10 full state occupied **102.3 GiB**;
final deployment weights occupied **23.1 GiB**. Verified metrics covered steps
1–20 without gaps/duplicates, with finite losses and gradient norms.

Limits: two H100s ran out of memory for this setup; four-GPU NCCL needed the
host-specific NVLS workaround. The intentionally stopped container exited with
code 1; recovery was verified from actual artifacts and resumed steps, since a
resume INFO log was missing. These results do not establish convergence,
benchmark quality, bitwise equivalence to uninterrupted training, changing-LR
scheduler recovery, full-dataset training, multi-node/RDMA or Cosmos GPU support.

## Later CPU/container checks

The local review image `openwam:docker-review`, ID
`sha256:088aee706a114117d7a134072f19a81641293335e352ae5dcf6e5df337fd0e01`,
was built from `b27be2f` plus the directory and review fixes. It passed pip/Ruff,
**1,984 CPU tests** and **12 Docker integration tests**, including custom Compose
image selection, relative mounts, Git worktrees, runtime UID/GID, CUDA library
selection under `exec`, and schema 3 offline delivery. A separate cold-load check
used the original schema 2 importer to verify backward compatibility.

These later checks are local CPU/container evidence, not new GPU acceptance.
Earlier iteration reports remain in Git history. For another revision, build it,
run `make docker-check` and `make docker-integration-check PYTHON=python3`, then
run GPU checks, real inference and training/resume on the destination host.
Record the image/revision and driver/Toolkit alongside results; CPU CI alone
does not establish GPU compatibility.
