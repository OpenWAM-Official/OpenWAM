# Docker acceptance record — 2026-09-10

The standard Wan Docker runtime was tested at commit
`9ef0cd98edb7899ddff1e38fbe7791005fd257ac`. This records that revision's
short training/recovery test, not a model-convergence or benchmark-success
evaluation. Subsequent project-isolation, runtime-user and bootstrap-index
changes have their own [follow-up validation record](docker_followup_validation.md);
the GPU/remote-CI results below do not claim to test those later changes.

## Environment and results

| Check | Result |
| --- | --- |
| Remote build and CPU checks | [GitHub Actions run 34466639632](https://github.com/KraHsu/OpenWAM-Official/actions/runs/34466639632): actual CUDA image built and loaded; pip/Ruff checks passed; **1,960 tests passed, 5 skipped, 21 GPU tests deselected**. |
| Docker integration | Same remote run: **4 tests passed**, covering image selection, development mounts/output persistence, offline round-trip, and validation image/GPU/port configuration. |
| Cold CI build | [Run 34463896096](https://github.com/KraHsu/OpenWAM-Official/actions/runs/34463896096) also passed before the rendezvous fix. |
| Offline runtime | Final CUDA archive imported on H100. SHA256 and image layers/configuration matched across Docker 29.8.0/containerd and Docker 28.1.1/overlay2. |
| GPU smoke | Two H100 80 GB GPUs: BF16, compiled forward/backward, DeepSpeed FusedAdam and NCCL all-reduce passed with networking disabled. |
| Nondefault serving port | Host networking, loopback **18848**, one H100: scheduled health, `make docker-health`, and the inference CLI passed without explicit URL arguments. Prediction returned 20 action values, followed by a successful reset. This used the original released checkpoint. |
| Training/recovery | **Four H100 80 GB GPUs**, BF16, ZeRO-2, batch size 1 per GPU, gradient checkpointing, CPU model initialization, no optimizer offload, `NCCL_NVLS_ENABLE=0`: steps 1–10, container stop, new container steps 11–20, exit code 0. |

The GPU host ran driver **550.54.14**, NVIDIA Container Toolkit **1.18.1** and
kernel **5.4.0**. The image uses Python 3.12, PyTorch 2.7.1+cu128, Accelerate
1.14.0 and DeepSpeed 0.18.9. These results describe this combination, rather
than establishing compatibility with every driver, topology or optional image.

## Real data and recovery evidence

The input was four original RoboTwin `adjust_bottle/aloha-agilex_clean_50`
episodes (`episode0.hdf5` through `episode3.hdf5`) with instructions and scene
metadata: **583 timesteps, 579 training windows**. The reader produced 9 video
frames and a finite action tensor of shape `[32, 80]`. Normalization statistics
were computed from this subset. These are benchmark simulation demonstrations,
not randomly generated tensors or physical-robot data.

The run used `training.debug=true`, `training.num_epochs=null`,
`training.dataset_num_workers=0` and
`training.save_full_states_for_resume=true`. Debug mode uses a constant learning
rate, so this does not validate recovery of a changing learning-rate scheduler.

The driver waited for the atomic completion marker before stopping the first
container. That marker contained:

```json
{"global_step": 10, "opt_step": 10, "epoch": 0}
```

The completed state contained four optimizer shards, four RNG-state files and
the model state, totaling **109,839,682,451 bytes (102.3 GiB)**. The first
container exited after the intentional stop; its exit code was 1. A distinct
container subsequently started with the same image/output directory,
`training.finetune_ckpt_path=null`, and `training.resume_ckpt_path` pointing to
the completed state.

The resumed container's debug records were exactly steps 11–20, with matching
optimizer-step counters. The combined CSV contained exactly steps 1–20 without
gaps or duplicates. All loss and gradient-norm values were finite. The resumed
container exited with code **0**. Final weights occupy **24,813,767,464 bytes
(23.1 GiB)** and differ from step 10. Normal completion removed resumable state
directories and retained deployment weights, matching the current trainer policy.

| Artifact | SHA256 |
| --- | --- |
| Dataset subset archive | `f3de081e82114d621b4ed5b191bc7bc42a6a8ef4126efc09164153410d5cb7b8` |
| Final Docker archive | `b46d8561416c03dd5c6747ecc2804db685f9339f4309745ee4769c4c07318a66` |
| Step-10 weights | `4e617189d7c1b2f93790ef16185f060d0a7fc07c53a3430d694fd8fa29e9bdf2` |
| Step-20 weights | `2009366b2a269e84eda0bc959465ff4d44e74dfab199a0db3c15dd70d935c660` |

## Issues found

- The no-network GPU smoke initially hung because `torchrun --standalone`
  advertised an unresolvable container hostname. Commit `9ef0cd9` uses static
  loopback rendezvous; the updated two-GPU smoke passed.
- Two 80 GB GPUs were insufficient for this full-model training configuration.
  The first Adam update requested another 11.22 GiB with only 9.02 GiB free.
  Four GPUs passed. This is not an absolute minimum for other configurations;
  optimizer offload and different models require separate testing.
- Four-GPU NCCL initialization hung with default settings. A minimal all-reduce
  reproduced it. Disabling cuMem host allocations alone did not help;
  **`NCCL_NVLS_ENABLE=0` alone passed**, returning 4.0 on every rank. Both
  training phases used that setting. It disables NVLink SHARP collective
  offload and may affect performance; see
  [NVIDIA's setting reference](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/env.html#nccl-nvls-enable).
- The resume INFO message did not appear on stdout. The original acceptance
  harness consequently rejected that log-only condition after training exited
  successfully. An independent verifier instead checked the actual completion
  marker, optimizer/RNG inventory, sequential container lifetimes, first resumed
  step, all 20 metrics, changed weights and exit status. The original diagnostic
  and successful verification are both retained; the console visibility issue
  remains an observability limitation.
- Full-state checkpointing needs substantially more storage than serving. Old
  and new checkpoint generations can coexist before pruning; allow space for
  both, plus the image, original weights, dataset and caches.

## Rechecking and retained evidence

Use the [Docker guide](docker.md) to select the image, mounts, GPU IDs and port:

```bash
make docker-check
make docker-integration-check PYTHON=python3
make docker-gpu-check
make docker-health
docker compose exec -T serve python scripts/inference_test/inference_single_test.py --test --state-dim 20
```

For a host reproducing the four-GPU NVLS issue, pass
`-e NCCL_NVLS_ENABLE=0` to both `docker compose run ... gpu-check` and each
`docker compose run ... train train ...` invocation. The guide includes an
example. Select free GPUs before testing.

Raw evidence is retained locally in `outputs/docker-validation-20260910/`:
CI logs, GPU/serving logs, the training CSV, acceptance scripts and
`training/training-verified.json`. These are excluded from Git and the image
build context. On H100, the checkpoint and training logs remain in
`/mnt/nvme4/zch/openwam-docker/acceptance-9ef0cd9-gpu4-nvls/`; inference and
communication diagnostics are in `acceptance-9ef0cd9/validation/` under the same
parent. All acceptance GPU processes and the temporary policy server have exited.

No uninterrupted control run established bitwise trajectory identity.
Multi-node/RDMA, complete-dataset training, changing learning-rate scheduler
recovery and optional Cosmos GPU behavior are outside this test.
