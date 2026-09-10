# Compose overrides and Git worktree validation — 2026-09-10

Both follow-up development workflow fixes are included in the locally built
image `openwam:compose-worktree-fixes`, ID
`sha256:436c0ce3e394d977d39bb3866151c0a067709e204de49e4efa0273e92543f6d7`.

| Requirement | Implementation | Evidence |
| --- | --- | --- |
| Make and Compose select the same image with custom overlays | Removed Make's fixed `-f compose.yaml` image lookup. Build, CPU checks, integration checks and export resolve the effective `serve` image. Make command-line `COMPOSE_FILE` is exported. | Real Compose resolution agrees with `make docker-image` for automatic `compose.override.yaml`, `.env`, exported `COMPOSE_FILE` and Make command-line selection. The test executes all four Make recipes with recording adapters and confirms their build/run/export arguments and integration-test environment select the override image. |
| Git works inside a linked-worktree development container | Added the optional `compose.worktree.yaml`, preserving the selected workspace's host paths and the existing source/output/asset aliases. Documented setup from a checkout. | Real containers run Git status, diff, add and commit in a temporary workspace with spaces in its path. Both `run` and `exec` work. The host sees the new commit on the feature branch; the main branch stays unchanged. Git pointer files are byte-identical, dry-run pruning finds nothing to remove, and outputs survive container removal. |

Validation against the new image:

- `make docker-check`: dependency checks, serve CLI and Ruff passed;
  **1,960 CPU tests passed**, 10 skipped and 21 GPU tests deselected.
- `make docker-integration-check PYTHON=python3`: **all 9 checks passed**,
  including prior isolation, runtime identity and offline bundle round-trip checks.
- The combined host-network/development/worktree configuration parsed successfully;
  the Docker workflow passed actionlint.
- SHA256 values of Makefile, both development overlays, integration tests,
  environment template and Docker guide match between the image and workspace.

Reproduce from a checkout with Docker access:

```bash
export OPENWAM_IMAGE=openwam:compose-worktree-fixes
make docker-build
make docker-check docker-integration-check PYTHON=python3
```

Raw build/check logs and file hashes are retained in
`outputs/docker-workflow-validation-20260910/` (ignored by Git and Docker).
The validation build reused the existing dependency cache with Aliyun as the
primary pip index and PyPI as fallback. No CUDA/dependency/runtime-user changes
were required. This is local CPU and development workflow validation; it does
not claim a new H100 training run or remote CI run. This report was written
after the image build; the shipped Docker usage guide contains the setup steps.

The worktree overlay deliberately exposes the workspace directory selected by
the developer, including its repositories and Git metadata, writable at the
same host path. It is opt-in and supplied with source checkouts. Standalone
offline deployment continues to use its bundled serving/training configuration.
