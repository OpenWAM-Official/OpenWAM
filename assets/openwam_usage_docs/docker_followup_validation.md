# Docker usability fixes — 2026-09-10

The three follow-up fixes were built into the complete CUDA image
`openwam:opensource-fixes` and validated locally. The test image ID is
`sha256:d62d3303f01337525e2b58e114c2b8fffe52ed443b997d64d133a63933d910d0`.
The Python dependency lock and CUDA base version were unchanged.

| Requirement | Change | Verification |
| --- | --- | --- |
| Independent working copies | Removed the fixed Compose project name; restored directory-based naming. Documented per-checkout `COMPOSE_PROJECT_NAME` for identical basenames and migration of existing deployments. | Started containers from two checkouts, confirmed distinct project/container identities, stopped the second project and confirmed the first container was still running. Also checked `.env` overrides for two `OpenWAM` directories. |
| Arbitrary runtime UID/GID | Persisted `HOME=/cache/home`; exposed logical user/group `openwam` through private NSS account files without changing numeric identity or system account files. Account initialization precedes the shell and uses `exec` to preserve process behavior. | Tested existing UID 1000 and absent UID 23456 (GID 23457): `whoami`, `pwd`, `grp`, `getpass`, writable Git global configuration, persistence across container recreation, and name resolution under `docker exec`. No NSS startup error output. |
| Bootstrap fallback | Moved the fallback build argument before bootstrap; the actual bootstrap pip installer now accepts the primary and fallback indexes. CUDA torch wheels retain their dedicated index. | In a network-disabled container, a fresh venv failed against an empty primary index; enabling the fallback installed a real local test wheel. With versions 1.0 and 2.0 available, the constraint correctly selected 1.0. |

`make docker-integration-check` passed **all 7 checks**, also covering the prior
image selection, editable source/output mounts, offline delivery and GPU/port
configuration behavior. `make docker-check` passed dependency checks, the serve
CLI and Ruff, plus **1,960 CPU tests** (8 skips, 21 GPU tests deselected).
The Docker workflow includes the new tests and bootstrap-script syntax check;
the updated workflow passed actionlint.

Reproduce from a checkout with Docker access:

```bash
export OPENWAM_IMAGE=openwam:opensource-fixes
make docker-build
make docker-check
make docker-integration-check PYTHON=python3
```

The validation build used Aliyun as `PIP_INDEX_URL` and PyPI as
`PIP_FALLBACK_INDEX_URL`. Bootstrap pip searches both sources without priority;
the uv phase retains its existing primary-first behavior. Use trusted indexes.

Raw build and test logs are retained under
`outputs/docker-usability-validation-20260910/` (excluded from Git and the image
context). These follow-up checks validate the three usability changes; no new
H100 training or remote CI run is claimed. Earlier GPU evidence is recorded
separately in [the original acceptance record](docker_validation.md).
