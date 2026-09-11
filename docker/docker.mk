.PHONY: docker-build docker-check docker-export docker-image docker-integration-check
.PHONY: docker-gpu-check docker-health docker-help

DOCKER ?= docker
DOCKER_BUNDLE ?= dist/openwam-offline
DOCKER_BUILD_ARGS ?=
VCS_REF ?= $(shell git rev-parse HEAD)$(if $(shell git status --porcelain --untracked-files=normal),-dirty)
# Compose owns image interpolation, including .env. Do not reimplement dotenv in Make.
ifneq ($(origin OPENWAM_IMAGE), undefined)
export OPENWAM_IMAGE
endif
ifneq ($(origin COMPOSE_FILE), undefined)
export COMPOSE_FILE
endif
ifneq ($(filter-out docker-help,$(filter docker-%,$(MAKECMDGOALS))),)
ifneq ($(origin DOCKER_IMAGE), undefined)
$(error DOCKER_IMAGE was replaced by OPENWAM_IMAGE; set OPENWAM_IMAGE in .env or export it)
endif
endif
DOCKER_IMAGE_REF = "$$($(DOCKER) compose config --images serve)"

docker-help:
	@echo "make docker-build     - build the CUDA 12.8 image (network required)"
	@echo "make docker-check     - validate Compose and run container CPU checks"
	@echo "make docker-export    - save image + run config as an offline bundle"
	@echo "make docker-image     - show the image selected by Compose and Make"
	@echo "make docker-integration-check - test image selection, dev mounts and offline delivery"
	@echo "make docker-gpu-check - test selected image/GPUs, compiler, optimizer and NCCL"
	@echo "make docker-health    - ping the running policy server at its configured address"

docker-build:
	$(DOCKER) build --platform linux/amd64 --target openwam --build-arg VCS_REF="$(VCS_REF)" $(DOCKER_BUILD_ARGS) -f docker/Dockerfile -t $(DOCKER_IMAGE_REF) .

docker-image:
	@$(DOCKER) compose config --images serve

docker-gpu-check:
	$(DOCKER) compose run --rm -T gpu-check

docker-health:
	$(DOCKER) compose exec -T serve python /opt/openwam/docker/healthcheck.py

docker-check:
	$(DOCKER) compose config --quiet
	$(DOCKER) compose -f compose.yaml -f docker/compose.host.yaml config --quiet
	$(DOCKER) compose -f compose.yaml -f docker/compose.dev.yaml --profile dev --profile train config --quiet
	$(DOCKER) compose -f compose.yaml -f docker/compose.host.yaml -f docker/compose.dev.yaml --profile dev --profile train config --quiet
	OPENWAM_WORKSPACE_DIR=/tmp/openwam-workspace OPENWAM_SOURCE_DIR=/tmp/openwam-workspace/feature $(DOCKER) compose -f compose.yaml -f docker/compose.dev.yaml -f docker/compose.worktree.yaml --profile dev --profile train config --quiet
	$(DOCKER) run --rm --pull=never --network none $(DOCKER_IMAGE_REF) python -m pip check
	$(DOCKER) run --rm --pull=never --network none $(DOCKER_IMAGE_REF) serve --help
	$(DOCKER) run --rm --pull=never --network none --tmpfs /opt/openwam/tests/dataloader/.cache:mode=1777 $(DOCKER_IMAGE_REF) make all CORE_TEST_ARGS='tests docker/tests --ignore=tests/test_tri_system_smoke.py -o cache_dir=/cache/pytest'

docker-integration-check:
	openwam_image=$(DOCKER_IMAGE_REF) && OPENWAM_DOCKER_TEST_IMAGE="$$openwam_image" OPENWAM_DOCKER_COMMAND="$(DOCKER)" $(PYTHON) -m unittest discover -s docker/tests -p test_docker_integration.py -v

docker-export:
	$(PYTHON) docker/offline.py --docker "$(DOCKER)" export $(DOCKER_IMAGE_REF) "$(DOCKER_BUNDLE)"
