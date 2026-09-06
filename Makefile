.PHONY: test test-full tri-system-smoke check compile clean lint format all help

PYTHON ?= python
CORE_TEST_ARGS ?= tests --ignore=tests/test_tri_system_smoke.py
TRI_SYSTEM_SMOKE_ARGS ?= tests/test_tri_system_smoke.py

help:
	@echo "make test             - run the core OpenWAM test suite"
	@echo "make test-full        - run the full OpenWAM test suite"
	@echo "make tri-system-smoke - run tri-system smoke tests"
	@echo "make lint             - check code with ruff"
	@echo "make format           - auto-format code with ruff"
	@echo "make compile          - syntax-check Python sources with compileall"
	@echo "make check            - run compile checks and the core test suite"
	@echo "make all              - lint + core test"
	@echo "make clean            - remove Python cache files"

test:
	$(PYTHON) -m pytest -q -m "not gpu" $(CORE_TEST_ARGS)

test-full:
	$(PYTHON) -m pytest -q tests

tri-system-smoke:
	$(PYTHON) -m pytest -q $(TRI_SYSTEM_SMOKE_ARGS)

lint:
	$(PYTHON) -m ruff check openwam/ scripts/ tests/

format:
	$(PYTHON) -m ruff format openwam/ scripts/ tests/
	$(PYTHON) -m ruff check --fix openwam/ scripts/ tests/

compile:
	$(PYTHON) -m compileall openwam scripts tests

check: compile test

all: lint test

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
