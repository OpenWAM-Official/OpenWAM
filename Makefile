.PHONY: test check compile clean lint format all help

PYTHON ?= python

help:
	@echo "make test    - run the core OpenWAM test suite"
	@echo "make lint    - check code with ruff"
	@echo "make format  - auto-format code with ruff"
	@echo "make compile - syntax-check Python sources with compileall"
	@echo "make check   - run compile checks and the core test suite"
	@echo "make all     - lint + test (full validation)"
	@echo "make clean   - remove Python cache files"

test:
	$(PYTHON) -m pytest -q tests

lint:
	$(PYTHON) -m ruff check open_wam/ scripts/ tests/

format:
	$(PYTHON) -m ruff format open_wam/ scripts/ tests/
	$(PYTHON) -m ruff check --fix open_wam/ scripts/ tests/

compile:
	$(PYTHON) -m compileall open_wam scripts tests

check: compile test

all: lint test

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
