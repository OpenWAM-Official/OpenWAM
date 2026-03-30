.PHONY: test check compile clean

PYTHON ?= python

help:
	@echo "make test    - run the core OpenWAM test suite"
	@echo "make compile - syntax-check Python sources with compileall"
	@echo "make check   - run compile checks and the core test suite"
	@echo "make clean   - remove Python cache files"

test:
	$(PYTHON) -m pytest -q tests

compile:
	$(PYTHON) -m compileall open_wam scripts tests

check: compile test

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
