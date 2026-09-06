# Contributing to OpenWAM

## Dev setup

Install the base environment first, then install the development toolchain:

~~~bash
pip install -e '.[dev]'
~~~

Install the pre-commit hooks if desired:

~~~bash
pre-commit install
~~~

## Common commands

~~~bash
make lint      # check code quality with ruff
make format    # auto-format code
make check     # compile check
make all       # lint and compile check
~~~

## Before submitting a PR

1. Run make all and resolve any failures.
2. Update README.md when user-visible behavior changes.
3. Keep commits focused: one logical change per commit.
