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

## Running GPU tests

`make test` runs the CPU suite (`-m "not gpu"`). Tests that need real weights are
marked `gpu` and resolve their assets through `tests/_assets.py`: each asset is
looked up first in an `OPENWAM_<ASSET>` environment variable, then at the
directory the downloader in `scripts/download_assets/` writes to by default. So
after downloading with the repo scripts nothing needs to be set:

~~~bash
python scripts/download_assets/download_video_backbone.py --name Wan2.2-TI2V-5B --yes
pytest -m gpu tests/test_video_backbone_consistency.py
~~~

To point at weights stored elsewhere, export the matching variable, e.g.
`OPENWAM_WAN22_TI2V_5B=/data/Wan2.2-TI2V-5B`. The full list of names and default
locations is in `tests/_assets.py`.

## Before submitting a PR

1. Run make all and resolve any failures.
2. Update README.md when user-visible behavior changes.
3. Keep commits focused: one logical change per commit.
