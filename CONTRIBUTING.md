# Contributing to OpenWAM

## Development Setup

```bash
# Clone and install in editable mode
git clone https://github.com/KraHsu/OpenWAM.git
cd OpenWAM
pip install -e .

# Install dev tools
pip install ruff pytest pre-commit

# Set up pre-commit hooks (optional but recommended)
pre-commit install
```

## Common Commands

```bash
make test      # run the test suite
make lint      # check code quality with ruff
make format    # auto-format code
make check     # compile check + tests
make all       # lint + tests (full validation)
```

## Before Submitting a PR

1. Run `make all` and ensure it passes
2. Add tests for new functionality in `tests/`
3. If you changed user-visible behavior, update `README.md`
4. Keep commits focused: one logical change per commit

## Code Style

- Ruff handles linting and formatting (configured in `pyproject.toml`)
- Line length limit: 120 characters
- Import sorting: ruff isort (first-party = `open_wam`)
- `third_party/` is excluded from linting

## Project Structure

- `open_wam/` - main package (all new code goes here)
- `scripts/` - Hydra entrypoints (train, infer, eval)
- `configs/` - Hydra config groups
- `tests/` - pytest test suite
- `third_party/` - vendored dependencies (do not modify unless necessary)

## Adding a New Component

### New dataset
1. Create `open_wam/data/my_dataset.py` inheriting from `BaseActionDataset`
2. Register it in `open_wam/data/registry.py`
3. Add a config in `configs/data/my_dataset.yaml`

### New architecture
1. Create `open_wam/models/architectures/my_arch.py` inheriting from `BaseWAMArchitecture`
2. Register with `@register_architecture("my_arch")`
3. Add a config in `configs/model/architecture/my_arch.yaml`

### New evaluator
1. Create `open_wam/evaluation/my_evaluator.py` inheriting from `BaseEvaluator`
2. Register in `open_wam/evaluation/registry.py`
3. Add a config in `configs/eval/my_eval.yaml`
