# Developer Guide

This guide covers setting up a development environment, running tests, and
building the documentation. For a broader introduction to best practices for
scientific Python packages, see the
[Scientific Python Developer Guide](https://learn.scientific-python.org/development/).

## Quick development

The recommended way to develop kwneuro is with [uv](https://docs.astral.sh/uv/).
Install uv following the
[official instructions](https://docs.astral.sh/uv/getting-started/installation/),
then clone the repo and sync:

```bash
git clone https://github.com/brain-microstructure-exploration-tools/kwneuro.git
cd kwneuro
uv sync --extra dev
```

This creates a virtual environment under `.venv/`, installs kwneuro in editable
mode with all development dependencies (including all optional extras), and
respects the committed `uv.lock` for reproducible installs.

```{note}
CI uses `uv sync --extra test`, which installs only the lightweight optional
deps needed for tests (neuroCombat). The heavier optional deps (HD-BET,
TractSeg, AMICO) are mocked in tests and not required in CI.
```

Common tasks:

```bash
uv run pytest                            # Run tests
uv run pytest --cov=kwneuro              # Run tests with coverage
uv run pre-commit run --all-files        # Format + fast lint
uv run pylint kwneuro                    # Thorough lint (slow)
```

## Setting up without uv

You can also set up a development environment with plain pip:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -e ".[dev]"
```

## Pre-commit

This project uses pre-commit for all style checking. Install the hook so it runs
automatically on each commit:

```bash
uv run pre-commit install
```

You can also run it manually:

```bash
uv run pre-commit run --all-files
```

## Testing

```bash
uv run pytest
```

## Coverage

```bash
uv run pytest --cov=kwneuro
```

## Building docs

```bash
uv run --extra docs sphinx-build -n -T docs docs/_build/html
```

To live-preview while editing:

```bash
uv run --extra docs sphinx-autobuild -n -T docs docs/_build/html
```

### Rebuilding tutorial pages from notebooks

The tutorial pages under `docs/tutorials/` are pre-rendered Markdown files
generated from the Jupytext notebooks in `notebooks/`. To regenerate them (e.g.,
after editing a notebook), use the conversion script:

```bash
# Rebuild all tutorials (needs all optional extras installed)
uv run --extra notebooks --extra all python scripts/update-notebook-pages.py

# Rebuild just one tutorial
uv run --extra notebooks --extra combat python scripts/update-notebook-pages.py \
    notebooks/example-harmonization.py
```

The script executes each notebook and exports the results as Markdown with
images. The output files in `docs/tutorials/` should be committed to git so that
the docs build on ReadTheDocs does not need to execute notebooks.

## Running the example notebooks

The `notebooks/` directory contains Jupytext percent-format `.py` files. Each
notebook may require different optional extras. See `notebooks/README.md` for
per-notebook instructions, or install everything at once:

```bash
uv sync --extra notebooks --extra all
cd notebooks
uv run jupytext --to ipynb example-harmonization.py
uv run jupyter notebook example-harmonization.ipynb
```
