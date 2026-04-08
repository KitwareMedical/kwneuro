See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Quick development

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

Note: CI uses `uv sync --extra test`, which installs only the lightweight
optional deps needed for tests (neuroCombat). The heavier optional deps (HD-BET,
TractSeg, AMICO) are mocked in tests and not required in CI.

Common tasks:

```bash
uv run pytest                            # Run tests
uv run pytest --cov=kwneuro              # Run tests with coverage
uv run pre-commit run --all-files        # Format + fast lint
uv run pylint kwneuro                    # Thorough lint (slow)
```

# Setting up without uv

You can also set up a development environment with plain pip:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install -e ".[dev]"
```

# Pre-commit

This project uses pre-commit for all style checking. Install the hook so it runs
automatically on each commit:

```bash
uv run pre-commit install
```

You can also run it manually:

```bash
uv run pre-commit run --all-files
```

# Testing

```bash
uv run pytest
```

# Coverage

```bash
uv run pytest --cov=kwneuro
```

# Building docs

```bash
uv run --extra docs sphinx-build -n -T docs docs/_build/html
```

To live-preview while editing:

```bash
uv run --extra docs sphinx-autobuild -n -T docs docs/_build/html
```
