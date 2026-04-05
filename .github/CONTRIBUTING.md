See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Quick development

The recommended way to develop kwneuro is with [uv](https://docs.astral.sh/uv/).
Install uv following the
[official instructions](https://docs.astral.sh/uv/getting-started/installation/),
then clone the repo and sync:

```console
$ git clone https://github.com/brain-microstructure-exploration-tools/kwneuro.git
$ cd kwneuro
$ uv sync --extra dev
```

This creates a virtual environment under `.venv/`, installs kwneuro in editable
mode with all development dependencies, and respects the committed `uv.lock` for
reproducible installs.

Common tasks:

```console
$ uv run pytest                            # Run tests
$ uv run pytest --cov=kwneuro              # Run tests with coverage
$ uv run pre-commit run --all-files        # Format + fast lint
$ uv run pylint kwneuro                    # Thorough lint (slow)
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

```console
$ uv run pre-commit install
```

You can also run it manually:

```console
$ uv run pre-commit run --all-files
```

# Testing

```console
$ uv run pytest
```

# Coverage

```console
$ uv run pytest --cov=kwneuro
```

# Building docs

```console
$ uv run --extra docs sphinx-apidoc -o docs --separate --module-first -d 2 --force src
$ uv run --extra docs sphinx-build -n -T docs docs/_build/html
```

To live-preview while editing:

```console
$ uv run --extra docs sphinx-autobuild -n -T docs docs/_build/html
```
