# Notebooks

Example notebooks demonstrating `kwneuro` pipeline usage. These are stored as
percent-format Python scripts (compatible with Jupytext) and are excluded from
CI linting.

## Running a notebook

Install the notebook dependencies with
[uv](https://docs.astral.sh/uv/getting-started/installation/), then open a
notebook:

```bash
uv sync --extra notebooks
cd notebooks
uv run jupytext --to notebook example-pipeline.py
uv run jupyter notebook example-pipeline.ipynb
```

### Without uv

```bash
pip install -e ".[notebooks]"
cd notebooks
jupytext --to notebook example-pipeline.py
jupyter notebook example-pipeline.ipynb
```
