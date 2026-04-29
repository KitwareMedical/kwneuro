# Notebooks

Example notebooks demonstrating `kwneuro` pipeline usage. These are stored as
percent-format Python scripts (compatible with Jupytext) and are excluded from
CI linting.

## Running a notebook

From the `notebooks/` directory, first convert the Jupytext `.py` files to
`.ipynb` (one-time step, no project dependencies needed):

```bash
cd notebooks
uv run --with jupytext jupytext --to ipynb *.py
```

Then open a specific notebook with exactly the extras it needs:

```bash
uv run --extra notebooks --extra hdbet --extra noddi --extra tractseg \
    jupyter notebook example-pipeline.ipynb

uv run --extra notebooks --extra combat \
    jupyter notebook example-harmonization.ipynb

uv run --extra notebooks \
    jupyter notebook example-group-template.ipynb

uv run --extra notebooks --extra hdbet --extra antspynet \
    jupyter notebook example-region-analysis.ipynb
```

### Without uv

```bash
pip install -e ".[notebooks,all]"
cd notebooks
jupytext --to notebook example-pipeline.py
jupyter notebook example-pipeline.ipynb
```
