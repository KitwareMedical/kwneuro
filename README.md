# kwneuro

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/brain-microstructure-exploration-tools/kwneuro/workflows/CI/badge.svg
[actions-link]:             https://github.com/brain-microstructure-exploration-tools/kwneuro/actions
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/brain-microstructure-exploration-tools/kwneuro/discussions
[pypi-link]:                https://pypi.org/project/kwneuro/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/kwneuro
[pypi-version]:             https://img.shields.io/pypi/v/kwneuro
[rtd-badge]:                https://readthedocs.org/projects/kwneuro/badge/?version=latest
[rtd-link]:                 https://kwneuro.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

A Python-native toolkit for diffusion MRI analysis -- `pip install` and go from
raw dMRI data to microstructure maps, fiber orientations, and tract
segmentations without wrestling with multi-tool installations.

> **Early phase, under active development.** The API may change between
> releases.

## Why kwneuro?

Diffusion MRI analysis typically requires stitching together several packages
(FSL, MRtrix3, DIPY, AMICO, ANTs, ...), each with its own installation story,
file conventions, and coordinate quirks. kwneuro wraps the best of these tools
behind a single, pip-installable Python interface so you can:

- **Get started fast** -- no system-level dependencies to configure.
- **Swap models easily** -- go from DTI to NODDI to CSD without rewriting your
  script.
- **Work lazily or eagerly** -- data stays on disk until you call `.load()`, so
  you control memory usage.

kwneuro is not (yet) a replacement for the full power of FSL or MRtrix3. It is a
lightweight layer for researchers who want standard dMRI analyses with minimal
friction.

## Installation

```bash
pip install kwneuro
```

Requires Python 3.10+.

## Quick start

```python
from kwneuro.dwi import Dwi
from kwneuro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource

# Load DWI data into memory
dwi = Dwi(
    NiftiVolumeResource("sub-01_dwi.nii.gz"),
    FslBvalResource("sub-01_dwi.bval"),
    FslBvecResource("sub-01_dwi.bvec"),
).load()

# Denoise and extract a brain mask
dwi = dwi.denoise()
mask = dwi.extract_brain()

# Fit DTI and get FA / MD maps
dti = dwi.estimate_dti(mask=mask)
fa, md = dti.get_fa_md()

# Fit NODDI (needs multi-shell data)
noddi = dwi.estimate_noddi(mask=mask)

# Save everything to disk
dti.save("output/dti.nii.gz")
NiftiVolumeResource.save(fa, "output/fa.nii.gz")
noddi.save("output/noddi.nii.gz")
```

## What's included

| Capability             | What it does                                                      | Powered by |
| ---------------------- | ----------------------------------------------------------------- | ---------- |
| **Denoising**          | Patch2Self self-supervised denoising                              | DIPY       |
| **Brain extraction**   | Deep-learning brain masking from mean b=0                         | HD-BET     |
| **DTI**                | Tensor fitting, FA, MD, eigenvalue decomposition                  | DIPY       |
| **NODDI**              | Neurite density, orientation dispersion, free water fraction      | AMICO      |
| **CSD**                | Fiber orientation distributions and peak extraction               | DIPY       |
| **Tract segmentation** | 72 white-matter bundles from CSD peaks                            | TractSeg   |
| **Registration**       | Pairwise registration (rigid, affine, SyN)                        | ANTs       |
| **Template building**  | Iterative unbiased population templates (single- or multi-metric) | ANTs       |

## Example notebooks

The [`notebooks/`](notebooks/) directory contains Jupytext notebooks you can run
end-to-end:

- **[example-pipeline.py](notebooks/example-pipeline.py)** -- Single-subject
  walkthrough: loading, denoising, brain extraction, DTI, NODDI, CSD, and
  TractSeg.
- **[example-group-template.py](notebooks/example-group-template.py)** --
  Multi-subject FA/MD template construction using iterative registration.

## Contributing

Contributions are welcome! Set up a dev environment with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pre-commit install
```

Run the tests and linter:

```bash
uv run pytest
uv run ruff check .
```

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the full guide, including a
non-uv setup option.

See the [GitHub Discussions][github-discussions-link] for questions and ideas,
or open an
[issue](https://github.com/brain-microstructure-exploration-tools/kwneuro/issues)
for bugs and feature requests.

## Acknowledgements

This work is supported by the National Institutes of Health under Award Number
1R21MH132982. The content is solely the responsibility of the authors and does
not necessarily represent the official views of the National Institutes of
Health.
