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

- **Get started fast** -- core analysis (DTI, CSD, registration, template
  building) works out of the box. Additional tools (brain extraction, NODDI,
  tract segmentation, harmonization) are available as optional extras.
- **Swap models easily** -- go from DTI to NODDI to CSD without rewriting your
  script.
- **Work lazily or eagerly** -- data stays on disk until you call `.load()`, so
  you control memory usage.

kwneuro is not (yet) a replacement for the full power of FSL or MRtrix3. It is a
lightweight layer for researchers who want standard dMRI analyses with minimal
friction.

<!-- GETTING-STARTED-START -->

## Installation

```bash
pip install kwneuro             # Core (DTI, CSD, registration, templates)
pip install kwneuro[all]        # Everything including optional extras
```

Individual optional extras can also be installed separately:

```bash
pip install kwneuro[hdbet]      # Brain extraction (HD-BET)
pip install kwneuro[noddi]      # NODDI estimation (AMICO)
pip install kwneuro[tractseg]   # Tract segmentation (TractSeg)
pip install kwneuro[combat]     # ComBat harmonization (neuroCombat)
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

# Denoise and fit DTI (core -- no extras needed)
dwi = dwi.denoise()
dti = dwi.estimate_dti()
fa, md = dti.get_fa_md()

# Brain extraction and NODDI require optional extras:
#   pip install kwneuro[hdbet,noddi]
mask = dwi.extract_brain()
noddi = dwi.estimate_noddi(mask=mask)

# Save everything to disk
dti.save("output/dti.nii.gz")
NiftiVolumeResource.save(fa, "output/fa.nii.gz")
noddi.save("output/noddi.nii.gz")
```

## What's included

| Capability             | What it does                                                      | Powered by  | Extra        |
| ---------------------- | ----------------------------------------------------------------- | ----------- | ------------ |
| **Denoising**          | Patch2Self self-supervised denoising                              | DIPY        |              |
| **Brain extraction**   | Deep-learning brain masking from mean b=0                         | HD-BET      | `[hdbet]`    |
| **DTI**                | Tensor fitting, FA, MD, eigenvalue decomposition                  | DIPY        |              |
| **NODDI**              | Neurite density, orientation dispersion, free water fraction      | AMICO       | `[noddi]`    |
| **CSD**                | Fiber orientation distributions and peak extraction               | DIPY        |              |
| **Tract segmentation** | 72 white-matter bundles from CSD peaks                            | TractSeg    | `[tractseg]` |
| **Registration**       | Pairwise registration (rigid, affine, SyN)                        | ANTs        |              |
| **Template building**  | Iterative unbiased population templates (single- or multi-metric) | ANTs        |              |
| **Harmonization**      | ComBat site-effect removal for multi-site scalar maps             | neuroCombat | `[combat]`   |

<!-- GETTING-STARTED-END -->

## Example notebooks

The
[`notebooks/`](https://github.com/brain-microstructure-exploration-tools/kwneuro/tree/main/notebooks)
directory contains Jupytext notebooks you can run end-to-end:

- **[example-pipeline.py](https://github.com/brain-microstructure-exploration-tools/kwneuro/blob/main/notebooks/example-pipeline.py)**
  -- Single-subject walkthrough: loading, denoising, brain extraction, DTI,
  NODDI, CSD, and TractSeg.
- **[example-group-template.py](https://github.com/brain-microstructure-exploration-tools/kwneuro/blob/main/notebooks/example-group-template.py)**
  -- Multi-subject FA/MD template construction using iterative registration.
- **[example-harmonization.py](https://github.com/brain-microstructure-exploration-tools/kwneuro/blob/main/notebooks/example-harmonization.py)**
  -- ComBat harmonization of multi-site scalar maps.

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

See
[CONTRIBUTING.md](https://github.com/brain-microstructure-exploration-tools/kwneuro/blob/main/.github/CONTRIBUTING.md)
for the full guide, including a non-uv setup option.

See the [GitHub Discussions][github-discussions-link] for questions and ideas,
or open an
[issue](https://github.com/brain-microstructure-exploration-tools/kwneuro/issues)
for bugs and feature requests.

## Acknowledgements

This work is supported by the National Institutes of Health under Award Number
1R21MH132982. The content is solely the responsibility of the authors and does
not necessarily represent the official views of the National Institutes of
Health.
