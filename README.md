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

A Python-native toolkit for neuroimage processing -- `pip install` and go from
raw structural images to tissue segmentations, microstructure maps, fiber
orientations, and tract segmentations without wrestling with multi-tool
installations.

## Why kwneuro?

Neuroimaging analysis typically requires stitching together several packages
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

kwneuro is not intended to be a replacement for the full power of FSL or
MRtrix3. It is a lightweight layer for researchers who want standard structural
or dMRI analyses with minimal friction.

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

Some optional extras install PyTorch-backed tools. If the selected PyTorch wheel
does not match your GPU or NVIDIA driver, install a CPU-only or CUDA-specific
PyTorch build appropriate for that machine before installing the extra, or use
the matching PyTorch package index when syncing with uv.

Requires Python 3.10+.

## Quick start

```python
from kwneuro import read_dwi_fsl, write_volume

# Load DWI data into memory. The .bval and .bvec sidecars are inferred
# from the DWI NIfTI path when they use the same BIDS-style stem.
dwi = read_dwi_fsl("sub-01_dwi.nii.gz").load()

# Denoise and fit DTI (core -- no extras needed)
dwi = dwi.denoise()
dti = dwi.estimate_dti()
fa, md = dti.get_fa_md()

# Brain extraction and NODDI require optional extras:
#   pip install kwneuro[hdbet,noddi]
mask = dwi.extract_brain()
noddi = dwi.estimate_noddi(mask=mask)

# Save everything to disk
write_volume(dti.volume, "output/dti.nii.gz")
write_volume(fa, "output/fa.nii.gz")
write_volume(md, "output/md.nii.gz")
write_volume(noddi.volume, "output/noddi.nii.gz")
```

Pass explicit `bval=` or `bvec=` paths to `read_dwi_fsl()` when your sidecars do
not share the DWI NIfTI stem.

These file helpers are convenience adapters for quick file-first workflows. The
resource model remains the core API for custom pipelines and reusable Python
code.

## Command line

Common one-step workflows are also exposed through the `kwneuro` command:

```bash
kwneuro --help
kwneuro dwi mean-b0 --dwi sub-01_dwi.nii.gz --out output/mean_b0.nii.gz
kwneuro dwi dti --dwi sub-01_dwi.nii.gz --out-dti output/dti.nii.gz --out-fa output/fa.nii.gz --out-md output/md.nii.gz
kwneuro mask dwi-batch --inputs bids-or-fsl-dir --outputs masks
kwneuro structural bias-correct --image sub-01_T1w.nii.gz --out output/t1_bias_corrected.nii.gz
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
uv run pre-commit run -a
```

See the
[Developer Guide](https://kwneuro.readthedocs.io/en/latest/developer-guide.html)
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
