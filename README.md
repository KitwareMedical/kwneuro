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

kwneuro is a Python package for building MRI analysis workflows around diffusion
microstructure and structural MRI data.

It works with ordinary NIfTI images and FSL-style `.bval`/`.bvec` files, then
gives you composable Python objects for common steps: denoising, DTI, NODDI,
CSD, brain masks, T1 bias correction, tissue segmentation, parcellation,
registration, template building, harmonization, and cached reruns of expensive
work.

## What kwneuro is for

kwneuro is designed for workflows that:

- fit diffusion microstructure models and write scalar maps such as FA, MD, NDI,
  ODI, and free-water fraction;
- combine DWI and structural MRI data through masking, registration,
  segmentation, and region-level analysis;
- build study-specific templates or harmonize scalar maps across sites;
- write Python pipeline code without managing each backend tool's file formats
  and coordinate conventions yourself.

kwneuro is not intended to be a replacement for the full power of FSL or
MRtrix3. It is a lightweight layer for researchers who want standard structural
or dMRI analyses with minimal friction.

<!-- GETTING-STARTED-START -->

## Installation

```bash
pip install kwneuro              # Core DWI, structural, registration, templates
pip install "kwneuro[all]"       # All optional backends
```

Individual optional extras can also be installed separately:

```bash
pip install "kwneuro[hdbet]"      # Brain extraction (HD-BET)
pip install "kwneuro[noddi]"      # NODDI estimation (AMICO)
pip install "kwneuro[tractseg]"   # Tract segmentation (TractSeg)
pip install "kwneuro[combat]"     # ComBat harmonization (neuroCombat)
pip install "kwneuro[antspynet]"  # Deep structural segmentation/parcellation
```

Some optional extras install PyTorch- or TensorFlow-backed tools. If the
selected wheel does not match your GPU or driver, install the CPU-only or
CUDA-specific backend appropriate for that machine before installing the extra.

Requires Python 3.10-3.13.

## Quick starts

### DWI microstructure

```python
from kwneuro import read_dwi_fsl, write_volume

# .bval and .bvec are inferred from the DWI NIfTI path when they
# use the same BIDS-style stem.
dwi = read_dwi_fsl("sub-01_dwi.nii.gz").load()

dwi_denoised = dwi.denoise()
dti = dwi_denoised.estimate_dti()
fa, md = dti.get_fa_md()

write_volume(dti.volume, "output/dti.nii.gz")
write_volume(fa, "output/fa.nii.gz")
write_volume(md, "output/md.nii.gz")
```

Pass explicit `bval=` or `bvec=` paths to `read_dwi_fsl()` when those files do
not share the DWI NIfTI stem.

### Structural MRI

```python
from kwneuro import read_structural, write_structural, write_volume

t1 = read_structural("sub-01_T1w.nii.gz").load()

t1_corrected = t1.correct_bias()
tissue_labels = t1_corrected.segment_tissues()

write_structural(t1_corrected, "output/sub-01_T1w_n4.nii.gz")
write_volume(tissue_labels, "output/sub-01_tissues.nii.gz")
```

The file helpers are convenience adapters for file-first scripts and notebooks.
The resource model remains the core API for reusable pipelines.

## Caching

Wrap pipeline code in a `Cache` context to persist outputs and reuse them when
the same inputs and parameters are seen again:

```python
from kwneuro import Cache, read_dwi_fsl

dwi = read_dwi_fsl("sub-01_dwi.nii.gz")

with Cache("cache/sub-01"):
    dwi_denoised = dwi.denoise()
    dti = dwi_denoised.estimate_dti()
```

Outside a `Cache` context, the same functions run normally.

## Command line

Common one-step workflows are also exposed through the `kwneuro` command:

```bash
kwneuro --help

kwneuro dwi dti \
  --dwi sub-01_dwi.nii.gz \
  --out-dti output/dti.nii.gz \
  --out-fa output/fa.nii.gz \
  --out-md output/md.nii.gz

kwneuro structural segment-tissues \
  --image sub-01_T1w.nii.gz \
  --out output/tissues.nii.gz

kwneuro registration dwi-to-structural \
  --dwi sub-01_dwi.nii.gz \
  --structural sub-01_T1w.nii.gz \
  --out-transform output/dwi_to_t1_transform
```

## What's included

| Area            | Capability                                             | Powered by  | Install       |
| --------------- | ------------------------------------------------------ | ----------- | ------------- |
| **DWI**         | Patch2Self denoising, DTI, FA/MD, CSD response/peaks   | DIPY        | core          |
| **DWI**         | NODDI maps: NDI, ODI, free-water fraction              | AMICO       | `[noddi]`     |
| **DWI**         | White-matter tract segmentation from CSD peaks         | TractSeg    | `[tractseg]`  |
| **Structural**  | N4 bias correction and Atropos tissue segmentation     | ANTsPy      | core          |
| **Structural**  | Brain extraction                                       | HD-BET      | `[hdbet]`     |
| **Structural**  | Deep Atropos segmentation and DKT parcellation         | ANTsPyNet   | `[antspynet]` |
| **Cross-modal** | DWI/T1 registration, transforms, and template building | ANTsPy      | core          |
| **Group data**  | ComBat harmonization of scalar maps                    | neuroCombat | `[combat]`    |
| **Pipelines**   | Lazy resources, file helpers, CLI commands, caching    | kwneuro     | core          |

<!-- GETTING-STARTED-END -->

## Tutorials

Start with the tutorial that matches the job:

- [Run a single-subject DWI microstructure pipeline](https://kwneuro.readthedocs.io/en/latest/tutorials/example-pipeline.html)
- [Combine T1 and DWI data for region-level analysis](https://kwneuro.readthedocs.io/en/latest/tutorials/example-region-analysis.html)
- [Build a population template](https://kwneuro.readthedocs.io/en/latest/tutorials/example-group-template.html)
- [Harmonize multi-site scalar maps](https://kwneuro.readthedocs.io/en/latest/tutorials/example-harmonization.html)
- [Use one kwneuro step in an existing non-kwneuroworkflow](https://kwneuro.readthedocs.io/en/latest/tutorials/example-single-step.html)
- [Add a custom non-kwneuro step to a kwneuro pipeline](https://kwneuro.readthedocs.io/en/latest/tutorials/example-custom-step.html)

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
