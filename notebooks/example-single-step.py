# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # One-Step File Workflow
#
# This notebook demonstrates the quickest path from DWI files to scalar maps.
# It uses tiny synthetic data so it can run anywhere without downloads or
# optional pipeline dependencies.

# %% [markdown]
# ## Create a small FSL-style DWI input
#
# In real use, these files already come from a scanner, BIDS dataset, or
# preprocessing step. Here we create them only so the example is self-contained.

# %%
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from kwneuro.dwi import Dwi
from kwneuro.files import read_dwi_fsl, write_dwi_fsl, write_volume
from kwneuro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
)


def make_synthetic_dwi() -> Dwi:
    shape = (5, 5, 5)
    bvals = np.array([0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    bvecs = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    bvecs[4:] = bvecs[4:] / np.linalg.norm(bvecs[4:], axis=1, keepdims=True)

    grid = np.indices(shape).sum(axis=0)
    s0 = 1000.0 + 20.0 * grid
    tensor = np.diag([1.5e-3, 0.5e-3, 0.3e-3])
    signal = np.empty((*shape, bvals.size), dtype=np.float32)

    for i, (bval, bvec) in enumerate(zip(bvals, bvecs, strict=True)):
        adc = float(bvec @ tensor @ bvec)
        signal[..., i] = s0 * np.exp(-bval * adc)

    header = nib.Nifti1Header()
    header.set_xyzt_units("mm")

    return Dwi(
        volume=InMemoryVolumeResource(
            array=signal,
            affine=np.diag([2.0, 2.0, 2.0, 1.0]),
            metadata=dict(header),
        ),
        bval=InMemoryBvalResource(bvals),
        bvec=InMemoryBvecResource(bvecs),
    )


tmpdir = TemporaryDirectory()
work_dir = Path(tmpdir.name)
input_dwi_path = work_dir / "sub-01_dwi.nii.gz"

write_dwi_fsl(make_synthetic_dwi(), input_dwi_path)
print(f"Wrote example inputs under {work_dir}")

# %% [markdown]
# ## Read the files and fit DTI
#
# `kwneuro.files` is a convenience boundary for quick file-first workflows:
# it builds the same lazy resource objects used by the rest of `kwneuro`, but
# keeps the call site focused on paths. The resource model remains the core API
# for reusable or custom pipelines.

# %%
dwi = read_dwi_fsl(input_dwi_path)
dti = dwi.estimate_dti()
fa, md = dti.get_fa_md()

print(f"DWI shape: {dwi.volume.get_array().shape}")
print(f"FA range: {fa.get_array().min():.3f} to {fa.get_array().max():.3f}")
print(f"MD mean: {md.get_array().mean():.6f}")

# %% [markdown]
# ## Write derived volumes
#
# The file helpers write a resource to disk and return a lazy on-disk resource.

# %%
output_dir = work_dir / "outputs"
dti_path = output_dir / "dti.nii.gz"
fa_path = output_dir / "fa.nii.gz"
md_path = output_dir / "md.nii.gz"

write_volume(dti.volume, dti_path)
write_volume(fa, fa_path)
write_volume(md, md_path)

print(f"Wrote DTI tensor image: {dti_path.name}")
print(f"Wrote scalar maps: {fa_path.name}, {md_path.name}")

# %% [markdown]
# ## CLI equivalent
#
# The same one-step workflow is available from the command line:
#
# ```bash
# kwneuro dwi dti \
#   --dwi sub-01_dwi.nii.gz \
#   --bval sub-01_dwi.bval \
#   --bvec sub-01_dwi.bvec \
#   --out-dti dti.nii.gz \
#   --out-fa fa.nii.gz \
#   --out-md md.nii.gz
# ```
#
# If the `.bval` and `.bvec` files share the DWI basename, the sidecar options
# can be omitted:
#
# ```bash
# kwneuro dwi dti \
#   --dwi sub-01_dwi.nii.gz \
#   --out-dti dti.nii.gz \
#   --out-fa fa.nii.gz \
#   --out-md md.nii.gz
# ```

# %% tags=["remove-cell"]
tmpdir.cleanup()
