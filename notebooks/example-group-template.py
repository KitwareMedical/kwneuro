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
# # Population Template Construction
#
# This notebook demonstrates how to build population-level templates from
# individual subject volumes using the `kwneuro` registration tools.
#
# ## Pipeline overview
#
# 0. Download example data (3 subjects from OpenNeuro ds000221)
# 1. Load per-subject DWI data and estimate FA / MD via DTI
# 2. Build a single-metric population template
# 3. Build a multi-metric population template
# 4. Save templates to disk

# %% [markdown]
# ### Spatial subsampling
#
# Configure the following to spatially subsample the data and run through the demo more quickly at lower spatial resolution.

# %%
SUBSAMPLE = True
SUBSAMPLE_FACTOR = 2

# %% [markdown]
# ## 0. Download example data
#
# We download 3 subjects from the MPI-Leipzig Mind-Brain-Body dataset
# ([OpenNeuro ds000221](https://openneuro.org/datasets/ds000221)). This dataset
# contains 64-direction single-shell DWI data (b ~ 1000 s/mm²), which is
# sufficient for DTI-based template construction.
#
# Total download size: ~250 MB for 3 subjects.

# %%
from pathlib import Path

import openneuro as on

# %%
DATA_DIR = Path("example_data/ds000221")
SUBJECTS = ["sub-010002", "sub-010005", "sub-010006"]

# Download if not already present
if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
    print("Downloading example data from OpenNeuro ds000221...")
    print("This may take a few minutes (~250 MB total).")
    on.download(
        dataset="ds000221",
        target_dir=str(DATA_DIR),
        include=[
            "dataset_description.json",
            "participants.tsv",
        ] + [f"{s}/ses-01/dwi/*" for s in SUBJECTS],
    )
    print("Download complete!")
else:
    print(f"Using existing data at {DATA_DIR.resolve()}")

# %% [markdown]
# ## 1. Load DWI data and compute per-subject FA / MD
#
# We construct `Dwi` objects from each subject's DWI files, fit a diffusion
# tensor, and extract FA and MD maps. These scalar maps are then used for
# template construction.

# %%
import matplotlib.pyplot as plt
import numpy as np
import tempfile

from kwneuro.dwi import Dwi
from kwneuro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource
from kwneuro.masks import brain_extract_batch
from kwneuro.resource import VolumeResource

# Load all DWIs from the downloaded data
dwis: list[Dwi] = []
for subject in SUBJECTS:
    dwi_dir = DATA_DIR / subject / "ses-01" / "dwi"
    dwi_nifti = next(dwi_dir.glob("*_dwi.nii.gz"))
    basename = dwi_nifti.name.removesuffix(".nii.gz")

    dwis.append(Dwi(
        NiftiVolumeResource(dwi_dir / f"{basename}.nii.gz"),
        FslBvalResource(dwi_dir / f"{basename}.bval"),
        FslBvecResource(dwi_dir / f"{basename}.bvec"),
    ).load())

print(f"Loaded {len(dwis)} subjects")

if SUBSAMPLE:
    from kwneuro.dwi import subsample_dwi

    dwis = [subsample_dwi(d, SUBSAMPLE_FACTOR) for d in dwis]
    print(f"Subsampled by factor {SUBSAMPLE_FACTOR}")

# Brain extraction in one batch (HD-BET initializes once)
with tempfile.TemporaryDirectory() as tmpdir:
    cases = [(dwi, Path(tmpdir) / f"mask_{i}.nii.gz") for i, dwi in enumerate(dwis)]
    # Load masks into memory before the temp directory is cleaned up
    masks = [m.load() for m in brain_extract_batch(cases)]

# Estimate DTI and extract FA/MD
fa_volumes: list[VolumeResource] = []
md_volumes: list[VolumeResource] = []

for dwi, mask in zip(dwis, masks):
    dti = dwi.estimate_dti(mask=mask)
    fa_vol, md_vol = dti.get_fa_md()
    fa_volumes.append(fa_vol)
    md_volumes.append(md_vol)

# %%
# (This cell fixes a few notebook output issues caused by the HD-BET masking step above)

# %matplotlib inline

import sys
from IPython import get_ipython

kernel = get_ipython().kernel
sys.stdout = kernel._stdout

# %% [markdown]
# Preview a slice of the individual FA and MD maps.

# %%
print(f"Computed FA/MD for {len(fa_volumes)} subjects")

n_show = min(len(fa_volumes), 4)
fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
if n_show == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    arr = fa_volumes[i].get_array()
    mid = arr.shape[2] // 2
    ax.imshow(arr[:, :, mid].T, cmap="hot", origin="lower", vmin=0, vmax=1)
    ax.set_title(f"Subject {i + 1}")
    ax.axis("off")
plt.suptitle("Individual FA maps")
plt.tight_layout()
plt.show()

n_show = min(len(md_volumes), 4)
fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
if n_show == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    arr = md_volumes[i].get_array()
    mid = arr.shape[2] // 2
    ax.imshow(arr[:, :, mid].T, cmap="viridis", origin="lower", vmin=0, vmax=3e-3)
    ax.set_title(f"Subject {i + 1}")
    ax.axis("off")
plt.suptitle("Individual MD maps")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Single-metric population template
#
# `build_template` constructs an unbiased group-wise mean template via
# iterative ANTs SyN registration. Each iteration:
#
# 1. Registers every subject to the current template
# 2. Averages the warped images
# 3. Corrects for mean shape bias using the inverse average transform
# 4. Sharpens the result
#
# The initial template is the simple voxel-wise average (no registration).

# %%
from kwneuro.build_template import average_volumes, build_template

initial_avg = average_volumes(fa_volumes)

# %%
fa_template_1it = build_template(fa_volumes, initial_template=initial_avg, iterations=1)

# %%
fa_template_4it = build_template(fa_volumes, initial_template=fa_template_1it, iterations=3)

# %% [markdown]
# Compare the naive average with the registration-based template.

# %%
avg_arr = initial_avg.get_array()
tmpl_arr_1it = fa_template_1it.get_array()
tmpl_arr_4it = fa_template_4it.get_array()
mid = avg_arr.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(avg_arr[:, :, mid].T, cmap="hot", origin="lower", vmin=0)
axes[0].set_title("Simple average (no registration)")
axes[1].imshow(tmpl_arr_1it[:, :, mid].T, cmap="hot", origin="lower", vmin=0, vmax=1)
axes[1].set_title("Population template (1 iteration)")
axes[2].imshow(tmpl_arr_4it[:, :, mid].T, cmap="hot", origin="lower", vmin=0, vmax=1)
axes[2].set_title("Population template (4 iterations)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Multi-metric population template
#
# `build_multi_metric_template` uses multiple modalities simultaneously to
# drive the registration. The first modality is used for the affine step;
# all modalities contribute to the deformable (SyN) step.
#
# Each subject is passed as a dictionary mapping modality names to volumes.

# %%
from kwneuro.build_template import build_multi_metric_template

subject_list = []
for fa, md in zip(fa_volumes, md_volumes):
    subject_list.append({"FA": fa, "MD": md})

# %%
multi_template_1it = build_multi_metric_template(
    subject_list,
    weights={"FA": 1.0, "MD": 1.0},
    iterations=1,
)

# %%
multi_template_4it = build_multi_metric_template(
    subject_list,
    weights={"FA": 1.0, "MD": 1.0},
    iterations=3,
    initial_template=multi_template_1it,
)

# %% [markdown]
# Visualise the multi-metric template for each modality.

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

fa_tmpl_1it = multi_template_1it["FA"].get_array()
md_tmpl_1it = multi_template_1it["MD"].get_array()
fa_tmpl_4it = multi_template_4it["FA"].get_array()
md_tmpl_4it = multi_template_4it["MD"].get_array()
mid = fa_tmpl_1it.shape[2] // 2

im0 = axes[0,0].imshow(fa_tmpl_1it[:, :, mid].T, cmap="hot", origin="lower", vmin=0, vmax=1)
axes[0,0].set_title("FA template (1 iteration)")
plt.colorbar(im0, ax=axes[0,0], fraction=0.046)

im1 = axes[0,1].imshow(md_tmpl_1it[:, :, mid].T, cmap="viridis", origin="lower", vmin=0, vmax=3e-3)
axes[0,1].set_title("MD template (1 iteration)")
plt.colorbar(im1, ax=axes[0,1], fraction=0.046)

im2 = axes[1,0].imshow(fa_tmpl_4it[:, :, mid].T, cmap="hot", origin="lower", vmin=0, vmax=1)
axes[1,0].set_title("FA template (4 iterations)")
plt.colorbar(im2, ax=axes[1,0], fraction=0.046)

im3 = axes[1,1].imshow(md_tmpl_4it[:, :, mid].T, cmap="viridis", origin="lower", vmin=0, vmax=3e-3)
axes[1,1].set_title("MD template (4 iterations)")
plt.colorbar(im3, ax=axes[1,1], fraction=0.046)

for ax in axes.flatten():
    ax.axis("off")
plt.suptitle("Multi-metric population templates")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Save templates to disk

# %%
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

NiftiVolumeResource.save(fa_template_4it, output_dir / "fa_template.nii.gz")

for name, vol in multi_template_4it.items():
    NiftiVolumeResource.save(vol, output_dir / f"{name.lower()}_template_multi.nii.gz")

print(f"Templates saved to {output_dir.resolve()}")
