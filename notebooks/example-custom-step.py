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
# # Custom Cached Pipeline Step
#
# This notebook shows how to add a small project-specific step without changing
# `kwneuro` itself. The step accepts a `VolumeResource`, returns an
# `InMemoryVolumeResource`, and can use the same `Cache` context as built-in
# pipeline stages.

# %% [markdown]
# ## Create a synthetic volume

# %%
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from kwneuro.cache import Cache, cacheable
from kwneuro.resource import InMemoryVolumeResource, VolumeResource
from kwneuro.util import update_volume_metadata


def make_synthetic_volume() -> InMemoryVolumeResource:
    x, y, z = np.indices((6, 6, 6))
    data = (x + 2 * y + 3 * z).astype(np.float32)
    data = data / data.max()

    header = nib.Nifti1Header()
    header.set_xyzt_units("mm")

    return InMemoryVolumeResource(
        array=data,
        affine=np.diag([1.5, 1.5, 1.5, 1.0]),
        metadata=dict(header),
    )


volume = make_synthetic_volume()
print(f"Input shape: {volume.get_array().shape}")

# %% [markdown]
# ## Define a custom step
#
# The function below is deliberately small: it loads the resource once,
# computes a binary high-intensity mask, then preserves spatial metadata using
# `update_volume_metadata()`.

# %%
CALL_COUNT = 0


@cacheable
def high_intensity_mask(
    volume: VolumeResource,
    threshold: float = 0.6,
) -> InMemoryVolumeResource:
    global CALL_COUNT
    CALL_COUNT += 1

    loaded = volume.load()
    mask = (loaded.get_array() >= threshold).astype(np.uint8)
    metadata = update_volume_metadata(loaded.get_metadata(), mask)

    return InMemoryVolumeResource(
        array=mask,
        affine=loaded.get_affine(),
        metadata=metadata,
    )


mask = high_intensity_mask(volume)
print(f"Mask voxels: {int(mask.get_array().sum())}")
CALL_COUNT = 0

# %% [markdown]
# ## Use the pipeline cache
#
# Outside a `Cache` context, `@cacheable` has no effect. Inside a context, the
# first call writes the result to disk, and later calls with the same inputs
# load the saved output.

# %%
tmpdir = TemporaryDirectory()
cache_dir = Path(tmpdir.name) / "cache"

with Cache(cache_dir) as cache:
    mask_1 = high_intensity_mask(volume, threshold=0.6)
    print(cache.status([high_intensity_mask]))

print(f"Executions after first cached call: {CALL_COUNT}")

with Cache(cache_dir):
    mask_2 = high_intensity_mask(volume, threshold=0.6)

print(f"Executions after cache hit: {CALL_COUNT}")
print(f"Cached output matches: {np.array_equal(mask_1.get_array(), mask_2.get_array())}")

with Cache(cache_dir, force={"high_intensity_mask"}):
    high_intensity_mask(volume, threshold=0.6)

print(f"Executions after forced recompute: {CALL_COUNT}")

# %% [markdown]
# Changing a scalar parameter also changes the cache fingerprint, so the next
# call recomputes and stores a new result.

# %%
with Cache(cache_dir):
    lower_threshold_mask = high_intensity_mask(volume, threshold=0.4)

print(f"Executions after parameter change: {CALL_COUNT}")
print(f"Lower-threshold voxels: {int(lower_threshold_mask.get_array().sum())}")

# %% tags=["remove-cell"]
tmpdir.cleanup()
