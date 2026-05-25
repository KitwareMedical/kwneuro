# Custom Cached Pipeline Step

This notebook shows how to add a small project-specific step without changing
`kwneuro` itself. The step accepts a `VolumeResource`, returns an
`InMemoryVolumeResource`, and can use the same `Cache` context as built-in
pipeline stages.

## Create a synthetic volume


```python
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
```

    Input shape: (6, 6, 6)


## Define a custom step

The function below is deliberately small: it loads the resource once,
computes a binary high-intensity mask, then preserves spatial metadata using
`update_volume_metadata()`.


```python
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
```

    Mask voxels: 78


## Use the pipeline cache

Outside a `Cache` context, `@cacheable` has no effect. Inside a context, the
first call writes the result to disk, and later calls with the same inputs
load the saved output.


```python
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
```

    {'high_intensity_mask': True}
    Executions after first cached call: 1
    Executions after cache hit: 1
    Cached output matches: True
    Executions after forced recompute: 2


Changing a scalar parameter also changes the cache fingerprint, so the next
call recomputes and stores a new result.


```python
with Cache(cache_dir):
    lower_threshold_mask = high_intensity_mask(volume, threshold=0.4)

print(f"Executions after parameter change: {CALL_COUNT}")
print(f"Lower-threshold voxels: {int(lower_threshold_mask.get_array().sum())}")
```

    Executions after parameter change: 3
    Lower-threshold voxels: 149

