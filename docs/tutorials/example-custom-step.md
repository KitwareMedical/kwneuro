# Add a Custom Step to a kwneuro Pipeline

This notebook is for projects that use `kwneuro` as the main pipeline driver
and need to add a project-specific step without changing `kwneuro` itself.
The step accepts a `VolumeResource`, returns an `InMemoryVolumeResource`, and
can use the same `Cache` context as built-in pipeline stages.

## Create a synthetic volume


```python
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from kwneuro.cache import Cache, cacheable
from kwneuro.external import temporary_volume_file
from kwneuro.files import read_volume
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


## Interoperate with file-based external tools

Some tools only accept filesystem paths. `kwneuro.external` provides temporary
file helpers for those boundaries: write a fresh copy, call the external tool
while the context is open, then explicitly re-enter `kwneuro` with the file
helpers.


```python
def external_threshold_tool(input_path: Path, output_path: Path) -> None:
    image = nib.load(input_path)
    data = image.get_fdata()
    thresholded = (data > data.mean()).astype(np.uint8)
    nib.save(nib.Nifti1Image(thresholded, image.affine, image.header), output_path)


with temporary_volume_file(volume) as input_path:
    external_output_path = input_path.with_name("external_mask.nii.gz")
    external_threshold_tool(input_path, external_output_path)
    external_mask = read_volume(external_output_path).load()

print(f"External-tool mask voxels: {int(external_mask.get_array().sum())}")
print(f"Temporary input cleaned up: {not input_path.exists()}")
```

    External-tool mask voxels: 102
    Temporary input cleaned up: True

