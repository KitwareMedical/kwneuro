"""Common utility functions"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from nibabel.nifti1 import Nifti1Header
from numpy.typing import NDArray

from kwneuro.resource import InMemoryVolumeResource, VolumeResource

PathLike = Path | str


def normalize_path(path_input: PathLike) -> Path:
    """
    Accepts a path as a string or Path object, expands the tilde (~),
    and returns a resolved, absolute Path object.
    """
    return Path(path_input).expanduser().resolve()


def ensure_output_dir(path: PathLike) -> Path:
    """Normalize *path*, assert it is a directory (not a file), create it, and return it."""
    resolved = normalize_path(path)
    if resolved.exists() and not resolved.is_dir():
        msg = "`path` should be the desired save directory"
        raise ValueError(msg)
    resolved.mkdir(exist_ok=True, parents=True)
    return resolved


def create_estimate_volume_resource(
    array: NDArray[Any], reference_volume: VolumeResource, intent_name: str
) -> VolumeResource:
    """
    Creates an InMemoryVolumeResource from a numpy array of scalar estimates,
    using the affine and metadata of a reference volume.
    """
    return InMemoryVolumeResource(
        array=array,
        affine=reference_volume.get_affine(),
        metadata=update_volume_metadata(
            reference_volume.get_metadata(),
            array,
            intent_code="NIFTI_INTENT_ESTIMATE",
            intent_name=intent_name,
        ),
    )


def update_volume_metadata(
    metadata: dict[str, Any],
    volume_data_array: np.ndarray,
    intent_code: int | str | None = None,
    intent_params: Any = (),
    intent_name: str = "",
) -> dict[str, Any]:
    """Use the convenience of nibabel's header class to update volume metadata.
    If intent_code is not provided then we don't modify the intent parameters.
    """
    header = Nifti1Header()  # convert to a nibabel header in order to get convenience functions like set_data_shape
    for key, val in metadata.items():
        header[key] = val
    header.set_data_dtype(volume_data_array.dtype)
    header.set_data_shape(volume_data_array.shape)
    if intent_code is not None:
        header.set_intent(intent_code, intent_params, intent_name)
    return dict(header)


def subsample_volume(volume: VolumeResource, factor: int = 2) -> InMemoryVolumeResource:
    """Spatially subsample a volume by taking every Nth voxel along each spatial axis.

    This is stride-based subsampling (not interpolated resampling).  It is mainly
    useful for quickly reducing resolution in demo notebooks so that expensive
    downstream steps run faster.

    Args:
        volume: The volume to subsample.
        factor: Take every *factor*-th voxel along each of the first three
            (spatial) dimensions.  Extra dimensions are left untouched.

    Returns:
        A new InMemoryVolumeResource with the subsampled data and an
        updated affine that reflects the coarser voxel grid.
    """
    loaded = volume.load()
    arr = loaded.get_array()
    affine = loaded.get_affine()

    slices: tuple[slice, ...] = (slice(None, None, factor),) * 3
    if arr.ndim > 3:
        slices = slices + (slice(None),) * (arr.ndim - 3)
    new_arr = arr[slices]

    new_affine = affine.copy()
    new_affine[:3, :3] *= factor

    return InMemoryVolumeResource(
        array=new_arr,
        affine=new_affine,
        metadata=update_volume_metadata(loaded.get_metadata(), new_arr),
    )


def deep_equal_allclose(obj1: Any, obj2: Any) -> bool:
    """
    Recursively compares two objects, including nested lists, tuples,
    and dicts. Uses np.allclose for numpy arrays. NaN's are considered equal.
    """
    if type(obj1) is not type(obj2):
        return False

    if isinstance(obj1, np.ndarray):
        if obj1.shape != obj2.shape:
            return False
        if obj1.dtype != obj2.dtype:
            return False
        # if the arrays are numeric, compare them with a tolerance.
        if np.issubdtype(obj1.dtype, np.number) and np.issubdtype(
            obj2.dtype, np.number
        ):
            return bool(np.allclose(obj1, obj2, equal_nan=True))
        # Otherwise (e.g., for string arrays), require exact equality.
        return bool(np.array_equal(obj1, obj2))

    if isinstance(obj1, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(deep_equal_allclose(obj1[k], obj2[k]) for k in obj1)

    if isinstance(obj1, list | tuple):
        if len(obj1) != len(obj2):
            return False
        return all(
            deep_equal_allclose(item1, item2)
            for item1, item2 in zip(obj1, obj2, strict=False)
        )

    # catch the case of two single python or numpy NaNs (here mypy seems to have an issue with the isinstance, thinking it's always true)
    if isinstance(obj1, float | np.floating) and np.isnan(obj1) and np.isnan(obj2):
        return True

    # for all other types (int, str, etc.)
    return bool(obj1 == obj2)
