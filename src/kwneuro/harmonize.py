"""ComBat harmonization for multi-site scalar brain volumes."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from kwneuro.resource import InMemoryVolumeResource, VolumeResource
from kwneuro.util import update_volume_metadata


@dataclass
class CombatEstimates:
    """ComBat parameter estimates from a harmonization run.

    These estimates can be used to inspect the fitted model or, in the future,
    to harmonize new data using the same model via ``neuroCombatFromTraining``.
    """

    estimates: dict[str, Any]
    """Empirical Bayes estimates of batch effects (gamma/delta parameters,
    pooled variance, etc.)."""

    info: dict[str, Any]
    """Metadata about the harmonization (batch levels, sample counts,
    design matrix, etc.)."""


def _flatten_volumes(
    loaded_volumes: list[InMemoryVolumeResource],
    mask_array: NDArray[np.number[Any]],
) -> tuple[NDArray[np.floating[Any]], tuple[NDArray[np.intp], ...]]:
    """Flatten 3D volumes into a 2D (n_voxels, n_samples) array for neuroCombat.

    Returns the data matrix and the mask indices for later reconstruction.
    """
    mask_indices = np.nonzero(mask_array > 0)
    n_voxels = mask_indices[0].shape[0]
    n_samples = len(loaded_volumes)

    dat = np.empty((n_voxels, n_samples), dtype=np.float64)
    for i, vol in enumerate(loaded_volumes):
        dat[:, i] = vol.get_array()[mask_indices]

    return dat, mask_indices


def _unflatten_to_volumes(
    harmonized_data: NDArray[np.floating[Any]],
    mask_indices: tuple[NDArray[np.intp], ...],
    reference_volumes: list[InMemoryVolumeResource],
    volume_shape: tuple[int, ...],
    *,
    preserve_out_of_mask: bool,
) -> list[InMemoryVolumeResource]:
    """Reshape harmonized 2D data back into 3D VolumeResources."""
    result: list[InMemoryVolumeResource] = []
    for i, ref_vol in enumerate(reference_volumes):
        if preserve_out_of_mask:
            arr = ref_vol.get_array().astype(harmonized_data.dtype, copy=True)
        else:
            arr = np.zeros(volume_shape, dtype=harmonized_data.dtype)
        arr[mask_indices] = harmonized_data[:, i]
        result.append(
            InMemoryVolumeResource(
                array=arr,
                affine=ref_vol.get_affine(),
                metadata=update_volume_metadata(ref_vol.get_metadata(), arr),
            )
        )
    return result


def _validate_inputs(
    loaded_volumes: list[InMemoryVolumeResource],
    covars: pd.DataFrame,
    batch_col: str,
    mask_array: NDArray[np.number[Any]],
    categorical_cols: list[str] | None,
    continuous_cols: list[str] | None,
) -> None:
    """Validate all inputs before running ComBat."""
    if len(loaded_volumes) < 2:
        msg = "At least 2 volumes are required for harmonization."
        raise ValueError(msg)

    volume_shape = loaded_volumes[0].get_array().shape
    for i, vol in enumerate(loaded_volumes):
        arr = vol.get_array()
        if arr.ndim != 3:
            msg = f"All volumes must be 3D. Volume at index {i} has {arr.ndim} dimensions."
            raise ValueError(msg)
        if arr.shape != volume_shape:
            msg = (
                f"All volumes must have the same shape. Expected {volume_shape}, "
                f"but volume at index {i} has shape {arr.shape}."
            )
            raise ValueError(msg)

    if mask_array.shape != volume_shape:
        msg = (
            f"Mask shape {mask_array.shape} does not match volume shape {volume_shape}."
        )
        raise ValueError(msg)

    if np.count_nonzero(mask_array) == 0:
        msg = "Mask is empty (all zeros). Cannot harmonize without brain voxels."
        raise ValueError(msg)

    if len(covars) != len(loaded_volumes):
        msg = (
            f"Number of rows in covars ({len(covars)}) does not match "
            f"number of volumes ({len(loaded_volumes)})."
        )
        raise ValueError(msg)

    if batch_col not in covars.columns:
        msg = f"Batch column '{batch_col}' not found in covars columns: {list(covars.columns)}"
        raise ValueError(msg)

    n_batches = covars[batch_col].nunique()
    if n_batches < 2:
        msg = (
            f"At least 2 batches are required for harmonization, "
            f"but only {n_batches} found in column '{batch_col}'."
        )
        raise ValueError(msg)

    for col_list_name, col_list in [
        ("categorical_cols", categorical_cols),
        ("continuous_cols", continuous_cols),
    ]:
        if col_list is not None:
            for col in col_list:
                if col not in covars.columns:
                    msg = (
                        f"Column '{col}' from {col_list_name} not found in covars "
                        f"columns: {list(covars.columns)}"
                    )
                    raise ValueError(msg)


def harmonize_volumes(
    volumes: Sequence[VolumeResource],
    covars: pd.DataFrame,
    batch_col: str,
    mask: VolumeResource,
    *,
    categorical_cols: list[str] | None = None,
    continuous_cols: list[str] | None = None,
    eb: bool = True,
    parametric: bool = True,
    mean_only: bool = False,
    ref_batch: str | None = None,
    preserve_out_of_mask: bool = False,
) -> tuple[list[InMemoryVolumeResource], CombatEstimates]:
    """Harmonize 3D scalar brain volumes across scanner sites using ComBat.

    All volumes must be in the same voxel space (same shape). A mask selects
    which voxels to harmonize; out-of-mask voxels are zeroed by default or
    preserved from the original volumes if ``preserve_out_of_mask`` is True.

    Args:
        volumes: Sequence of 3D scalar VolumeResource objects, one per
            subject/scan. Must all have identical shape.
        covars: A pandas DataFrame with one row per volume. Must contain at
            least the batch column. Row order must correspond to ``volumes``.
        batch_col: Column name in ``covars`` identifying the scanner/site.
        mask: A 3D VolumeResource. Voxels where mask > 0 are included in
            harmonization. Must have the same shape as the input volumes.
        categorical_cols: Column names in ``covars`` for categorical covariates
            to preserve (e.g. ``["sex"]``).
        continuous_cols: Column names in ``covars`` for continuous covariates
            to preserve (e.g. ``["age"]``).
        eb: Whether to use Empirical Bayes estimation. Default True.
        parametric: Whether to use parametric adjustments. Default True.
        mean_only: Whether to only adjust batch means (not variances).
        ref_batch: Optional reference batch whose data is preserved as-is.
        preserve_out_of_mask: If True, voxels outside the mask retain their
            original values. If False (default), they are set to zero.

    Returns:
        A tuple of (harmonized_volumes, combat_estimates). The first element
        is a list of harmonized InMemoryVolumeResources in the same order as
        the input. The second is a CombatEstimates object containing the
        fitted model parameters.

    Raises:
        ValueError: If inputs are invalid (shape mismatch, missing columns,
            fewer than 2 batches, empty mask, etc.).
    """
    loaded_volumes = [v.load() for v in volumes]
    mask_loaded = mask.load()
    mask_array = mask_loaded.get_array()

    _validate_inputs(
        loaded_volumes, covars, batch_col, mask_array, categorical_cols, continuous_cols
    )

    volume_shape = loaded_volumes[0].get_array().shape
    n_batches = covars[batch_col].nunique()

    logging.info(
        "Running ComBat harmonization on %d volumes across %d batches",
        len(loaded_volumes),
        n_batches,
    )

    try:
        from neuroCombat import neuroCombat as _run_combat
    except ImportError:
        msg = (
            "neuroCombat is required for harmonization but is not installed. "
            "Install it with: pip install kwneuro[combat]"
        )
        raise ImportError(msg) from None

    dat, mask_indices = _flatten_volumes(loaded_volumes, mask_array)

    result = _run_combat(
        dat=dat,
        covars=covars,
        batch_col=batch_col,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        eb=eb,
        parametric=parametric,
        mean_only=mean_only,
        ref_batch=ref_batch,
    )

    harmonized_volumes = _unflatten_to_volumes(
        result["data"],
        mask_indices,
        loaded_volumes,
        volume_shape,
        preserve_out_of_mask=preserve_out_of_mask,
    )

    return harmonized_volumes, CombatEstimates(
        estimates=result["estimates"],
        info=result["info"],
    )
