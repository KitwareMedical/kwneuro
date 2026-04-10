from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from kwneuro.cache import _active_cache, _compute_fingerprint, _save_params
from kwneuro.csd import combine_csd_peaks_to_vector_volume, compute_csd_peaks
from kwneuro.io import NiftiVolumeResource
from kwneuro.resource import ResponseFunctionResource, VolumeResource
from kwneuro.util import create_estimate_volume_resource

if TYPE_CHECKING:
    import numpy as np

    from kwneuro.dwi import Dwi


def _call_tractseg(data: np.ndarray, output_type: str) -> np.ndarray:
    """Call TractSeg, handling the lazy import."""
    try:
        from tractseg.python_api import run_tractseg
    except ImportError:
        msg = (
            "TractSeg is required for tract segmentation but is not installed. "
            "Install it with: pip install kwneuro[tractseg]"
        )
        raise ImportError(msg) from None
    return run_tractseg(data=data, output_type=output_type)


def extract_tractseg(
    dwi: Dwi,
    mask: VolumeResource,
    response: ResponseFunctionResource | None = None,
    output_type: str = "tract_segmentation",
    csd_peaks: tuple[VolumeResource, VolumeResource] | None = None,
) -> VolumeResource:
    """Run TractSeg on a DWI dataset to segment white matter tracts.

    Args:
        dwi: The Diffusion Weighted Imaging (DWI) dataset.
        mask: A binary brain mask volume.
        response (Optional): The single-fiber response function. If `None`, the
            response function is estimated automatically. Ignored when
            ``csd_peaks`` is provided.
        output_type: TractSeg can segment not only bundles, but also the end regions of bundles.
            Moreover it can create Tract Orientation Maps (TOM).
            'tract_segmentation' [DEFAULT]: Segmentation of bundles (72 bundles).
            'endings_segmentation': Segmentation of bundle end regions (72 bundles).
            'TOM': Tract Orientation Maps (20 bundles).
        csd_peaks (Optional): Pre-computed CSD peaks as a ``(directions, values)`` tuple
            of VolumeResources. When provided the CSD computation step is skipped.

    Returns: A volume resource containing a 4D numpy array with the output of tractseg
        for tract_segmentation:     [x, y, z, nr_of_bundles]
        for endings_segmentation:   [x, y, z, 2*nr_of_bundles]
        for TOM:                    [x, y, z, 3*nr_of_bundles]
    """
    # Inline cache check — uses a dynamic filename so a decorator cannot be used.
    _cache = _active_cache.get()
    _cache_file = f"tractseg_{output_type}.nii.gz"
    _scalars: dict[str, Any] | None = None
    _hashes: dict[str, str] | None = None
    if _cache is not None:
        _scalars = {"output_type": output_type}
        h: dict[str, str] = {}
        for name, val in [
            ("dwi", dwi),
            ("mask", mask),
            ("response", response),
            ("csd_peaks", csd_peaks),
        ]:
            fp = _compute_fingerprint(val)
            if fp is not None:
                h[name] = fp
        _hashes = h or None
        if _cache.is_cached("extract_tractseg", [_cache_file], _scalars, _hashes):
            logging.info("extract_tractseg: cache hit (%s)", output_type)
            return NiftiVolumeResource(_cache.cache_dir / _cache_file)

    if csd_peaks is None:
        # Compute CSD peaks (MRtrix3 convention: 3 peaks).
        # compute_csd_peaks is @cacheable, so it uses _cache automatically.
        csd_peaks = compute_csd_peaks(
            dwi,
            mask,
            response=response,
            flip_bvecs_x=True,
            n_peaks=3,
        )

    csd_peaks_vector = combine_csd_peaks_to_vector_volume(
        csd_peaks_dirs=csd_peaks[0], csd_peaks_values=csd_peaks[1]
    )
    logging.info("Running tractseg...")
    segmentation = _call_tractseg(
        data=csd_peaks_vector.get_array(), output_type=output_type
    )

    result = create_estimate_volume_resource(
        array=segmentation,
        reference_volume=dwi.volume,
        intent_name="TRACTSEG",
    )

    if _cache is not None:
        logging.info("extract_tractseg: saving cache (%s)", output_type)
        _save_params(_cache.cache_dir, "extract_tractseg", _scalars, _hashes)
        return NiftiVolumeResource.save(result, _cache.cache_dir / _cache_file)

    return result
