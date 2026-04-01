from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tractseg.python_api import run_tractseg

from kwneuro.cache import _active_cache
from kwneuro.csd import combine_csd_peaks_to_vector_volume, compute_csd_peaks
from kwneuro.io import NiftiVolumeResource
from kwneuro.resource import ResponseFunctionResource, VolumeResource
from kwneuro.util import create_estimate_volume_resource

if TYPE_CHECKING:
    from kwneuro.dwi import Dwi


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
    if _cache is not None and _cache.is_cached("extract_tractseg", [_cache_file]):
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
    segmentation = run_tractseg(
        data=csd_peaks_vector.get_array(), output_type=output_type
    )

    result = create_estimate_volume_resource(
        array=segmentation,
        reference_volume=dwi.volume,
        intent_name="TRACTSEG",
    )

    if _cache is not None:
        logging.info("extract_tractseg: saving cache (%s)", output_type)
        return NiftiVolumeResource.save(result, _cache.cache_dir / _cache_file)

    return result
