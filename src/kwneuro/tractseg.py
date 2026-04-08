from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from kwneuro.csd import combine_csd_peaks_to_vector_volume, compute_csd_peaks
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
) -> VolumeResource:
    """Run TractSeg on a DWI dataset to segment white matter tracts.

    Args:
        dwi: The Diffusion Weighted Imaging (DWI) dataset.
        mask: A binary brain mask volume.
        response (Optional): The single-fiber response function. If `None`, the response function is estimated using an ROI in the center of the brain mask.
        output_type: TractSeg can segment not only bundles, but also the end regions of bundles.
            Moreover it can create Tract Orientation Maps (TOM).
            'tract_segmentation' [DEFAULT]: Segmentation of bundles (72 bundles).
            'endings_segmentation': Segmentation of bundle end regions (72 bundles).
            'TOM': Tract Orientation Maps (20 bundles).

    Returns: A volume resource containing a 4D numpy array with the output of tractseg
        for tract_segmentation:     [x, y, z, nr_of_bundles]
        for endings_segmentation:   [x, y, z, 2*nr_of_bundles]
        for TOM:                    [x, y, z, 3*nr_of_bundles]
    """

    # Compute CSD peaks
    csd_peaks = compute_csd_peaks(
        dwi,
        mask,
        response=response,
        flip_bvecs_x=True,
        n_peaks=3,  # MRtrix3 uses 3 peaks
    )
    csd_peaks_vector = combine_csd_peaks_to_vector_volume(
        csd_peaks_dirs=csd_peaks[0], csd_peaks_values=csd_peaks[1]
    )
    # Run TractSeg
    logging.info("Running tractseg...")
    segmentation = _call_tractseg(
        data=csd_peaks_vector.get_array(), output_type=output_type
    )

    return create_estimate_volume_resource(
        array=segmentation,
        reference_volume=dwi.volume,
        intent_name="TRACTSEG",
    )
