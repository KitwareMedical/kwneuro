from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import (
    ConstrainedSphericalDeconvModel,
    mask_for_response_ssst,
    response_from_mask_ssst,
)
from dipy.reconst.shm import convert_sh_descoteaux_tournier

from kwneuro.cache import CacheSpec, cacheable
from kwneuro.io import NiftiVolumeResource
from kwneuro.resource import (
    InMemoryResponseFunctionResource,
    ResponseFunctionResource,
    VolumeResource,
)
from kwneuro.util import create_estimate_volume_resource

if TYPE_CHECKING:
    from kwneuro.dwi import Dwi


def combine_response_functions(
    responses: list[ResponseFunctionResource],
) -> InMemoryResponseFunctionResource:
    """
    Combines multiple response functions into a single group average using the MRtrix3 approach.

    This function harmonizes subjects by scaling their SH coefficients so that the
    isotropic component (L=0) of each subject matches the group's mean L=0 intensity.
    This ensures that differences in scanner gain or global signal intensity do not
    bias the shape of the resulting group-average response.

    Args:
        responses: List of ResponseFunctionResource objects to be combined.
    Returns: An InMemoryResponseFunctionResource containing the combined group-average response function.
    """

    if not responses:
        error_msg = "The list of responses cannot be empty."
        raise ValueError(error_msg)

    # Loads responses if not in memory
    responses = [r.load() for r in responses]

    # Get all coefficients and average signals
    num_coeffs = len(responses[0].get()[0])
    all_coeffs = []
    all_avg_signals = []
    for r in responses:
        sh_coeffs, avg_signal = r.get()
        if len(sh_coeffs) != num_coeffs:
            error_msg = "All response functions must have the same number of SH coefficients to be combined."
            raise ValueError(error_msg)
        all_coeffs.append(sh_coeffs)
        all_avg_signals.append(avg_signal)
    all_coeffs = np.array(all_coeffs)

    # Based on the approach in MRtrix3 responsemean, we scale each subject's response function
    # so that the L=0 coefficient matches the group average L=0 coefficient.
    # Average L = 0 across subjects to compute scaling factors
    target_l0 = np.mean(all_coeffs[:, 0])
    if np.any(all_coeffs[:, 0] == 0):
        error_msg = "Response function with zero L=0 coefficient cannot be combined."
        raise ValueError(error_msg)
    multipliers = target_l0 / all_coeffs[:, 0]

    # Scale each subject's entire row
    normalized_coeffs = all_coeffs * multipliers[:, np.newaxis]

    # Average coefficients across subjects
    group_sh_coeffs = np.mean(normalized_coeffs, axis=0)

    # Average the S0 signal
    avg_s0 = np.mean(all_avg_signals)

    return InMemoryResponseFunctionResource(
        sh_coeffs=group_sh_coeffs, avg_signal=avg_s0
    )


@cacheable
def estimate_response_function(
    dwi: Dwi,
    mask: VolumeResource,
    flip_bvecs_x: bool = True,
    fa_thr: float = 0.8,
    sh_order_max: int = 8,
) -> InMemoryResponseFunctionResource:
    """Estimate the single-shell single-tissue response function from a DWI dataset using the SSST method.
    Args:
        dwi: The Diffusion Weighted Imaging (DWI) dataset.
        mask: A binary brain mask volume. This is used to extract an ROI at the center of the brain.
        flip_bvecs_x: Whether to flip the x-component of the b-vectors to match MRtrix3 convention.
        fa_thr: FA threshold for calculating the response function.
        sh_order_max: Maximum spherical harmonic order to use for the basis model. Default is 8.
    Returns: A resource containing the estimated single-tissue response function.
    """

    # Load data as numpy arrays
    bvals = dwi.bval.get()
    bvecs = dwi.bvec.get()
    volume_data = dwi.volume.get_array()
    mask_data = mask.get_array().astype(int)

    bvecs_mod = bvecs.copy()
    if flip_bvecs_x:
        bvecs_mod[:, 0] = -bvecs_mod[:, 0]

    # b-values above 1200 aren't great for DTI estimation. dipy uses DTI to model response functions.
    low_b_mask = bvals <= 1200
    gtab_low_b = gradient_table(bvals[low_b_mask], bvecs=bvecs_mod[low_b_mask])
    data_low_b = volume_data[..., low_b_mask]

    # determine roi center based on brain mask (aiming for corpus callosum)
    i, j, k = np.where(mask_data)
    roi_center = np.round([i.mean(), j.mean(), k.mean()]).astype(int)  # COM of mask
    roi_radii = [(indices.max() - indices.min()) // 4 for indices in (i, j, k)]

    # get mask of voxels to use for response estimate
    mask_for_response = mask_for_response_ssst(
        gtab_low_b,
        data_low_b,
        roi_center=roi_center,
        roi_radii=roi_radii,
        fa_thr=fa_thr,
    )
    mask_for_response *= mask_data  # ensure we stay inside brain mask (almost certainly we already were but just in case)

    if mask_for_response.sum() < 100:
        logging.warning(
            "There are less than 100 voxels in the mask used to estimate the response function."
        )
        # If this happens maybe we need to decrease fa_thr or check what might be the issue

    # perform response function estimate using the selected voxels
    response, ratio = response_from_mask_ssst(
        gtab_low_b,
        data_low_b,
        mask_for_response,
    )

    if ratio > 0.3:
        logging.warning(
            "Ratio of response diffusion tensor eigenvalues is greater than 0.3. For a response function we expect more prolateness. Something may be wrong."
        )

    return InMemoryResponseFunctionResource.from_prolate_tensor(
        response, gtab=gtab_low_b, sh_order_max=sh_order_max
    )


def compute_csd_fods(
    dwi: Dwi,
    mask: VolumeResource,
    response: ResponseFunctionResource | None = None,
    flip_bvecs_x: bool = True,
    mrtrix_format: bool = False,
    sh_order_max: int = 8,
) -> np.ndarray:
    """Computes Fiber Orientation Distributions (FODs) from a DWI dataset using Constrained Spherical Deconvolution (CSD).

    Args:
        dwi: The Diffusion Weighted Imaging (DWI) dataset.
        mask: A binary brain mask volume. FODs are computed only within this mask.
        response (Optional): The single-fiber response function. If `None`, the response function is estimated using an ROI in the center of the brain mask.
        flip_bvecs_x: Whether to flip the x-component of the b-vectors to match MRtrix3 convention. Default is True.
        mrtrix_format: If True, converts SH coefficients between legacy-descoteaux07 and tournier07.
        sh_order_max: Maximum spherical harmonic order to use in the CSD model. Default is 8.
    Returns: Array containing the spherical harmonic coefficients of the obtained FODs.
    """

    # Load data as numpy arrays
    bvals = dwi.bval.get()
    bvecs = dwi.bvec.get()
    volume_data = dwi.volume.get_array()
    mask_data = mask.get_array().astype(int)

    bvecs_mod = bvecs.copy()
    if flip_bvecs_x:
        bvecs_mod[:, 0] = -bvecs_mod[:, 0]

    if not response:
        logging.info("Estimating single-shell single-tissue response function...")
        response = estimate_response_function(
            dwi=dwi, mask=mask, flip_bvecs_x=flip_bvecs_x
        )

    gtab = gradient_table(bvals, bvecs=bvecs_mod)
    csd_model = ConstrainedSphericalDeconvModel(
        gtab, response.get_dipy_object(), sh_order_max=sh_order_max
    )
    csd_fit = csd_model.fit(volume_data, mask=mask_data)

    coeffs = csd_fit.shm_coeff

    if mrtrix_format:
        coeffs = convert_sh_descoteaux_tournier(csd_fit.shm_coeff)

    return coeffs


@cacheable(  # type: ignore[untyped-decorator]
    CacheSpec(
        files=["csd_peak_dirs.nii.gz", "csd_peak_values.nii.gz"],
        save=lambda result, d: (  # type: ignore[arg-type]
            NiftiVolumeResource.save(result[0], d / "csd_peak_dirs.nii.gz"),
            NiftiVolumeResource.save(result[1], d / "csd_peak_values.nii.gz"),
        ),
        load=lambda d: (
            NiftiVolumeResource(d / "csd_peak_dirs.nii.gz"),
            NiftiVolumeResource(d / "csd_peak_values.nii.gz"),
        ),
    )
)
def compute_csd_peaks(
    dwi: Dwi,
    mask: VolumeResource,
    response: ResponseFunctionResource | None = None,
    flip_bvecs_x: bool = True,
    n_peaks: int = 5,
    relative_peak_threshold: float = 0.5,
    min_separation_angle: float = 25,
) -> tuple[VolumeResource, VolumeResource]:
    """
    Compute Constrained Spherical Deconvolution peaks from a DWI resource. This involves
    estimating the response function, fitting the CSD model, and extracting the peaks.
    Args:
        dwi: The Diffusion Weighted Imaging (DWI) dataset.
        mask: A binary brain mask volume. CSD is computed only within this mask.
        response (Optional): The single-fiber response function. If `None`, the response function is estimated using an ROI in the center of the brain mask.
        flip_bvecs_x: Whether to flip the x-component of the b-vectors to match MRtrix3 convention.
        n_peaks: Number of peaks to extract per voxel. Default is 5.
        relative_peak_threshold: Only return peaks greater than relative_peak_threshold * m where m is the largest peak.
        min_separation_angle: The minimum distance between directions. If two peaks are too close only the larger of the two is
        returned. Must be in range [0,90]
    Returns: A tuple of VolumeResources containing the CSD peak directions stored as a 5-D array of shape [x,y,z,n_peaks,3],
    and the corresponding peak values stored as a 4D array of shape [x,y,z,n_peaks].
    """
    # Load data as numpy arrays
    bvals = dwi.bval.get()
    bvecs = dwi.bvec.get()
    volume_data = dwi.volume.get_array()
    mask_data = mask.get_array().astype(int)

    bvecs_mod = bvecs.copy()
    if flip_bvecs_x:
        bvecs_mod[:, 0] = -bvecs_mod[:, 0]  # flip x axis to match MRtrix3 convention

    if not response:
        logging.info("Estimating single-shell single-tissue response function...")
        response = estimate_response_function(
            dwi=dwi, mask=mask, flip_bvecs_x=flip_bvecs_x
        )

    gtab = gradient_table(bvals, bvecs=bvecs_mod)
    csd_model = ConstrainedSphericalDeconvModel(
        gtab, response.get_dipy_object(), sh_order_max=8
    )

    logging.info("Computing peaks...")
    csd_peaks = peaks_from_model(
        model=csd_model,
        data=volume_data,
        mask=mask_data,
        sphere=default_sphere,
        relative_peak_threshold=relative_peak_threshold,
        min_separation_angle=min_separation_angle,  # minimum angular distance required between any two peaks. Filtering step after peaks are found
        npeaks=n_peaks,
    )

    return (
        create_estimate_volume_resource(
            array=csd_peaks.peak_dirs,
            reference_volume=dwi.volume,
            intent_name="CSD_PEAK_DIRS",
        ),
        create_estimate_volume_resource(
            array=csd_peaks.peak_values,
            reference_volume=dwi.volume,
            intent_name="CSD_PEAK_VALS",
        ),
    )


def combine_csd_peaks_to_vector_volume(
    csd_peaks_dirs: VolumeResource, csd_peaks_values: VolumeResource
) -> VolumeResource:
    """
    Convert CSD peaks from separate direction and value VolumeResources (e.g. Dipy format) into a single volume
    containing a 4D array, where the last dimension is a n_peaks*3-length vector containing
    the x,y,z components of each peak (e.g. MRtrix3 format). The magnitude of each vector (peak value) is stored as its length (norm).
    Args:
        csd_peaks_dirs: VolumeResource containing peak directions (unit vectors).
        csd_peaks_values: VolumeResource containing peak values (amplitudes).
    Returns: VolumeResource with 4D array of shape [x,y,z,n_peaks*3].
    """
    dirs = csd_peaks_dirs.get_array()
    values = csd_peaks_values.get_array()

    values_expanded = np.expand_dims(values, axis=-1)
    csd_peaks = dirs * values_expanded
    spatial_dims = csd_peaks.shape[:-2]
    n_peak_dims = csd_peaks.shape[-2] * 3
    peaks_array = csd_peaks.reshape(*spatial_dims, n_peak_dims)

    return create_estimate_volume_resource(
        array=peaks_array,
        reference_volume=csd_peaks_dirs,
        intent_name="CSD_PEAK_VECTORS",
    )
