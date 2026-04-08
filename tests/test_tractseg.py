from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from kwneuro.dwi import Dwi
from kwneuro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryResponseFunctionResource,
    InMemoryVolumeResource,
)
from kwneuro.tractseg import extract_tractseg


@pytest.fixture
def affine_random():
    rng = np.random.default_rng(7562)
    affine = np.eye(4, dtype=float)
    affine[:3, 3] = 100 * (
        rng.random(size=(3,), dtype=float) - 0.5
    )  # random translation
    random_3by3 = rng.random(size=(3, 3), dtype=float) - 0.5
    affine[:3, :3] = scipy.linalg.expm(
        random_3by3 - random_3by3.T
    )  # exponentiate a random skew-symmetric matrix to get some orthogonal matrix
    return affine


@pytest.fixture
def dwi_data_small_random(affine_random) -> Dwi:
    rng = np.random.default_rng(1337)
    dwi_data = rng.random(size=(3, 4, 5, 6))
    bvals = np.array([0, 1000, 500, 0, 0, 500], dtype=float)
    bvecs = rng.random(size=(6, 3))
    bvecs = bvecs / np.sqrt((bvecs**2).sum(axis=1, keepdims=True))
    return Dwi(
        volume=InMemoryVolumeResource(array=dwi_data, affine=affine_random),
        bval=InMemoryBvalResource(bvals),
        bvec=InMemoryBvecResource(bvecs),
    )


@pytest.fixture
def response_function() -> InMemoryResponseFunctionResource:
    rng = np.random.default_rng(13)
    sh_coeffs = rng.random(size=(10,))  # random array of coeeficients
    avg_signal = rng.random(size=(), dtype=np.float32)
    return InMemoryResponseFunctionResource(sh_coeffs=sh_coeffs, avg_signal=avg_signal)


@pytest.mark.parametrize("response_is_none", [True, False])
def test_tractseg(
    mocker,
    dwi_data_small_random,
    response_function,
    response_is_none,
):
    # Create mock csd peaks data
    rng = np.random.default_rng(17)
    vol = dwi_data_small_random.volume
    vol_shape = vol.get_array().shape[:-1]  #  x,y,z, dimensions only
    n_peaks = 3
    peak_dir_data = rng.random(size=(*vol_shape, n_peaks, 3))
    peak_value_data = rng.random(size=(*vol_shape, n_peaks))
    csd_peak_dir = InMemoryVolumeResource(array=peak_dir_data, affine=vol.get_affine())
    csd_peak_value = InMemoryVolumeResource(
        array=peak_value_data, affine=vol.get_affine()
    )

    mock_csd_peaks_result = (csd_peak_dir, csd_peak_value)
    mocker_csd_peaks = mocker.patch(
        "kwneuro.tractseg.compute_csd_peaks", return_value=mock_csd_peaks_result
    )

    # Mock tractseg output
    mock_tractseg_output = np.ones((*vol_shape, 72))
    mocker_run_tract_seg = mocker.patch(
        "kwneuro.tractseg._call_tractseg", return_value=mock_tractseg_output
    )

    response = None if response_is_none else response_function

    mock_mask = mocker.Mock()
    seg_volume = extract_tractseg(
        dwi_data_small_random,
        mask=mock_mask,
        response=response,
        output_type="tract_segmentation",
    )

    mocker_csd_peaks.assert_called_once_with(
        dwi_data_small_random,
        mock_mask,
        response=response,
        flip_bvecs_x=True,
        n_peaks=n_peaks,
    )
    mocker_run_tract_seg.assert_called_once()

    assert np.allclose(seg_volume.get_array(), mock_tractseg_output)
