from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

from kwneuro.dwi import Dwi
from kwneuro.masks import (
    brain_extract_batch,
)
from kwneuro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
)


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


@pytest.mark.parametrize("extension", ["nii.gz", "nii"])
def test_brain_extract_batch(
    mocker,
    extension,
    dwi_data_small_random,
):
    with tempfile.TemporaryDirectory() as work_dir:
        mask_out_path = Path(work_dir) / f"aaa_mask.{extension}"

        mock_run_hd_bet = mocker.patch("kwneuro.masks._run_hd_bet")

        brain_extract_batch(
            cases=[
                (dwi_data_small_random, mask_out_path),
            ],
        )
        mock_run_hd_bet.assert_called_once()  # HD_BET should have been called
        # Default is parallel (sequential=False).
        assert mock_run_hd_bet.call_args.kwargs.get("sequential") is False


def test_brain_extract_batch_propagates_sequential_flag(
    mocker,
    dwi_data_small_random,
):
    """``sequential=True`` reaches ``_run_hd_bet`` so the no-multiprocessing
    code path is used. Required by callers like the kwneuro Slicer
    extension that can't safely ``multiprocessing.spawn``."""
    with tempfile.TemporaryDirectory() as work_dir:
        mask_out_path = Path(work_dir) / "aaa_mask.nii.gz"
        mock_run_hd_bet = mocker.patch("kwneuro.masks._run_hd_bet")

        brain_extract_batch(
            cases=[(dwi_data_small_random, mask_out_path)],
            sequential=True,
        )
        mock_run_hd_bet.assert_called_once()
        assert mock_run_hd_bet.call_args.kwargs["sequential"] is True
