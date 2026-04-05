from __future__ import annotations

import numpy as np
import pytest

from kwneuro.dwi import Dwi, subsample_dwi
from kwneuro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
)
from kwneuro.util import subsample_volume


@pytest.fixture
def affine_2mm():
    """A simple diagonal affine with 2 mm voxel spacing."""
    aff = np.eye(4)
    np.fill_diagonal(aff[:3, :3], 2.0)
    return aff


@pytest.fixture
def volume_3d(affine_2mm):
    arr = np.arange(12 * 14 * 10, dtype=np.float64).reshape(12, 14, 10)
    return InMemoryVolumeResource(array=arr, affine=affine_2mm, metadata={})


@pytest.fixture
def volume_4d(affine_2mm):
    arr = np.arange(12 * 14 * 10 * 3, dtype=np.float64).reshape(12, 14, 10, 3)
    return InMemoryVolumeResource(array=arr, affine=affine_2mm, metadata={})


class TestSubsampleVolume:
    def test_3d_shape(self, volume_3d):
        result = subsample_volume(volume_3d, factor=2)
        assert result.get_array().shape == (6, 7, 5)

    def test_4d_shape(self, volume_4d):
        result = subsample_volume(volume_4d, factor=2)
        assert result.get_array().shape == (6, 7, 5, 3)

    def test_4d_extra_dim_untouched(self, volume_4d):
        result = subsample_volume(volume_4d, factor=2)
        original = volume_4d.get_array()
        # Every output voxel should match the strided input voxel
        np.testing.assert_array_equal(
            result.get_array()[1, 2, 3, :],
            original[2, 4, 6, :],
        )

    def test_values_are_exact_subset(self, volume_3d):
        result = subsample_volume(volume_3d, factor=2)
        original = volume_3d.get_array()
        np.testing.assert_array_equal(
            result.get_array(),
            original[::2, ::2, ::2],
        )

    def test_affine_scaling(self, volume_3d, affine_2mm):
        result = subsample_volume(volume_3d, factor=3)
        expected_affine = affine_2mm.copy()
        expected_affine[:3, :3] *= 3
        np.testing.assert_array_equal(result.get_affine(), expected_affine)

    def test_affine_preserves_origin(self, volume_3d, affine_2mm):
        affine_2mm[:3, 3] = [10.0, 20.0, 30.0]
        vol = InMemoryVolumeResource(
            array=volume_3d.get_array(), affine=affine_2mm, metadata={}
        )
        result = subsample_volume(vol, factor=2)
        np.testing.assert_array_equal(result.get_affine()[:3, 3], [10.0, 20.0, 30.0])

    def test_world_coords_preserved(self, volume_3d):
        """World coordinate at output voxel (i,j,k) should equal world coordinate
        at input voxel (i*f, j*f, k*f)."""
        factor = 2
        result = subsample_volume(volume_3d, factor=factor)
        orig_affine = volume_3d.get_affine()
        new_affine = result.get_affine()
        # Check a few voxel indices
        for idx in [(0, 0, 0), (1, 2, 3), (3, 5, 2)]:
            orig_world = orig_affine @ np.array([i * factor for i in idx] + [1])
            new_world = new_affine @ np.array([*list(idx), 1])
            np.testing.assert_allclose(new_world, orig_world)

    def test_factor_1_is_identity(self, volume_3d):
        result = subsample_volume(volume_3d, factor=1)
        np.testing.assert_array_equal(result.get_array(), volume_3d.get_array())
        np.testing.assert_array_equal(result.get_affine(), volume_3d.get_affine())

    def test_factor_3(self, volume_3d):
        result = subsample_volume(volume_3d, factor=3)
        assert result.get_array().shape == (4, 5, 4)


class TestSubsampleDwi:
    @pytest.fixture
    def dwi(self, volume_4d):
        n_dirs = volume_4d.get_array().shape[3]
        bvals = np.array([0.0, 1000.0, 1000.0])[:n_dirs]
        # Unit vectors for non-zero b-values
        bvecs = np.zeros((n_dirs, 3))
        for i in range(n_dirs):
            if bvals[i] > 0:
                bvecs[i] = [1.0, 0.0, 0.0]
        return Dwi(
            volume=volume_4d,
            bval=InMemoryBvalResource(bvals),
            bvec=InMemoryBvecResource(bvecs),
        )

    def test_returns_dwi(self, dwi):
        result = subsample_dwi(dwi, factor=2)
        assert isinstance(result, Dwi)

    def test_volume_subsampled(self, dwi):
        result = subsample_dwi(dwi, factor=2)
        assert result.volume.get_array().shape == (6, 7, 5, 3)

    def test_bval_preserved(self, dwi):
        result = subsample_dwi(dwi, factor=2)
        np.testing.assert_array_equal(result.bval.get(), dwi.bval.get())

    def test_bvec_preserved(self, dwi):
        result = subsample_dwi(dwi, factor=2)
        np.testing.assert_array_equal(result.bvec.get(), dwi.bvec.get())
