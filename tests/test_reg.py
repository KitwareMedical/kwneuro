from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import ants
import nibabel as nib
import numpy as np
import pytest

from kwneuro import Cache
from kwneuro.dwi import Dwi
from kwneuro.reg import TransformResource, register_dwi_to_structural, register_volumes
from kwneuro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
)
from kwneuro.structural import StructuralImage


@pytest.fixture
def structural() -> StructuralImage:
    return StructuralImage(volume=InMemoryVolumeResource(array=np.zeros((2, 2, 2))))


@pytest.fixture
def small_nifti_header():
    hdr = nib.Nifti1Header()
    hdr["descrip"] = b"a kwneuro unit test header description"
    hdr.set_xyzt_units(xyz="mm")
    return hdr


@pytest.fixture
def dwi1(small_nifti_header) -> Dwi:
    rng = np.random.default_rng(2656542)
    bvals = np.array([0, 1000, 500, 0, 0, 500], dtype=float)
    bvecs = rng.random(size=(6, 3))
    bvecs = bvecs / np.sqrt((bvecs**2).sum(axis=1, keepdims=True))

    # Gaussian blob on a low-noise background. Smooth gradients give ANTs
    # clear features to align without the sharp edges that amplify float32
    # precision errors when transforms are round-tripped through disk.
    shape_3d = (16, 16, 16)
    volume_array = rng.random(size=(*shape_3d, 6)) * 0.01
    center = np.array([8.0, 8.0, 8.0])
    sigma = 3.0
    coords = np.stack(np.mgrid[0:16, 0:16, 0:16], axis=-1).astype(float)
    gauss = np.exp(-np.sum((coords - center) ** 2, axis=-1) / (2 * sigma**2))
    for i in (0, 3, 4):  # b0 frames
        volume_array[:, :, :, i] += gauss

    return Dwi(
        volume=InMemoryVolumeResource(
            array=volume_array, affine=np.eye(4), metadata=dict(small_nifti_header)
        ),
        bval=InMemoryBvalResource(bvals),
        bvec=InMemoryBvecResource(bvecs),
    )


@pytest.fixture
def dwi2(small_nifti_header) -> Dwi:
    rng = np.random.default_rng(26540)
    bvals = np.array([0, 1000, 500, 0, 0, 500], dtype=float)
    bvecs = rng.random(size=(6, 3))
    bvecs = bvecs / np.sqrt((bvecs**2).sum(axis=1, keepdims=True))

    # Shifted Gaussian blob — different shape to test cross-resolution
    # registration.
    shape_3d = (20, 14, 16)
    volume_array = rng.random(size=(*shape_3d, 6)) * 0.01
    center = np.array([12.0, 8.0, 8.0])
    sigma = 3.0
    coords = np.stack(np.mgrid[0:20, 0:14, 0:16], axis=-1).astype(float)
    gauss = np.exp(-np.sum((coords - center) ** 2, axis=-1) / (2 * sigma**2))
    for i in (0, 3, 4):  # b0 frames
        volume_array[:, :, :, i] += gauss

    return Dwi(
        volume=InMemoryVolumeResource(
            array=volume_array, affine=np.eye(4), metadata=dict(small_nifti_header)
        ),
        bval=InMemoryBvalResource(bvals),
        bvec=InMemoryBvecResource(bvecs),
    )


def test_register_volumes(dwi1: Dwi, dwi2: Dwi, tmp_path):
    fixed_scalar_volume = dwi1.compute_mean_b0()
    moving_scalar_volume = dwi2.compute_mean_b0()

    registered_volume, transform = register_volumes(
        fixed=fixed_scalar_volume,
        moving=moving_scalar_volume,
    )

    # Check transforms
    assert len(transform._ants_fwd_paths) == 2
    assert len(transform._ants_inv_paths) == 2

    assert transform.matrices is not None
    assert transform.warp_fields is not None
    assert len(transform.matrices) == 1
    assert len(transform.warp_fields) == 1

    assert isinstance(registered_volume, InMemoryVolumeResource)
    assert isinstance(transform.matrices[0], ants.ANTsTransform)
    assert np.allclose(
        transform.warp_fields[0].get_affine(), fixed_scalar_volume.get_affine()
    )

    # Check that the registered volume has the same shape as the fixed
    assert registered_volume.get_array().shape == fixed_scalar_volume.get_array().shape

    # Check saving
    transform.save(tmp_path)
    for file in transform._ants_fwd_paths + transform._ants_inv_paths:
        assert (tmp_path / Path(file).name).exists()

    # Check application
    applied_volume = transform.apply(
        fixed=fixed_scalar_volume,
        moving=moving_scalar_volume,
    )

    # The ANTs warpedmovout uses the in-memory transform, while
    # transform.apply() reads it back from disk (float32 .mat / .nii.gz).
    # The float32 round-trip introduces small coordinate errors.
    assert np.allclose(
        registered_volume.get_array(), applied_volume.get_array(), atol=1e-5
    )

    applied_volume_invert = transform.apply(
        fixed=moving_scalar_volume,
        moving=applied_volume,
        invert=True,
    )

    # Allclose on the array data fails because of interpolation differences and information loss
    assert (
        applied_volume_invert.get_array().shape
        == moving_scalar_volume.get_array().shape
    )

    # Test with masks
    f_mask = InMemoryVolumeResource(
        array=np.ones(fixed_scalar_volume.get_array().shape),
        affine=fixed_scalar_volume.get_affine(),
        metadata=fixed_scalar_volume.get_metadata(),
    )
    m_mask = InMemoryVolumeResource(
        array=np.ones(moving_scalar_volume.get_array().shape),
        affine=moving_scalar_volume.get_affine(),
        metadata=moving_scalar_volume.get_metadata(),
    )

    reg_vol, transform = register_volumes(
        fixed=fixed_scalar_volume,
        moving=moving_scalar_volume,
        mask=f_mask,
        moving_mask=m_mask,
    )

    assert reg_vol.get_array().shape == fixed_scalar_volume.get_array().shape


def test_register_volumes_without_warps(
    dwi1: Dwi,
    dwi2: Dwi,
):
    fixed = dwi1.compute_mean_b0()
    moving = dwi2.compute_mean_b0()

    registered_volume, transform = register_volumes(
        fixed=fixed,
        moving=moving,
        type_of_transform="Affine",
    )

    # Check transforms
    assert len(transform._ants_fwd_paths) == 1
    assert len(transform._ants_inv_paths) == 1

    assert transform.matrices is not None
    assert transform.warp_fields == []
    assert len(transform.matrices) == 1

    assert isinstance(registered_volume, InMemoryVolumeResource)
    assert isinstance(transform.matrices[0], ants.ANTsTransform)

    # Check that the registered volume has the same shape as the fixed
    assert registered_volume.get_array().shape == fixed.get_array().shape

    # Check application
    applied_volume = transform.apply(
        fixed=fixed,
        moving=moving,
    )

    # The ANTs warpedmovout uses the in-memory transform, while
    # transform.apply() reads it back from disk (float32 .mat / .nii.gz).
    # The float32 round-trip introduces small coordinate errors.
    assert np.allclose(
        registered_volume.get_array(), applied_volume.get_array(), atol=1e-5
    )

    applied_volume_invert = transform.apply(
        fixed=moving,
        moving=applied_volume,
        invert=True,
    )

    # Minimal interpolation errors expected with affine only
    assert np.allclose(
        applied_volume_invert.get_array().shape, moving.get_array().shape
    )


def test_transform_save_writes_reloadable_manifest(tmp_path: Path) -> None:
    affine_path = tmp_path / "0GenericAffine.mat"
    warp_path = tmp_path / "1Warp.nii.gz"
    affine_path.write_text("affine")
    warp_path.write_text("warp")
    transform = TransformResource(
        _ants_fwd_paths=[str(warp_path), str(affine_path)],
        _ants_inv_paths=[str(affine_path)],
    )

    saved = transform.save(tmp_path / "saved_transform")
    manifest_path = tmp_path / "saved_transform" / "transform.json"
    manifest = json.loads(manifest_path.read_text())
    loaded = TransformResource.load(tmp_path / "saved_transform")

    assert manifest == {
        "fwd": ["1Warp.nii.gz", "0GenericAffine.mat"],
        "inv": ["0GenericAffine.mat"],
    }
    assert saved._ants_fwd_paths == loaded._ants_fwd_paths
    assert saved._ants_inv_paths == loaded._ants_inv_paths


def test_register_dwi_to_structural_caches_transform_files(
    dwi1: Dwi, structural: StructuralImage, mocker, tmp_path: Path
) -> None:
    source_dir = tmp_path / "source_transform"
    source_dir.mkdir()
    affine_path = source_dir / "0GenericAffine.mat"
    warp_path = source_dir / "1Warp.nii.gz"
    affine_path.write_text("affine")
    warp_path.write_text("warp")
    source_transform = TransformResource(
        _ants_fwd_paths=[str(warp_path), str(affine_path)],
        _ants_inv_paths=[str(affine_path)],
    )
    mock_reg = mocker.patch(
        "kwneuro.reg.register_volumes",
        return_value=(MagicMock(), source_transform),
    )
    cache = Cache(tmp_path)

    with cache:
        first = register_dwi_to_structural(dwi=dwi1, structural=structural)

    cache_dir = tmp_path / "register_dwi_to_structural_transform"
    assert mock_reg.call_count == 1
    assert cache.status([register_dwi_to_structural])["register_dwi_to_structural"]
    assert json.loads((cache_dir / "transform.json").read_text()) == {
        "fwd": ["1Warp.nii.gz", "0GenericAffine.mat"],
        "inv": ["0GenericAffine.mat"],
    }
    assert [Path(p).name for p in first._ants_fwd_paths] == [
        "1Warp.nii.gz",
        "0GenericAffine.mat",
    ]
    assert [Path(p).name for p in first._ants_inv_paths] == ["0GenericAffine.mat"]
    for path in first._ants_fwd_paths + first._ants_inv_paths:
        assert Path(path).exists()
        assert Path(path).parent == cache_dir

    mock_reg.reset_mock()
    with cache:
        second = register_dwi_to_structural(dwi=dwi1, structural=structural)

    mock_reg.assert_not_called()
    assert second._ants_fwd_paths == first._ants_fwd_paths
    assert second._ants_inv_paths == first._ants_inv_paths


def test_register_volumes_with_incorrect_mask(dwi1: Dwi, dwi2: Dwi, small_nifti_header):
    fixed = dwi1.compute_mean_b0()
    moving = dwi2.compute_mean_b0()

    # Create a mask with the wrong shape
    wrong_mask = InMemoryVolumeResource(
        array=np.ones(moving.get_array().shape),
        affine=moving.get_affine(),
        metadata=dict(small_nifti_header),
    )

    with pytest.raises(ValueError, match="Fixed mask dimensions do not match"):
        register_volumes(fixed=fixed, moving=moving, mask=wrong_mask)

    # Create a mask with the wrong shape
    wrong_mask = InMemoryVolumeResource(
        array=np.ones(fixed.get_array().shape),
        affine=moving.get_affine(),
        metadata=dict(small_nifti_header),
    )

    with pytest.raises(ValueError, match="Moving mask dimensions do not match"):
        register_volumes(fixed=fixed, moving=moving, moving_mask=wrong_mask)


def test_register_dwi_to_structural(
    dwi1: Dwi, structural: StructuralImage, mocker, tmp_path: Path
):
    stub_transform = TransformResource(_ants_fwd_paths=[], _ants_inv_paths=[])
    mock_reg = mocker.patch(
        "kwneuro.reg.register_volumes", return_value=(MagicMock(), stub_transform)
    )

    result = register_dwi_to_structural(
        dwi=dwi1, structural=structural, type_of_transform="Rigid"
    )
    mock_reg.assert_called_once()
    call_kwargs = mock_reg.call_args.kwargs
    assert call_kwargs["fixed"] is structural.volume
    assert np.array_equal(
        call_kwargs["moving"].get_array(), dwi1.compute_mean_b0().get_array()
    )
    assert call_kwargs["type_of_transform"] == "Rigid"
    assert call_kwargs["mask"] is None
    assert call_kwargs["moving_mask"] is None
    assert type(result) is TransformResource

    # caching: second call in same Cache context is a hit
    mock_reg.reset_mock()
    with Cache(tmp_path):
        register_dwi_to_structural(dwi=dwi1, structural=structural)
    assert mock_reg.call_count == 1

    mock_reg.reset_mock()
    with Cache(tmp_path):
        register_dwi_to_structural(dwi=dwi1, structural=structural)
    mock_reg.assert_not_called()


def test_register_volumes_with_incorrect_dimensions(
    dwi1: Dwi,
    dwi2: Dwi,
):
    correct_dim = dwi1.compute_mean_b0()

    with pytest.raises(ValueError, match="Input volume dimensions must be"):
        register_volumes(fixed=correct_dim, moving=dwi2.volume)

    with pytest.raises(ValueError, match="Input volume dimensions must be"):
        register_volumes(fixed=dwi2.volume, moving=correct_dim)
