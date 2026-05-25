from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import kwneuro
from kwneuro import Dwi, StructuralImage
from kwneuro.files import (
    read_dwi_fsl,
    read_structural,
    read_volume,
    write_dwi_fsl,
    write_structural,
    write_volume,
)
from kwneuro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource
from kwneuro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
)


@pytest.fixture
def dwi() -> Dwi:
    rng = np.random.default_rng(73912)
    volume = InMemoryVolumeResource(
        array=rng.random(size=(3, 4, 5, 4)),
        affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        metadata=dict(nib.Nifti1Header()),
    )
    return Dwi(
        volume=volume,
        bval=InMemoryBvalResource(np.array([0.0, 1000.0, 2000.0, 1000.0])),
        bvec=InMemoryBvecResource(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        ),
    )


@pytest.fixture
def structural() -> StructuralImage:
    rng = np.random.default_rng(8129)
    return StructuralImage(
        volume=InMemoryVolumeResource(
            array=rng.random(size=(3, 4, 5)),
            affine=np.diag([1.5, 1.5, 1.5, 1.0]),
            metadata=dict(nib.Nifti1Header()),
        )
    )


def assert_dwi_allclose(actual: Dwi, expected: Dwi) -> None:
    actual_loaded = actual.load()
    expected_loaded = expected.load()
    assert np.allclose(
        actual_loaded.volume.get_array(), expected_loaded.volume.get_array()
    )
    assert np.allclose(
        actual_loaded.volume.get_affine(), expected_loaded.volume.get_affine()
    )
    assert np.allclose(actual_loaded.bval.get(), expected_loaded.bval.get())
    assert np.allclose(actual_loaded.bvec.get(), expected_loaded.bvec.get())


def test_read_dwi_fsl_infers_bids_style_sidecars(dwi: Dwi, tmp_path: Path) -> None:
    volume_path = tmp_path / "sub-01_ses-01_acq-test_dwi.nii.gz"

    saved = write_dwi_fsl(dwi, volume_path)

    assert (tmp_path / "sub-01_ses-01_acq-test_dwi.bval").exists()
    assert (tmp_path / "sub-01_ses-01_acq-test_dwi.bvec").exists()
    assert isinstance(saved.volume, NiftiVolumeResource)
    assert isinstance(saved.bval, FslBvalResource)
    assert isinstance(saved.bvec, FslBvecResource)

    reloaded = read_dwi_fsl(volume_path)
    assert isinstance(reloaded.volume, NiftiVolumeResource)
    assert isinstance(reloaded.bval, FslBvalResource)
    assert isinstance(reloaded.bvec, FslBvecResource)
    assert reloaded.volume.path == volume_path.resolve()
    assert reloaded.bval.path == (
        tmp_path / "sub-01_ses-01_acq-test_dwi.bval"
    ).resolve()
    assert reloaded.bvec.path == (
        tmp_path / "sub-01_ses-01_acq-test_dwi.bvec"
    ).resolve()
    assert_dwi_allclose(reloaded, dwi)


def test_read_dwi_fsl_infers_sidecars_from_nii_path(
    dwi: Dwi, tmp_path: Path
) -> None:
    volume_path = tmp_path / "sub-01_dwi.nii"

    write_dwi_fsl(dwi, volume_path)

    assert (tmp_path / "sub-01_dwi.bval").exists()
    assert (tmp_path / "sub-01_dwi.bvec").exists()
    assert_dwi_allclose(read_dwi_fsl(volume_path), dwi)


def test_read_write_dwi_fsl_accepts_explicit_sidecar_paths(
    dwi: Dwi, tmp_path: Path
) -> None:
    volume_path = tmp_path / "dwi.nii.gz"
    bval_path = tmp_path / "gradients" / "run-1.bvals"
    bvec_path = tmp_path / "gradients" / "run-1.bvecs"

    write_dwi_fsl(dwi, volume_path, bval=bval_path, bvec=bvec_path)

    assert bval_path.exists()
    assert bvec_path.exists()
    assert not (tmp_path / "dwi.bval").exists()
    assert not (tmp_path / "dwi.bvec").exists()
    assert_dwi_allclose(read_dwi_fsl(volume_path, bval=bval_path, bvec=bvec_path), dwi)


def test_read_dwi_fsl_requires_nifti_path_for_sidecar_inference(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Expected a .nii or .nii.gz path"):
        read_dwi_fsl(tmp_path / "dwi.img")


def test_read_write_volume_round_trip(dwi: Dwi, tmp_path: Path) -> None:
    volume_path = tmp_path / "nested" / "volume.nii.gz"

    saved = write_volume(dwi.volume, volume_path)
    reloaded = read_volume(volume_path)

    assert isinstance(saved, NiftiVolumeResource)
    assert isinstance(reloaded, NiftiVolumeResource)
    assert not saved.is_loaded
    assert not reloaded.is_loaded
    assert np.allclose(reloaded.get_array(), dwi.volume.get_array())
    assert np.allclose(reloaded.get_affine(), dwi.volume.get_affine())


def test_read_write_structural_round_trip(
    structural: StructuralImage, tmp_path: Path
) -> None:
    structural_path = tmp_path / "sub-01_T1w.nii.gz"

    saved = write_structural(structural, structural_path)
    reloaded = read_structural(structural_path)

    assert isinstance(saved, StructuralImage)
    assert isinstance(reloaded, StructuralImage)
    assert isinstance(reloaded.volume, NiftiVolumeResource)
    assert np.allclose(reloaded.volume.get_array(), structural.volume.get_array())
    assert np.allclose(reloaded.volume.get_affine(), structural.volume.get_affine())


def test_top_level_file_imports() -> None:
    assert kwneuro.Dwi is Dwi
    assert kwneuro.StructuralImage is StructuralImage
    assert kwneuro.read_volume is read_volume
    assert kwneuro.write_volume is write_volume
    assert kwneuro.read_dwi_fsl is read_dwi_fsl
    assert kwneuro.write_dwi_fsl is write_dwi_fsl
    assert kwneuro.read_structural is read_structural
    assert kwneuro.write_structural is write_structural
