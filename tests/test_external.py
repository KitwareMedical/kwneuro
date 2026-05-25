from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import kwneuro
from kwneuro import Dwi, StructuralImage
from kwneuro.external import (
    TemporaryDwiFiles,
    temporary_dwi_files,
    temporary_structural_file,
    temporary_volume_file,
)
from kwneuro.files import (
    read_dwi_fsl,
    read_structural,
    read_volume,
    write_volume,
)
from kwneuro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
    VolumeResource,
)


@pytest.fixture
def volume() -> InMemoryVolumeResource:
    rng = np.random.default_rng(8285)
    header = nib.Nifti1Header()
    header.set_xyzt_units("mm")
    return InMemoryVolumeResource(
        array=rng.random(size=(3, 4, 5), dtype=np.float32),
        affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        metadata=dict(header),
    )


@pytest.fixture
def dwi() -> Dwi:
    rng = np.random.default_rng(7812)
    header = nib.Nifti1Header()
    header.set_xyzt_units("mm")
    return Dwi(
        volume=InMemoryVolumeResource(
            array=rng.random(size=(3, 4, 5, 4), dtype=np.float32),
            affine=np.diag([2.0, 2.0, 2.0, 1.0]),
            metadata=dict(header),
        ),
        bval=InMemoryBvalResource(np.array([0.0, 1000.0, 1000.0, 1000.0])),
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
def structural(volume: InMemoryVolumeResource) -> StructuralImage:
    return StructuralImage(volume=volume)


def test_temporary_volume_file_exists_only_inside_context(
    volume: InMemoryVolumeResource,
) -> None:
    with temporary_volume_file(volume) as path:
        assert path.name == "volume.nii.gz"
        assert path.exists()
        assert_volume_allclose(read_volume(path), volume)
        temporary_path = path

    assert not temporary_path.exists()
    assert not temporary_path.parent.exists()


def test_temporary_dwi_files_exist_only_inside_context(dwi: Dwi) -> None:
    with temporary_dwi_files(dwi, basename="sub-01_dwi") as files:
        assert isinstance(files, TemporaryDwiFiles)
        assert files.volume.name == "sub-01_dwi.nii.gz"
        assert files.bval.name == "sub-01_dwi.bval"
        assert files.bvec.name == "sub-01_dwi.bvec"
        assert files.volume.exists()
        assert files.bval.exists()
        assert files.bvec.exists()
        assert_dwi_allclose(
            read_dwi_fsl(files.volume, bval=files.bval, bvec=files.bvec),
            dwi,
        )
        temporary_files = files

    assert not temporary_files.volume.exists()
    assert not temporary_files.bval.exists()
    assert not temporary_files.bvec.exists()
    assert not temporary_files.volume.parent.exists()


def test_temporary_structural_file_exists_only_inside_context(
    structural: StructuralImage,
) -> None:
    with temporary_structural_file(structural, filename="sub-01_T1w.nii.gz") as path:
        assert path.name == "sub-01_T1w.nii.gz"
        assert path.exists()
        reloaded = read_structural(path)
        assert_volume_allclose(reloaded.volume, structural.volume)
        temporary_path = path

    assert not temporary_path.exists()
    assert not temporary_path.parent.exists()


def test_external_file_function_can_reenter_kwneuro(
    volume: InMemoryVolumeResource,
) -> None:
    with temporary_volume_file(volume) as input_path:
        output_path = input_path.with_name("thresholded.nii.gz")
        _external_threshold(input_path, output_path)
        thresholded = read_volume(output_path).load()

        expected = (volume.get_array() > volume.get_array().mean()).astype(np.uint8)
        assert output_path.exists()
        assert np.array_equal(thresholded.get_array(), expected)
        assert np.allclose(thresholded.get_affine(), volume.get_affine())


def test_modifying_temporary_copy_does_not_mutate_original_disk_source(
    volume: InMemoryVolumeResource,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.nii.gz"
    source = write_volume(volume, source_path)
    original = source.get_array().copy()

    with temporary_volume_file(source) as copy_path:
        assert copy_path != source_path.resolve()
        image = nib.load(copy_path)
        zeros = np.zeros_like(image.get_fdata())
        nib.save(nib.Nifti1Image(zeros, image.affine, image.header), copy_path)
        assert np.allclose(read_volume(copy_path).get_array(), zeros)

    assert np.allclose(read_volume(source_path).get_array(), original)


def test_temporary_file_rejects_absolute_temporary_names(
    volume: InMemoryVolumeResource,
    tmp_path: Path,
) -> None:
    with (
        pytest.raises(ValueError, match="must be relative"),
        temporary_volume_file(volume, filename=str(tmp_path / "bad.nii.gz")),
    ):
        pass


def test_temporary_file_rejects_paths_outside_temporary_directory(
    volume: InMemoryVolumeResource,
) -> None:
    with (
        pytest.raises(ValueError, match="within the temporary directory"),
        temporary_volume_file(volume, filename="../bad.nii.gz"),
    ):
        pass


def test_temporary_file_helpers_are_not_top_level_exports() -> None:
    assert not hasattr(kwneuro, "temporary_volume_file")
    assert not hasattr(kwneuro, "temporary_dwi_files")
    assert not hasattr(kwneuro, "temporary_structural_file")


def assert_volume_allclose(actual: VolumeResource, expected: VolumeResource) -> None:
    actual_loaded = actual.load()
    expected_loaded = expected.load()
    assert np.allclose(actual_loaded.get_array(), expected_loaded.get_array())
    assert np.allclose(actual_loaded.get_affine(), expected_loaded.get_affine())


def assert_dwi_allclose(actual: Dwi, expected: Dwi) -> None:
    actual_loaded = actual.load()
    expected_loaded = expected.load()
    assert_volume_allclose(actual_loaded.volume, expected_loaded.volume)
    assert np.allclose(actual_loaded.bval.get(), expected_loaded.bval.get())
    assert np.allclose(actual_loaded.bvec.get(), expected_loaded.bvec.get())


def _external_threshold(input_path: Path, output_path: Path) -> None:
    image = nib.load(input_path)
    data = image.get_fdata()
    thresholded = (data > data.mean()).astype(np.uint8)
    nib.save(nib.Nifti1Image(thresholded, image.affine, image.header), output_path)
