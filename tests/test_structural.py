from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import ANY, MagicMock

import nibabel as nib
import numpy as np
import pytest
from scipy.linalg import expm

from kwneuro import Cache
from kwneuro.io import NiftiVolumeResource
from kwneuro.masks import brain_extract, brain_extract_structural_batch
from kwneuro.resource import InMemoryVolumeResource
from kwneuro.structural import StructuralImage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_affine() -> np.ndarray:
    rng = np.random.default_rng(18653)
    affine = np.eye(4)
    affine[:3, :3] = expm((lambda A: (A - A.T) / 2)(rng.normal(size=(3, 3))))
    affine[:3, 3] = rng.random(3)
    return affine


@pytest.fixture
def nifti_header_mm():
    """A minimal NIfTI header with spatial units set to mm (required by to_ants_image)."""
    hdr = nib.Nifti1Header()
    hdr.set_xyzt_units("mm")
    return hdr


@pytest.fixture
def structural(nifti_header_mm) -> StructuralImage:
    rng = np.random.default_rng(42)
    array = rng.random(size=(8, 8, 8))
    return StructuralImage(
        volume=InMemoryVolumeResource(
            array=array,
            affine=np.eye(4),
            metadata=dict(nifti_header_mm),
        )
    )


@pytest.fixture
def mock_ants_vol(structural: StructuralImage):
    """A fake InMemoryVolumeResource to use as a stand-in ANTs round-trip result."""
    return InMemoryVolumeResource(
        array=structural.volume.get_array().copy(),
        affine=structural.volume.get_affine(),
        metadata=structural.volume.get_metadata(),
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def test_structural_save(structural: StructuralImage, tmp_path: Path):
    saved = structural.save(path=tmp_path, basename="test")
    assert not saved.volume.is_loaded
    assert (tmp_path / "test.nii.gz").exists()


def test_structural_save_load(
    structural: StructuralImage, random_affine, tmp_path: Path
):
    # Use the random affine for a more thorough round-trip
    vol = InMemoryVolumeResource(
        array=structural.volume.get_array(),
        affine=random_affine,
        metadata=structural.volume.get_metadata(),
    )
    s = StructuralImage(volume=vol)
    reloaded = s.save(path=tmp_path, basename="test").load()
    assert reloaded.volume.is_loaded
    assert np.allclose(reloaded.volume.get_array(), s.volume.get_array())
    assert np.allclose(reloaded.volume.get_affine(), s.volume.get_affine())


# ---------------------------------------------------------------------------
# Cache protocol
# ---------------------------------------------------------------------------


def test_structural_cache_files():
    assert StructuralImage._cache_files("step") == ["step.nii.gz"]


# ---------------------------------------------------------------------------
# correct_bias
# ---------------------------------------------------------------------------


def test_correct_bias(
    structural: StructuralImage, mock_ants_vol: InMemoryVolumeResource, mocker
):
    mock_ants_img = MagicMock()
    mocker.patch.object(
        InMemoryVolumeResource, "to_ants_image", return_value=mock_ants_img
    )
    mock_n4 = mocker.patch(
        "kwneuro.structural.ants.n4_bias_field_correction", return_value=mock_ants_img
    )
    mocker.patch.object(
        InMemoryVolumeResource, "from_ants_image", return_value=mock_ants_vol
    )

    result = structural.correct_bias()
    mock_n4.assert_called_once()
    assert isinstance(result, StructuralImage)
    assert result.volume.get_array().shape == structural.volume.get_array().shape


def test_correct_bias_caching(
    structural: StructuralImage,
    mock_ants_vol: InMemoryVolumeResource,
    mocker,
    tmp_path: Path,
):
    mock_ants_img = MagicMock()
    mocker.patch.object(
        InMemoryVolumeResource, "to_ants_image", return_value=mock_ants_img
    )
    mock_n4 = mocker.patch(
        "kwneuro.structural.ants.n4_bias_field_correction", return_value=mock_ants_img
    )
    mocker.patch.object(
        InMemoryVolumeResource, "from_ants_image", return_value=mock_ants_vol
    )

    with Cache(tmp_path):
        structural.correct_bias()
    assert mock_n4.call_count == 1
    assert (tmp_path / "correct_bias.nii.gz").exists()

    mock_n4.reset_mock()
    with Cache(tmp_path):
        structural.correct_bias()
    mock_n4.assert_not_called()  # cache hit


# ---------------------------------------------------------------------------
# extract_brain
# ---------------------------------------------------------------------------


def test_extract_brain_structural(structural: StructuralImage, mocker, tmp_path: Path):
    rng = np.random.default_rng(18653)
    mask_in_memory = InMemoryVolumeResource(
        array=rng.random((8, 8, 8)), affine=np.eye(4)
    )
    mask_on_disk = NiftiVolumeResource.save(mask_in_memory, tmp_path / "mask.nii.gz")
    mock_brain_extract = mocker.patch(
        "kwneuro.masks.brain_extract", return_value=mask_on_disk
    )

    mask = structural.extract_brain()
    mock_brain_extract.assert_called_once_with(volume=ANY, output_path=ANY)
    assert np.allclose(mask.get_array(), mask_in_memory.get_array())


# ---------------------------------------------------------------------------
# segment_tissues
# ---------------------------------------------------------------------------


def test_segment_tissues_atropos(
    structural: StructuralImage, mock_ants_vol: InMemoryVolumeResource, mocker
):
    mock_ants_img = MagicMock()
    mocker.patch.object(
        InMemoryVolumeResource, "to_ants_image", return_value=mock_ants_img
    )
    mocker.patch("kwneuro.structural.ants.get_mask", return_value=mock_ants_img)
    mock_atropos = mocker.patch(
        "kwneuro.structural.ants.atropos",
        return_value={"segmentation": mock_ants_img},
    )
    mocker.patch.object(
        InMemoryVolumeResource, "from_ants_image", return_value=mock_ants_vol
    )

    result = structural.segment_tissues(method="atropos")
    mock_atropos.assert_called_once()
    assert isinstance(result, InMemoryVolumeResource)


def test_segment_tissues_atropos_with_mask(
    structural: StructuralImage, mock_ants_vol: InMemoryVolumeResource, mocker
):
    mock_ants_img = MagicMock()
    mocker.patch.object(
        InMemoryVolumeResource, "to_ants_image", return_value=mock_ants_img
    )
    mock_atropos = mocker.patch(
        "kwneuro.structural.ants.atropos",
        return_value={"segmentation": mock_ants_img},
    )
    mocker.patch.object(
        InMemoryVolumeResource, "from_ants_image", return_value=mock_ants_vol
    )

    mask = InMemoryVolumeResource(
        array=np.ones((8, 8, 8)),
        affine=np.eye(4),
        metadata=structural.volume.get_metadata(),
    )
    result = structural.segment_tissues(mask=mask, method="atropos")
    mock_atropos.assert_called_once()
    assert isinstance(result, InMemoryVolumeResource)


def test_segment_tissues_deep_atropos(
    structural: StructuralImage, mock_ants_vol: InMemoryVolumeResource, mocker
):
    mock_ants_img = MagicMock()
    mocker.patch.object(
        InMemoryVolumeResource, "to_ants_image", return_value=mock_ants_img
    )
    mocker.patch.object(
        InMemoryVolumeResource, "from_ants_image", return_value=mock_ants_vol
    )
    mock_antspynet = MagicMock()
    mock_antspynet.deep_atropos.return_value = {"segmentation_image": mock_ants_img}
    mocker.patch.dict(sys.modules, {"antspynet": mock_antspynet})

    result = structural.segment_tissues(method="deep_atropos")
    mock_antspynet.deep_atropos.assert_called_once()
    assert isinstance(result, InMemoryVolumeResource)


def test_segment_tissues_deep_atropos_not_installed(
    structural: StructuralImage, mocker
):
    mocker.patch.dict(sys.modules, {"antspynet": None})

    with pytest.raises(ImportError, match="kwneuro\\[antspynet\\]"):
        structural.segment_tissues(method="deep_atropos")


def test_segment_tissues_unknown_method(structural: StructuralImage):
    with pytest.raises(ValueError, match="Unknown segmentation method"):
        structural.segment_tissues(method="bad_method")


def test_segment_tissues_caching(
    structural: StructuralImage,
    mock_ants_vol: InMemoryVolumeResource,
    mocker,
    tmp_path: Path,
):
    mock_ants_img = MagicMock()
    mocker.patch.object(
        InMemoryVolumeResource, "to_ants_image", return_value=mock_ants_img
    )
    mocker.patch("kwneuro.structural.ants.get_mask", return_value=mock_ants_img)
    mock_atropos = mocker.patch(
        "kwneuro.structural.ants.atropos",
        return_value={"segmentation": mock_ants_img},
    )
    mocker.patch.object(
        InMemoryVolumeResource, "from_ants_image", return_value=mock_ants_vol
    )

    with Cache(tmp_path):
        structural.segment_tissues(method="atropos")
    assert mock_atropos.call_count == 1
    assert (tmp_path / "segment_tissues.nii.gz").exists()

    mock_atropos.reset_mock()
    with Cache(tmp_path):
        structural.segment_tissues(method="atropos")
    mock_atropos.assert_not_called()  # cache hit


def test_segment_tissues_caching_writes_method_param(
    structural: StructuralImage,
    mock_ants_vol: InMemoryVolumeResource,
    mocker,
    tmp_path: Path,
):
    """The method parameter is stored in the cache sidecar so method changes cause a cache miss."""
    mock_ants_img = MagicMock()
    mocker.patch.object(
        InMemoryVolumeResource, "to_ants_image", return_value=mock_ants_img
    )
    mocker.patch("kwneuro.structural.ants.get_mask", return_value=mock_ants_img)
    mocker.patch(
        "kwneuro.structural.ants.atropos",
        return_value={"segmentation": mock_ants_img},
    )
    mocker.patch.object(
        InMemoryVolumeResource, "from_ants_image", return_value=mock_ants_vol
    )

    with Cache(tmp_path):
        structural.segment_tissues(method="atropos")

    params = json.loads((tmp_path / "segment_tissues.params.json").read_text())
    assert params["scalars"]["method"] == "atropos"


# ---------------------------------------------------------------------------
# parcellate
# ---------------------------------------------------------------------------


def test_parcellate_dkt(
    structural: StructuralImage, mock_ants_vol: InMemoryVolumeResource, mocker
):
    mock_ants_img = MagicMock()
    mocker.patch.object(
        InMemoryVolumeResource, "to_ants_image", return_value=mock_ants_img
    )
    mocker.patch.object(
        InMemoryVolumeResource, "from_ants_image", return_value=mock_ants_vol
    )
    mock_antspynet = MagicMock()
    mock_antspynet.desikan_killiany_tourville_labeling.return_value = {
        "parcellation_segmentation": mock_ants_img
    }
    mocker.patch.dict(sys.modules, {"antspynet": mock_antspynet})

    result = structural.parcellate(method="dkt")
    mock_antspynet.desikan_killiany_tourville_labeling.assert_called_once()
    assert isinstance(result, InMemoryVolumeResource)


def test_parcellate_not_installed(structural: StructuralImage, mocker):
    mocker.patch.dict(sys.modules, {"antspynet": None})

    with pytest.raises(ImportError, match="kwneuro\\[antspynet\\]"):
        structural.parcellate(method="dkt")


def test_parcellate_unknown_method(structural: StructuralImage):
    with pytest.raises(ValueError, match="Unknown parcellation method"):
        structural.parcellate(method="bad_method")


def test_parcellate_caching(
    structural: StructuralImage,
    mock_ants_vol: InMemoryVolumeResource,
    mocker,
    tmp_path: Path,
):
    mock_ants_img = MagicMock()
    mocker.patch.object(
        InMemoryVolumeResource, "to_ants_image", return_value=mock_ants_img
    )
    mocker.patch.object(
        InMemoryVolumeResource, "from_ants_image", return_value=mock_ants_vol
    )
    mock_antspynet = MagicMock()
    mock_antspynet.desikan_killiany_tourville_labeling.return_value = {
        "parcellation_segmentation": mock_ants_img
    }
    mocker.patch.dict(sys.modules, {"antspynet": mock_antspynet})

    with Cache(tmp_path):
        structural.parcellate(method="dkt")
    assert mock_antspynet.desikan_killiany_tourville_labeling.call_count == 1
    assert (tmp_path / "parcellate.nii.gz").exists()

    mock_antspynet.desikan_killiany_tourville_labeling.reset_mock()
    mocker.patch.dict(sys.modules, {"antspynet": mock_antspynet})
    with Cache(tmp_path):
        structural.parcellate(method="dkt")
    mock_antspynet.desikan_killiany_tourville_labeling.assert_not_called()  # cache hit


# ---------------------------------------------------------------------------
# brain_extract (masks.py core function)
# ---------------------------------------------------------------------------


def test_brain_extract(structural: StructuralImage, mocker, tmp_path: Path):
    output_path = tmp_path / "mask.nii.gz"
    mock_run_hd_bet = mocker.patch("kwneuro.masks._run_hd_bet")

    brain_extract(volume=structural.volume, output_path=output_path)
    mock_run_hd_bet.assert_called_once()


def test_brain_extract_structural_batch(
    structural: StructuralImage, mocker, tmp_path: Path
):
    output_path = tmp_path / "mask.nii.gz"
    mock_run_hd_bet = mocker.patch("kwneuro.masks._run_hd_bet")

    brain_extract_structural_batch(cases=[(structural, output_path)])
    mock_run_hd_bet.assert_called_once()
