"""Tests for `kwneuro.util`."""

from __future__ import annotations

import numpy as np

from kwneuro.util import update_volume_metadata


def test_update_volume_metadata_preserves_unknown_keys() -> None:
    """Non-NIfTI keys pass through to the output dict unchanged.

    VolumeResource implementations may carry custom metadata alongside
    the standard NIfTI fields (e.g. a Slicer scene node ID, a user-
    supplied processing tag). Those keys should survive pipeline stages
    that route through `update_volume_metadata` rather than being
    dropped or raising.
    """
    array = np.zeros((3, 4, 5), dtype=np.float32)
    metadata = {
        "qform_code": 1,  # valid NIfTI field
        "slicer_node_id": "vtkMRMLScalarVolumeNode1",  # not a NIfTI field
        "arbitrary_tool_annotation": "anything",  # not a NIfTI field
    }

    result = update_volume_metadata(metadata, array)

    # Custom keys survive verbatim.
    assert result["slicer_node_id"] == "vtkMRMLScalarVolumeNode1"
    assert result["arbitrary_tool_annotation"] == "anything"
    # NIfTI keys are still normalised through the header.
    assert int(result["qform_code"]) == 1


def test_update_volume_metadata_preserves_known_keys() -> None:
    """Valid NIfTI keys from the input metadata survive the update."""
    array = np.zeros((3, 4, 5), dtype=np.float32)
    metadata = {"qform_code": 2, "sform_code": 1}

    result = update_volume_metadata(metadata, array)

    assert int(result["qform_code"]) == 2
    assert int(result["sform_code"]) == 1


def test_update_volume_metadata_sets_shape_and_dtype() -> None:
    """The function overrides dim/data-shape based on the passed array."""
    array = np.zeros((6, 7, 8), dtype=np.int16)
    result = update_volume_metadata({}, array)

    # dim is (ndims, *shape, padding), padded to 8 entries.
    dim = result["dim"]
    assert int(dim[0]) == 3
    assert (int(dim[1]), int(dim[2]), int(dim[3])) == (6, 7, 8)


def test_update_volume_metadata_empty_metadata_ok() -> None:
    """Empty input metadata is a no-op on the field-setting phase."""
    array = np.zeros((3, 4, 5), dtype=np.float32)
    result = update_volume_metadata({}, array)
    assert isinstance(result, dict)
    assert "dim" in result  # set by set_data_shape


def test_update_volume_metadata_intent_applied_when_provided() -> None:
    """intent_code / intent_name flow through to the header."""
    array = np.zeros((3, 4, 5), dtype=np.float32)
    result = update_volume_metadata(
        {},
        array,
        intent_code="NIFTI_INTENT_ESTIMATE",
        intent_name="FA",
    )
    # The header's intent_code / intent_name should reflect what we passed.
    # intent_code is stored as an int; NIFTI_INTENT_ESTIMATE is 1001.
    assert int(result["intent_code"]) == 1001
    # intent_name is stored as a fixed-length string array; strip and decode.
    name = result["intent_name"]
    name_str = (
        name.tobytes().rstrip(b"\x00").decode()
        if hasattr(name, "tobytes")
        else str(name)
    )
    assert "FA" in name_str
