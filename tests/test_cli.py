from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from click.testing import CliRunner, Result

from kwneuro.cli import kwneuro
from kwneuro.dti import Dti
from kwneuro.dwi import Dwi
from kwneuro.files import (
    read_dwi_fsl,
    read_structural,
    read_volume,
    write_dwi_fsl,
    write_structural,
    write_volume,
)
from kwneuro.io import NiftiVolumeResource
from kwneuro.reg import TransformResource
from kwneuro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
)
from kwneuro.structural import StructuralImage


@pytest.fixture
def dwi() -> Dwi:
    rng = np.random.default_rng(2656542)
    return Dwi(
        volume=InMemoryVolumeResource(
            array=rng.random(size=(2, 3, 4, 4)),
            affine=np.diag([2.0, 2.0, 2.0, 1.0]),
            metadata=dict(nib.Nifti1Header()),
        ),
        bval=InMemoryBvalResource(np.array([0.0, 0.0, 1000.0, 2000.0])),
        bvec=InMemoryBvecResource(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            )
        ),
    )


@pytest.fixture
def structural() -> StructuralImage:
    rng = np.random.default_rng(7224)
    return StructuralImage(
        InMemoryVolumeResource(
            array=rng.random(size=(3, 4, 5)),
            affine=np.diag([1.5, 1.5, 1.5, 1.0]),
            metadata=dict(nib.Nifti1Header()),
        )
    )


def invoke_ok(runner: CliRunner, args: list[str]) -> Result:
    result = runner.invoke(kwneuro, args)
    assert result.exit_code == 0, result.output
    return result


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


def test_dwi_mean_b0_command(dwi: Dwi, tmp_path: Path) -> None:
    dwi_path = tmp_path / "sub-01_dwi.nii.gz"
    out_path = tmp_path / "mean_b0.nii.gz"
    write_dwi_fsl(dwi, dwi_path)

    invoke_ok(
        CliRunner(),
        ["dwi", "mean-b0", "--dwi", str(dwi_path), "--out", str(out_path)],
    )

    expected = dwi.volume.get_array()[..., :2].mean(axis=3)
    assert np.allclose(read_volume(out_path).get_array(), expected)


def test_dwi_denoise_command_writes_dwi(dwi: Dwi, mocker, tmp_path: Path) -> None:
    dwi_path = tmp_path / "sub-01_dwi.nii.gz"
    out_path = tmp_path / "denoised.nii.gz"
    denoised = Dwi(
        volume=InMemoryVolumeResource(
            array=dwi.volume.get_array() + 1.0,
            affine=dwi.volume.get_affine(),
            metadata=dwi.volume.get_metadata(),
        ),
        bval=dwi.bval,
        bvec=dwi.bvec,
    )
    write_dwi_fsl(dwi, dwi_path)
    mock_denoise = mocker.patch.object(Dwi, "denoise", return_value=denoised)

    invoke_ok(
        CliRunner(),
        ["dwi", "denoise", "--dwi", str(dwi_path), "--out-dwi", str(out_path)],
    )

    mock_denoise.assert_called_once()
    assert (tmp_path / "denoised.bval").exists()
    assert (tmp_path / "denoised.bvec").exists()
    assert_dwi_allclose(read_dwi_fsl(out_path), denoised)


def test_dwi_dti_command_writes_requested_outputs(
    dwi: Dwi, mocker, tmp_path: Path
) -> None:
    dwi_path = tmp_path / "sub-01_dwi.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    out_dti = tmp_path / "dti.nii.gz"
    out_fa = tmp_path / "fa.nii.gz"
    out_md = tmp_path / "md.nii.gz"
    write_dwi_fsl(dwi, dwi_path)
    write_volume(
        InMemoryVolumeResource(
            array=np.ones(dwi.volume.get_array().shape[:3]),
            affine=dwi.volume.get_affine(),
            metadata=dwi.volume.get_metadata(),
        ),
        mask_path,
    )

    dti_result = Dti(
        InMemoryVolumeResource(
            array=np.ones((*dwi.volume.get_array().shape[:3], 6)),
            affine=dwi.volume.get_affine(),
            metadata=dwi.volume.get_metadata(),
        )
    )
    fa = InMemoryVolumeResource(
        array=np.full(dwi.volume.get_array().shape[:3], 0.5),
        affine=dwi.volume.get_affine(),
        metadata=dwi.volume.get_metadata(),
    )
    md = InMemoryVolumeResource(
        array=np.full(dwi.volume.get_array().shape[:3], 0.001),
        affine=dwi.volume.get_affine(),
        metadata=dwi.volume.get_metadata(),
    )
    mock_estimate = mocker.patch.object(Dwi, "estimate_dti", return_value=dti_result)
    mock_get_fa_md = mocker.patch.object(Dti, "get_fa_md", return_value=(fa, md))

    invoke_ok(
        CliRunner(),
        [
            "dwi",
            "dti",
            "--dwi",
            str(dwi_path),
            "--mask",
            str(mask_path),
            "--out-dti",
            str(out_dti),
            "--out-fa",
            str(out_fa),
            "--out-md",
            str(out_md),
        ],
    )

    mock_estimate.assert_called_once()
    assert isinstance(mock_estimate.call_args.kwargs["mask"], NiftiVolumeResource)
    mock_get_fa_md.assert_called_once()
    assert np.allclose(read_volume(out_dti).get_array(), dti_result.volume.get_array())
    assert np.allclose(read_volume(out_fa).get_array(), fa.get_array())
    assert np.allclose(read_volume(out_md).get_array(), md.get_array())


def test_mask_dwi_batch_command(mocker, dwi: Dwi, tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    case1_dir = input_dir / "a/b1/c"
    case2_dir = input_dir / "a/b2"
    case1_dir.mkdir(parents=True)
    case2_dir.mkdir(parents=True)
    write_dwi_fsl(dwi, case1_dir / "an_image_dwi.nii.gz")
    write_dwi_fsl(dwi, case2_dir / "another_image_dwi.nii.gz")
    mock_brain_extract_batch = mocker.patch("kwneuro.cli.masks.brain_extract_dwi_batch")

    invoke_ok(
        CliRunner(),
        [
            "mask",
            "dwi-batch",
            "--inputs",
            str(input_dir),
            "--outputs",
            str(output_dir),
            "--sequential",
        ],
    )

    mock_brain_extract_batch.assert_called_once()
    assert mock_brain_extract_batch.call_args.kwargs["sequential"] is True
    cases = mock_brain_extract_batch.call_args.args[0]
    output_paths = {case[1] for case in cases}
    assert len(cases) == 2
    assert all(isinstance(case[0], Dwi) for case in cases)
    assert output_paths == {
        output_dir / "a/b1/c/an_image_dwi_mask.nii.gz",
        output_dir / "a/b2/another_image_dwi_mask.nii.gz",
    }


def test_mask_dwi_batch_command_rejects_missing_input() -> None:
    result = CliRunner().invoke(
        kwneuro,
        [
            "mask",
            "dwi-batch",
            "--inputs",
            "/does/not/exist",
            "--outputs",
            "/tmp/out",
        ],
    )

    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_mask_structural_batch_command(
    mocker, structural: StructuralImage, tmp_path: Path
) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    case1_dir = input_dir / "a/b1/c"
    case2_dir = input_dir / "a/b2"
    case1_dir.mkdir(parents=True)
    case2_dir.mkdir(parents=True)
    write_structural(structural, case1_dir / "sub-01_T1w.nii.gz")
    write_structural(structural, case2_dir / "sub-02_T2w.nii.gz")
    mock_brain_extract_batch = mocker.patch(
        "kwneuro.cli.masks.brain_extract_structural_batch"
    )

    invoke_ok(
        CliRunner(),
        [
            "mask",
            "structural-batch",
            "--inputs",
            str(input_dir),
            "--outputs",
            str(output_dir),
            "--sequential",
        ],
    )

    mock_brain_extract_batch.assert_called_once()
    assert mock_brain_extract_batch.call_args.kwargs["sequential"] is True
    cases = mock_brain_extract_batch.call_args.args[0]
    output_paths = {case[1] for case in cases}
    assert len(cases) == 2
    assert all(isinstance(case[0], StructuralImage) for case in cases)
    assert output_paths == {
        output_dir / "a/b1/c/sub-01_T1w_mask.nii.gz",
        output_dir / "a/b2/sub-02_T2w_mask.nii.gz",
    }


def test_registration_volumes_command_writes_outputs(
    mocker, structural: StructuralImage, tmp_path: Path
) -> None:
    fixed_path = tmp_path / "fixed.nii.gz"
    moving_path = tmp_path / "moving.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    moving_mask_path = tmp_path / "moving_mask.nii.gz"
    out_path = tmp_path / "registered.nii.gz"
    out_transform = tmp_path / "transform"
    registered = InMemoryVolumeResource(
        array=structural.volume.get_array() + 1.0,
        affine=structural.volume.get_affine(),
        metadata=structural.volume.get_metadata(),
    )
    write_volume(structural.volume, fixed_path)
    write_volume(structural.volume, moving_path)
    write_volume(
        InMemoryVolumeResource(
            array=np.ones(structural.volume.get_array().shape),
            affine=structural.volume.get_affine(),
            metadata=structural.volume.get_metadata(),
        ),
        mask_path,
    )
    write_volume(
        InMemoryVolumeResource(
            array=np.ones(structural.volume.get_array().shape),
            affine=structural.volume.get_affine(),
            metadata=structural.volume.get_metadata(),
        ),
        moving_mask_path,
    )
    transform = make_transform(tmp_path, "volumes")
    mock_register = mocker.patch(
        "kwneuro.cli.register_volumes", return_value=(registered, transform)
    )

    invoke_ok(
        CliRunner(),
        [
            "registration",
            "volumes",
            "--fixed",
            str(fixed_path),
            "--moving",
            str(moving_path),
            "--out",
            str(out_path),
            "--out-transform",
            str(out_transform),
            "--transform",
            "Affine",
            "--mask",
            str(mask_path),
            "--moving-mask",
            str(moving_mask_path),
        ],
    )

    mock_register.assert_called_once()
    call_kwargs = mock_register.call_args.kwargs
    assert isinstance(call_kwargs["fixed"], NiftiVolumeResource)
    assert isinstance(call_kwargs["moving"], NiftiVolumeResource)
    assert isinstance(call_kwargs["mask"], NiftiVolumeResource)
    assert isinstance(call_kwargs["moving_mask"], NiftiVolumeResource)
    assert call_kwargs["type_of_transform"] == "Affine"
    assert np.allclose(read_volume(out_path).get_array(), registered.get_array())
    assert (out_transform / "transform.json").exists()


def test_registration_dwi_to_structural_command_saves_transform(
    mocker, dwi: Dwi, structural: StructuralImage, tmp_path: Path
) -> None:
    dwi_path = tmp_path / "sub-01_dwi.nii.gz"
    structural_path = tmp_path / "sub-01_T1w.nii.gz"
    dwi_mask_path = tmp_path / "dwi_mask.nii.gz"
    structural_mask_path = tmp_path / "structural_mask.nii.gz"
    out_transform = tmp_path / "dwi_to_structural_transform"
    write_dwi_fsl(dwi, dwi_path)
    write_structural(structural, structural_path)
    write_volume(
        InMemoryVolumeResource(
            array=np.ones(dwi.volume.get_array().shape[:3]),
            affine=dwi.volume.get_affine(),
            metadata=dwi.volume.get_metadata(),
        ),
        dwi_mask_path,
    )
    write_volume(
        InMemoryVolumeResource(
            array=np.ones(structural.volume.get_array().shape),
            affine=structural.volume.get_affine(),
            metadata=structural.volume.get_metadata(),
        ),
        structural_mask_path,
    )
    transform = make_transform(tmp_path, "dwi_to_structural")
    mock_register = mocker.patch(
        "kwneuro.cli.register_dwi_to_structural", return_value=transform
    )

    invoke_ok(
        CliRunner(),
        [
            "registration",
            "dwi-to-structural",
            "--dwi",
            str(dwi_path),
            "--structural",
            str(structural_path),
            "--out-transform",
            str(out_transform),
            "--transform",
            "Rigid",
            "--dwi-mask",
            str(dwi_mask_path),
            "--structural-mask",
            str(structural_mask_path),
        ],
    )

    mock_register.assert_called_once()
    call_kwargs = mock_register.call_args.kwargs
    assert isinstance(call_kwargs["dwi"], Dwi)
    assert isinstance(call_kwargs["structural"], StructuralImage)
    assert isinstance(call_kwargs["dwi_mask"], NiftiVolumeResource)
    assert isinstance(call_kwargs["structural_mask"], NiftiVolumeResource)
    assert call_kwargs["type_of_transform"] == "Rigid"
    assert (out_transform / "transform.json").exists()


def test_registration_apply_command_writes_transformed_volume(
    mocker, structural: StructuralImage, tmp_path: Path
) -> None:
    transform_dir = tmp_path / "transform"
    transform_dir.mkdir()
    fixed_path = tmp_path / "fixed.nii.gz"
    moving_path = tmp_path / "moving.nii.gz"
    out_path = tmp_path / "warped.nii.gz"
    transformed = InMemoryVolumeResource(
        array=structural.volume.get_array() + 3.0,
        affine=structural.volume.get_affine(),
        metadata=structural.volume.get_metadata(),
    )
    write_volume(structural.volume, fixed_path)
    write_volume(structural.volume, moving_path)
    mock_transform = mocker.Mock()
    mock_transform.apply.return_value = transformed
    mock_load = mocker.patch.object(
        TransformResource, "load", return_value=mock_transform
    )

    invoke_ok(
        CliRunner(),
        [
            "registration",
            "apply",
            "--transform",
            str(transform_dir),
            "--fixed",
            str(fixed_path),
            "--moving",
            str(moving_path),
            "--out",
            str(out_path),
            "--invert",
            "--interpolation",
            "genericLabel",
        ],
    )

    mock_load.assert_called_once_with(transform_dir)
    mock_transform.apply.assert_called_once()
    call_kwargs = mock_transform.apply.call_args.kwargs
    assert isinstance(call_kwargs["fixed"], NiftiVolumeResource)
    assert isinstance(call_kwargs["moving"], NiftiVolumeResource)
    assert call_kwargs["invert"] is True
    assert call_kwargs["interpolation"] == "genericLabel"
    assert np.allclose(read_volume(out_path).get_array(), transformed.get_array())


def test_structural_bias_correct_command(
    structural: StructuralImage, mocker, tmp_path: Path
) -> None:
    image_path = tmp_path / "sub-01_T1w.nii.gz"
    out_path = tmp_path / "bias_corrected.nii.gz"
    write_structural(structural, image_path)
    corrected = StructuralImage(
        InMemoryVolumeResource(
            array=structural.volume.get_array() + 2.0,
            affine=structural.volume.get_affine(),
            metadata=structural.volume.get_metadata(),
        )
    )
    mock_correct_bias = mocker.patch.object(
        StructuralImage, "correct_bias", return_value=corrected
    )

    invoke_ok(
        CliRunner(),
        [
            "structural",
            "bias-correct",
            "--image",
            str(image_path),
            "--out",
            str(out_path),
        ],
    )

    mock_correct_bias.assert_called_once()
    assert np.allclose(
        read_structural(out_path).volume.get_array(),
        corrected.volume.get_array(),
    )


def test_structural_extract_brain_command(
    structural: StructuralImage, mocker, tmp_path: Path
) -> None:
    image_path = tmp_path / "sub-01_T1w.nii.gz"
    out_mask = tmp_path / "brain_mask.nii.gz"
    write_structural(structural, image_path)
    mask = InMemoryVolumeResource(
        array=np.ones(structural.volume.get_array().shape),
        affine=structural.volume.get_affine(),
        metadata=structural.volume.get_metadata(),
    )
    mock_extract_brain = mocker.patch.object(
        StructuralImage, "extract_brain", return_value=mask
    )

    invoke_ok(
        CliRunner(),
        [
            "structural",
            "extract-brain",
            "--image",
            str(image_path),
            "--out-mask",
            str(out_mask),
        ],
    )

    mock_extract_brain.assert_called_once()
    assert np.allclose(read_volume(out_mask).get_array(), mask.get_array())


def test_structural_segment_tissues_command(
    structural: StructuralImage, mocker, tmp_path: Path
) -> None:
    image_path = tmp_path / "sub-01_T1w.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    out_path = tmp_path / "segmentation.nii.gz"
    write_structural(structural, image_path)
    write_volume(
        InMemoryVolumeResource(
            array=np.ones(structural.volume.get_array().shape),
            affine=structural.volume.get_affine(),
            metadata=structural.volume.get_metadata(),
        ),
        mask_path,
    )
    segmentation = InMemoryVolumeResource(
        array=np.ones(structural.volume.get_array().shape),
        affine=structural.volume.get_affine(),
        metadata=structural.volume.get_metadata(),
    )
    mock_segment = mocker.patch.object(
        StructuralImage, "segment_tissues", return_value=segmentation
    )

    invoke_ok(
        CliRunner(),
        [
            "structural",
            "segment-tissues",
            "--image",
            str(image_path),
            "--out",
            str(out_path),
            "--mask",
            str(mask_path),
            "--method",
            "atropos",
        ],
    )

    mock_segment.assert_called_once()
    call_kwargs = mock_segment.call_args.kwargs
    assert isinstance(call_kwargs["mask"], NiftiVolumeResource)
    assert call_kwargs["method"] == "atropos"
    assert np.allclose(read_volume(out_path).get_array(), segmentation.get_array())


def test_structural_parcellate_command(
    structural: StructuralImage, mocker, tmp_path: Path
) -> None:
    image_path = tmp_path / "sub-01_T1w.nii.gz"
    out_path = tmp_path / "parcellation.nii.gz"
    write_structural(structural, image_path)
    parcellation = InMemoryVolumeResource(
        array=np.ones(structural.volume.get_array().shape),
        affine=structural.volume.get_affine(),
        metadata=structural.volume.get_metadata(),
    )
    mock_parcellate = mocker.patch.object(
        StructuralImage, "parcellate", return_value=parcellation
    )

    invoke_ok(
        CliRunner(),
        [
            "structural",
            "parcellate",
            "--image",
            str(image_path),
            "--out",
            str(out_path),
            "--method",
            "dkt",
        ],
    )

    mock_parcellate.assert_called_once_with(method="dkt")
    assert np.allclose(read_volume(out_path).get_array(), parcellation.get_array())


def make_transform(tmp_path: Path, name: str) -> TransformResource:
    source_dir = tmp_path / f"{name}_source_transform"
    source_dir.mkdir()
    affine_path = source_dir / "0GenericAffine.mat"
    warp_path = source_dir / "1Warp.nii.gz"
    affine_path.write_text("affine")
    warp_path.write_text("warp")
    return TransformResource(
        _ants_fwd_paths=[str(warp_path), str(affine_path)],
        _ants_inv_paths=[str(affine_path)],
    )
