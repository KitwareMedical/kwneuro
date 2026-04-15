from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from kwneuro.dwi import Dwi
from kwneuro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
)
from kwneuro.run import gen_masks


@pytest.fixture
def volume_array():
    rng = np.random.default_rng(2656542)
    return rng.random(size=(2, 3, 2, 3), dtype=float)


@pytest.fixture
def dwi(volume_array) -> Dwi:
    """An example in-memory Dwi"""
    return Dwi(
        volume=InMemoryVolumeResource(array=volume_array),
        bval=InMemoryBvalResource(np.array([500.0, 1000.0])),
        bvec=InMemoryBvecResource(
            np.array(
                [
                    [
                        1.0 / np.sqrt(3),
                        1.0 / np.sqrt(3),
                        1.0 / np.sqrt(3),
                    ],
                    [2.0 / np.sqrt(20), 0.0, -4.0 / np.sqrt(20)],
                ]
            )
        ),
    )


def test_gen_masks(mocker, dwi):
    runner = CliRunner()
    mock_brain_extract_batch = mocker.patch("kwneuro.masks.brain_extract_dwi_batch")
    with tempfile.TemporaryDirectory() as work_dir:
        input_dir = Path(work_dir) / "inputs"
        output_dir = Path(work_dir) / "outputs"
        case1_dir = input_dir / "a/b1/c"
        case2_dir = input_dir / "a/b2/"
        img1_name = "an_image"
        img2_name = "another_image"
        for case_dir, img_name in [(case1_dir, img1_name), (case2_dir, img2_name)]:
            case_dir.mkdir(parents=True)
            dwi.save(case_dir, f"{img_name}_dwi")  # saves to nii.gz

        runner.invoke(
            gen_masks,
            ["--inputs", str(input_dir), "--outputs", str(output_dir)],
        )
        mock_brain_extract_batch.assert_called_once()
        cases = mock_brain_extract_batch.call_args.args[0]
        assert len(cases) == 2  # ensure both cases were found for processing


def test_gen_masks_nonexistent_inputs():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as work_dir:
        input_dir = Path(work_dir) / "dir_that_does_not_exist"
        output_dir = Path(work_dir) / "outputs"
        output_dir.mkdir()

        # We will check that an exception is raised, but we cannot use "with pytests.raises..."
        # because the clock CliRunner swallows the exception. We have to check the result.
        click_result = runner.invoke(
            gen_masks,
            ["--inputs", str(input_dir), "--outputs", str(output_dir)],
        )
        assert isinstance(click_result.exception, FileNotFoundError)
