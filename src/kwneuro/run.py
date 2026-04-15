"""
.. click:: kwneuro.run:gen_masks
    :prog: gen_masks
    :nested: short
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click

from kwneuro import masks
from kwneuro.dwi import Dwi
from kwneuro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARN"))


@click.command("gen_masks")
@click.option(
    "--inputs",
    "-i",
    required=True,
    type=Path,
    help="Root directory to search for inputs.",
)
@click.option(
    "--outputs",
    "-o",
    required=True,
    type=Path,
    help="Root directory to write outputs.",
)
def gen_masks(inputs: Path, outputs: Path) -> None:
    """
    Recursively find and process dwi images and create hd_bet masks for each.
    Preserves directory structure in output.

    Searches for input files: ``<ID>_dwi.nii.gz``, ``<ID>.bval``, ``<ID>.bvec``

    Produces output files: ``<ID>_mask.nii.gz``
    \f
    See :func:`kwneuro.masks.brain_extract_dwi_batch` for details.
    """

    cases: list[tuple[Dwi, Path]] = []

    if not inputs.exists():
        error_message = f"Input directory {inputs} not found."
        raise FileNotFoundError(error_message)

    for dwi_input_path in inputs.rglob("*_dwi.nii.gz"):
        base = dwi_input_path.with_name(dwi_input_path.name.removesuffix(".nii.gz"))
        base_out = outputs.joinpath(base.relative_to(inputs))

        cases.append(
            (
                Dwi(
                    volume=NiftiVolumeResource(base.with_suffix(".nii.gz")),
                    bval=FslBvalResource(base.with_suffix(".bval")),
                    bvec=FslBvecResource(base.with_suffix(".bvec")),
                ),
                base_out.with_name(base_out.name + "_mask.nii.gz"),
            )
        )
        base_out.parent.mkdir(parents=True, exist_ok=True)

    masks.brain_extract_dwi_batch(cases)
