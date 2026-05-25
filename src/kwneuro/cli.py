from __future__ import annotations

import logging
import os
from pathlib import Path

import click

from kwneuro import masks
from kwneuro.dwi import Dwi
from kwneuro.files import (
    read_dwi_fsl,
    read_structural,
    read_volume,
    write_dwi_fsl,
    write_structural,
    write_volume,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARN"))

INPUT_FILE = click.Path(
    exists=True,
    file_okay=True,
    dir_okay=False,
    path_type=Path,
)
OUTPUT_FILE = click.Path(
    file_okay=True,
    dir_okay=False,
    path_type=Path,
)
INPUT_DIR = click.Path(
    exists=True,
    file_okay=False,
    dir_okay=True,
    path_type=Path,
)
OUTPUT_DIR = click.Path(
    file_okay=False,
    dir_okay=True,
    path_type=Path,
)


@click.group()
def kwneuro() -> None:
    """Run common kwneuro file-based workflows."""


@kwneuro.group()
def dwi() -> None:
    """DWI commands."""


@kwneuro.group()
def mask() -> None:
    """Brain mask commands."""


@kwneuro.group()
def structural() -> None:
    """Structural MRI commands."""


@dwi.command("mean-b0")
@click.option("--dwi", "dwi_path", required=True, type=INPUT_FILE)
@click.option("--bval", type=INPUT_FILE)
@click.option("--bvec", type=INPUT_FILE)
@click.option("--out", "out_path", required=True, type=OUTPUT_FILE)
def mean_b0(
    dwi_path: Path,
    bval: Path | None,
    bvec: Path | None,
    out_path: Path,
) -> None:
    """Compute the mean b=0 image from a DWI."""
    dwi_image = _read_dwi(dwi_path, bval, bvec)
    write_volume(dwi_image.compute_mean_b0(), out_path)


@dwi.command("denoise")
@click.option("--dwi", "dwi_path", required=True, type=INPUT_FILE)
@click.option("--bval", type=INPUT_FILE)
@click.option("--bvec", type=INPUT_FILE)
@click.option("--out-dwi", required=True, type=OUTPUT_FILE)
def denoise(
    dwi_path: Path,
    bval: Path | None,
    bvec: Path | None,
    out_dwi: Path,
) -> None:
    """Denoise a DWI and write NIfTI/FSL outputs."""
    dwi_image = _read_dwi(dwi_path, bval, bvec)
    write_dwi_fsl(dwi_image.denoise(), out_dwi)


@dwi.command("dti")
@click.option("--dwi", "dwi_path", required=True, type=INPUT_FILE)
@click.option("--bval", type=INPUT_FILE)
@click.option("--bvec", type=INPUT_FILE)
@click.option("--mask", "mask_path", type=INPUT_FILE)
@click.option("--out-dti", required=True, type=OUTPUT_FILE)
@click.option("--out-fa", type=OUTPUT_FILE)
@click.option("--out-md", type=OUTPUT_FILE)
def dti(
    dwi_path: Path,
    bval: Path | None,
    bvec: Path | None,
    mask_path: Path | None,
    out_dti: Path,
    out_fa: Path | None,
    out_md: Path | None,
) -> None:
    """Fit DTI and optionally write FA/MD maps."""
    dwi_image = _read_dwi(dwi_path, bval, bvec)
    brain_mask = read_volume(mask_path) if mask_path is not None else None
    dti_image = dwi_image.estimate_dti(mask=brain_mask)
    write_volume(dti_image.volume, out_dti)

    if out_fa is not None or out_md is not None:
        fa, md = dti_image.get_fa_md()
        if out_fa is not None:
            write_volume(fa, out_fa)
        if out_md is not None:
            write_volume(md, out_md)


@mask.command("dwi-batch")
@click.option("--inputs", required=True, type=INPUT_DIR)
@click.option("--outputs", required=True, type=OUTPUT_DIR)
@click.option(
    "--sequential",
    is_flag=True,
    help="Disable HD-BET multiprocessing for embedded Python environments.",
)
def dwi_batch(inputs: Path, outputs: Path, sequential: bool) -> None:
    """Batch brain extraction for DWI files under an input directory."""
    cases = _find_dwi_mask_cases(inputs, outputs)
    masks.brain_extract_dwi_batch(cases, sequential=sequential)


@structural.command("bias-correct")
@click.option("--image", "image_path", required=True, type=INPUT_FILE)
@click.option("--out", "out_path", required=True, type=OUTPUT_FILE)
def bias_correct(image_path: Path, out_path: Path) -> None:
    """Apply N4 bias-field correction to a structural image."""
    structural_image = read_structural(image_path)
    write_structural(structural_image.correct_bias(), out_path)


def _read_dwi(
    dwi_path: Path,
    bval: Path | None,
    bvec: Path | None,
) -> Dwi:
    return read_dwi_fsl(dwi_path, bval=bval, bvec=bvec)


def _find_dwi_mask_cases(inputs: Path, outputs: Path) -> list[tuple[Dwi, Path]]:
    cases: list[tuple[Dwi, Path]] = []
    for dwi_input_path in inputs.rglob("*_dwi.nii.gz"):
        base = dwi_input_path.with_name(dwi_input_path.name.removesuffix(".nii.gz"))
        base_out = outputs.joinpath(base.relative_to(inputs))
        output_path = base_out.with_name(base_out.name + "_mask.nii.gz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cases.append((read_dwi_fsl(dwi_input_path), output_path))
    return cases


__all__ = [
    "kwneuro",
]
