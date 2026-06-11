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
from kwneuro.reg import TransformResource, register_dwi_to_structural, register_volumes
from kwneuro.resource import VolumeResource
from kwneuro.structural import StructuralImage

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
def registration() -> None:
    """Registration and transform commands."""


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


@mask.command("structural-batch")
@click.option("--inputs", required=True, type=INPUT_DIR)
@click.option("--outputs", required=True, type=OUTPUT_DIR)
@click.option(
    "--sequential",
    is_flag=True,
    help="Disable HD-BET multiprocessing for embedded Python environments.",
)
def structural_batch(
    inputs: Path,
    outputs: Path,
    sequential: bool,
) -> None:
    """Batch brain extraction for structural NIfTI files under an input directory."""
    cases = _find_structural_mask_cases(inputs, outputs)
    masks.brain_extract_structural_batch(cases, sequential=sequential)


@registration.command("volumes")
@click.option("--fixed", "fixed_path", required=True, type=INPUT_FILE)
@click.option("--moving", "moving_path", required=True, type=INPUT_FILE)
@click.option("--out", "out_path", required=True, type=OUTPUT_FILE)
@click.option("--out-transform", required=True, type=OUTPUT_DIR)
@click.option("--transform", "type_of_transform", default="SyN", show_default=True)
@click.option("--mask", "mask_path", type=INPUT_FILE)
@click.option("--moving-mask", "moving_mask_path", type=INPUT_FILE)
def register_volume_command(
    fixed_path: Path,
    moving_path: Path,
    out_path: Path,
    out_transform: Path,
    type_of_transform: str,
    mask_path: Path | None,
    moving_mask_path: Path | None,
) -> None:
    """Register a scalar moving volume to a fixed reference volume."""
    registered, transform = register_volumes(
        fixed=read_volume(fixed_path),
        moving=read_volume(moving_path),
        type_of_transform=type_of_transform,
        mask=_read_optional_volume(mask_path),
        moving_mask=_read_optional_volume(moving_mask_path),
    )
    write_volume(registered, out_path)
    transform.save(out_transform)


@registration.command("dwi-to-structural")
@click.option("--dwi", "dwi_path", required=True, type=INPUT_FILE)
@click.option("--bval", type=INPUT_FILE)
@click.option("--bvec", type=INPUT_FILE)
@click.option("--structural", "structural_path", required=True, type=INPUT_FILE)
@click.option("--out-transform", required=True, type=OUTPUT_DIR)
@click.option("--transform", "type_of_transform", default="Rigid", show_default=True)
@click.option("--dwi-mask", "dwi_mask_path", type=INPUT_FILE)
@click.option("--structural-mask", "structural_mask_path", type=INPUT_FILE)
def register_dwi_to_structural_command(
    dwi_path: Path,
    bval: Path | None,
    bvec: Path | None,
    structural_path: Path,
    out_transform: Path,
    type_of_transform: str,
    dwi_mask_path: Path | None,
    structural_mask_path: Path | None,
) -> None:
    """Register a DWI mean b0 image to a structural image."""
    transform = register_dwi_to_structural(
        dwi=_read_dwi(dwi_path, bval, bvec),
        structural=read_structural(structural_path),
        type_of_transform=type_of_transform,
        dwi_mask=_read_optional_volume(dwi_mask_path),
        structural_mask=_read_optional_volume(structural_mask_path),
    )
    transform.save(out_transform)


@registration.command("apply")
@click.option("--transform", "transform_dir", required=True, type=INPUT_DIR)
@click.option("--fixed", "fixed_path", required=True, type=INPUT_FILE)
@click.option("--moving", "moving_path", required=True, type=INPUT_FILE)
@click.option("--out", "out_path", required=True, type=OUTPUT_FILE)
@click.option("--invert", is_flag=True)
@click.option("--interpolation", default="linear", show_default=True)
def apply_transform_command(
    transform_dir: Path,
    fixed_path: Path,
    moving_path: Path,
    out_path: Path,
    invert: bool,
    interpolation: str,
) -> None:
    """Apply a saved transform to a moving volume."""
    transform = TransformResource.load(transform_dir)
    transformed = transform.apply(
        fixed=read_volume(fixed_path),
        moving=read_volume(moving_path),
        invert=invert,
        interpolation=interpolation,
    )
    write_volume(transformed, out_path)


@structural.command("bias-correct")
@click.option("--image", "image_path", required=True, type=INPUT_FILE)
@click.option("--out", "out_path", required=True, type=OUTPUT_FILE)
def bias_correct(image_path: Path, out_path: Path) -> None:
    """Apply N4 bias-field correction to a structural image."""
    structural_image = read_structural(image_path)
    write_structural(structural_image.correct_bias(), out_path)


@structural.command("extract-brain")
@click.option("--image", "image_path", required=True, type=INPUT_FILE)
@click.option("--out-mask", required=True, type=OUTPUT_FILE)
def structural_extract_brain(image_path: Path, out_mask: Path) -> None:
    """Extract a brain mask from a structural image."""
    structural_image = read_structural(image_path)
    write_volume(structural_image.extract_brain(), out_mask)


@structural.command("segment-tissues")
@click.option("--image", "image_path", required=True, type=INPUT_FILE)
@click.option("--out", "out_path", required=True, type=OUTPUT_FILE)
@click.option("--mask", "mask_path", type=INPUT_FILE)
@click.option(
    "--method",
    type=click.Choice(["atropos", "deep_atropos"]),
    default="atropos",
    show_default=True,
)
def structural_segment_tissues(
    image_path: Path,
    out_path: Path,
    mask_path: Path | None,
    method: str,
) -> None:
    """Segment structural image tissues into labeled classes."""
    structural_image = read_structural(image_path)
    write_volume(
        structural_image.segment_tissues(
            mask=_read_optional_volume(mask_path), method=method
        ),
        out_path,
    )


@structural.command("parcellate")
@click.option("--image", "image_path", required=True, type=INPUT_FILE)
@click.option("--out", "out_path", required=True, type=OUTPUT_FILE)
@click.option(
    "--method",
    type=click.Choice(["dkt"]),
    default="dkt",
    show_default=True,
)
def structural_parcellate(image_path: Path, out_path: Path, method: str) -> None:
    """Parcellate a structural image."""
    structural_image = read_structural(image_path)
    write_volume(structural_image.parcellate(method=method), out_path)


def _read_dwi(
    dwi_path: Path,
    bval: Path | None,
    bvec: Path | None,
) -> Dwi:
    return read_dwi_fsl(dwi_path, bval=bval, bvec=bvec)


def _read_optional_volume(path: Path | None) -> VolumeResource | None:
    return read_volume(path) if path is not None else None


def _find_dwi_mask_cases(inputs: Path, outputs: Path) -> list[tuple[Dwi, Path]]:
    cases: list[tuple[Dwi, Path]] = []
    for dwi_input_path in inputs.rglob("*_dwi.nii.gz"):
        base = dwi_input_path.with_name(dwi_input_path.name.removesuffix(".nii.gz"))
        base_out = outputs.joinpath(base.relative_to(inputs))
        output_path = base_out.with_name(base_out.name + "_mask.nii.gz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cases.append((read_dwi_fsl(dwi_input_path), output_path))
    return cases


def _find_structural_mask_cases(
    inputs: Path, outputs: Path
) -> list[tuple[StructuralImage, Path]]:
    cases: list[tuple[StructuralImage, Path]] = []
    for structural_input_path in inputs.rglob("*.nii.gz"):
        base = structural_input_path.with_name(
            structural_input_path.name.removesuffix(".nii.gz")
        )
        base_out = outputs.joinpath(base.relative_to(inputs))
        output_path = base_out.with_name(base_out.name + "_mask.nii.gz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cases.append((read_structural(structural_input_path), output_path))
    return cases


__all__ = [
    "kwneuro",
]
