"""User-facing file-first helpers that adapt paths to kwneuro resources.

This module is a convenience layer for scripts, notebooks, and CLI-style
workflows. The internal on-disk resource implementations live in
``kwneuro.io``.
"""

from __future__ import annotations

from pathlib import Path

from kwneuro.dwi import Dwi
from kwneuro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource
from kwneuro.resource import VolumeResource
from kwneuro.structural import StructuralImage
from kwneuro.util import PathLike, normalize_path


def read_volume(path: PathLike) -> NiftiVolumeResource:
    """Create a lazy volume resource from a NIfTI path."""
    return NiftiVolumeResource(path)


def write_volume(volume: VolumeResource, path: PathLike) -> NiftiVolumeResource:
    """Write a volume resource to a NIfTI path and return the on-disk resource."""
    output_path = _prepare_output_path(path)
    return NiftiVolumeResource.save(volume, output_path)


def read_dwi_fsl(
    volume: PathLike,
    bval: PathLike | None = None,
    bvec: PathLike | None = None,
) -> Dwi:
    """Create a DWI from a NIfTI volume and FSL b-value/b-vector files.

    When ``bval`` or ``bvec`` are omitted, they are inferred from the DWI volume
    path by replacing a ``.nii`` or ``.nii.gz`` suffix with ``.bval`` and
    ``.bvec``. This covers common BIDS-style DWI sidecars when the NIfTI path is
    passed directly.
    """
    volume_path = normalize_path(volume)
    bval_path = (
        normalize_path(bval) if bval is not None else _infer_sidecar(volume_path, "bval")
    )
    bvec_path = (
        normalize_path(bvec) if bvec is not None else _infer_sidecar(volume_path, "bvec")
    )

    return Dwi(
        volume=NiftiVolumeResource(volume_path),
        bval=FslBvalResource(bval_path),
        bvec=FslBvecResource(bvec_path),
    )


def write_dwi_fsl(
    dwi: Dwi,
    volume: PathLike,
    bval: PathLike | None = None,
    bvec: PathLike | None = None,
) -> Dwi:
    """Write a DWI to NIfTI plus FSL b-value/b-vector files.

    When ``bval`` or ``bvec`` are omitted, sidecar paths are inferred from the
    output volume path in the same way as :func:`read_dwi_fsl`.
    """
    volume_path = _prepare_output_path(volume)
    bval_path = (
        _prepare_output_path(bval)
        if bval is not None
        else _prepare_output_path(_infer_sidecar(volume_path, "bval"))
    )
    bvec_path = (
        _prepare_output_path(bvec)
        if bvec is not None
        else _prepare_output_path(_infer_sidecar(volume_path, "bvec"))
    )

    return Dwi(
        volume=NiftiVolumeResource.save(dwi.volume, volume_path),
        bval=FslBvalResource.save(dwi.bval, bval_path),
        bvec=FslBvecResource.save(dwi.bvec, bvec_path),
    )


def read_structural(path: PathLike) -> StructuralImage:
    """Create a structural image from a lazy NIfTI volume resource."""
    return StructuralImage(volume=NiftiVolumeResource(path))


def write_structural(
    structural: StructuralImage, path: PathLike
) -> StructuralImage:
    """Write a structural image to a NIfTI path and return the on-disk image."""
    output_path = _prepare_output_path(path)
    return StructuralImage(
        volume=NiftiVolumeResource.save(structural.volume, output_path)
    )


def _prepare_output_path(path: PathLike) -> Path:
    output_path = normalize_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _infer_sidecar(volume_path: Path, extension: str) -> Path:
    base_path = _strip_nifti_suffix(volume_path)
    return base_path.with_name(f"{base_path.name}.{extension}")


def _strip_nifti_suffix(path: Path) -> Path:
    if path.name.endswith(".nii.gz"):
        return path.with_name(path.name.removesuffix(".nii.gz"))
    if path.suffix == ".nii":
        return path.with_suffix("")

    msg = (
        f"Cannot infer FSL sidecar paths from {path}. "
        "Expected a .nii or .nii.gz path."
    )
    raise ValueError(msg)
