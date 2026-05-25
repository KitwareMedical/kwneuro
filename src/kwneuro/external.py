from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from kwneuro.dwi import Dwi
from kwneuro.files import write_dwi_fsl, write_structural, write_volume
from kwneuro.resource import VolumeResource
from kwneuro.structural import StructuralImage


@dataclass(frozen=True)
class TemporaryDwiFiles:
    """Temporary file paths for a DWI volume and FSL gradient sidecars."""

    volume: Path
    bval: Path
    bvec: Path


@contextmanager
def temporary_volume_file(
    volume: VolumeResource,
    *,
    filename: str = "volume.nii.gz",
) -> Iterator[Path]:
    """Write a volume to a temporary NIfTI file for an external file-based tool."""
    with TemporaryDirectory() as tmpdir:
        path = _temp_path(tmpdir, filename)
        write_volume(volume, path)
        yield path


@contextmanager
def temporary_dwi_files(
    dwi: Dwi,
    *,
    basename: str = "dwi",
) -> Iterator[TemporaryDwiFiles]:
    """Write a DWI to temporary NIfTI/FSL files for an external file-based tool."""
    with TemporaryDirectory() as tmpdir:
        volume_path = _temp_path(tmpdir, f"{basename}.nii.gz")
        bval_path = _temp_path(tmpdir, f"{basename}.bval")
        bvec_path = _temp_path(tmpdir, f"{basename}.bvec")
        write_dwi_fsl(dwi, volume_path, bval=bval_path, bvec=bvec_path)
        yield TemporaryDwiFiles(volume=volume_path, bval=bval_path, bvec=bvec_path)


@contextmanager
def temporary_structural_file(
    structural: StructuralImage,
    *,
    filename: str = "structural.nii.gz",
) -> Iterator[Path]:
    """Write a structural image to a temporary NIfTI file for an external tool."""
    with TemporaryDirectory() as tmpdir:
        path = _temp_path(tmpdir, filename)
        write_structural(structural, path)
        yield path


def _temp_path(tmpdir: str, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        msg = "Temporary file paths must be relative."
        raise ValueError(msg)

    tmpdir_path = Path(tmpdir).resolve()
    candidate = (tmpdir_path / path).resolve()
    if not candidate.is_relative_to(tmpdir_path):
        msg = "Temporary file paths must stay within the temporary directory."
        raise ValueError(msg)
    return candidate
