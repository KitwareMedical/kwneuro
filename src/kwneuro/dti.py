from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import dipy.reconst.dti

from kwneuro.cache import cacheable
from kwneuro.io import NiftiVolumeResource
from kwneuro.resource import InMemoryVolumeResource, VolumeResource
from kwneuro.util import PathLike, update_volume_metadata

if TYPE_CHECKING:
    from kwneuro.dwi import Dwi


@dataclass
class Dti:
    """A diffusion tesnor image."""

    volume: VolumeResource
    """ The DTI image volume.
    It is a 4D volume, with the first three dimensions being spatial and the final dimension indexing
    the lower triangular entries of a symmetric matrix, in dipy order (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz).
    """

    def load(self) -> Dti:
        """Load any on-disk resources into memory and return a DTI with all loadable resources loaded."""
        return Dti(
            volume=self.volume.load(),
        )

    # ------------------------------------------------------------------
    # Cache protocol
    # ------------------------------------------------------------------

    @classmethod
    def _cache_files(cls, step_name: str) -> list[str]:
        return [f"{step_name}.nii.gz"]

    def _cache_save(self, cache_dir: Path, step_name: str) -> None:
        self.save(cache_dir / f"{step_name}.nii.gz")

    @classmethod
    def _cache_load(cls, cache_dir: Path, step_name: str) -> Dti:
        return cls(NiftiVolumeResource(cache_dir / f"{step_name}.nii.gz"))

    def save(self, path: PathLike) -> Dti:
        """Save all resources to disk and return a Dti with all resources being on-disk.

        Args:
            path: The desired file save location, a nifti file path.

        Returns: A Dti with its internal resources being on-disk.
        """
        return Dti(
            volume=NiftiVolumeResource.save(self.volume, path),
        )

    @staticmethod
    @cacheable
    def estimate_dti(dwi: Dwi, mask: VolumeResource | None = None) -> Dti:
        """Estimate a DTI from a DWI.

        Args:
            dwi: The source DWI
            mask: Optionally, a boolean 3D volume that has indicates where the fit should take place,
                such as a brain mask.
        """

        dwi_data_array = dwi.volume.get_array()

        tensor_model = dipy.reconst.dti.TensorModel(dwi.get_gtab())
        tensor_fit = tensor_model.fit(
            data=dwi_data_array, mask=mask.get_array() if mask is not None else None
        )
        dti_data_array = tensor_fit.lower_triangular()

        metadata = update_volume_metadata(
            metadata=dwi.volume.get_metadata(),
            volume_data_array=dti_data_array,
            intent_code="symmetric matrix",
            intent_params=(6,),
            intent_name="DTI",
        )

        return Dti(
            InMemoryVolumeResource(
                array=dti_data_array,
                affine=dwi.volume.get_affine(),
                metadata=metadata,
            )
        )

    def get_eig(self) -> tuple[VolumeResource, VolumeResource]:
        """Get eigenvalues and eigenvectors of the diffusion tensors. Returns 3D volumes with the same
        spatial shape as the DTI.

        Returns eigenvalues (evals), eigenvectors (evecs). Each is returned as a VolumeResource.

        The evals have shape (H,W,D,3).

        The evecs have shape (H,W,D,9), where the final axis provides the three components of the  eigenvector that
        goes with the first eigenvalue, followed by the three components of the eigenvector that goes with the second value,
        and so on for a total of 9 components.
        """
        dti_vol = self.volume.load()
        eig_info = dipy.reconst.dti.eig_from_lo_tri(
            dti_vol.get_array()
        )  # shape (H,W,D,12)
        evals = eig_info[..., :3]  # shape (H,W,D,3)
        evecs = eig_info[..., 3:]  # shape (H,W,D,9)
        return (
            InMemoryVolumeResource(
                evals,
                dti_vol.get_affine(),
                update_volume_metadata(
                    dti_vol.get_metadata(), evals, "vector", (), "eigenvalues"
                ),
            ),
            InMemoryVolumeResource(
                evecs,
                dti_vol.get_affine(),
                update_volume_metadata(
                    dti_vol.get_metadata(), evecs, "vector", (), "eigenvectors"
                ),
            ),
        )

    def get_fa_md(self) -> tuple[VolumeResource, VolumeResource]:
        """Get fractional anisotropy and mean diffusivity images.

        Returns 3D volumes for FA and MD, as VolumeResources.
        """
        evals, _ = self.get_eig()
        evals = evals.load()
        fa = dipy.reconst.dti.fractional_anisotropy(evals.get_array())
        md = dipy.reconst.dti.mean_diffusivity(evals.get_array())
        return (
            InMemoryVolumeResource(
                fa,
                evals.get_affine(),
                update_volume_metadata(evals.get_metadata(), fa, "estimate", (), "FA"),
            ),
            InMemoryVolumeResource(
                md,
                evals.get_affine(),
                update_volume_metadata(evals.get_metadata(), md, "estimate", (), "MD"),
            ),
        )
