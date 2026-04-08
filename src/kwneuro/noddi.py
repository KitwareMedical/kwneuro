from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kwneuro.cache import cacheable
from kwneuro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource
from kwneuro.resource import InMemoryVolumeResource, VolumeResource
from kwneuro.util import (
    PathLike,
    create_estimate_volume_resource,
    update_volume_metadata,
)

if TYPE_CHECKING:
    from kwneuro.dwi import Dwi


@dataclass
class Noddi:
    """A Noddi result"""

    volume: VolumeResource
    """ The NODDI image volume.
    It is a 4D volume, with the first three dimensions being spatial and the final dimension indexing
    the noddi outputs.
    """

    directions: VolumeResource
    """ The NODDI image volume.
    It is a 4D volume, with the first three dimensions being spatial and the final dimension indexing
    the directions.
    """

    def load(self) -> Noddi:
        """Load any on-disk resources into memory and return a DTI with all loadable resources loaded."""
        return Noddi(volume=self.volume.load(), directions=self.directions.load())

    # ------------------------------------------------------------------
    # Cache protocol
    # ------------------------------------------------------------------

    @classmethod
    def _cache_files(cls, step_name: str) -> list[str]:
        return [f"{step_name}.nii.gz", f"{step_name}_directions.nii.gz"]

    def _cache_save(self, cache_dir: Path, step_name: str) -> None:
        NiftiVolumeResource.save(self.volume, cache_dir / f"{step_name}.nii.gz")
        NiftiVolumeResource.save(
            self.directions, cache_dir / f"{step_name}_directions.nii.gz"
        )

    @classmethod
    def _cache_load(cls, cache_dir: Path, step_name: str) -> Noddi:
        return cls(
            NiftiVolumeResource(cache_dir / f"{step_name}.nii.gz"),
            NiftiVolumeResource(cache_dir / f"{step_name}_directions.nii.gz"),
        )

    def save(self, path: PathLike) -> Noddi:
        """Save all resources to disk and return a Noddi with all resources being on-disk.

        Args:
            path: The desired file save location, a nifti file path.

        Returns: A Noddi with its internal resources being on-disk.
        """

        directions_path = (
            str(path)
            .replace(".nii", "_directions.nii")
            .replace(".nii.gz", "_directions.nii.gz")
        )
        return Noddi(
            volume=NiftiVolumeResource.save(self.volume, path),
            directions=NiftiVolumeResource.save(self.directions, directions_path),
        )

    @staticmethod
    @cacheable
    def estimate_noddi(
        dwi: Dwi,
        mask: VolumeResource | None = None,
        dpar: float = 1.7e-3,
        n_kernel_dirs: int = 500,
    ) -> Noddi:
        """Estimate Noddi from a DWI.

        Args:
            dwi: The source DWI
            mask: Optionally, a boolean 3D volume that has indicates where the fit should take place,
                such as a brain mask.
            dpar: The parallel diffusivity to be used in the model fitting. If not provided, the default value of
                1.7e-3 mm^2/s is used, which is suitable for white matter. For gray matter, a value of 1.3e-3 mm^2/s is recommended.
            n_kernel_dirs: The number of directions to use when generating the AMICO NODDI kernels. This value represents the total
                count of possible orientations for the response functions across the half-sphere. Default: 500.

        Returns: A Noddi resource containing the estimated parameters.
        """
        try:
            import amico
        except ImportError:
            msg = (
                "AMICO is required for NODDI estimation but is not installed. "
                "Install it with: pip install kwneuro[noddi]"
            )
            raise ImportError(msg) from None

        amico.setup()
        with tempfile.TemporaryDirectory() as tmpdir:
            ae = amico.Evaluation(output_path=tmpdir)

            # Force the kernels to be written to the temp dir
            ae.set_config("ATOMS_path", str(Path(tmpdir) / "AMICO_kernels"))

            scheme_output_path = Path(tmpdir) / "amico_scheme.scheme"

            # Save DWI to file to be read by AMICO
            volume = NiftiVolumeResource.save(
                dwi.volume, Path(tmpdir) / "amico_volume.nii.gz"
            )
            bval = FslBvalResource.save(dwi.bval, Path(tmpdir) / "amico.bval")
            bvec = FslBvecResource.save(dwi.bvec, Path(tmpdir) / "amico.bvec")

            amico.util.fsl2scheme(
                bval.path,
                bvec.path,
                schemeFilename=scheme_output_path,
            )

            # Write mask to file
            if mask is not None:
                brain_mask_output_path = Path(tmpdir) / "brain_mask.nii.gz"
                NiftiVolumeResource.save(mask, brain_mask_output_path)
            else:
                brain_mask_output_path = None

            ae.load_data(
                volume.path,
                scheme_filename=scheme_output_path,
                mask_filename=brain_mask_output_path,
            )  # Additional parameters that can be set: b0_thr=0, b0_min_signal=0, replace_bad_voxels=None

            ae.set_model("NODDI")
            ae.model.dPar = dpar

            regenerate_kernels = True
            ae.generate_kernels(regenerate=regenerate_kernels, ndirs=n_kernel_dirs)
            ae.load_kernels()
            ae.fit()

        noddi_data_array = ae.RESULTS["MAPs"]
        directions_array = ae.RESULTS["DIRs"]
        # AMICO also has options for RMSE, NRMSE - how the predicted signal differs from DWI signal.
        # Needs residual info so cannot be computed later. If needed, this can be saved as an additional volume.

        volume_metadata = update_volume_metadata(
            metadata=dwi.volume.get_metadata(),
            volume_data_array=noddi_data_array,
            intent_code="NIFTI_INTENT_ESTIMATE",
            intent_name="-".join(ae.model.maps_name),
        )

        directions_metadata = update_volume_metadata(
            metadata=dwi.volume.get_metadata(),
            volume_data_array=directions_array,
            intent_code="NIFTI_INTENT_ESTIMATE",
            intent_name="-".join(ae.model.maps_name),
        )

        return Noddi(
            volume=InMemoryVolumeResource(
                array=noddi_data_array,
                affine=dwi.volume.get_affine(),
                metadata=volume_metadata,
            ),
            directions=InMemoryVolumeResource(
                array=directions_array,
                affine=dwi.volume.get_affine(),
                metadata=directions_metadata,
            ),
        )

    @property
    def ndi(self) -> VolumeResource:
        """Neurite Density Index (NDI) map as a 3D volume."""
        array = self.volume.get_array()[..., 0]
        return create_estimate_volume_resource(
            array=array, reference_volume=self.volume, intent_name="NDI"
        )

    @property
    def odi(self) -> VolumeResource:
        """Orientation Dispersion Index (ODI) map as a 3D volume."""
        array = self.volume.get_array()[..., 1]
        return create_estimate_volume_resource(
            array=array, reference_volume=self.volume, intent_name="ODI"
        )

    @property
    def fwf(self) -> VolumeResource:
        """Free Water Fraction (FWF) map as a 3D volume."""
        array = self.volume.get_array()[..., 2]
        return create_estimate_volume_resource(
            array=array, reference_volume=self.volume, intent_name="FWF"
        )

    def get_modulated_ndi_odi(self) -> tuple[VolumeResource, VolumeResource]:
        """Compute the modulated maps, NDI*TF and ODI*TF, where TF = 1 - FWF.

        Returns:
            Returns 3D volumes for modulated NDI and ODI maps, as VolumeResources.
        """

        tf = 1.0 - self.fwf.get_array()
        modulated_ndi = self.ndi.get_array() * tf
        modulated_odi = self.odi.get_array() * tf

        return (
            create_estimate_volume_resource(
                array=modulated_ndi, reference_volume=self.volume, intent_name="modNDI"
            ),
            create_estimate_volume_resource(
                array=modulated_odi, reference_volume=self.volume, intent_name="modODI"
            ),
        )
