from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import dipy.core.gradients
import numpy as np

from kwneuro.cache import cacheable
from kwneuro.denoise import denoise_dwi
from kwneuro.dti import Dti
from kwneuro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource
from kwneuro.masks import brain_extract
from kwneuro.noddi import Noddi
from kwneuro.reg import TransformResource, register_volumes
from kwneuro.resource import (
    BvalResource,
    BvecResource,
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryVolumeResource,
    VolumeResource,
)
from kwneuro.util import (
    PathLike,
    deep_equal_allclose,
    normalize_path,
    subsample_volume,
    update_volume_metadata,
)

if TYPE_CHECKING:
    from kwneuro.structural import StructuralImage


@dataclass
class Dwi:
    """A diffusion weighted image."""

    volume: VolumeResource
    """ The DWI image volume.
    It is assumed to be a 4D volume, with the first three dimensions being spatial and the final dimension indexing
    the diffusion weightings.
    """

    bval: BvalResource
    """The DWI b-values"""

    bvec: BvecResource
    """The DWI b-vectors"""

    structural: StructuralImage | None = None
    """An optional structural image (e.g. T1w) from the same imaging session."""

    def __post_init__(self) -> None:
        # Check that b-vectors are unit vectors whenever the b-value isn't 0
        bvecs_to_check = self.bvec.get()[self.bval.get() != 0]
        if not np.allclose(np.linalg.norm(bvecs_to_check, axis=1), 1.0):
            msg = "All b-vectors with nonzero b-values must be unit vectors."
            raise ValueError(msg)

    def load(self) -> Dwi:
        """Load any on-disk resources into memory and return a Dwi with all loadable resources loaded."""
        return Dwi(
            volume=self.volume.load(),
            bval=self.bval.load(),
            bvec=self.bvec.load(),
            structural=self.structural.load() if self.structural is not None else None,
        )

    # ------------------------------------------------------------------
    # Cache protocol
    # ------------------------------------------------------------------

    @classmethod
    def _cache_files(cls, step_name: str) -> list[str]:
        return [f"{step_name}.nii.gz", f"{step_name}.bval", f"{step_name}.bvec"]

    def _cache_save(self, cache_dir: Path, step_name: str) -> None:
        self.save(cache_dir, step_name)

    @classmethod
    def _cache_load(cls, cache_dir: Path, step_name: str) -> Dwi:
        return cls(
            NiftiVolumeResource(cache_dir / f"{step_name}.nii.gz"),
            FslBvalResource(cache_dir / f"{step_name}.bval"),
            FslBvecResource(cache_dir / f"{step_name}.bvec"),
        )

    def save(self, path: PathLike, basename: str) -> Dwi:
        """Save all resources to disk and return a Dwi with all resources being on-disk.

        Args:
            path: The desired save directory.
            basename: The desired file basenames, i.e. without an extension.

        Returns: A Dwi with its internal resources being on-disk.
        """
        path = normalize_path(path)
        if path.exists() and not path.is_dir():
            msg = "`path` should be the desired save directory"
            raise ValueError(msg)
        path.mkdir(exist_ok=True, parents=True)
        return Dwi(
            volume=NiftiVolumeResource.save(self.volume, path / f"{basename}.nii.gz"),
            bval=FslBvalResource.save(self.bval, path / f"{basename}.bval"),
            bvec=FslBvecResource.save(self.bvec, path / f"{basename}.bvec"),
        )

    def get_gtab(self) -> dipy.core.gradients.GradientTable:
        """Get the GradientTable for this DWI."""
        return dipy.core.gradients.gradient_table(
            bvals=self.bval.get(), bvecs=self.bvec.get()
        )

    def compute_mean_b0(self) -> InMemoryVolumeResource:
        """Compute the mean of the b=0 images of a DWI."""
        gtab = self.get_gtab()
        mean_b0_array = self.volume.get_array()[:, :, :, gtab.b0s_mask].mean(axis=3)

        dwi_metadata = self.volume.get_metadata()
        metadata = update_volume_metadata(
            dwi_metadata,
            mean_b0_array,
            intent_code=0,  # "none"
            intent_name="mean_b0",
        )
        metadata["descrip"] = "Mean b0 image extracted from a DWI."

        return InMemoryVolumeResource(
            array=mean_b0_array,
            affine=self.volume.get_affine(),
            metadata=metadata,
        )

    @staticmethod
    def concatenate(dwis: list[Dwi]) -> Dwi:
        """Concatenate a list of ``Dwi``\\s into a single (loaded) DWI.

        The affine and metadata of the first ``Dwi`` in the list will be used to concatenate volumes.
        """
        if len(dwis) == 0:
            msg = "Cannot concatenate an empty list of DWIs."
            raise ValueError(msg)

        # ensure all DWI resources are loaded into memory
        loaded_dwis = [d.load() for d in dwis]

        # use the first DWI as the reference for metadata
        ref_dwi = loaded_dwis[0]
        ref_affine = ref_dwi.volume.get_affine()
        ref_metadata = ref_dwi.volume.get_metadata()

        # check for metadata consistency across all DWIs and log warnings
        for i, dwi in enumerate(loaded_dwis[1:], start=1):
            if not np.allclose(dwi.volume.get_affine(), ref_affine):
                logging.warning(
                    "Affine mismatch: Using affine from DWI 0, but DWI %s has a different affine.",
                    i,
                )

            dwi_metadata = dwi.volume.get_metadata()
            keys_that_are_allowed_to_not_match = ["dim"]
            common_keys = [k for k in ref_metadata if k in dwi_metadata]
            if not (
                set(dwi_metadata.keys()) == set(ref_metadata.keys())
                and all(
                    deep_equal_allclose(dwi_metadata[k], ref_metadata[k])
                    for k in common_keys
                    if k not in keys_that_are_allowed_to_not_match
                )
            ):
                logging.warning(
                    "Metadata mismatch: Using metadata from DWI 0, but DWI %s has different metadata:",
                    i,
                )
                for key in common_keys:
                    if key not in dwi_metadata:
                        logging.warning(
                            "DWI 0 header has key '%s', but DWI %s header does not.",
                            key,
                            i,
                        )
                    elif not deep_equal_allclose(dwi_metadata[key], ref_metadata[key]):
                        logging.warning(
                            "At key '%s', values differ between DWI 0 and DWI %s.\n--> DWI 0: %s\n--> DWI %s: %s",
                            key,
                            i,
                            ref_metadata[key],
                            i,
                            dwi_metadata[key],
                        )

        # extract the numpy arrays from each resource.
        all_volumes_data = [d.volume.get_array() for d in loaded_dwis]
        all_bvals_data = [d.bval.get() for d in loaded_dwis]
        all_bvecs_data = [d.bvec.get() for d in loaded_dwis]

        # concatenate the data.
        concatenated_volume_data = np.concatenate(all_volumes_data, axis=-1)
        concatenated_bval_data = np.concatenate(all_bvals_data)
        concatenated_bvec_data = np.concatenate(all_bvecs_data, axis=0)

        if concatenated_volume_data.ndim != 4:
            msg = "Concatenated DWI was expected to be a 4D array"
            raise RuntimeError(msg)

        # create new in-memory resources for the concatenated data.
        concatenated_volume = InMemoryVolumeResource(
            array=concatenated_volume_data,
            affine=ref_affine,
            metadata=update_volume_metadata(
                ref_metadata,
                concatenated_volume_data,
            ),
        )
        concatenated_bval = InMemoryBvalResource(array=concatenated_bval_data)
        concatenated_bvec = InMemoryBvecResource(array=concatenated_bvec_data)

        # return a new Dwi object with the concatenated, in-memory data.
        return Dwi(
            volume=concatenated_volume,
            bval=concatenated_bval,
            bvec=concatenated_bvec,
            structural=dwis[0].structural,
        )

    def denoise(self) -> Dwi:
        """Denoise using Patch2Self from DIPY."""

        denoised_volume = denoise_dwi(self)

        return Dwi(
            volume=denoised_volume,
            bval=self.bval,
            bvec=self.bvec,
            structural=self.structural,
        )

    def extract_brain(self) -> InMemoryVolumeResource:
        """Extract brain mask. This is meant to be convenient rather than efficient.
        Using this in a loop could result in unnecessary repetition of file I/O operations.
        For efficiency, see :func:`kwneuro.masks.brain_extract_dwi_batch`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "brain_mask.nii.gz"
            brain_mask = brain_extract(
                volume=self.compute_mean_b0(), output_path=output_path
            )
            return brain_mask.load()

    def estimate_dti(self, mask: VolumeResource | None = None) -> Dti:
        """Estimate diffusion tensor image from this DWI"""
        return Dti.estimate_dti(self, mask)  # type: ignore[no-any-return]

    def estimate_noddi(
        self,
        mask: VolumeResource | None = None,
        dpar: float = 1.7e-3,
        n_kernel_dirs: int = 500,
    ) -> Noddi:
        """Estimate NODDI model parameters from this DWI. See :meth:`kwneuro.noddi.Noddi.estimate_noddi` for details."""
        return Noddi.estimate_noddi(self, mask, dpar, n_kernel_dirs)  # type: ignore[no-any-return]

    def register_to_structural(
        self,
        type_of_transform: str = "Rigid",
        mask: VolumeResource | None = None,
        structural_mask: VolumeResource | None = None,
    ) -> TransformResource:
        """Register this DWI (mean b0) to the associated structural (T1) image. This is a
        convenience wrapper around :func:`kwneuro.reg.register_volumes`. Requires
        ``self.structural`` to be set.

        The returned :class:`~kwneuro.reg.TransformResource` maps DWI space → T1 space.
        To warp T1-derived labels into DWI space, call
        ``transform.apply(..., invert=True, interpolation="genericLabel")``.
        See :meth:`kwneuro.reg.TransformResource.apply` for details.
        """
        if self.structural is None:
            msg = "register_to_structural requires a structural image; set Dwi.structural first."
            raise ValueError(msg)

        _, transform = register_volumes(
            fixed=self.structural.volume,
            moving=self.compute_mean_b0(),
            type_of_transform=type_of_transform,
            mask=structural_mask,
            moving_mask=mask,
        )
        return transform


def subsample_dwi(dwi: Dwi, factor: int = 2) -> Dwi:
    """Spatially subsample a DWI by taking every Nth voxel along each spatial axis.

    Convenience wrapper around :func:`kwneuro.util.subsample_volume` that returns a new
    Dwi with the subsampled volume and the original b-values and b-vectors.
    """
    return Dwi(
        volume=subsample_volume(dwi.volume, factor),
        bval=dwi.bval,
        bvec=dwi.bvec,
    )


Dwi.denoise = cacheable(Dwi.denoise)  # type: ignore[method-assign]
Dwi.extract_brain = cacheable(Dwi.extract_brain)  # type: ignore[method-assign]
Dwi.register_to_structural = cacheable(Dwi.register_to_structural)  # type: ignore[method-assign]
