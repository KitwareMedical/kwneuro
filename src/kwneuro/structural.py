from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import ants

from kwneuro.cache import cacheable
from kwneuro.io import NiftiVolumeResource
from kwneuro.resource import InMemoryVolumeResource, VolumeResource
from kwneuro.util import PathLike, normalize_path

if TYPE_CHECKING:
    pass


@dataclass
class StructuralImage:
    """A structural (e.g. T1-weighted) MRI image."""

    volume: VolumeResource
    """The structural MRI volume. Expected to be a 3D volume."""

    def load(self) -> StructuralImage:
        """Load any on-disk resources into memory and return a StructuralImage with all resources loaded."""
        return StructuralImage(volume=self.volume.load())

    # ------------------------------------------------------------------
    # Cache protocol
    # ------------------------------------------------------------------

    @classmethod
    def _cache_files(cls, step_name: str) -> list[str]:
        return [f"{step_name}.nii.gz"]

    def _cache_save(self, cache_dir: Path, step_name: str) -> None:
        NiftiVolumeResource.save(self.volume, cache_dir / f"{step_name}.nii.gz")

    @classmethod
    def _cache_load(cls, cache_dir: Path, step_name: str) -> StructuralImage:
        return cls(NiftiVolumeResource(cache_dir / f"{step_name}.nii.gz"))

    def save(self, path: PathLike, basename: str) -> StructuralImage:
        """Save the volume to disk and return a StructuralImage with an on-disk resource.

        Args:
            path: The desired save directory.
            basename: The desired file basename (without extension).

        Returns: A StructuralImage with its internal resource being on-disk.
        """
        path = normalize_path(path)
        if path.exists() and not path.is_dir():
            msg = "`path` should be the desired save directory"
            raise ValueError(msg)
        path.mkdir(exist_ok=True, parents=True)
        return StructuralImage(
            volume=NiftiVolumeResource.save(self.volume, path / f"{basename}.nii.gz"),
        )

    def correct_bias(self) -> StructuralImage:
        """Apply N4 bias field correction using ANTsPy.

        Returns: A new StructuralImage with the bias-corrected volume.
        """
        loaded = self.volume.load()
        corrected = ants.n4_bias_field_correction(loaded.to_ants_image())
        return StructuralImage(volume=InMemoryVolumeResource.from_ants_image(corrected))

    def extract_brain(self) -> InMemoryVolumeResource:
        """Extract a brain mask using HD-BET.

        This is meant to be convenient rather than efficient.
        Using this in a loop could result in unnecessary repetition of file I/O operations.
        For efficiency, see :func:`kwneuro.masks.brain_extract_structural_batch`.
        """
        from kwneuro.masks import brain_extract  # deferred to avoid circular import

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "brain_mask.nii.gz"
            brain_mask = brain_extract(volume=self.volume, output_path=output_path)
            return brain_mask.load()

    def segment_tissues(
        self, mask: VolumeResource | None = None, method: str = "atropos"
    ) -> InMemoryVolumeResource:
        """Segment brain tissues into labeled classes.

        Args:
            mask: Optional brain mask. Used only with ``method="atropos"``; when omitted
                a mask is generated automatically via :func:`ants.get_mask`.
                Ignored for ``method="deep_atropos"``, which handles preprocessing internally.
            method: Segmentation method. One of:

                - ``"atropos"`` *(default)*: Classical ANTsPy k-means Atropos.
                  Returns a 3-class labeled volume (1=CSF, 2=GM, 3=WM).
                  No extra installation required.
                - ``"deep_atropos"``: Deep-learning segmentation via ANTsPyNet.
                  Returns a 6-class labeled volume (1=CSF, 2=GM, 3=WM, 4=deep GM,
                  5=cerebellum, 6=brainstem). Requires ``pip install kwneuro[antspynet]``.

        Returns: A labeled segmentation volume.
        """
        if method == "atropos":
            return self._segment_atropos(mask)
        if method == "deep_atropos":
            return self._segment_deep_atropos()
        msg = f"Unknown segmentation method: {method!r}. Expected 'atropos' or 'deep_atropos'."
        raise ValueError(msg)

    def _segment_atropos(self, mask: VolumeResource | None) -> InMemoryVolumeResource:
        loaded = self.volume.load()
        ants_image = loaded.to_ants_image()
        ants_mask = (
            mask.load().to_ants_image()
            if mask is not None
            else ants.get_mask(ants_image)
        )
        result = ants.atropos(
            a=ants_image, m="[0.2,1x1x1]", c="[5,0]", i="kmeans[3]", x=ants_mask
        )
        return InMemoryVolumeResource.from_ants_image(result["segmentation"])

    def _segment_deep_atropos(self) -> InMemoryVolumeResource:
        try:
            import antspynet
        except ImportError:
            msg = (
                "deep_atropos requires ANTsPyNet but it is not installed. "
                "Install it with: pip install kwneuro[antspynet]"
            )
            raise ImportError(msg) from None
        loaded = self.volume.load()
        ants_image = loaded.to_ants_image()
        result = antspynet.deep_atropos(t1=ants_image, do_preprocessing=True)
        return InMemoryVolumeResource.from_ants_image(result["segmentation_image"])

    def parcellate(self, method: str = "dkt") -> InMemoryVolumeResource:
        """Cortical parcellation.

        Args:
            method: Parcellation method. Currently only ``"dkt"`` is supported:
                Desikan-Killiany-Tourville (DKT) cortical labeling via ANTsPyNet
                (~84 regions). Requires ``pip install kwneuro[antspynet]``.

        Returns: A DKT-labeled parcellation volume.
        """
        if method == "dkt":
            return self._parcellate_dkt()
        msg = f"Unknown parcellation method: {method!r}. Expected 'dkt'."
        raise ValueError(msg)

    def _parcellate_dkt(self) -> InMemoryVolumeResource:
        try:
            import antspynet
        except ImportError:
            msg = (
                "parcellate requires ANTsPyNet but it is not installed. "
                "Install it with: pip install kwneuro[antspynet]"
            )
            raise ImportError(msg) from None
        loaded = self.volume.load()
        ants_image = loaded.to_ants_image()
        result = antspynet.desikan_killiany_tourville_labeling(
            t1=ants_image, do_preprocessing=True
        )
        return InMemoryVolumeResource.from_ants_image(
            result["parcellation_segmentation"]
        )


StructuralImage.correct_bias = cacheable(StructuralImage.correct_bias)  # type: ignore[method-assign]
StructuralImage.extract_brain = cacheable(StructuralImage.extract_brain)  # type: ignore[method-assign]
StructuralImage.segment_tissues = cacheable(StructuralImage.segment_tissues)  # type: ignore[method-assign]
StructuralImage.parcellate = cacheable(StructuralImage.parcellate)  # type: ignore[method-assign]
