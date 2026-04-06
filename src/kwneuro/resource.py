from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from kwneuro.io import JsonResponseFunctionResource, NiftiVolumeResource

import ants
import nibabel as nib
import numpy as np
from dipy.core.geometry import cart2sphere
from dipy.core.gradients import GradientTable
from dipy.reconst.csdeconv import AxSymShResponse, estimate_response
from dipy.reconst.shm import (
    lazy_index,
    real_sh_descoteaux_from_index,
    sph_harm_ind_list,
)
from numpy.typing import NDArray


class Resource(ABC):
    """Base class for all Resources. A Resource is a piece of data that could live in memory or on disk."""

    is_loaded: ClassVar[bool] = True
    """Whether a resource corresponds to in-memory data, rather than for example on-disk data."""

    @abstractmethod
    def load(self) -> Resource:
        """Load a Resource. Specific functionality depends on the Resource subclass, but if the resource is considered to be
        'loaded' then this should be a no-op, returning the same Resource back."""


class VolumeResource(Resource):
    """Base class for resources representing a volume or volume stack.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""

    @abstractmethod
    def get_array(self) -> NDArray[np.number]:
        """Get the underlying volume data array"""

    @abstractmethod
    def get_affine(self) -> NDArray[np.floating]:
        """Get the 4x4 affine matrix that maps index space to patient/scanner space"""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Get the volume image metadata"""

    @abstractmethod
    def load(self) -> InMemoryVolumeResource:
        """Load volume into memory"""

    # ------------------------------------------------------------------
    # Cache protocol — used by @cacheable when the return type is a
    # VolumeResource subclass.  Subclasses may override for custom names.
    # ------------------------------------------------------------------

    @classmethod
    def _cache_files(cls, step_name: str) -> list[str]:
        return [f"{step_name}.nii.gz"]

    def _cache_save(self, cache_dir: Path, step_name: str) -> None:
        from kwneuro.io import NiftiVolumeResource

        NiftiVolumeResource.save(self, cache_dir / f"{step_name}.nii.gz")

    @classmethod
    def _cache_load(cls, cache_dir: Path, step_name: str) -> NiftiVolumeResource:
        from kwneuro.io import NiftiVolumeResource

        return NiftiVolumeResource(cache_dir / f"{step_name}.nii.gz")


@dataclass
class InMemoryVolumeResource(VolumeResource):
    """A volume resource that is loaded into memory.
    An n-D array where n >= 3 and where three of the dimensions are spatial
    and have associated header information describing a patient coordinate system."""

    is_loaded: ClassVar[bool] = True

    array: NDArray[np.number]
    affine: NDArray[np.floating] = field(default_factory=lambda: np.eye(4))
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_array(self) -> NDArray[Any]:
        return self.array

    def get_affine(self) -> NDArray[np.floating]:
        return self.affine

    def get_metadata(self) -> dict[Any, Any]:
        return self.metadata

    def load(self) -> InMemoryVolumeResource:
        return self

    def to_ants_image(self) -> ants.ANTsImage:
        """Convert to an ANTsImage.
        This conversion creates a deep copy of the underlying data.
        The coordinate system is also converted from Nibabel's RAS+ to ANTs/ LPS+ orientation.
        """

        header = nib.Nifti1Header()
        for key, val in self.get_metadata().items():
            header[key] = val

        current_units = header.get_xyzt_units()
        if current_units[0] != "mm":
            error_msg = (
                "Volume must have spatial units in 'mm' for conversion to ANTsImage."
            )
            raise ValueError(error_msg)

        nib_image = nib.Nifti1Image(
            dataobj=self.get_array(),
            affine=self.get_affine(),
            header=header,
        )

        return ants.from_nibabel_nifti(nib_image)

    @staticmethod
    def from_ants_image(ants_image: ants.ANTsImage) -> InMemoryVolumeResource:
        """Create an InMemoryVolumeResource from an ANTsImage."""
        nib_image = ants.to_nibabel_nifti(ants_image)

        return InMemoryVolumeResource(
            array=nib_image.get_fdata(),
            affine=nib_image.affine,
            metadata=dict(nib_image.header),
        )


class BvalResource(Resource):
    """Base class for resources representing a list of b-values associated with a 4D DWI
    volume stack."""

    @abstractmethod
    def get(self) -> NDArray[np.floating]:
        """Get the underlying array of b-values"""

    def load(self) -> BvalResource:
        return self


@dataclass
class InMemoryBvalResource(BvalResource):
    """A b-value list that is loaded into memory."""

    is_loaded: ClassVar[bool] = True

    array: NDArray[np.floating]
    """The underlying array of b-values"""

    def get(self) -> NDArray[np.floating]:
        return self.array


class BvecResource(Resource):
    """Base class for resources representing a list of b-vectors associated with a 4D DWI
    volume stack."""

    @abstractmethod
    def get(self) -> NDArray[np.floating]:
        """Get the underlying array of b-vectors of shape (N,3)"""

    def load(self) -> BvecResource:
        return self


@dataclass
class InMemoryBvecResource(BvecResource):
    """A b-vector list that is loaded into memory."""

    is_loaded: ClassVar[bool] = True

    array: NDArray[np.floating]
    """The underlying array of b-vectors"""

    def __post_init__(self) -> None:
        # Check that b-vectors have the expected shape
        if self.array.ndim != 2 or self.array.shape[1] != 3:
            msg = f"Encountered wrong b-vector array shape {self.array.shape}. Expected shape (N,3)."
            raise ValueError(msg)

    def get(self) -> NDArray[np.floating]:
        return self.array


class ResponseFunctionResource(Resource):
    """Base class for resources representing a response function associated with a DWI."""

    @abstractmethod
    def get(self) -> tuple[NDArray, np.floating]:
        """Returns the underlying response function components as (sh_coeffs, avg_signal).

        Returns:
            A tuple of (sh_coeffs, avg_signal) where sh_coeffs is an array of m=0 coefficients
            for even degrees l = [0, 2, ..., sh_order] with length (sh_order / 2) + 1,
            and avg_signal is the mean signal intensity across the sphere.
        """

    @abstractmethod
    def get_dipy_object(self) -> AxSymShResponse:
        """Get the underlying response function in a format compatible with Dipy"""

    def load(self) -> ResponseFunctionResource:
        return self


@dataclass
class InMemoryResponseFunctionResource(ResponseFunctionResource):
    """A response function that is loaded into memory."""

    is_loaded: ClassVar[bool] = True

    sh_coeffs: NDArray[np.floating]
    """
    Spherical harmonic coefficients of the response function for an axially symmetric,
    even-degree model. Following the Dipy convention for symmetric signals, only even degrees (l = 0, 2, 4, ...,
    sh_order) are included. Under the assumption of axial symmetry, only the m = 0 coefficients are included.
    The coefficients are ordered by increasing degree l:

    - Index 0: l=0, m=0 (proportional to the average signal or avg_signal)
    - Index 1: l=2, m=0
    - Index 2: l=4, m=0
    - ...
    - Index M-1: l=sh_order, m=0

    The total number of coefficients M is (sh_order / 2) + 1.
    """

    avg_signal: np.floating
    """ The average non-diffusion weighted signal within the voxels used to calculate the response function"""

    def get(self) -> tuple[NDArray, np.floating]:
        """Returns the underlying response function components as (sh_coeffs, avg_signal).

        Returns:
            A tuple of (sh_coeffs, avg_signal) where sh_coeffs is an array of m=0 coefficients
            for even degrees l = [0, 2, ..., sh_order] with length (sh_order / 2) + 1,
            and avg_signal is the mean signal intensity across the sphere.
        """
        return (self.sh_coeffs, self.avg_signal)

    @staticmethod
    def from_prolate_tensor(
        response: tuple[NDArray, np.floating],
        gtab: GradientTable,
        sh_order_max: int = 8,
    ) -> InMemoryResponseFunctionResource:
        """Convert a legacy DIPY prolate-tensor response `(evals, S0)` into
        spherical harmonic coefficients using the approach from DIPY's
        `csdeconv` module:
        https://github.com/dipy/dipy/blob/f7b863f1485cd3fa6329c8e8f3388d8f58863f0d/dipy/reconst/csdeconv.py#L168.

        Args:
            response: Response function output by DIPY.
            gtab: Gradient table used to estimate response
            sh_order_max: Maximum spherical harmonic order to use for the basis model. Default is 8.

        Returns: InMemoryResponseFunctionResource
        """

        eig_vals, s0 = response

        if not isinstance(eig_vals, np.ndarray) or eig_vals.shape != (3,):
            error_msg = "the first element of response should be a numpy array (listing three eigenvalues)"
            raise ValueError(error_msg)

        if gtab is None or not hasattr(gtab, "gradients"):
            error_msg = "Invalid GradientTable provided."
            raise ValueError(error_msg)

        m_values, l_values = sph_harm_ind_list(sh_order_max)
        _where_dwi = lazy_index(~gtab.b0s_mask)

        x, y, z = gtab.gradients[_where_dwi].T
        _, theta, phi = cart2sphere(x, y, z)
        b_dwi = real_sh_descoteaux_from_index(
            m_values, l_values, theta[:, None], phi[:, None]
        )

        s_r = estimate_response(gtab, eig_vals, s0)
        sh_coeffs = np.linalg.lstsq(b_dwi, s_r[_where_dwi], rcond=-1)[0]

        return InMemoryResponseFunctionResource(sh_coeffs=sh_coeffs, avg_signal=s0)

    @staticmethod
    def from_dipy_object(obj: AxSymShResponse) -> InMemoryResponseFunctionResource:
        """Construct from a DIPY `AxSymShResponse` instance."""

        if not isinstance(obj, AxSymShResponse):
            error_msg = "from_dipy_object requires an AxSymShResponse instance."
            raise TypeError(error_msg)

        return InMemoryResponseFunctionResource(
            sh_coeffs=obj.dwi_response, avg_signal=obj.S0
        )

    def get_dipy_object(self) -> AxSymShResponse:
        """Return the stored response function as a DIPY `AxSymShResponse`."""

        return AxSymShResponse(S0=self.avg_signal, dwi_response=self.sh_coeffs)

    # ------------------------------------------------------------------
    # Cache protocol
    # ------------------------------------------------------------------

    @classmethod
    def _cache_files(cls, step_name: str) -> list[str]:
        return [f"{step_name}.json"]

    def _cache_save(self, cache_dir: Path, step_name: str) -> None:
        from kwneuro.io import JsonResponseFunctionResource

        JsonResponseFunctionResource.save(self, cache_dir / f"{step_name}.json")

    @classmethod
    def _cache_load(
        cls, cache_dir: Path, step_name: str
    ) -> JsonResponseFunctionResource:
        from kwneuro.io import JsonResponseFunctionResource

        return JsonResponseFunctionResource(cache_dir / f"{step_name}.json")
