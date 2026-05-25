"""
Copyright (c) 2024 Kitware. All rights reserved.

kwneuro: Processing pipelines to extract brain microstructure from diffusion MRI
"""

from __future__ import annotations

from ._version import version as __version__
from .cache import Cache, CacheSpec, cacheable
from .dwi import Dwi
from .files import (
    read_dwi_fsl,
    read_structural,
    read_volume,
    write_dwi_fsl,
    write_structural,
    write_volume,
)
from .structural import StructuralImage

__all__ = [
    "Cache",
    "CacheSpec",
    "Dwi",
    "StructuralImage",
    "__version__",
    "cacheable",
    "read_dwi_fsl",
    "read_structural",
    "read_volume",
    "write_dwi_fsl",
    "write_structural",
    "write_volume",
]
