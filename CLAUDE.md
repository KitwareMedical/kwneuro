# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

This is a Python package (`kwneuro`) for extracting brain microstructure
parameters from diffusion MRI (dMRI) data. It provides components for building
pipelines that perform denoising, brain extraction, registration, template
building, tract segmentation, and microstructure estimation (DTI, NODDI, and CSD
models).

The package was formerly called `abcdmicro` / `abcd-microstructure-pipelines`
and was specific to the ABCD Study. It has been renamed and generalized for
broader diffusion MRI use. Source code lives under `src/kwneuro/`.

## Development Commands

### Setup

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run a single test file
pytest tests/test_dwi.py

# Run a specific test
pytest tests/test_dwi.py::test_dwi_load
```

### Code Quality

```bash
# Run pre-commit checks manually (ruff, mypy, etc.)
pre-commit run --all-files

# Ruff linting
ruff check .

# Ruff formatting
ruff format .

# Type checking with mypy
mypy src tests
```

### Documentation

```bash
# Build docs
cd docs
make html

# Live rebuild docs
sphinx-autobuild docs docs/_build/html
```

### Package Build

```bash
# Build distribution
python -m build

# Version is managed by setuptools_scm based on git tags
```

## Architecture

### The Resource Abstraction Pattern

The `Resource` abstraction (in `src/kwneuro/resource.py`) is the foundation for
lazy loading and polymorphic storage:

- **`Resource`**: Abstract base with `load()` method
- **In-Memory Resources**: `InMemoryVolumeResource`, `InMemoryBvalResource`,
  `InMemoryBvecResource`, `InMemoryResponseFunctionResource`
  - `is_loaded = True` (class variable)
  - `load()` returns self (no-op)
- **On-Disk Resources**: `NiftiVolumeResource`, `FslBvalResource`,
  `FslBvecResource`, `JsonResponseFunctionResource` (in `src/kwneuro/io.py`)
  - `is_loaded = False`
  - `load()` reads from disk and returns corresponding InMemory resource
  - Static `save()` method writes to disk and returns on-disk Resource

**Critical Pattern**: Call `load()` once and reuse the result. The `get_array()`
method on disk resources re-loads data every time, which is inefficient:

```python
# Inefficient - loads 3 times:
vol.get_array()
vol.get_affine()
vol.get_metadata()

# Efficient - load once:
vol_loaded = vol.load()
arr = vol_loaded.get_array()
affine = vol_loaded.get_affine()
```

#### ANTs Interop on VolumeResource

`InMemoryVolumeResource` has `to_ants_image()` and `from_ants_image()` methods
for converting to/from ANTsImage. The conversion handles the RAS+ (nibabel) to
LPS+ (ANTs) coordinate system change automatically. These are used extensively
by the registration and template building modules.

#### ResponseFunctionResource

The `ResponseFunctionResource` hierarchy (in `src/kwneuro/resource.py` and
`src/kwneuro/io.py`) stores CSD response functions as spherical harmonic
coefficients plus an average signal value. `InMemoryResponseFunctionResource`
includes factory methods `from_prolate_tensor()` (converting DIPY's legacy
format) and `from_dipy_object()`, plus a `get_dipy_object()` method for interop
with DIPY's `AxSymShResponse`.

### The Dwi Class: Central Orchestrator

The `Dwi` class (in `src/kwneuro/dwi.py`) bundles the three resources needed for
diffusion imaging:

- `volume: VolumeResource` - 4D array (x, y, z, diffusion weightings)
- `bval: BvalResource` - b-values
- `bvec: BvecResource` - b-vectors (unit vectors when bval ≠ 0)

**Key Pattern**: Both `load()` and `save()` return NEW `Dwi` objects rather than
modifying in place. This functional style ensures resource state is explicit.

The `Dwi` class provides a fluent interface for pipeline operations:

```python
dwi.denoise() -> Dwi                    # Returns new Dwi with denoised volume
dwi.extract_brain() -> VolumeResource   # Returns brain mask
dwi.estimate_dti(mask) -> Dti           # Returns DTI model
dwi.estimate_noddi(mask, ...) -> Noddi  # Returns NODDI model
dwi.compute_mean_b0() -> VolumeResource # Utility for brain extraction
```

### Pipeline Stages

Pipeline functions typically return Resources, while wrapper methods on domain
objects return new domain objects:

1. **Denoising** (`src/kwneuro/denoise.py`):

   - `denoise_dwi(dwi: Dwi) -> InMemoryVolumeResource`
   - Uses DIPY's Patch2Self algorithm

2. **Masking** (`src/kwneuro/masks.py`):

   - `brain_extract_batch(cases: list[tuple[Dwi, Path]]) -> list[NiftiVolumeResource]`
   - `brain_extract_single(dwi: Dwi, output_path: PathLike) -> NiftiVolumeResource`
   - Uses HD-BET (deep learning) on mean b0 images
   - **Important**: Always prefer batch processing - HD-BET initialization is
     expensive

3. **DTI Estimation** (`src/kwneuro/dti.py`):

   - `Dti.estimate_from_dwi(dwi: Dwi, mask: VolumeResource | None) -> Dti`
   - Uses DIPY's TensorModel
   - Returns 6 values per voxel (lower triangular of symmetric tensor)
   - Provides derived maps: `get_fa_md()`, `get_eig()`

4. **NODDI Estimation** (`src/kwneuro/noddi.py`):

   - `Noddi.estimate_from_dwi(dwi: Dwi, mask, dpar, n_kernel_dirs) -> Noddi`
   - Uses AMICO library
   - Outputs NDI (neurite density), ODI (orientation dispersion), FWF (free
     water fraction) via `ndi`, `odi`, `fwf` properties
   - `get_modulated_ndi_odi()` computes tissue-fraction-modulated maps
   - **Important**: AMICO writes kernels to disk; code redirects to temp
     directory via `set_config("ATOMS_path", tmpdir)`

5. **CSD / Fiber Orientation** (`src/kwneuro/csd.py`):

   - `estimate_response_function(dwi, mask, ...) -> InMemoryResponseFunctionResource`
     - Uses DIPY's SSST method on low-b (<=1200) data
   - `combine_response_functions(responses) -> InMemoryResponseFunctionResource`
     - Averages multiple response functions using MRtrix3-style L=0
       normalization
   - `compute_csd_fods(dwi, mask, response, ...) -> np.ndarray`
     - Constrained Spherical Deconvolution via DIPY
   - `compute_csd_peaks(dwi, mask, response, ...) -> tuple[VolumeResource, VolumeResource]`
     - Peak directions and values from CSD
   - `combine_csd_peaks_to_vector_volume(dirs, values) -> VolumeResource`
     - Converts Dipy peak format to MRtrix3-style vector volume

6. **Registration** (`src/kwneuro/reg.py`):

   - `register_volumes(fixed, moving, type_of_transform, mask, moving_mask) -> tuple[InMemoryVolumeResource, TransformResource]`
     - Wraps ANTs registration; supports Rigid, Affine, SyN, etc.
   - `TransformResource` wraps ANTs transform files (affine .mat and warp .nii)
     - `apply(fixed, moving, invert, interpolation)` applies the transform
     - `save(output_dir)` persists temporary ANTs files to a permanent location
     - Properties `matrices` and `warp_fields` for lazy access to transform
       components

7. **Template Building** (`src/kwneuro/build_template.py`):

   - `average_volumes(volume_list, normalize) -> InMemoryVolumeResource`
   - `build_template(volume_list, initial_template, iterations) -> InMemoryVolumeResource`
     - Iterative unbiased groupwise registration (SyN + affine averaging +
       sharpening)
   - `build_multi_metric_template(subject_list, ...) -> Mapping[str, InMemoryVolumeResource]`
     - Multi-modality variant using multivariate ANTs registration

8. **Tract Segmentation** (`src/kwneuro/tractseg.py`):

   - `extract_tractseg(dwi, mask, response, output_type) -> VolumeResource`
     - Computes CSD peaks, then runs TractSeg
     - `output_type`: `"tract_segmentation"`, `"endings_segmentation"`, or
       `"TOM"`

### CLI Integration

The `gen_masks` command (`src/kwneuro/run.py`) demonstrates the batch processing
pattern:

1. Recursively finds `*_dwi.nii.gz` files
2. Constructs `Dwi` objects with on-disk resources (doesn't load!)
3. Batches all cases and calls `brain_extract_batch()` with single HD-BET
   initialization
4. Preserves directory structure in output

## Extension Points

### Adding a New Pipeline Stage

1. Create a function in a new module:

   ```python
   def correct_distortion(dwi: Dwi, fieldmap: VolumeResource) -> InMemoryVolumeResource:
       # Implementation
       return corrected_volume
   ```

2. Add a method to `Dwi` class:
   ```python
   def correct_distortion(self, fieldmap: VolumeResource) -> Dwi:
       corrected_volume = sdc.correct_distortion(self, fieldmap)
       return Dwi(
           volume=corrected_volume,
           bval=self.bval,  # Reuse unchanged resources
           bvec=self.bvec,
       )
   ```

### Adding a New Model Type

Follow the `Dti`/`Noddi` pattern:

1. Create a dataclass with `VolumeResource`(s)
2. Implement `load()` and `save()` methods that return new instances
3. Add static `estimate_from_dwi()` method
4. Add convenience method to `Dwi` class

### Adding a New Resource Type

1. Create abstract base inheriting from `Resource`
2. Create in-memory implementation
3. Create on-disk implementation with static `save()` method in `io.py`
4. Update relevant domain classes

## Important Conventions

### Type Checking

- **Strict mypy** is enforced for `src/kwneuro/*` (disallow_untyped_defs = true)
- Tests have relaxed type checking
- All files must import `from __future__ import annotations` (enforced by ruff)

### Metadata Management

- Use `update_volume_metadata()` in `src/kwneuro/util.py` to update NIfTI
  metadata
- Use `create_estimate_volume_resource()` in `src/kwneuro/util.py` as a
  shorthand for creating scalar estimate volumes with proper metadata
- Automatically updates `dim` field to match array shape
- Sets intent codes (e.g., "symmetric matrix" for DTI)
- Metadata propagates from input `Dwi` through all derived volumes

### Validation

- Validation happens at construction (`__post_init__`)
- B-vectors must be unit vectors when bval ≠ 0
- B-vector shape must be (N, 3)
- Errors are raised early; warnings for metadata inconsistencies

### The Concatenation Pattern

When combining multiple `Dwi` objects:

- First `Dwi`'s metadata/affine becomes reference
- Warnings logged for mismatches (doesn't fail)
- `dim` field allowed to differ (updated for concatenated result)
- Uses `deep_equal_allclose()` for NumPy-aware comparison

## Critical Non-Obvious Details

1. **Resource.get_array() on disk resources re-loads every time** - always cache
   loaded results
2. **brain_extract_batch is strongly preferred over brain_extract_single** -
   HD-BET initialization is expensive
3. **Dwi.concatenate uses first Dwi as reference** - order matters for multi-run
   data
4. **NODDI requires temporary directory** - AMICO writes kernels, code redirects
   to tmpdir
5. **All save() methods return new objects** - functional style, never mutate
   originals
6. **gen_masks operates on mean b0 images** - not the full DWI volume
7. **CSD response estimation uses only b<=1200 data** - higher b-values aren't
   good for the DTI model used internally by DIPY's response estimation
8. **TransformResource files start in temp directories** - must call `save()` to
   persist ANTs transform files before the temp dir is cleaned up
9. **ANTs coordinate conventions differ from nibabel** - the `to_ants_image()`/
   `from_ants_image()` methods handle the RAS+ to LPS+ conversion

## Testing Strategy

- Use `pytest-mock` for mocking expensive operations (HD-BET, AMICO)
- Test data fixtures use synthetic volumes with known properties
- Warn/error filters accommodate dependencies (see pyproject.toml)
- Coverage target excludes TYPE_CHECKING blocks and ellipsis

## Dependencies

### Core

- **dipy** (>=1.9): Diffusion imaging toolkit
- **dmri-amico** (==2.1.0): NODDI model fitting
- **hd-bet** (==2.0.1): Deep learning brain extraction
- **nibabel**: NIfTI file I/O
- **click**: CLI framework
- **antspyx** (>=0.6.2): Registration and template building
- **TractSeg**: White matter tract segmentation

### Pinned Versions

- AMICO and HD-BET are pinned due to API stability concerns
- `backports.tarfile` required for older amico/setuptools compatibility
