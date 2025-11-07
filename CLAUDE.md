# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

This is a Python package (`abcdmicro`) for extracting brain microstructure
parameters from diffusion MRI (dMRI) data from the ABCD Study. It provides
components for building pipelines that perform denoising, brain extraction, and
microstructure estimation (DTI and NODDI models).

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

The `Resource` abstraction (in `src/abcdmicro/resource.py`) is the foundation
for lazy loading and polymorphic storage:

- **`Resource`**: Abstract base with `load()` method
- **In-Memory Resources**: `InMemoryVolumeResource`, `InMemoryBvalResource`,
  `InMemoryBvecResource`
  - `is_loaded = True` (class variable)
  - `load()` returns self (no-op)
- **On-Disk Resources**: `NiftiVolumeResource`, `FslBvalResource`,
  `FslBvecResource`
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

### The Dwi Class: Central Orchestrator

The `Dwi` class (in `src/abcdmicro/dwi.py`) bundles the three resources needed
for diffusion imaging:

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

1. **Denoising** (`src/abcdmicro/denoise.py`):

   - `denoise_dwi(dwi: Dwi) -> InMemoryVolumeResource`
   - Uses DIPY's Patch2Self algorithm

2. **Masking** (`src/abcdmicro/masks.py`):

   - `brain_extract_batch(cases: list[tuple[Dwi, Path]]) -> list[NiftiVolumeResource]`
   - `brain_extract_single(dwi: Dwi, output_path: PathLike) -> NiftiVolumeResource`
   - Uses HD-BET (deep learning) on mean b0 images
   - **Important**: Always prefer batch processing - HD-BET initialization is
     expensive

3. **DTI Estimation** (`src/abcdmicro/dti.py`):

   - `Dti.estimate_from_dwi(dwi: Dwi, mask: VolumeResource | None) -> Dti`
   - Uses DIPY's TensorModel
   - Returns 6 values per voxel (lower triangular of symmetric tensor)
   - Provides derived maps: `get_fa_md()`, `get_eig()`

4. **NODDI Estimation** (`src/abcdmicro/noddi.py`):
   - `Noddi.estimate_from_dwi(dwi: Dwi, mask, dpar, n_kernel_dirs) -> Noddi`
   - Uses AMICO library
   - Outputs NDI (neurite density), ODI (orientation dispersion), FWF (free
     water fraction)
   - **Important**: AMICO writes kernels to disk; code redirects to temp
     directory via `set_config("ATOMS_path", tmpdir)`

### ABCD Study Integration

The `AbcdEvent` class (`src/abcdmicro/event.py`) encapsulates ABCD-specific
directory structure:

- `get_dwis()` handles scanner-specific logic (Philips returns multiple DWIs
  that need concatenation)
- `get_table()` provides lazy-loaded access to tabular data
- Tables are cached at class level in `_tables: ClassVar` - shared across all
  instances with same version

### CLI Integration

The `gen_masks` command (`src/abcdmicro/run.py`) demonstrates the batch
processing pattern:

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
           event=self.event,
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
3. Create on-disk implementation with static `save()` method
4. Update relevant domain classes

## Important Conventions

### Type Checking

- **Strict mypy** is enforced for `src/abcdmicro/*` (disallow_untyped_defs =
  true)
- Tests have relaxed type checking
- All files must import `from __future__ import annotations` (enforced by ruff)

### Metadata Management

- Use `update_volume_metadata()` in `src/abcdmicro/util.py` to update NIfTI
  metadata
- Automatically updates `dim` field to match array shape
- Sets intent codes (e.g., "symmetric matrix" for DTI)
- Metadata propagates from input `Dwi` through all derived volumes

### Validation

- Validation happens at construction (`__post_init__`)
- B-vectors must be unit vectors when bval ≠ 0
- B-vector shape must be (N, 3)
- Errors are raised early; warnings for metadata inconsistencies

### The Concatenation Pattern

When combining multiple `Dwi` objects (Philips scanner workflow):

- First `Dwi`'s metadata/affine/event becomes reference
- Warnings logged for mismatches (doesn't fail)
- `dim` field allowed to differ (updated for concatenated result)
- Uses `deep_equal_allclose()` for NumPy-aware comparison

## Critical Non-Obvious Details

1. **Resource.get_array() on disk resources re-loads every time** - always cache
   loaded results
2. **brain_extract_batch is strongly preferred over brain_extract_single** -
   HD-BET initialization is expensive
3. **Dwi.concatenate uses first Dwi as reference** - order matters for Philips
   multi-run data
4. **NODDI requires temporary directory** - AMICO writes kernels, code redirects
   to tmpdir
5. **AbcdEvent tables are class-level cached** - shared across all instances
   with same version
6. **All save() methods return new objects** - functional style, never mutate
   originals
7. **gen_masks operates on mean b0 images** - not the full DWI volume

## Testing Strategy

- Use `pytest-mock` for mocking expensive operations (HD-BET, AMICO)
- Test data fixtures use synthetic volumes with known properties
- Warn/error filters accommodate AMICO dependencies (see pyproject.toml)
- Coverage target excludes TYPE_CHECKING blocks and ellipsis

## Dependencies

### Core

- **dipy** (>=1.9): Diffusion imaging toolkit
- **dmri-amico** (==2.1.0): NODDI model fitting
- **hd-bet** (==2.0.1): Deep learning brain extraction
- **nibabel**: NIfTI file I/O
- **click**: CLI framework

### Pinned Versions

- AMICO and HD-BET are pinned due to API stability concerns
- `backports.tarfile` required for older amico/setuptools compatibility
