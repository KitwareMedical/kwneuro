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

All commands use [uv](https://docs.astral.sh/uv/). The project has a committed
`uv.lock` for reproducible installs.

### Setup

```bash
# Install in editable mode with dev dependencies (includes all optional extras)
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

Note: `uv sync --extra dev` installs all optional extras (HD-BET, TractSeg,
AMICO, neuroCombat) via the `all` extra. CI uses `uv sync --extra test` which
only installs neuroCombat (the other optional deps are mocked in tests).

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run a single test file
uv run pytest tests/test_dwi.py

# Run a specific test
uv run pytest tests/test_dwi.py::test_dwi_load
```

### Code Quality

Ruff (lint + format) and mypy are run via pre-commit hooks — they are not
installed as direct dev dependencies, so invoke them through pre-commit rather
than `uv run ruff ...` / `uv run mypy ...`.

```bash
# Run pre-commit checks (ruff, mypy, etc.) — fast
uv run pre-commit run -a

# Run PyLint — thorough, slow
uv run pylint kwneuro
```

### Documentation

```bash
# Build docs
uv run --extra docs sphinx-build -n -T docs docs/_build/html

# Live rebuild docs
uv run --extra docs sphinx-autobuild -n -T docs docs/_build/html

# Rebuild tutorial pages from notebooks (needs notebook extras)
uv run --extra notebooks --extra all python scripts/update-notebook-pages.py
```

The developer guide lives at `docs/developer-guide.md` (canonical source;
`.github/CONTRIBUTING.md` is a thin redirect). Tutorial pages under
`docs/tutorials/` are pre-rendered from `notebooks/*.py` via the
`scripts/update-notebook-pages.py` script and committed to git.

### Package Build

```bash
# Build distribution
uv build

# Version is managed by setuptools_scm based on git tags
```

### Verification checklist

After a chunk of development, run all three before committing:

1. `uv run --extra dev pre-commit run -a` — linting, formatting, type checking
2. `uv run --extra dev pytest` — tests
3. `uv run --extra docs sphinx-build -n -T docs docs/_build/html` — docs build (catches broken cross-references)

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

```text
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

2. **Masking** (`src/kwneuro/masks.py`) — requires `kwneuro[hdbet]`:
   - `brain_extract_dwi_batch(cases: list[tuple[Dwi, Path]]) -> list[NiftiVolumeResource]`
   - `brain_extract_structural_batch(cases: list[tuple[StructuralImage, Path]]) -> list[NiftiVolumeResource]`
   - `brain_extract(volume: VolumeResource, output_path: PathLike) -> NiftiVolumeResource`
   - Uses HD-BET (deep learning) on mean b0 images (DWI) or the volume directly (structural)
   - **Important**: Always prefer batch processing - HD-BET initialization is
     expensive

3. **DTI Estimation** (`src/kwneuro/dti.py`):
   - `Dti.estimate_dti(dwi: Dwi, mask: VolumeResource | None) -> Dti`
   - Uses DIPY's TensorModel
   - Returns 6 values per voxel (lower triangular of symmetric tensor)
   - Provides derived maps: `get_fa_md()`, `get_eig()`

4. **NODDI Estimation** (`src/kwneuro/noddi.py`) — requires `kwneuro[noddi]`:
   - `Noddi.estimate_noddi(dwi: Dwi, mask, dpar, n_kernel_dirs) -> Noddi`
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
   - `register_dwi_to_structural(dwi, structural, type_of_transform, dwi_mask, structural_mask) -> TransformResource`
     - Convenience wrapper that registers the mean b0 (moving) to a
       `StructuralImage` (fixed). Returned transform maps DWI → structural; use
       `transform.apply(..., invert=True, interpolation="genericLabel")` to warp
       T1-space labels back into DWI space.
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

8. **Tract Segmentation** (`src/kwneuro/tractseg.py`) — requires `kwneuro[tractseg]`:
   - `extract_tractseg(dwi, mask, response, output_type) -> VolumeResource`
     - Computes CSD peaks, then runs TractSeg
     - `output_type`: `"tract_segmentation"`, `"endings_segmentation"`, or
       `"TOM"`

9. **Harmonization** (`src/kwneuro/harmonize.py`) — requires `kwneuro[combat]`:
   - `harmonize_volumes(volumes, covars, batch_col, mask, ...) -> tuple[list[InMemoryVolumeResource], CombatEstimates]`
     - Wraps neuroCombat for multi-site ComBat harmonization
     - Operates on 3D scalar maps (FA, MD, NDI, etc.) in common space
     - Flattens masked voxels to features, runs ComBat, reshapes back
     - `preserve_out_of_mask` flag to retain or zero out-of-mask voxels
   - `CombatEstimates` dataclass wraps neuroCombat's estimates and info dicts
   - **Important**: All input volumes must be in the same voxel space (same
     shape/affine)
   - **Important**: This is a group-level operation (like build_template), not a
     per-subject stage

10. **Structural Imaging** (`src/kwneuro/structural.py`):
    - `StructuralImage` is the structural-side analogue of `Dwi`: a small
      dataclass wrapping a single 3D `VolumeResource` (typically a T1w) with the
      same `load()` / `save()` / cache-protocol pattern.
    - Methods (each `@cacheable`):
      - `correct_bias() -> StructuralImage` — ANTsPy N4 bias field correction.
      - `extract_brain() -> InMemoryVolumeResource` — HD-BET brain mask. Single
        call only; for batches use
        `kwneuro.masks.brain_extract_structural_batch`.
      - `segment_tissues(mask=None, method="atropos" | "deep_atropos") -> InMemoryVolumeResource`
        - `"atropos"`: ANTsPy k-means, 3 classes (1=CSF, 2=GM, 3=WM). Uses
          `mask` if provided, else `ants.get_mask`.
        - `"deep_atropos"`: ANTsPyNet deep segmentation, 6 classes (adds deep
          GM, cerebellum, brainstem). **Ignores `mask`** — preprocessing is
          handled inside the model. Requires `kwneuro[antspynet]`.
      - `parcellate(method="dkt") -> InMemoryVolumeResource` — DKT cortical
        labeling via ANTsPyNet. Requires `kwneuro[antspynet]`.
    - `register_dwi_to_structural` (in `reg.py`, listed under stage 6) is the
      bridge between this stage and DWI-space analyses.

### Pipeline Caching (`src/kwneuro/cache.py`)

`Cache(cache_dir, force)` is a context manager that activates transparent disk
caching for all `@cacheable`-decorated functions within its `with` block.
Outside a `Cache` context, decorated functions run normally with zero overhead.

#### Activating the cache

```python
with Cache(cache_dir="my_cache/sub-01", force=False) as cache:
    dti = dwi.estimate_dti(mask)  # computed and saved on first run
    dti = dwi.estimate_dti(mask)  # loaded from disk on second run
```

Each subject should use a distinct `cache_dir` to avoid collisions.
`cache_dir` is created automatically if it does not exist.

#### The `@cacheable` decorator — two forms

**Bare `@cacheable`**: use when the return type implements the cache protocol
(`_cache_files`, `_cache_save`, `_cache_load` class/static methods). All result
classes (`Dti`, `Noddi`, etc.) implement this protocol.

```python
@cacheable
def estimate_dti(dwi: Dwi, mask: VolumeResource | None = None) -> Dti:
    ...
```

**`@cacheable(CacheSpec(...))`**: use when the return type is a tuple or other
type that cannot carry the protocol.

```python
@cacheable(
    CacheSpec(
        files=["peaks_dirs.nii.gz", "peaks_values.nii.gz"],
        save=lambda result, d: ...,
        load=lambda d: ...,
        step_name="compute_csd_peaks",  # optional; defaults to fn.__name__
    )
)
def compute_csd_peaks() -> tuple[VolumeResource, VolumeResource]:
    ...
```

`CacheSpec.step_name` overrides the default step name (the function's
`__name__`). This is useful to avoid collisions when two functions would
otherwise share a name.

#### Return-type resolution is deferred

For bare `@cacheable`, the return type annotation is resolved at the **first
call**, not at decoration time. This allows `@cacheable` to be stacked directly
on `@staticmethod` inside a class body, where the class name is not yet in the
module namespace at decoration time.

#### Cache miss always returns the disk-loaded result

After a cache miss the function runs, the result is saved, and then
`_cache_load()` is called before returning — not the in-memory result directly.
This guarantees the caller always receives a disk-round-tripped object,
ensuring consistency between hit and miss paths.

#### Fingerprinting

Each call's arguments are classified into two buckets stored in a sidecar file
`{step_name}.params.json`:

- **`"scalars"`** — `bool`, `int`, `float`, `str`, `None` values stored
  verbatim as human-readable JSON. Note: `None` is treated as a scalar, not
  fingerprinted.
- **`"hashes"`** — everything else is content-fingerprinted with sha256.
  Supported types: numpy arrays (bytes + shape + dtype), numpy scalars, `Path`,
  `dict` (sorted by key), `list`/`tuple` (ordered), and **dataclass instances**
  (recursively, field by field). This covers the full resource hierarchy (`Dwi`,
  `VolumeResource`, etc.) without those classes needing to know about caching.

A mismatch in either section triggers a cache miss. Arguments of unrecognised
types cannot be tracked and trigger a `UserWarning`; use
`force={"step_name"}` to force recomputation when such an argument changes.

#### `force` parameter

`force` accepts `bool` or `set[str]`:

```python
Cache(cache_dir=..., force=True)  # recompute everything
Cache(cache_dir=..., force={"estimate_noddi"})  # recompute only this step
```

#### `cache.status()`

```python
cache.status([dwi.estimate_dti, dwi.estimate_noddi])
# -> {"Dwi.estimate_dti": True, "Dwi.estimate_noddi": False}
```

Returns a `dict[str, bool]` mapping each step's `fn.__qualname__` to whether
all its cached output files exist on disk. Non-decorated callables passed to
`status()` are silently skipped.

#### Thread / async safety

`_active_cache` is a `contextvars.ContextVar`, so each thread or asyncio
coroutine has its own independent cache context — nested or concurrent
pipelines do not interfere with each other.

### CLI Integration

The `gen_masks` command (`src/kwneuro/run.py`) demonstrates the batch processing
pattern:

1. Recursively finds `*_dwi.nii.gz` files
2. Constructs `Dwi` objects with on-disk resources (doesn't load!)
3. Batches all cases and calls `brain_extract_dwi_batch()` with single HD-BET
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
3. Add a static `estimate_<model>()` method (e.g. `Dti.estimate_dti`,
   `Noddi.estimate_noddi`) decorated with `@cacheable`
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
2. **brain_extract_dwi_batch / brain_extract_structural_batch are strongly preferred over brain_extract** -
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
10. **neuroCombat expects (features, samples) layout** - our wrapper handles the
    transpose, but if interacting with neuroCombat directly, remember that rows
    are voxels and columns are subjects
11. **neuroCombat prints to stdout** - brief progress messages that we don't
    suppress; this is expected behavior
12. **neuroHarmonize's dependency declaration is broken** - neuroHarmonize
    (which wraps neuroCombat) fails to declare neuroCombat as a dependency; we
    depend on neuroCombat directly to avoid this issue

## Testing Strategy

- Use `pytest-mock` for mocking expensive operations (HD-BET, AMICO)
- Test data fixtures use synthetic volumes with known properties
- Warn/error filters accommodate dependencies (see pyproject.toml)
- Coverage target excludes TYPE_CHECKING blocks and ellipsis

## Dependencies

### Core

- **dipy** (>=1.9): Diffusion imaging toolkit
- **nibabel**: NIfTI file I/O
- **click**: CLI framework
- **antspyx** (>=0.6.2): Registration and template building

### Optional (pip extras)

These are heavier or more fragile dependencies that are opt-in. Each has a
corresponding pip extra:

- **hd-bet** (==2.0.1): Deep learning brain extraction — `pip install kwneuro[hdbet]`
- **dmri-amico** (==2.1.1): NODDI model fitting — `pip install kwneuro[noddi]`
- **TractSeg**: White matter tract segmentation — `pip install kwneuro[tractseg]`
- **neuroCombat** (==0.2.12): ComBat harmonization — `pip install kwneuro[combat]`
- **antspynet**: Deep-learning structural segmentation / parcellation
  (`StructuralImage.segment_tissues(method="deep_atropos")`,
  `StructuralImage.parcellate`) — `pip install kwneuro[antspynet]`. Pulls in
  TensorFlow as a transitive dep.
- Install all at once with `pip install kwneuro[all]`

Source modules use lazy imports: the optional dependency is only imported inside
the function that needs it, and raises `ImportError` with install instructions
if missing.

### Pinned Versions

- AMICO and HD-BET are pinned due to API stability concerns
- `backports.tarfile` required for older amico/setuptools compatibility (in the
  `noddi` extra)
- neuroCombat is pinned because the library is dormant (no releases since 2021)
  and we want Dependabot to flag any unexpected new release
