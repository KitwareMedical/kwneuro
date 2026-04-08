from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import nibabel as nib
import numpy as np
import pytest
from scipy.linalg import expm

from kwneuro.cache import (
    Cache,
    CacheSpec,
    _active_cache,
    cacheable,
)
from kwneuro.csd import compute_csd_peaks
from kwneuro.dti import Dti
from kwneuro.dwi import Dwi
from kwneuro.noddi import Noddi
from kwneuro.resource import (
    InMemoryBvalResource,
    InMemoryBvecResource,
    InMemoryResponseFunctionResource,
    InMemoryVolumeResource,
)


@pytest.fixture
def random_affine() -> np.ndarray:
    rng = np.random.default_rng(18653)
    affine = np.eye(4)
    affine[:3, :3] = expm((lambda A: (A - A.T) / 2)(rng.normal(size=(3, 3))))
    affine[:3, 3] = rng.random(3)
    return affine


@pytest.fixture
def small_nifti_header():
    hdr = nib.Nifti1Header()
    hdr["descrip"] = b"an kwneuro unit test header description"
    return hdr


@pytest.fixture
def dwi3(random_affine, small_nifti_header) -> Dwi:
    """An example in-memory Dwi with 6 volumes, 3 of which have b=0"""
    n_vols = 6
    rng = np.random.default_rng(7816)
    bvec_array = rng.random(size=(n_vols, 3), dtype=np.float32)
    bvec_array = bvec_array / np.sqrt((bvec_array**2).sum(axis=1, keepdims=True))
    return Dwi(
        volume=InMemoryVolumeResource(
            array=rng.random(size=(3, 4, 5, n_vols), dtype=np.float32),
            affine=random_affine,
            metadata=dict(small_nifti_header),
        ),
        bval=InMemoryBvalResource(array=np.array([0, 1000, 3000, 0, 0, 2000])),
        bvec=InMemoryBvecResource(array=bvec_array),
    )


@pytest.fixture
def dwi4(random_affine, small_nifti_header) -> Dwi:
    """An example in-memory Dwi with 10 volumes."""
    n_vols = 10
    rng = np.random.default_rng(4616)
    bvec_array = rng.random(size=(n_vols, 3), dtype=np.float32)
    bvec_array = bvec_array / np.sqrt((bvec_array**2).sum(axis=1, keepdims=True))
    return Dwi(
        volume=InMemoryVolumeResource(
            array=rng.random(size=(3, 4, 5, n_vols), dtype=np.float32),
            affine=random_affine,
            metadata=dict(small_nifti_header),
        ),
        bval=InMemoryBvalResource(array=rng.integers(0, 3000, n_vols).astype(float)),
        bvec=InMemoryBvecResource(array=bvec_array),
    )


# ---------------------------------------------------------------------------
# Cache.is_forced
# ---------------------------------------------------------------------------


def test_is_forced_false(tmp_path: Path) -> None:
    pc = Cache(tmp_path, force=False)
    assert not pc.is_forced("any_step")


def test_is_forced_true(tmp_path: Path) -> None:
    pc = Cache(tmp_path, force=True)
    assert pc.is_forced("any_step")


def test_is_forced_set_match(tmp_path: Path) -> None:
    pc = Cache(tmp_path, force={"step_a"})
    assert pc.is_forced("step_a")


def test_is_forced_set_no_match(tmp_path: Path) -> None:
    pc = Cache(tmp_path, force={"step_b"})
    assert not pc.is_forced("step_a")


# ---------------------------------------------------------------------------
# Cache.is_cached
# ---------------------------------------------------------------------------


def test_is_cached_missing_file(tmp_path: Path) -> None:
    pc = Cache(tmp_path)
    assert not pc.is_cached("step", ["missing.txt"])


def test_is_cached_all_files_present(tmp_path: Path) -> None:
    (tmp_path / "out.txt").write_text("x")
    pc = Cache(tmp_path, force=False)
    assert pc.is_cached("step", ["out.txt"])


def test_is_cached_forced_overrides_present(tmp_path: Path) -> None:
    (tmp_path / "out.txt").write_text("x")
    pc = Cache(tmp_path, force=True)
    assert not pc.is_cached("step", ["out.txt"])


def test_is_cached_scalars_match(tmp_path: Path) -> None:
    (tmp_path / "out.txt").write_text("x")
    (tmp_path / "step.params.json").write_text(
        json.dumps({"scalars": {"x": 1}}, sort_keys=True)
    )
    pc = Cache(tmp_path)
    assert pc.is_cached("step", ["out.txt"], scalars={"x": 1})


def test_is_cached_scalars_mismatch(tmp_path: Path) -> None:
    (tmp_path / "out.txt").write_text("x")
    (tmp_path / "step.params.json").write_text(
        json.dumps({"scalars": {"x": 1}}, sort_keys=True)
    )
    pc = Cache(tmp_path)
    assert not pc.is_cached("step", ["out.txt"], scalars={"x": 99})


def test_is_cached_params_file_missing_is_miss(tmp_path: Path) -> None:
    (tmp_path / "out.txt").write_text("x")
    pc = Cache(tmp_path)
    assert not pc.is_cached("step", ["out.txt"], scalars={"x": 1})


def test_is_cached_hashes_match(tmp_path: Path) -> None:
    (tmp_path / "out.txt").write_text("x")
    (tmp_path / "step.params.json").write_text(
        json.dumps({"hashes": {"arr": "abc123"}}, sort_keys=True)
    )
    pc = Cache(tmp_path)
    assert pc.is_cached("step", ["out.txt"], hashes={"arr": "abc123"})


def test_is_cached_hashes_mismatch(tmp_path: Path) -> None:
    (tmp_path / "out.txt").write_text("x")
    (tmp_path / "step.params.json").write_text(
        json.dumps({"hashes": {"arr": "abc123"}}, sort_keys=True)
    )
    pc = Cache(tmp_path)
    assert not pc.is_cached("step", ["out.txt"], hashes={"arr": "different"})


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_sets_active_cache(tmp_path: Path) -> None:
    assert _active_cache.get() is None
    pc = Cache(tmp_path)
    with pc:
        assert _active_cache.get() is pc
    assert _active_cache.get() is None


def test_context_manager_restores_on_exception(tmp_path: Path) -> None:
    pc = Cache(tmp_path)
    try:
        with pc:
            error = "Error"
            raise RuntimeError(error)
    except RuntimeError:
        pass
    assert _active_cache.get() is None


def test_context_manager_creates_cache_dir(tmp_path: Path) -> None:
    cache_dir = tmp_path / "nested" / "cache"
    assert not cache_dir.exists()
    with Cache(cache_dir):
        assert cache_dir.is_dir()


# ---------------------------------------------------------------------------
# @cacheable mechanism
# ---------------------------------------------------------------------------


def test_cacheable_raises_on_no_return_annotation(tmp_path: Path) -> None:
    """TypeError is raised on first call within a cache context if no return annotation is present."""

    @cacheable
    def bad_fn(x: int):
        pass

    with (
        pytest.raises(TypeError, match="no resolvable return type"),
        Cache(tmp_path),
    ):
        bad_fn(1)


def test_cacheable_raises_on_missing_protocol(tmp_path: Path) -> None:
    """TypeError is raised on first call within a cache context if return type lacks cache protocol.

    Uses int as the return type: it is resolvable from the module namespace but does not
    implement _cache_files / _cache_save / _cache_load.
    """

    @cacheable
    def bad_fn(x: int) -> int:
        return x

    with (
        pytest.raises(TypeError, match="does not implement the cache protocol"),
        Cache(tmp_path),
    ):
        bad_fn(1)


def test_cacheable_preserves_function_name() -> None:
    """@cacheable preserves __name__ on the wrapped function."""
    assert Dti.estimate_dti.__name__ == "estimate_dti"
    assert Noddi.estimate_noddi.__name__ == "estimate_noddi"
    assert Dwi.denoise.__name__ == "denoise"


def test_cacheable_no_op_outside_context(dwi3: Dwi) -> None:
    """@cacheable is a transparent pass-through when no Cache context is active."""
    dti = dwi3.estimate_dti()
    assert isinstance(dti, Dti)


# ---------------------------------------------------------------------------
# Cache.status
# ---------------------------------------------------------------------------


def test_status_missing_shows_false(tmp_path: Path) -> None:
    pc = Cache(tmp_path)
    result = pc.status([Dti.estimate_dti])
    assert result["Dti.estimate_dti"] is False


def test_status_present_shows_true(dwi3: Dwi, tmp_path: Path) -> None:
    pc = Cache(tmp_path)
    with pc:
        dwi3.estimate_dti()
    result = pc.status([Dti.estimate_dti])
    assert result["Dti.estimate_dti"] is True


def test_status_non_cacheable_skipped(tmp_path: Path) -> None:
    def plain() -> None:
        pass

    pc = Cache(tmp_path)
    assert pc.status([plain]) == {}


def test_status_multiple_steps(dwi3: Dwi, tmp_path: Path) -> None:
    """status reflects which steps have been cached and which have not."""
    pc = Cache(tmp_path)
    with pc:
        dwi3.estimate_dti()
    result = pc.status([Dti.estimate_dti, Noddi.estimate_noddi])
    assert result["Dti.estimate_dti"] is True
    assert result["Noddi.estimate_noddi"] is False


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_estimate_dti_with_caching(dwi3: Dwi, tmp_path: Path) -> None:
    """DTI results are saved to cache and reloaded correctly on the second call."""
    pc = Cache(tmp_path)
    with pc:
        dti = dwi3.estimate_dti()

    assert np.allclose(dti.volume.get_affine(), dwi3.volume.get_affine())
    assert dti.volume.get_array().shape[:3] == (3, 4, 5)
    assert (tmp_path / "estimate_dti.nii.gz").exists()
    assert pc.status([Dti.estimate_dti])["Dti.estimate_dti"] is True

    # Second call returns the same result loaded from cache
    with pc:
        dti_cached = dwi3.estimate_dti()
    assert np.allclose(dti_cached.volume.get_array(), dti.volume.get_array())


def test_estimate_dti_force_recompute(dwi3: Dwi, tmp_path: Path, mocker) -> None:
    """force={"estimate_dti"} causes DTI to recompute even when a cache exists."""
    pc = Cache(tmp_path)
    with pc:
        dwi3.estimate_dti()

    save_spy = mocker.spy(Dti, "_cache_save")
    with Cache(tmp_path, force={"estimate_dti"}):
        dwi3.estimate_dti()

    save_spy.assert_called_once()


def _make_amico_mock(mocker, rng: np.random.Generator) -> MagicMock:
    """Patch AMICO internals so Noddi.estimate_noddi runs without the full AMICO stack.

    The @cacheable wrapper on estimate_noddi is left intact so that caching
    behaviour can be tested end-to-end.

    AMICO is an optional pip extra. A fake `amico` module is injected into
    `sys.modules` so these tests run even when AMICO is not installed.
    """
    mock_ae = MagicMock()
    mock_ae.RESULTS = {
        "MAPs": rng.random(size=(3, 4, 5, 3)).astype(np.float32),
        "DIRs": rng.random(size=(3, 4, 5, 3)).astype(np.float32),
    }
    mock_ae.model.maps_name = ["NDI", "ODI", "FWF"]
    # amico is imported inside Noddi.estimate_noddi as ``import amico``. Since
    # amico is an optional pip extra that may not be installed, we inject fake
    # module objects into sys.modules first so that ``import amico`` succeeds,
    # then patch the individual attributes the function actually calls.
    mock_amico_module = MagicMock()
    mock_amico_util = MagicMock()
    mocker.patch.dict(
        "sys.modules", {"amico": mock_amico_module, "amico.util": mock_amico_util}
    )
    mock_amico_module.Evaluation.return_value = mock_ae
    mock_amico_module.util = mock_amico_util
    return mock_ae


def test_estimate_noddi_with_caching(dwi3: Dwi, mocker, tmp_path: Path) -> None:
    """NODDI results are saved to cache and reloaded correctly on the second call."""
    mock_ae = _make_amico_mock(mocker, np.random.default_rng(18653))

    pc = Cache(tmp_path)
    with pc:
        noddi = dwi3.estimate_noddi()

    assert np.allclose(noddi.volume.get_array(), mock_ae.RESULTS["MAPs"])
    assert np.allclose(noddi.directions.get_array(), mock_ae.RESULTS["DIRs"])
    assert np.allclose(noddi.volume.get_affine(), dwi3.volume.get_affine())
    assert (tmp_path / "estimate_noddi.nii.gz").exists()
    assert (tmp_path / "estimate_noddi_directions.nii.gz").exists()
    assert pc.status([Noddi.estimate_noddi])["Noddi.estimate_noddi"] is True

    # Second call loads from cache — AMICO fit is not re-invoked
    mock_ae.fit.reset_mock()
    with pc:
        dwi3.estimate_noddi()
    mock_ae.fit.assert_not_called()


def test_estimate_noddi_parameter_change_triggers_recompute(
    dwi3: Dwi, mocker, tmp_path: Path
) -> None:
    """Changing a scalar parameter (dpar) invalidates the NODDI cache and triggers recomputation."""
    mock_ae = _make_amico_mock(mocker, np.random.default_rng(18653))

    pc = Cache(tmp_path)
    with pc:
        dwi3.estimate_noddi(dpar=1.7e-3)

    mock_ae.fit.reset_mock()
    with pc:
        dwi3.estimate_noddi(dpar=1.3e-3)  # different dpar → cache miss

    mock_ae.fit.assert_called_once()


def test_estimate_noddi_same_params_no_recompute(
    dwi3: Dwi, mocker, tmp_path: Path
) -> None:
    """Calling estimate_noddi with unchanged parameters reuses the cached result."""
    mock_ae = _make_amico_mock(mocker, np.random.default_rng(18653))

    pc = Cache(tmp_path)
    with pc:
        dwi3.estimate_noddi(dpar=1.7e-3)

    mock_ae.fit.reset_mock()
    with pc:
        dwi3.estimate_noddi(dpar=1.7e-3)  # same params → cache hit

    mock_ae.fit.assert_not_called()


def test_compute_csd_peaks_with_caching(dwi3: Dwi, mocker, tmp_path: Path) -> None:
    """CSD peaks are saved to cache via CacheSpec and reloaded correctly on the second call.

    compute_csd_peaks uses @cacheable(CacheSpec(...)) rather than the bare @cacheable
    decorator, so this test exercises the explicit-spec caching path with fixed output
    filenames and custom save/load lambdas.
    """
    rng = np.random.default_rng(18653)
    n_peaks = 5

    # Mock the DIPY CSD model and peak-finding to avoid running the full pipeline,
    mock_peaks = MagicMock()
    mock_peaks.peak_dirs = rng.random(size=(3, 4, 5, n_peaks, 3)).astype(np.float32)
    mock_peaks.peak_values = rng.random(size=(3, 4, 5, n_peaks)).astype(np.float32)
    mocker.patch("kwneuro.csd.ConstrainedSphericalDeconvModel")
    mock_peaks_from_model = mocker.patch(
        "kwneuro.csd.peaks_from_model", return_value=mock_peaks
    )

    mask = InMemoryVolumeResource(
        array=np.ones((3, 4, 5), dtype=np.float32),
        affine=dwi3.volume.get_affine(),
    )
    # Use a real InMemoryResponseFunctionResource so it is content-fingerprinted
    # rather than triggering an untracked-parameter warning (warnings are errors
    # in this test suite).
    mock_response = InMemoryResponseFunctionResource(
        sh_coeffs=np.zeros(5, dtype=np.float64),
        avg_signal=np.float32(1.0),
    )

    pc = Cache(tmp_path)
    with pc:
        peak_dirs, peak_values = compute_csd_peaks(dwi3, mask, response=mock_response)

    # Result contains the expected arrays
    assert np.allclose(peak_dirs.get_array(), mock_peaks.peak_dirs)
    assert np.allclose(peak_values.get_array(), mock_peaks.peak_values)

    # Cache files (fixed names from CacheSpec) exist and status reports cached
    assert (tmp_path / "csd_peak_dirs.nii.gz").exists()
    assert (tmp_path / "csd_peak_values.nii.gz").exists()
    assert pc.status([compute_csd_peaks])["compute_csd_peaks"] is True

    # Second call loads from cache — peaks_from_model is not re-invoked
    mock_peaks_from_model.reset_mock()
    with pc:
        compute_csd_peaks(dwi3, mask, response=mock_response)
    mock_peaks_from_model.assert_not_called()


def test_denoise_with_caching(dwi4: Dwi, tmp_path: Path) -> None:
    """Denoised DWI is saved to cache and reloaded correctly on the second call."""
    pc = Cache(tmp_path)
    with pc:
        denoised = dwi4.denoise()

    assert isinstance(denoised, Dwi)
    assert denoised.volume.get_array().shape == dwi4.volume.get_array().shape
    assert np.allclose(denoised.volume.get_affine(), dwi4.volume.get_affine())
    assert (tmp_path / "denoise.nii.gz").exists()
    assert (tmp_path / "denoise.bval").exists()
    assert (tmp_path / "denoise.bvec").exists()
    assert pc.status([Dwi.denoise])["Dwi.denoise"] is True

    # Second call returns the same volume loaded from cache
    with pc:
        denoised_cached = dwi4.denoise()
    assert np.allclose(denoised_cached.volume.get_array(), denoised.volume.get_array())


# ---------------------------------------------------------------------------
# Content fingerprinting
# ---------------------------------------------------------------------------


def test_params_json_has_scalars_and_hashes(dwi3: Dwi, tmp_path: Path) -> None:
    """The params sidecar contains separate 'scalars' and 'hashes' sections."""
    with Cache(tmp_path):
        dwi3.estimate_dti()

    params = json.loads((tmp_path / "estimate_dti.params.json").read_text())
    # Scalar arguments (mask=None) are human-readable in the "scalars" section.
    assert "scalars" in params
    assert params["scalars"]["mask"] is None
    # Non-scalar arguments (dwi, the Dwi dataclass) are sha256 fingerprints.
    assert "hashes" in params
    assert "dwi" in params["hashes"]
    assert len(params["hashes"]["dwi"]) == 64  # sha256 hex digest is 64 chars


def test_data_change_triggers_recompute(dwi3: Dwi, tmp_path: Path, mocker) -> None:
    """Modifying the imaging array invalidates the cache even when scalar params are unchanged."""
    pc = Cache(tmp_path)
    with pc:
        dwi3.estimate_dti()

    save_spy = mocker.spy(Dti, "_cache_save")

    # Mutate the underlying array — same scalar params, different data fingerprint.
    assert isinstance(dwi3.volume, InMemoryVolumeResource)
    dwi3.volume.array[0, 0, 0, 0] += 1.0

    with pc:
        dwi3.estimate_dti()

    save_spy.assert_called_once()  # recomputed because the data hash changed


def test_untracked_param_warns(tmp_path: Path) -> None:
    """A UserWarning is issued for parameters that cannot be fingerprinted."""

    class Opaque:
        """Not a scalar, not a dataclass, not a numpy type — cannot be fingerprinted."""

    def _save(result: str, d: Path) -> None:
        (d / "out.txt").write_text(result)

    @cacheable(  # type: ignore[untyped-decorator]
        CacheSpec(
            files=["out.txt"],
            save=_save,
            load=lambda d: (d / "out.txt").read_text(),
        )
    )
    def fn(_x: object) -> str:
        return "result"

    with pytest.warns(UserWarning, match="cannot be fingerprinted"), Cache(tmp_path):
        fn(Opaque())
