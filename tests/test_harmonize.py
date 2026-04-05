from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from kwneuro.harmonize import CombatEstimates, harmonize_volumes
from kwneuro.resource import InMemoryVolumeResource


@pytest.fixture
def small_nifti_header():
    hdr = nib.Nifti1Header()
    hdr["descrip"] = b"a kwneuro unit test header description"
    hdr.set_xyzt_units(xyz="mm")
    return hdr


@pytest.fixture
def synthetic_site_data(small_nifti_header):
    """Create 3 sites x 5 subjects with known site biases and an age effect."""
    rng = np.random.default_rng(42)
    shape = (10, 12, 8)
    n_per_site = 5
    sites = ["site_A", "site_B", "site_C"]

    # Base signal: smooth gradient (biological ground truth)
    base = np.linspace(0.3, 0.7, shape[0])[:, None, None] * np.ones(shape)

    # Site biases
    site_shifts = {"site_A": 0.0, "site_B": 0.3, "site_C": -0.2}
    site_scales = {"site_A": 1.0, "site_B": 1.5, "site_C": 0.7}

    # Mask: central region
    mask_array = np.zeros(shape, dtype=np.int8)
    mask_array[2:8, 2:10, 2:6] = 1

    volumes = []
    site_labels = []
    ages = []
    for site in sites:
        for _ in range(n_per_site):
            age = rng.uniform(20, 80)
            age_effect = age * 0.002
            noise = rng.normal(0, 0.02, shape)
            vol_data = (base + age_effect + noise) * site_scales[site] + site_shifts[
                site
            ]
            volumes.append(
                InMemoryVolumeResource(
                    array=vol_data,
                    affine=np.eye(4),
                    metadata=dict(small_nifti_header),
                )
            )
            site_labels.append(site)
            ages.append(age)

    covars = pd.DataFrame({"site": site_labels, "age": ages})
    mask = InMemoryVolumeResource(
        array=mask_array, affine=np.eye(4), metadata=dict(small_nifti_header)
    )

    return volumes, covars, mask, site_shifts, site_scales


# --- Functional tests ---


def test_harmonize_basic_shapes(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data
    harmonized, _estimates = harmonize_volumes(volumes, covars, "site", mask)

    assert len(harmonized) == len(volumes)
    for orig, harm in zip(volumes, harmonized, strict=True):
        assert isinstance(harm, InMemoryVolumeResource)
        assert harm.get_array().shape == orig.get_array().shape


def test_harmonize_site_means_converge(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data
    mask_idx = np.nonzero(mask.get_array() > 0)

    # Per-site means before
    means_before = []
    for site in ["site_A", "site_B", "site_C"]:
        site_mask = covars["site"] == site
        site_vols = [v for v, m in zip(volumes, site_mask, strict=True) if m]
        site_mean = np.mean([v.get_array()[mask_idx].mean() for v in site_vols])
        means_before.append(site_mean)

    harmonized, _ = harmonize_volumes(volumes, covars, "site", mask)

    # Per-site means after
    means_after = []
    for site in ["site_A", "site_B", "site_C"]:
        site_mask = covars["site"] == site
        site_vols = [v for v, m in zip(harmonized, site_mask, strict=True) if m]
        site_mean = np.mean([v.get_array()[mask_idx].mean() for v in site_vols])
        means_after.append(site_mean)

    # Site means should be much closer after harmonization
    assert np.std(means_after) < np.std(means_before) * 0.3


def test_harmonize_site_variances_converge(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data
    mask_idx = np.nonzero(mask.get_array() > 0)

    # Per-site variances before
    vars_before = []
    for site in ["site_A", "site_B", "site_C"]:
        site_mask = covars["site"] == site
        site_vols = [v for v, m in zip(volumes, site_mask, strict=True) if m]
        site_var = np.mean([v.get_array()[mask_idx].var() for v in site_vols])
        vars_before.append(site_var)

    harmonized, _ = harmonize_volumes(volumes, covars, "site", mask)

    vars_after = []
    for site in ["site_A", "site_B", "site_C"]:
        site_mask = covars["site"] == site
        site_vols = [v for v, m in zip(harmonized, site_mask, strict=True) if m]
        site_var = np.mean([v.get_array()[mask_idx].var() for v in site_vols])
        vars_after.append(site_var)

    # Site variances should be closer after harmonization
    assert np.std(vars_after) < np.std(vars_before) * 0.5


def test_harmonize_age_effect_preserved(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data
    mask_idx = np.nonzero(mask.get_array() > 0)
    ages = covars["age"].to_numpy()

    # Correlation between age and mean voxel value before
    means_before = [v.get_array()[mask_idx].mean() for v in volumes]
    corr_before = np.corrcoef(ages, means_before)[0, 1]

    harmonized, _ = harmonize_volumes(
        volumes, covars, "site", mask, continuous_cols=["age"]
    )

    # Correlation after
    means_after = [v.get_array()[mask_idx].mean() for v in harmonized]
    corr_after = np.corrcoef(ages, means_after)[0, 1]

    # Age effect should be preserved (correlation should remain strong)
    assert abs(corr_after) > 0.5
    # And should not be weaker than before (ComBat should remove site confounds)
    assert abs(corr_after) >= abs(corr_before) * 0.5


def test_harmonize_mask_respected(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data
    harmonized, _ = harmonize_volumes(volumes, covars, "site", mask)
    mask_array = mask.get_array()

    for vol in harmonized:
        arr = vol.get_array()
        out_of_mask = arr[mask_array == 0]
        assert np.all(out_of_mask == 0.0)


def test_harmonize_preserve_out_of_mask(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data
    harmonized, _ = harmonize_volumes(
        volumes, covars, "site", mask, preserve_out_of_mask=True
    )
    mask_array = mask.get_array()

    for orig, harm in zip(volumes, harmonized, strict=True):
        orig_outside = orig.get_array()[mask_array == 0]
        harm_outside = harm.get_array()[mask_array == 0]
        np.testing.assert_array_almost_equal(harm_outside, orig_outside)


def test_harmonize_inputs_not_modified(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data

    originals = [v.get_array().copy() for v in volumes]
    mask_original = mask.get_array().copy()

    harmonize_volumes(volumes, covars, "site", mask)

    for orig_arr, vol in zip(originals, volumes, strict=True):
        np.testing.assert_array_equal(vol.get_array(), orig_arr)
    np.testing.assert_array_equal(mask.get_array(), mask_original)


def test_harmonize_returns_estimates(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data
    _harmonized, estimates = harmonize_volumes(volumes, covars, "site", mask)

    assert isinstance(estimates, CombatEstimates)
    assert isinstance(estimates.estimates, dict)
    assert isinstance(estimates.info, dict)

    # Check expected keys from neuroCombat
    assert "gamma.star" in estimates.estimates
    assert "delta.star" in estimates.estimates
    assert "var.pooled" in estimates.estimates
    assert "batch_levels" in estimates.info


# --- Validation error tests ---


def test_harmonize_mismatched_shapes(synthetic_site_data, small_nifti_header):
    volumes, covars, mask, _, _ = synthetic_site_data
    # Replace one volume with a different shape
    bad_vol = InMemoryVolumeResource(
        array=np.zeros((5, 5, 5)),
        affine=np.eye(4),
        metadata=dict(small_nifti_header),
    )
    bad_volumes = list(volumes)
    bad_volumes[0] = bad_vol

    with pytest.raises(ValueError, match="same shape"):
        harmonize_volumes(bad_volumes, covars, "site", mask)


def test_harmonize_wrong_mask_shape(synthetic_site_data, small_nifti_header):
    volumes, covars, _, _, _ = synthetic_site_data
    bad_mask = InMemoryVolumeResource(
        array=np.ones((5, 5, 5), dtype=np.int8),
        affine=np.eye(4),
        metadata=dict(small_nifti_header),
    )

    with pytest.raises(ValueError, match="Mask shape"):
        harmonize_volumes(volumes, covars, "site", bad_mask)


def test_harmonize_empty_mask(synthetic_site_data, small_nifti_header):
    volumes, covars, _, _, _ = synthetic_site_data
    shape = volumes[0].get_array().shape
    empty_mask = InMemoryVolumeResource(
        array=np.zeros(shape, dtype=np.int8),
        affine=np.eye(4),
        metadata=dict(small_nifti_header),
    )

    with pytest.raises(ValueError, match="empty"):
        harmonize_volumes(volumes, covars, "site", empty_mask)


def test_harmonize_single_site(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data
    # Use only site_A subjects
    site_a_mask = covars["site"] == "site_A"
    site_a_vols = [v for v, m in zip(volumes, site_a_mask, strict=True) if m]
    site_a_covars = covars[site_a_mask].reset_index(drop=True)

    with pytest.raises(ValueError, match="At least 2 batches"):
        harmonize_volumes(site_a_vols, site_a_covars, "site", mask)


def test_harmonize_covars_length_mismatch(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data
    # Drop a row from covars
    short_covars = covars.iloc[:-1]

    with pytest.raises(ValueError, match="does not match"):
        harmonize_volumes(volumes, short_covars, "site", mask)


def test_harmonize_missing_batch_col(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data

    with pytest.raises(ValueError, match="not found in covars"):
        harmonize_volumes(volumes, covars, "nonexistent_col", mask)


def test_harmonize_missing_covariate_col(synthetic_site_data):
    volumes, covars, mask, _, _ = synthetic_site_data

    with pytest.raises(ValueError, match="not found in covars"):
        harmonize_volumes(
            volumes, covars, "site", mask, continuous_cols=["nonexistent"]
        )

    with pytest.raises(ValueError, match="not found in covars"):
        harmonize_volumes(
            volumes, covars, "site", mask, categorical_cols=["nonexistent"]
        )


def test_harmonize_4d_volume_rejected(synthetic_site_data, small_nifti_header):
    volumes, covars, mask, _, _ = synthetic_site_data
    bad_vol = InMemoryVolumeResource(
        array=np.zeros((10, 12, 8, 5)),
        affine=np.eye(4),
        metadata=dict(small_nifti_header),
    )
    bad_volumes = list(volumes)
    bad_volumes[0] = bad_vol

    with pytest.raises(ValueError, match="3D"):
        harmonize_volumes(bad_volumes, covars, "site", mask)
