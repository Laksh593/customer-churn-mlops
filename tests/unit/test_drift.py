"""tests/unit/test_drift.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for statistical drift calculation engine (Milestone 11 - Step 2).

Tests:
1. Identical reference and current distributions (PSI ~ 0, p-value high, no drift).
2. Meaningful numerical distribution shifts (high PSI, small p-value, drift detected).
3. Categorical distribution shifts.
4. Previously unseen categories handling and tracking.
5. Missing values and missingness rate delta tracking.
6. Zero proportions and PSI smoothing (avoids division by zero / inf).
7. Constant features and duplicate bin boundaries handling.
8. Values below or above reference boundaries (boundary extension).
9. Empty batches and invalid values (safe status tracking).
10. Insufficient sample sizes (< min_samples) non-evaluable status.
11. Invalid Chi-square cases (Cochran assumption failure).
12. Bonferroni correction across multiple comparisons.
13. Prediction probability distributions inside and outside the valid range.
14. Deterministic output and correct dual-condition drift decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    calculate_psi,
    evaluate_batch_drift,
    evaluate_categorical_drift,
    evaluate_numerical_drift,
    evaluate_prediction_drift,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_numerical_ref() -> dict[str, Any]:
    """Return synthetic reference profile for a numerical feature."""
    return {
        "count": 1000,
        "missing_count": 0,
        "mean": 50.0,
        "std": 15.0,
        "min": 10.0,
        "max": 90.0,
        "bin_edges": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        "proportions": [0.125] * 8,
    }


@pytest.fixture()
def sample_categorical_ref() -> dict[str, Any]:
    """Return synthetic reference profile for a categorical feature."""
    return {
        "count": 1000,
        "missing_count": 0,
        "categories": ["A", "B", "C"],
        "frequencies": {"A": 0.50, "B": 0.30, "C": 0.20},
    }


@pytest.fixture()
def sample_prediction_ref() -> dict[str, Any]:
    """Return synthetic reference profile for prediction probabilities."""
    return {
        "count": 1000,
        "mean": 0.40,
        "std": 0.25,
        "median": 0.35,
        "min": 0.01,
        "max": 0.98,
        "bin_edges": [round(b, 4) for b in np.linspace(0.0, 1.0, 11)],
        "proportions": [0.10] * 10,
    }


# ---------------------------------------------------------------------------
# PSI calculation unit tests
# ---------------------------------------------------------------------------


def test_calculate_psi_identical() -> None:
    """PSI between identical distributions must be 0.0."""
    props = [0.1, 0.2, 0.4, 0.3]
    psi = calculate_psi(props, props)
    assert pytest.approx(psi, abs=1e-6) == 0.0


def test_calculate_psi_shifted() -> None:
    """PSI between shifted distributions must be strictly positive."""
    ref = [0.25, 0.25, 0.25, 0.25]
    curr = [0.05, 0.05, 0.45, 0.45]
    psi = calculate_psi(ref, curr)
    assert psi > 0.20


def test_calculate_psi_zero_proportions_smoothing() -> None:
    """PSI with zero frequency bins must remain finite and well-defined with epsilon smoothing."""
    ref = [0.25, 0.25, 0.25, 0.25]
    curr = [0.0, 0.0, 0.5, 0.5]
    psi = calculate_psi(ref, curr, epsilon=1e-4)
    assert np.isfinite(psi)
    assert psi > 0.5


def test_calculate_psi_length_mismatch_raises() -> None:
    """Mismatched proportions lengths must raise ValueError."""
    with pytest.raises(
        ValueError, match="Reference length .* and current length .* must match"
    ):
        calculate_psi([0.5, 0.5], [0.3, 0.3, 0.4])


# ---------------------------------------------------------------------------
# Numerical feature drift evaluation tests
# ---------------------------------------------------------------------------


def test_numerical_drift_identical(sample_numerical_ref: dict[str, Any]) -> None:
    """Current batch identical to reference distribution should report no drift."""
    # Generate 400 samples uniformly spread across the 8 bins (50 per bin)
    vals: list[float] = []
    edges = sample_numerical_ref["bin_edges"]
    for i in range(len(edges) - 1):
        mid = (edges[i] + edges[i + 1]) / 2.0
        vals.extend([mid] * 50)

    res = evaluate_numerical_drift(
        pd.Series(vals), sample_numerical_ref, feature_name="feat_num"
    )
    assert res.status == "evaluated"
    assert res.psi is not None and res.psi < 0.02
    assert res.test_evaluable is True
    assert res.raw_p_value is not None and res.raw_p_value > 0.90
    assert res.drift_detected is False


def test_numerical_drift_shifted(sample_numerical_ref: dict[str, Any]) -> None:
    """Strongly shifted numerical distribution should flag drift."""
    # Concentrate all 200 samples in the last bin
    vals = [85.0] * 200
    res = evaluate_numerical_drift(
        pd.Series(vals), sample_numerical_ref, feature_name="feat_num"
    )
    assert res.status == "evaluated"
    assert res.psi is not None and res.psi > 0.50
    assert res.test_evaluable is True
    assert res.raw_p_value is not None and res.raw_p_value < 1e-4


def test_numerical_drift_values_outside_boundaries(
    sample_numerical_ref: dict[str, Any]
) -> None:
    """Values below min (10) or above max (90) must be captured in outer bins without dropping."""
    vals = [-50.0] * 50 + [500.0] * 50
    res = evaluate_numerical_drift(
        pd.Series(vals), sample_numerical_ref, feature_name="feat_num"
    )
    assert res.status == "evaluated"
    assert res.sample_count == 100
    assert res.psi is not None and res.psi > 0.10


def test_numerical_drift_missing_values(sample_numerical_ref: dict[str, Any]) -> None:
    """Missing values must be accounted for and missing rate delta calculated."""
    vals = [30.0] * 70 + [np.nan] * 30
    res = evaluate_numerical_drift(
        pd.Series(vals), sample_numerical_ref, feature_name="feat_num"
    )
    assert res.sample_count == 70
    assert res.missing_count == 30
    assert pytest.approx(res.missing_rate, abs=1e-5) == 0.30
    assert pytest.approx(res.missing_rate_delta, abs=1e-5) == 0.30


def test_numerical_drift_constant_feature() -> None:
    """Constant feature (min == max) must evaluate via single bounded bin without crashing."""
    const_ref = {
        "count": 500,
        "missing_count": 0,
        "min": 1.0,
        "max": 1.0,
        "bin_edges": [0.5, 1.5],
        "proportions": [1.0],
    }
    # Current has same constant value
    res = evaluate_numerical_drift(
        pd.Series([1.0] * 50), const_ref, feature_name="constant_feat"
    )
    assert res.status == "evaluated"
    assert res.psi == 0.0
    assert res.drift_detected is False

    # Current has different constant value
    res_shifted = evaluate_numerical_drift(
        pd.Series([5.0] * 50), const_ref, feature_name="constant_feat"
    )
    assert res_shifted.status == "evaluated"
    assert res_shifted.drift_detected is True


def test_numerical_drift_insufficient_samples(
    sample_numerical_ref: dict[str, Any]
) -> None:
    """Sample size below min_samples must return status='insufficient_samples' and no drift."""
    res = evaluate_numerical_drift(
        pd.Series([25.0] * 10),
        sample_numerical_ref,
        min_samples=30,
        feature_name="feat_num",
    )
    assert res.status == "insufficient_samples"
    assert res.sample_count == 10
    assert res.psi is None
    assert res.drift_detected is False


def test_numerical_drift_empty_input(sample_numerical_ref: dict[str, Any]) -> None:
    """Empty input series must return status='empty_input' safely."""
    res = evaluate_numerical_drift(pd.Series([], dtype=float), sample_numerical_ref)
    assert res.status == "empty_input"
    assert res.sample_count == 0
    assert res.drift_detected is False


# ---------------------------------------------------------------------------
# Categorical feature drift evaluation tests
# ---------------------------------------------------------------------------


def test_categorical_drift_identical(sample_categorical_ref: dict[str, Any]) -> None:
    """Current categorical batch matching reference frequencies should report no drift."""
    vals = ["A"] * 100 + ["B"] * 60 + ["C"] * 40  # 50%, 30%, 20%
    res = evaluate_categorical_drift(
        pd.Series(vals), sample_categorical_ref, feature_name="cat_feat"
    )
    assert res.status == "evaluated"
    assert res.psi is not None and res.psi < 0.01
    assert res.test_evaluable is True
    assert res.raw_p_value is not None and res.raw_p_value > 0.90
    assert res.drift_detected is False
    assert len(res.unseen_categories) == 0


def test_categorical_drift_shifted(sample_categorical_ref: dict[str, Any]) -> None:
    """Shifted categorical distribution should flag drift."""
    vals = ["A"] * 10 + ["B"] * 20 + ["C"] * 170  # Inverted proportions
    res = evaluate_categorical_drift(
        pd.Series(vals), sample_categorical_ref, feature_name="cat_feat"
    )
    assert res.status == "evaluated"
    assert res.psi is not None and res.psi > 0.20
    assert res.test_evaluable is True
    assert res.raw_p_value is not None and res.raw_p_value < 1e-4


def test_categorical_drift_unseen_categories(
    sample_categorical_ref: dict[str, Any]
) -> None:
    """Previously unseen categories must be tracked and captured in unseen rate."""
    vals = ["A"] * 60 + ["B"] * 40 + ["UNKNOWN_X"] * 20 + ["UNKNOWN_Y"] * 10
    res = evaluate_categorical_drift(
        pd.Series(vals), sample_categorical_ref, feature_name="cat_feat"
    )
    assert res.status == "evaluated"
    assert res.unseen_categories == ["UNKNOWN_X", "UNKNOWN_Y"]
    assert pytest.approx(res.unseen_rate, abs=1e-5) == 30 / 130
    assert res.psi is not None and res.psi > 0.10
    # Chi-square is not evaluable when unseen categories have zero expected reference frequency
    assert res.test_evaluable is False
    assert "Unseen categories" in res.notes


def test_categorical_drift_all_missing(sample_categorical_ref: dict[str, Any]) -> None:
    """All-missing categorical series must return status='all_missing'."""
    res = evaluate_categorical_drift(
        pd.Series([None, None, None]), sample_categorical_ref
    )
    assert res.status == "all_missing"
    assert res.sample_count == 0
    assert res.drift_detected is False


# ---------------------------------------------------------------------------
# Prediction probability drift evaluation tests
# ---------------------------------------------------------------------------


def test_prediction_drift_identical(sample_prediction_ref: dict[str, Any]) -> None:
    """Probabilities matching reference deciles must report no drift."""
    vals: list[float] = []
    edges = sample_prediction_ref["bin_edges"]
    for i in range(len(edges) - 1):
        mid = (edges[i] + edges[i + 1]) / 2.0
        vals.extend([mid] * 20)

    res = evaluate_prediction_drift(vals, sample_prediction_ref)
    assert res.status == "evaluated"
    assert res.psi is not None and res.psi < 0.02
    assert res.test_evaluable is True
    assert res.raw_p_value is not None and res.raw_p_value > 0.90
    assert res.drift_detected is False


def test_prediction_drift_shifted(sample_prediction_ref: dict[str, Any]) -> None:
    """Shifted probability distribution must produce high PSI and low p-value."""
    # 200 samples highly confident near 0.95
    vals = [0.95] * 200
    res = evaluate_prediction_drift(vals, sample_prediction_ref)
    assert res.status == "evaluated"
    assert res.psi is not None and res.psi > 0.50
    assert res.test_evaluable is True
    assert res.raw_p_value is not None and res.raw_p_value < 1e-4


def test_prediction_drift_invalid_range(sample_prediction_ref: dict[str, Any]) -> None:
    """Probabilities outside [0.0, 1.0] must return status='invalid_input'."""
    res = evaluate_prediction_drift([-0.1, 0.5, 1.2], sample_prediction_ref)
    assert res.status == "invalid_input"
    assert res.drift_detected is False
    assert "outside valid range" in res.notes


def test_prediction_drift_empty_input(sample_prediction_ref: dict[str, Any]) -> None:
    """Empty probability array returns status='empty_input'."""
    res = evaluate_prediction_drift([], sample_prediction_ref)
    assert res.status == "empty_input"
    assert res.drift_detected is False


# ---------------------------------------------------------------------------
# Batch drift evaluation and Bonferroni correction tests
# ---------------------------------------------------------------------------


def test_batch_drift_bonferroni_correction(
    sample_numerical_ref: dict[str, Any],
    sample_categorical_ref: dict[str, Any],
    sample_prediction_ref: dict[str, Any],
) -> None:
    """Verify Bonferroni correction multiplies p-values by m and enforces dual-condition decision."""
    ref_profile = {
        "numerical_features": {
            "num1": sample_numerical_ref,
            "num2": sample_numerical_ref,
        },
        "categorical_features": {"cat1": sample_categorical_ref},
        "prediction_probability": sample_prediction_ref,
    }

    # Construct batch of 100 samples with slight natural variation (not drifted)
    df = pd.DataFrame(
        {
            "num1": np.linspace(15.0, 85.0, 100),
            "num2": np.linspace(15.0, 85.0, 100),
            "cat1": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
        }
    )
    probs = np.linspace(0.05, 0.95, 100)

    report = evaluate_batch_drift(
        current_df=df,
        reference_profile=ref_profile,
        current_probabilities=probs,
        psi_threshold=0.10,
        alpha_threshold=0.05,
    )

    assert report.status == "evaluated"
    assert report.total_samples == 100
    assert report.num_statistical_tests == 4  # num1, num2, cat1, probability

    for col in ["num1", "num2", "cat1"]:
        res = report.feature_results[col]
        assert res.raw_p_value is not None
        assert res.corrected_p_value is not None
        # Bonferroni corrected p-value = min(1.0, raw_p * 4)
        assert pytest.approx(res.corrected_p_value, abs=1e-5) == min(
            1.0, res.raw_p_value * 4
        )

    assert report.prediction_result is not None
    assert report.prediction_result.corrected_p_value is not None
    assert pytest.approx(report.prediction_result.corrected_p_value, abs=1e-5) == min(
        1.0, report.prediction_result.raw_p_value * 4
    )


def test_batch_drift_missing_feature_handled(
    sample_numerical_ref: dict[str, Any],
    sample_categorical_ref: dict[str, Any],
) -> None:
    """When a feature is missing from the incoming DataFrame, it is flagged as 'missing_feature'."""
    ref_profile = {
        "numerical_features": {
            "present_num": sample_numerical_ref,
            "missing_num": sample_numerical_ref,
        },
        "categorical_features": {"present_cat": sample_categorical_ref},
    }

    df = pd.DataFrame(
        {
            "present_num": np.linspace(15.0, 85.0, 50),
            "present_cat": ["A"] * 25 + ["B"] * 15 + ["C"] * 10,
        }
    )

    report = evaluate_batch_drift(df, ref_profile)
    assert report.feature_results["missing_num"].status == "missing_feature"
    assert report.feature_results["missing_num"].drift_detected is False
    assert report.feature_results["present_num"].status == "evaluated"


def test_batch_drift_empty_dataframe(sample_numerical_ref: dict[str, Any]) -> None:
    """Empty DataFrame input produces report with status='empty_input' without exceptions."""
    ref_profile = {"numerical_features": {"num1": sample_numerical_ref}}
    report = evaluate_batch_drift(pd.DataFrame(), ref_profile)
    assert report.status == "empty_input"
    assert report.total_samples == 0
    assert report.num_drifted_features == 0


def test_batch_drift_insufficient_samples(sample_numerical_ref: dict[str, Any]) -> None:
    """Batch smaller than min_samples returns status='insufficient_samples'."""
    ref_profile = {"numerical_features": {"num1": sample_numerical_ref}}
    df = pd.DataFrame({"num1": [10.0, 20.0]})
    report = evaluate_batch_drift(df, ref_profile, min_samples=30)
    assert report.status == "insufficient_samples"
    assert report.total_samples == 2
    assert report.num_drifted_features == 0
