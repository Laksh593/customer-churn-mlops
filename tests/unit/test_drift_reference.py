"""tests/unit/test_drift_reference.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for data drift reference profile generation (Milestone 11 - Step 1).

Tests:
1. Reproducible profile generation.
2. Correct training and validation split usage.
3. No test-set leakage (X_test is never accessed or used).
4. Absence of raw customer records, customer IDs, and target columns.
5. Duplicate bin boundaries and constant numerical features handling.
6. Categorical frequency normalization and ordering.
7. Prediction probability baseline calculation on validation data.
8. Missing or unavailable model artifacts error handling.
9. Missing or empty raw dataset error handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scripts.generate_drift_reference import (
    compute_categorical_reference,
    compute_numerical_reference,
    compute_prediction_reference,
    generate_reference_profile,
    save_reference_profile,
)

from src.training.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# ---------------------------------------------------------------------------
# Numerical feature reference calculation tests
# ---------------------------------------------------------------------------


def test_compute_numerical_reference_standard() -> None:
    """Verify standard numerical feature binning and summary statistics."""
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], name="test_num")
    result = compute_numerical_reference(series, num_quantiles=4)

    assert result["count"] == 5
    assert result["missing_count"] == 0
    assert result["mean"] == 30.0
    assert result["min"] == 10.0
    assert result["max"] == 50.0
    assert len(result["bin_edges"]) >= 2
    assert pytest.approx(sum(result["proportions"]), abs=1e-5) == 1.0
    assert all(p >= 0.0 for p in result["proportions"])


def test_compute_numerical_reference_constant_feature() -> None:
    """Verify constant feature produces a single safe bounding interval and proportion 1.0."""
    series = pd.Series([42.0, 42.0, 42.0, 42.0], name="constant_feat")
    result = compute_numerical_reference(series)

    assert result["count"] == 4
    assert result["min"] == 42.0
    assert result["max"] == 42.0
    assert result["mean"] == 42.0
    assert result["std"] == 0.0
    assert result["bin_edges"] == [41.5, 42.5]
    assert result["proportions"] == [1.0]


def test_compute_numerical_reference_duplicate_quantiles() -> None:
    """Verify series with repeated values deduplicates boundaries and computes actual proportions."""
    # Heavily skewed binary-like data where multiple quantiles collapse to 0
    series = pd.Series([0.0] * 8 + [1.0] * 2, name="skewed_feat")
    result = compute_numerical_reference(series, num_quantiles=10)

    # Must have strictly increasing bin edges
    edges = result["bin_edges"]
    assert len(edges) >= 2
    for i in range(len(edges) - 1):
        assert edges[i] < edges[i + 1]

    # Actual observed proportions (not hardcoded 10%)
    assert pytest.approx(sum(result["proportions"]), abs=1e-5) == 1.0
    # First bin enclosing 0.0 must have majority mass
    assert result["proportions"][0] >= 0.7


def test_compute_numerical_reference_with_missing_values() -> None:
    """Verify missing values are counted and excluded from distribution statistics."""
    series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, np.nan], name="feat_with_nan")
    result = compute_numerical_reference(series)

    assert result["count"] == 4
    assert result["missing_count"] == 2
    assert result["mean"] == 3.0
    assert pytest.approx(sum(result["proportions"]), abs=1e-5) == 1.0


def test_compute_numerical_reference_empty_raises() -> None:
    """Verify series with only missing values raises ValueError."""
    series = pd.Series([np.nan, np.nan], name="empty_num")
    with pytest.raises(ValueError, match="contains no valid non-null values"):
        compute_numerical_reference(series)


# ---------------------------------------------------------------------------
# Categorical feature reference calculation tests
# ---------------------------------------------------------------------------


def test_compute_categorical_reference_normalization() -> None:
    """Verify categorical frequencies sum to 1.0 and categories are tracked."""
    series = pd.Series(["A", "B", "A", "C", "A", "B"], name="test_cat")
    result = compute_categorical_reference(series)

    assert result["count"] == 6
    assert result["missing_count"] == 0
    assert result["categories"] == ["A", "B", "C"]
    assert pytest.approx(result["frequencies"]["A"], abs=1e-5) == 3 / 6
    assert pytest.approx(result["frequencies"]["B"], abs=1e-5) == 2 / 6
    assert pytest.approx(result["frequencies"]["C"], abs=1e-5) == 1 / 6
    assert pytest.approx(sum(result["frequencies"].values()), abs=1e-5) == 1.0


def test_compute_categorical_reference_with_missing() -> None:
    """Verify missing categorical values are tracked in missing_count."""
    series = pd.Series(["Yes", "No", None, "Yes"], name="cat_with_none")
    result = compute_categorical_reference(series)

    assert result["count"] == 3
    assert result["missing_count"] == 1
    assert pytest.approx(sum(result["frequencies"].values()), abs=1e-5) == 1.0


def test_compute_categorical_reference_empty_raises() -> None:
    """Verify all-null categorical series raises ValueError."""
    series = pd.Series([None, None], name="empty_cat")
    with pytest.raises(ValueError, match="contains no valid non-null values"):
        compute_categorical_reference(series)


# ---------------------------------------------------------------------------
# Prediction probability baseline calculation tests
# ---------------------------------------------------------------------------


def test_compute_prediction_reference_proportions() -> None:
    """Verify prediction probability statistics and decile bin proportions."""
    probs = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    result = compute_prediction_reference(probs, num_bins=10)

    assert result["evaluation_split"] == "validation"
    assert result["count"] == 10
    assert result["min"] == 0.05
    assert result["max"] == 0.95
    assert len(result["bin_edges"]) == 11
    assert result["bin_edges"] == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    # In this uniform case, each of the 10 bins has exactly 1 sample = 0.1 proportion
    assert len(result["proportions"]) == 10
    assert pytest.approx(sum(result["proportions"]), abs=1e-5) == 1.0
    assert all(pytest.approx(p, abs=1e-5) == 0.1 for p in result["proportions"])


def test_compute_prediction_reference_skewed() -> None:
    """Verify actual proportions reflect skewed probabilities rather than assuming 10% each."""
    # 8 samples in [0.0, 0.1], 2 samples in [0.9, 1.0]
    probs = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.95, 0.98])
    result = compute_prediction_reference(probs, num_bins=10)

    assert result["count"] == 10
    assert pytest.approx(sum(result["proportions"]), abs=1e-5) == 1.0
    assert (
        pytest.approx(result["proportions"][0], abs=1e-5) == 0.8
    )  # 80% in first decile
    assert (
        pytest.approx(result["proportions"][-1], abs=1e-5) == 0.2
    )  # 20% in last decile


def test_compute_prediction_reference_empty_raises() -> None:
    """Verify empty or all-NaN probabilities array raises ValueError."""
    with pytest.raises(ValueError, match="No valid prediction probabilities provided"):
        compute_prediction_reference(np.array([np.nan, np.nan]))


# ---------------------------------------------------------------------------
# End-to-end profile generation and data isolation tests
# ---------------------------------------------------------------------------


class MockModelPipeline:
    """Isolated mock model for testing profile generation."""

    def __init__(self) -> None:
        self.predict_proba_calls: list[pd.DataFrame] = []

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_proba_calls.append(X.copy())
        n = len(X)
        # Deterministic mock probabilities
        p1 = np.linspace(0.1, 0.8, n)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def _create_minimal_mock_df(n: int = 60) -> pd.DataFrame:
    """Create a minimal raw DataFrame for testing."""
    records: list[dict[str, Any]] = []
    for i in range(n):
        records.append(
            {
                "customerID": f"TEST-{i:04d}",
                "gender": "Female" if i % 2 == 0 else "Male",
                "SeniorCitizen": 0 if i % 3 != 0 else 1,
                "Partner": "Yes" if i % 2 == 0 else "No",
                "Dependents": "No",
                "tenure": i + 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL" if i % 2 == 0 else "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month" if i % 2 == 0 else "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 20.0 + float(i),
                "TotalCharges": str((20.0 + float(i)) * (i + 1)),
                "Churn": "Yes" if i % 4 == 0 else "No",
            }
        )
    return pd.DataFrame(records)


def test_generate_profile_split_usage_and_no_leakage() -> None:
    """Verify X_train is used for feature baselines, X_val for predictions, and X_test is unused."""
    mock_df = _create_minimal_mock_df(n=60)
    mock_model = MockModelPipeline()

    profile = generate_reference_profile(
        raw_df=mock_df,
        model=mock_model,
        model_name="test-predictor",
        model_version="1",
        random_state=42,
    )

    metadata = profile["metadata"]
    assert metadata["features_split"] == "train"
    assert metadata["prediction_probabilities_split"] == "validation"
    assert metadata["model_name"] == "test-predictor"
    assert metadata["model_version"] == "1"

    # Verify model predict_proba was invoked exactly once, and on the validation split length
    assert len(mock_model.predict_proba_calls) == 1
    val_call_df = mock_model.predict_proba_calls[0]
    assert len(val_call_df) == metadata["num_val_samples"]

    # Verify all expected 26 features exist in profile
    assert set(profile["numerical_features"].keys()) == set(NUMERICAL_FEATURES)
    assert set(profile["categorical_features"].keys()) == set(CATEGORICAL_FEATURES)


def test_generate_profile_no_customer_ids_or_raw_records() -> None:
    """Verify customerID and target Churn are strictly absent from all profile feature dictionaries."""
    mock_df = _create_minimal_mock_df(n=60)
    mock_model = MockModelPipeline()

    profile = generate_reference_profile(
        raw_df=mock_df,
        model=mock_model,
        random_state=42,
    )

    # Check keys
    assert "customerID" not in profile["numerical_features"]
    assert "customerID" not in profile["categorical_features"]
    assert "Churn" not in profile["numerical_features"]
    assert "Churn" not in profile["categorical_features"]

    # Check serialized JSON string for leaks
    json_str = json.dumps(profile)
    assert "customerID" not in json_str
    assert "TEST-" not in json_str


def test_generate_profile_deterministic() -> None:
    """Verify profile generation is fully deterministic given identical inputs."""
    mock_df = _create_minimal_mock_df(n=60)
    mock_model = MockModelPipeline()

    profile1 = generate_reference_profile(
        raw_df=mock_df,
        model=mock_model,
        random_state=42,
    )
    profile2 = generate_reference_profile(
        raw_df=mock_df,
        model=mock_model,
        random_state=42,
    )

    # Ignore timestamp
    p1 = {k: v for k, v in profile1.items() if k != "metadata"}
    p2 = {k: v for k, v in profile2.items() if k != "metadata"}
    assert p1 == p2


def test_missing_model_artifact_raises_error() -> None:
    """Verify clear error is raised when model artifact cannot be loaded."""
    mock_df = _create_minimal_mock_df(n=60)

    with patch(
        "scripts.generate_drift_reference.load_registered_model",
        side_effect=RuntimeError("Model not found in registry"),
    ):
        with pytest.raises(RuntimeError, match="Failed to load model artifact"):
            generate_reference_profile(
                raw_df=mock_df,
                model=None,
            )


def test_missing_raw_dataset_raises_error() -> None:
    """Verify clear FileNotFoundError is raised when raw data cannot be loaded."""
    with patch(
        "scripts.generate_drift_reference.load_raw",
        side_effect=FileNotFoundError("No raw CSV"),
    ):
        with pytest.raises(FileNotFoundError, match="Failed to load raw dataset"):
            generate_reference_profile(raw_df=None)


def test_save_reference_profile(tmp_path: Path) -> None:
    """Verify reference profile is correctly written to disk."""
    dummy_profile = {
        "metadata": {"test": True},
        "numerical_features": {},
        "categorical_features": {},
        "prediction_probability": {},
    }
    out_file = tmp_path / "test_profile.json"
    saved = save_reference_profile(dummy_profile, output_path=out_file)

    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8")) == dummy_profile
