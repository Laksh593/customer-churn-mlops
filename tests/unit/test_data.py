"""
tests/unit/test_data.py
────────────────────────
Unit tests for the Milestone 2 data pipeline modules.

All tests use synthetic, minimal DataFrames – they do NOT depend on the
actual Telco CSV being present on disk.  Tests that explicitly cover file
I/O use tmp_path fixtures or mocks.

Test classes
────────────
TestLoader             – file-not-found behaviour, basic load contract
TestValidation         – each individual validation check
TestCleaning           – TotalCharges conversion, Churn encoding, audit trail
TestFeatureEngineering – each engineered feature
TestSplitting          – reproducibility, stratification, ratio correctness
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.cleaning import clean_raw, encode_target, fix_total_charges
from src.data.constants import (
    REQUIRED_COLUMNS,
    TARGET_COL,
)
from src.data.features import (
    ENGINEERED_FEATURE_NAMES,
    add_estimated_annual_charges,
    add_has_online_security,
    add_has_tech_support,
    add_is_month_to_month,
    add_is_new_customer,
    add_tenure_group,
    add_total_services,
    engineer_features,
)
from src.data.splitting import DataSplit, split_data
from src.data.validation import (
    ValidationReport,
    check_categorical_values,
    check_customer_id_uniqueness,
    check_duplicate_rows,
    check_missing_values,
    check_not_empty,
    check_numerical_ranges,
    check_numerical_types,
    check_required_columns,
    check_target_column,
    validate_raw,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_minimal_df(n: int = 20, churn_as_int: bool = False) -> pd.DataFrame:
    """Create a minimal well-formed DataFrame for testing."""
    import numpy as np

    rng = np.random.default_rng(0)
    size = n
    half = size // 2

    churn_col = (
        ([0] * half + [1] * (size - half))
        if churn_as_int
        else (["No"] * half + ["Yes"] * (size - half))
    )

    return pd.DataFrame(
        {
            "customerID": [f"cust-{i:04d}" for i in range(size)],
            "gender": rng.choice(["Male", "Female"], size),
            "SeniorCitizen": rng.integers(0, 2, size),
            "Partner": rng.choice(["Yes", "No"], size),
            "Dependents": rng.choice(["Yes", "No"], size),
            "tenure": rng.integers(0, 73, size),
            "PhoneService": rng.choice(["Yes", "No"], size),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], size),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], size),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], size),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], size),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], size),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], size),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size),
            "PaperlessBilling": rng.choice(["Yes", "No"], size),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                size,
            ),
            "MonthlyCharges": rng.uniform(20.0, 120.0, size),
            "TotalCharges": [str(v) for v in rng.uniform(0.0, 8000.0, size)],
            "Churn": churn_col,
        }
    )


@pytest.fixture()
def minimal_df() -> pd.DataFrame:
    return _make_minimal_df(n=100)


@pytest.fixture()
def cleaned_df(minimal_df: pd.DataFrame) -> pd.DataFrame:
    cleaned, _ = clean_raw(minimal_df)
    return cleaned


# ---------------------------------------------------------------------------
# TestLoader
# ---------------------------------------------------------------------------


class TestLoader:
    """Tests for src.data.loader."""

    def test_file_not_found_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        from src.data.loader import load_raw

        missing = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError, match="data/raw"):
            load_raw(missing)

    def test_load_returns_dataframe(self, tmp_path: pytest.TempPathFactory) -> None:
        from src.data.loader import load_raw

        # Write a minimal valid CSV to tmp_path
        df = _make_minimal_df(n=10)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_raw(csv_path)
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == 10

    def test_load_preserves_all_columns(self, tmp_path: pytest.TempPathFactory) -> None:
        from src.data.loader import load_raw

        df = _make_minimal_df(n=5)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_raw(csv_path)
        for col in REQUIRED_COLUMNS:
            assert col in loaded.columns, f"Column '{col}' missing after load"

    def test_load_does_not_modify_raw(self, tmp_path: pytest.TempPathFactory) -> None:
        """Loading twice should return identical DataFrames."""
        from src.data.loader import load_raw

        df = _make_minimal_df(n=10)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        df1 = load_raw(csv_path)
        df2 = load_raw(csv_path)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for src.data.validation."""

    def test_check_not_empty_passes_on_non_empty(
        self, minimal_df: pd.DataFrame
    ) -> None:
        result = check_not_empty(minimal_df)
        assert result.passed

    def test_check_not_empty_fails_on_empty(self) -> None:
        result = check_not_empty(pd.DataFrame())
        assert not result.passed
        assert "empty" in result.message.lower()

    def test_check_required_columns_passes(self, minimal_df: pd.DataFrame) -> None:
        result = check_required_columns(minimal_df)
        assert result.passed

    def test_check_required_columns_fails_on_missing(
        self, minimal_df: pd.DataFrame
    ) -> None:
        df = minimal_df.drop(columns=["tenure", "Contract"])
        result = check_required_columns(df)
        assert not result.passed
        assert "tenure" in result.details["missing"]
        assert "Contract" in result.details["missing"]

    def test_check_customer_id_uniqueness_passes(
        self, minimal_df: pd.DataFrame
    ) -> None:
        result = check_customer_id_uniqueness(minimal_df)
        assert result.passed

    def test_check_customer_id_uniqueness_fails_on_duplicates(
        self, minimal_df: pd.DataFrame
    ) -> None:
        df_with_dupe = pd.concat([minimal_df, minimal_df.iloc[:1]], ignore_index=True)
        result = check_customer_id_uniqueness(df_with_dupe)
        assert not result.passed

    def test_check_duplicate_rows_passes(self, minimal_df: pd.DataFrame) -> None:
        result = check_duplicate_rows(minimal_df)
        assert result.passed

    def test_check_duplicate_rows_fails(self, minimal_df: pd.DataFrame) -> None:
        df_dupe = pd.concat([minimal_df, minimal_df.iloc[:2]], ignore_index=True)
        result = check_duplicate_rows(df_dupe)
        assert not result.passed
        assert result.details["duplicate_count"] >= 2

    def test_check_target_column_passes(self, minimal_df: pd.DataFrame) -> None:
        result = check_target_column(minimal_df)
        assert result.passed

    def test_check_target_column_fails_on_unexpected_value(
        self, minimal_df: pd.DataFrame
    ) -> None:
        df = minimal_df.copy()
        df.loc[0, TARGET_COL] = "Maybe"
        result = check_target_column(df)
        assert not result.passed
        assert "Maybe" in str(result.details["unexpected_values"])

    def test_check_target_column_fails_on_null(self, minimal_df: pd.DataFrame) -> None:
        df = minimal_df.copy()
        df.loc[0, TARGET_COL] = None
        result = check_target_column(df)
        assert not result.passed

    def test_check_missing_values_passes(self, minimal_df: pd.DataFrame) -> None:
        result = check_missing_values(minimal_df)
        assert result.passed

    def test_check_missing_values_detects_nulls(self, minimal_df: pd.DataFrame) -> None:
        df = minimal_df.copy()
        df.loc[0, "tenure"] = None
        result = check_missing_values(df)
        assert not result.passed
        assert "tenure" in result.details["missing_counts"]

    def test_check_numerical_types_detects_whitespace_total_charges(
        self, minimal_df: pd.DataFrame
    ) -> None:
        df = minimal_df.copy()
        df.loc[0, "TotalCharges"] = " "  # whitespace-only string
        result = check_numerical_types(df)
        assert not result.passed
        assert "TotalCharges" in result.details["issues"]

    def test_check_numerical_ranges_passes(self, minimal_df: pd.DataFrame) -> None:
        result = check_numerical_ranges(minimal_df)
        assert result.passed

    def test_check_categorical_values_passes(self, minimal_df: pd.DataFrame) -> None:
        result = check_categorical_values(minimal_df)
        assert result.passed

    def test_check_categorical_values_fails_on_unknown(
        self, minimal_df: pd.DataFrame
    ) -> None:
        df = minimal_df.copy()
        df.loc[0, "gender"] = "Other"
        result = check_categorical_values(df)
        assert not result.passed

    def test_validate_raw_returns_report(self, minimal_df: pd.DataFrame) -> None:
        report = validate_raw(minimal_df)
        assert isinstance(report, ValidationReport)
        assert len(report.results) > 0

    def test_validate_raw_passes_on_clean_data(self, minimal_df: pd.DataFrame) -> None:
        report = validate_raw(minimal_df)
        assert report.passed, report.summary()

    def test_report_raise_on_failure_raises_on_bad_data(self) -> None:
        report = validate_raw(pd.DataFrame())
        with pytest.raises(ValueError, match="validation failed"):
            report.raise_on_failure()


# ---------------------------------------------------------------------------
# TestCleaning
# ---------------------------------------------------------------------------


class TestCleaning:
    """Tests for src.data.cleaning."""

    def test_total_charges_converted_to_float(self, minimal_df: pd.DataFrame) -> None:
        cleaned, _ = clean_raw(minimal_df)
        assert cleaned["TotalCharges"].dtype == float

    def test_whitespace_total_charges_imputed(self, minimal_df: pd.DataFrame) -> None:
        df = minimal_df.copy()
        # Set first row to whitespace and give it a tenure > 0
        df.loc[0, "TotalCharges"] = " "
        df.loc[0, "tenure"] = 5
        df.loc[0, "MonthlyCharges"] = 50.0
        cleaned, audit = clean_raw(df)
        # Should be imputed: 5 * 50 = 250.0
        assert cleaned.loc[0, "TotalCharges"] == pytest.approx(250.0)
        assert audit.loc[0, "total_charges_imputed"]

    def test_zero_tenure_total_charges_set_to_zero(
        self, minimal_df: pd.DataFrame
    ) -> None:
        df = minimal_df.copy()
        df.loc[0, "TotalCharges"] = " "
        df.loc[0, "tenure"] = 0
        cleaned, _ = clean_raw(df)
        assert cleaned.loc[0, "TotalCharges"] == pytest.approx(0.0)

    def test_churn_encoded_as_int(self, minimal_df: pd.DataFrame) -> None:
        cleaned, _ = clean_raw(minimal_df)
        assert cleaned[TARGET_COL].dtype in (int, "int64", "Int64")
        assert set(cleaned[TARGET_COL].unique()) <= {0, 1}

    def test_churn_no_maps_to_zero(self, minimal_df: pd.DataFrame) -> None:
        df = minimal_df.copy()
        df[TARGET_COL] = "No"
        cleaned, _ = clean_raw(df)
        assert (cleaned[TARGET_COL] == 0).all()

    def test_churn_yes_maps_to_one(self, minimal_df: pd.DataFrame) -> None:
        df = minimal_df.copy()
        df[TARGET_COL] = "Yes"
        cleaned, _ = clean_raw(df)
        assert (cleaned[TARGET_COL] == 1).all()

    def test_encode_target_raises_on_unexpected_value(
        self, minimal_df: pd.DataFrame
    ) -> None:
        df = minimal_df.copy()
        df.loc[0, TARGET_COL] = "Unknown"
        with pytest.raises(ValueError, match="Unexpected values"):
            encode_target(df)

    def test_raw_not_modified(self, minimal_df: pd.DataFrame) -> None:
        original_churn = minimal_df[TARGET_COL].copy()
        clean_raw(minimal_df)
        pd.testing.assert_series_equal(minimal_df[TARGET_COL], original_churn)

    def test_no_rows_dropped(self, minimal_df: pd.DataFrame) -> None:
        cleaned, _ = clean_raw(minimal_df)
        assert len(cleaned) == len(minimal_df)

    def test_audit_has_same_length(self, minimal_df: pd.DataFrame) -> None:
        cleaned, audit = clean_raw(minimal_df)
        assert len(audit) == len(cleaned)

    def test_fix_total_charges_marks_affected_rows(
        self, minimal_df: pd.DataFrame
    ) -> None:
        df = minimal_df.copy()
        df.loc[2, "TotalCharges"] = "  "
        _, affected = fix_total_charges(df)
        assert affected.iloc[2]
        assert not affected.iloc[0]


# ---------------------------------------------------------------------------
# TestFeatureEngineering
# ---------------------------------------------------------------------------


class TestFeatureEngineering:
    """Tests for src.data.features."""

    def test_tenure_group_added(self, cleaned_df: pd.DataFrame) -> None:
        df = add_tenure_group(cleaned_df)
        assert "tenure_group" in df.columns

    def test_tenure_group_values_are_known_labels(
        self, cleaned_df: pd.DataFrame
    ) -> None:
        df = add_tenure_group(cleaned_df)
        valid = {"New", "Developing", "Established", "Loyal", "Champion"}
        assert set(df["tenure_group"].unique()).issubset(valid)

    def test_tenure_group_new_for_short_tenure(self) -> None:
        df = pd.DataFrame({"tenure": [0, 6, 12]})
        result = add_tenure_group(df)
        assert (result["tenure_group"] == "New").all()

    def test_total_services_non_negative(self, cleaned_df: pd.DataFrame) -> None:
        df = add_total_services(cleaned_df)
        assert (df["total_services"] >= 0).all()

    def test_total_services_bounded(self, cleaned_df: pd.DataFrame) -> None:
        df = add_total_services(cleaned_df)
        assert (df["total_services"] <= 8).all()

    def test_total_services_phone_service_and_multiple_lines(self) -> None:
        """PhoneService=Yes contributes 1 regardless of MultipleLines (Yes or No)."""
        df = pd.DataFrame(
            {
                "PhoneService": ["Yes", "Yes", "No"],
                "MultipleLines": ["No", "Yes", "No phone service"],
                "InternetService": ["No", "No", "No"],
                "OnlineSecurity": [
                    "No internet service",
                    "No internet service",
                    "No internet service",
                ],
                "OnlineBackup": [
                    "No internet service",
                    "No internet service",
                    "No internet service",
                ],
                "DeviceProtection": [
                    "No internet service",
                    "No internet service",
                    "No internet service",
                ],
                "TechSupport": [
                    "No internet service",
                    "No internet service",
                    "No internet service",
                ],
                "StreamingTV": [
                    "No internet service",
                    "No internet service",
                    "No internet service",
                ],
                "StreamingMovies": [
                    "No internet service",
                    "No internet service",
                    "No internet service",
                ],
            }
        )
        res = add_total_services(df)
        # PhoneService=Yes and MultipleLines=No contributes exactly 1
        assert res.loc[0, "total_services"] == 1
        # PhoneService=Yes and MultipleLines=Yes contributes exactly 1
        assert res.loc[1, "total_services"] == 1
        # PhoneService=No contributes 0
        assert res.loc[2, "total_services"] == 0

    def test_total_services_internet_service_contribution(self) -> None:
        """InternetService contributes 0 if 'No', and 1 if 'DSL' or 'Fiber optic'."""
        df = pd.DataFrame(
            {
                "PhoneService": ["No", "No", "No"],
                "MultipleLines": [
                    "No phone service",
                    "No phone service",
                    "No phone service",
                ],
                "InternetService": ["No", "DSL", "Fiber optic"],
                "OnlineSecurity": ["No internet service", "No", "No"],
                "OnlineBackup": ["No internet service", "No", "No"],
                "DeviceProtection": ["No internet service", "No", "No"],
                "TechSupport": ["No internet service", "No", "No"],
                "StreamingTV": ["No internet service", "No", "No"],
                "StreamingMovies": ["No internet service", "No", "No"],
            }
        )
        res = add_total_services(df)
        # InternetService=No contributes 0
        assert res.loc[0, "total_services"] == 0
        # InternetService=DSL contributes 1
        assert res.loc[1, "total_services"] == 1
        # InternetService=Fiber optic contributes 1
        assert res.loc[2, "total_services"] == 1

    def test_total_services_all_eight_services_counted(self) -> None:
        """Verify the calculation correctly counts all eight services."""
        # Case 1: All 8 subscribed
        all_subscribed = pd.DataFrame(
            {
                "PhoneService": ["Yes"],
                "MultipleLines": ["Yes"],  # Not counted separately
                "InternetService": ["Fiber optic"],  # Service 2
                "OnlineSecurity": ["Yes"],  # Service 3
                "OnlineBackup": ["Yes"],  # Service 4
                "DeviceProtection": ["Yes"],  # Service 5
                "TechSupport": ["Yes"],  # Service 6
                "StreamingTV": ["Yes"],  # Service 7
                "StreamingMovies": ["Yes"],  # Service 8
            }
        )
        res_all = add_total_services(all_subscribed)
        assert res_all.loc[0, "total_services"] == 8

        # Case 2: None subscribed
        none_subscribed = pd.DataFrame(
            {
                "PhoneService": ["No"],
                "MultipleLines": ["No phone service"],
                "InternetService": ["No"],
                "OnlineSecurity": ["No internet service"],
                "OnlineBackup": ["No internet service"],
                "DeviceProtection": ["No internet service"],
                "TechSupport": ["No internet service"],
                "StreamingTV": ["No internet service"],
                "StreamingMovies": ["No internet service"],
            }
        )
        res_none = add_total_services(none_subscribed)
        assert res_none.loc[0, "total_services"] == 0

        # Case 3: Each individual service tested separately
        service_cols = [
            ("PhoneService", "Yes"),
            ("InternetService", "DSL"),
            ("OnlineSecurity", "Yes"),
            ("OnlineBackup", "Yes"),
            ("DeviceProtection", "Yes"),
            ("TechSupport", "Yes"),
            ("StreamingTV", "Yes"),
            ("StreamingMovies", "Yes"),
        ]
        for col, val in service_cols:
            single_row = none_subscribed.copy()
            single_row[col] = val
            res = add_total_services(single_row)
            assert res.loc[0, "total_services"] == 1, f"Failed for service {col}={val}"

    def test_is_new_customer_binary(self, cleaned_df: pd.DataFrame) -> None:
        df = add_is_new_customer(cleaned_df)
        assert set(df["is_new_customer"].unique()).issubset({0, 1})

    def test_is_new_customer_correct_threshold(self) -> None:
        df = pd.DataFrame({"tenure": [0, 12, 13, 24]})
        result = add_is_new_customer(df)
        assert result.loc[0, "is_new_customer"] == 1
        assert result.loc[1, "is_new_customer"] == 1
        assert result.loc[2, "is_new_customer"] == 0

    def test_is_month_to_month_binary(self, cleaned_df: pd.DataFrame) -> None:
        df = add_is_month_to_month(cleaned_df)
        assert set(df["is_month_to_month"].unique()).issubset({0, 1})

    def test_is_month_to_month_correct(self) -> None:
        df = pd.DataFrame({"Contract": ["Month-to-month", "One year", "Two year"]})
        result = add_is_month_to_month(df)
        assert result.loc[0, "is_month_to_month"] == 1
        assert result.loc[1, "is_month_to_month"] == 0
        assert result.loc[2, "is_month_to_month"] == 0

    def test_has_online_security_binary(self, cleaned_df: pd.DataFrame) -> None:
        df = add_has_online_security(cleaned_df)
        assert set(df["has_online_security"].unique()).issubset({0, 1})

    def test_has_tech_support_binary(self, cleaned_df: pd.DataFrame) -> None:
        df = add_has_tech_support(cleaned_df)
        assert set(df["has_tech_support"].unique()).issubset({0, 1})

    def test_estimated_annual_charges_equals_monthly_times_12(self) -> None:
        df = pd.DataFrame({"MonthlyCharges": [100.0, 50.0, 75.5]})
        result = add_estimated_annual_charges(df)
        expected = [1200.0, 600.0, 906.0]
        for i, val in enumerate(expected):
            assert result.loc[i, "estimated_annual_charges"] == pytest.approx(val)

    def test_engineer_features_adds_all_features(
        self, cleaned_df: pd.DataFrame
    ) -> None:
        result = engineer_features(cleaned_df)
        for feat in ENGINEERED_FEATURE_NAMES:
            assert feat in result.columns, f"Feature '{feat}' missing"

    def test_engineer_features_does_not_drop_original_columns(
        self, cleaned_df: pd.DataFrame
    ) -> None:
        original_cols = set(cleaned_df.columns)
        result = engineer_features(cleaned_df)
        assert original_cols.issubset(set(result.columns))

    def test_engineer_features_does_not_modify_input(
        self, cleaned_df: pd.DataFrame
    ) -> None:
        original_len = len(cleaned_df.columns)
        engineer_features(cleaned_df)
        assert len(cleaned_df.columns) == original_len

    def test_engineer_features_is_deterministic(self, cleaned_df: pd.DataFrame) -> None:
        result1 = engineer_features(cleaned_df)
        result2 = engineer_features(cleaned_df)
        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# TestSplitting
# ---------------------------------------------------------------------------


class TestSplitting:
    """Tests for src.data.splitting."""

    @pytest.fixture()
    def clean_100(self) -> pd.DataFrame:
        """A cleaned 200-row DataFrame suitable for splitting."""
        return _make_minimal_df(n=200, churn_as_int=True)

    def test_split_returns_datasplit(self, clean_100: pd.DataFrame) -> None:
        result = split_data(clean_100)
        assert isinstance(result, DataSplit)

    def test_split_no_row_loss(self, clean_100: pd.DataFrame) -> None:
        result = split_data(clean_100)
        total = len(result.train) + len(result.val) + len(result.test)
        assert total == len(clean_100)

    def test_split_ratios_approximately_correct(self, clean_100: pd.DataFrame) -> None:
        result = split_data(clean_100)
        n = len(clean_100)
        assert abs(len(result.train) / n - 0.70) < 0.05
        assert abs(len(result.val) / n - 0.15) < 0.05
        assert abs(len(result.test) / n - 0.15) < 0.05

    def test_split_is_reproducible(self, clean_100: pd.DataFrame) -> None:
        s1 = split_data(clean_100, random_state=42)
        s2 = split_data(clean_100, random_state=42)
        pd.testing.assert_frame_equal(
            s1.train.reset_index(drop=True), s2.train.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            s1.test.reset_index(drop=True), s2.test.reset_index(drop=True)
        )

    def test_different_seeds_give_different_splits(
        self, clean_100: pd.DataFrame
    ) -> None:
        s1 = split_data(clean_100, random_state=0)
        s2 = split_data(clean_100, random_state=99)
        # Highly unlikely to be identical with different seeds
        assert not s1.train.index.equals(s2.train.index)

    def test_stratification_preserves_churn_rate(self, clean_100: pd.DataFrame) -> None:
        overall_rate = clean_100[TARGET_COL].mean()
        result = split_data(clean_100)
        for name, part in [
            ("train", result.train),
            ("val", result.val),
            ("test", result.test),
        ]:
            rate = part[TARGET_COL].mean()
            assert (
                abs(rate - overall_rate) < 0.10
            ), f"{name} churn rate {rate:.3f} deviates from overall {overall_rate:.3f}"

    def test_split_raises_on_bad_ratios(self, clean_100: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_data(clean_100, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_split_raises_on_missing_stratify_column(
        self, clean_100: pd.DataFrame
    ) -> None:
        df = clean_100.drop(columns=[TARGET_COL])
        with pytest.raises(ValueError, match="not found"):
            split_data(df)

    def test_partitions_are_disjoint(self, clean_100: pd.DataFrame) -> None:
        result = split_data(clean_100)
        train_idx = set(result.train.index)
        val_idx = set(result.val.index)
        test_idx = set(result.test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)
