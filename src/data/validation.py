"""
src/data/validation.py
──────────────────────
Reusable data-validation checks for the IBM Telco Customer Churn dataset.

Design principles:
- Validation *reports* problems; it never silently modifies or drops data.
- Every check returns a :class:`ValidationResult` so callers can decide how
  to handle failures (log, raise, continue).
- The :func:`validate_raw` convenience function runs the full suite and
  returns a :class:`ValidationReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.constants import (
    CATEGORICAL_VALID_VALUES,
    CUSTOMER_ID_COL,
    NUMERICAL_RANGES,
    REQUIRED_COLUMNS,
    TARGET_COL,
    TARGET_ENCODING,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of a single validation check."""

    name: str
    passed: bool
    message: str
    details: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


@dataclass
class ValidationReport:
    """Aggregated results of all validation checks."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return True only if every individual check passed."""
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[ValidationResult]:
        """Return only the checks that did not pass."""
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        """Return a human-readable summary of all results."""
        lines = [f"Validation Report – {'PASSED' if self.passed else 'FAILED'}"]
        lines.append("=" * 60)
        for r in self.results:
            lines.append(str(r))
        if not self.passed:
            lines.append("")
            lines.append(f"  {len(self.failures)} check(s) failed.")
        return "\n".join(lines)

    def raise_on_failure(self) -> None:
        """Raise :class:`ValueError` if any check failed.

        Raises
        ------
        ValueError
            With a full report of all failures.
        """
        if not self.passed:
            failure_messages = "\n".join(str(f) for f in self.failures)
            raise ValueError(
                f"Data validation failed with {len(self.failures)} error(s):\n"
                f"{failure_messages}"
            )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_required_columns(df: pd.DataFrame) -> ValidationResult:
    """Verify that all required columns are present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    unexpected = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    passed = len(missing) == 0
    if passed and not unexpected:
        msg = f"All {len(REQUIRED_COLUMNS)} required columns are present."
    elif passed:
        msg = (
            f"All required columns present. "
            f"Unexpected extra columns found: {unexpected}"
        )
    else:
        msg = f"Missing columns: {missing}"
    return ValidationResult(
        name="required_columns",
        passed=passed,
        message=msg,
        details={"missing": missing, "unexpected": unexpected},
    )


def check_not_empty(df: pd.DataFrame) -> ValidationResult:
    """Verify that the dataset contains at least one row."""
    passed = len(df) > 0
    return ValidationResult(
        name="not_empty",
        passed=passed,
        message=f"Dataset has {len(df):,} rows." if passed else "Dataset is empty.",
        details={"row_count": len(df)},
    )


def check_customer_id_uniqueness(df: pd.DataFrame) -> ValidationResult:
    """Verify that customerID is unique across all rows."""
    if CUSTOMER_ID_COL not in df.columns:
        return ValidationResult(
            name="customer_id_uniqueness",
            passed=False,
            message=f"Column '{CUSTOMER_ID_COL}' not found – cannot check uniqueness.",
        )
    n_total = len(df)
    n_unique = df[CUSTOMER_ID_COL].nunique()
    duplicated_ids = df[df[CUSTOMER_ID_COL].duplicated(keep=False)][
        CUSTOMER_ID_COL
    ].unique()
    passed = n_total == n_unique
    msg = (
        f"All {n_unique:,} customerIDs are unique."
        if passed
        else f"{n_total - n_unique:,} duplicate customerIDs found."
    )
    return ValidationResult(
        name="customer_id_uniqueness",
        passed=passed,
        message=msg,
        details={
            "total": n_total,
            "unique": n_unique,
            "duplicated": list(duplicated_ids),
        },
    )


def check_duplicate_rows(df: pd.DataFrame) -> ValidationResult:
    """Verify there are no fully duplicate rows."""
    n_dupes = int(df.duplicated().sum())
    passed = n_dupes == 0
    return ValidationResult(
        name="duplicate_rows",
        passed=passed,
        message=(
            "No duplicate rows found."
            if passed
            else f"{n_dupes:,} duplicate rows found."
        ),
        details={"duplicate_count": n_dupes},
    )


def check_target_column(df: pd.DataFrame) -> ValidationResult:
    """Verify that the target column exists and contains only expected values."""
    if TARGET_COL not in df.columns:
        return ValidationResult(
            name="target_column",
            passed=False,
            message=f"Target column '{TARGET_COL}' is missing.",
        )
    valid_values = set(TARGET_ENCODING.keys()) | set(TARGET_ENCODING.values())
    actual_values = set(df[TARGET_COL].dropna().unique())
    unexpected = actual_values - valid_values
    n_null = int(df[TARGET_COL].isnull().sum())
    passed = len(unexpected) == 0 and n_null == 0
    if passed:
        counts = df[TARGET_COL].value_counts().to_dict()
        msg = f"Target '{TARGET_COL}' is valid. Value counts: {counts}"
    else:
        parts = []
        if unexpected:
            parts.append(f"unexpected values {unexpected}")
        if n_null:
            parts.append(f"{n_null} null value(s)")
        msg = f"Target column problems: {'; '.join(parts)}"
    return ValidationResult(
        name="target_column",
        passed=passed,
        message=msg,
        details={
            "unexpected_values": list(unexpected),
            "null_count": n_null,
            "value_counts": (
                df[TARGET_COL].value_counts().to_dict() if TARGET_COL in df else {}
            ),
        },
    )


def check_missing_values(df: pd.DataFrame) -> ValidationResult:
    """Report missing values for every column.

    Note: TotalCharges whitespace-only entries show as non-null in pandas
    (they are strings). Those are caught by :func:`check_numerical_types`.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    passed = len(missing) == 0
    if passed:
        msg = "No missing (NaN) values found."
    else:
        details_str = missing.to_dict()
        msg = f"Missing values found in {len(missing)} column(s): {details_str}"
    return ValidationResult(
        name="missing_values",
        passed=passed,
        message=msg,
        details={"missing_counts": missing.to_dict()},
    )


def check_numerical_types(df: pd.DataFrame) -> ValidationResult:
    """Check that numerical columns can be parsed as numbers.

    TotalCharges is stored as string in the raw dataset and may contain
    whitespace-only entries that cannot be converted to float.
    """
    issues: dict[str, int] = {}
    for col in ["MonthlyCharges", "TotalCharges", "tenure", "SeniorCitizen"]:
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_invalid = int(coerced.isnull().sum() - df[col].isnull().sum())
        if n_invalid > 0:
            issues[col] = n_invalid
    passed = len(issues) == 0
    if passed:
        msg = "All numerical columns contain parseable values."
    else:
        msg = f"Non-numeric values found: {issues}"
    return ValidationResult(
        name="numerical_types",
        passed=passed,
        message=msg,
        details={"issues": issues},
    )


def check_numerical_ranges(df: pd.DataFrame) -> ValidationResult:
    """Check that numerical values fall within expected ranges."""
    issues: dict[str, dict[str, object]] = {}
    for col, (lo, hi) in NUMERICAL_RANGES.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        out_of_range = numeric[(numeric < lo) | (numeric > hi)]
        if len(out_of_range) > 0:
            issues[col] = {
                "count": len(out_of_range),
                "min_observed": float(numeric.min()),
                "max_observed": float(numeric.max()),
                "expected_range": (lo, hi),
            }
    passed = len(issues) == 0
    msg = (
        "All numerical values are within expected ranges."
        if passed
        else f"Out-of-range values found in {len(issues)} column(s): {list(issues.keys())}"
    )
    return ValidationResult(
        name="numerical_ranges",
        passed=passed,
        message=msg,
        details={"issues": issues},
    )


def check_categorical_values(df: pd.DataFrame) -> ValidationResult:
    """Check that categorical columns contain only known valid values."""
    issues: dict[str, dict[str, object]] = {}
    for col, valid in CATEGORICAL_VALID_VALUES.items():
        if col not in df.columns:
            continue
        actual = set(df[col].dropna().unique())
        # If target column was encoded to integers (0, 1), check against encoded values
        if col == TARGET_COL and actual.issubset(set(TARGET_ENCODING.values())):
            continue
        unexpected = actual - valid
        if unexpected:
            issues[col] = {"unexpected": list(unexpected)}
    passed = len(issues) == 0
    msg = (
        "All categorical columns contain only valid values."
        if passed
        else f"Unexpected values found in {len(issues)} column(s): {list(issues.keys())}"
    )
    return ValidationResult(
        name="categorical_values",
        passed=passed,
        message=msg,
        details={"issues": issues},
    )


# ---------------------------------------------------------------------------
# Full validation suite
# ---------------------------------------------------------------------------


def validate_raw(df: pd.DataFrame) -> ValidationReport:
    """Run the full validation suite against the raw DataFrame.

    Parameters
    ----------
    df:
        Raw DataFrame loaded by :func:`src.data.loader.load_raw`.

    Returns
    -------
    ValidationReport
        Contains individual :class:`ValidationResult` objects for every
        check that was executed.
    """
    checks = [
        check_not_empty,
        check_required_columns,
        check_customer_id_uniqueness,
        check_duplicate_rows,
        check_target_column,
        check_missing_values,
        check_numerical_types,
        check_numerical_ranges,
        check_categorical_values,
    ]
    report = ValidationReport()
    for check in checks:
        report.results.append(check(df))
    return report
