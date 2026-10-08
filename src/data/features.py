"""
src/data/features.py
────────────────────
Reusable feature-engineering module for the Telco Customer Churn dataset.

Design principles:
- Every engineered feature is documented with its definition and ML rationale.
- All transformations are deterministic: given the same input, the output is
  always identical.
- No target variable (Churn) is used in any transformation.
- No future information is used; every feature can be computed at inference
  time from the same columns available in the raw dataset.
- No statistical fitting occurs here (e.g., no mean/std, no encoders that
  require training data). This module is safe to call before or after the
  train/test split.

Engineered features
-------------------
tenure_group : str
    Categorical bucket of customer tenure in months.
    Bins: New (0-12 months), Developing (13-24), Established (25-48),
    Loyal (49-72), Champion (>72).
    Rationale: churn risk typically differs across life-cycle stages;
    binning makes non-linear tenure effects easier for linear models to
    capture.

total_services : int
    Count of actual subscribed paid services (0 to 8).
    Services counted:
    1. PhoneService == "Yes"
    2. InternetService != "No" (DSL or Fiber optic)
    3. OnlineSecurity == "Yes"
    4. OnlineBackup == "Yes"
    5. DeviceProtection == "Yes"
    6. TechSupport == "Yes"
    7. StreamingTV == "Yes"
    8. StreamingMovies == "Yes"
    Note: MultipleLines is not counted as a separate service.
    Rationale: customers with more services are generally more embedded and
    may show different churn patterns.

is_new_customer : int  (0 or 1)
    1 if tenure <= 12, 0 otherwise.
    Rationale: new customers churning early is a distinct phenomenon.

is_month_to_month : int  (0 or 1)
    1 if Contract == 'Month-to-month', 0 otherwise.
    Rationale: month-to-month contract is a strong churn predictor.

has_online_security : int  (0 or 1)
    1 if OnlineSecurity == 'Yes', 0 otherwise.
    Rationale: security add-on purchase may indicate engagement.

has_tech_support : int  (0 or 1)
    1 if TechSupport == 'Yes', 0 otherwise.
    Rationale: tech-support subscribers may have different satisfaction levels.

estimated_annual_charges : float
    MonthlyCharges × 12.
    Rationale: annual cost framing is useful for understanding price
    sensitivity, especially compared to TotalCharges which is
    tenure-dependent.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Tenure bin boundaries in months (right-exclusive except the last)
_TENURE_BINS: list[int] = [0, 12, 24, 48, 72, 10_000]
_TENURE_LABELS: list[str] = ["New", "Developing", "Established", "Loyal", "Champion"]


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``tenure_group`` categorical column.

    Parameters
    ----------
    df:
        Must contain ``tenure`` as a numeric column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``tenure_group`` added.
    """
    df = df.copy()
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=_TENURE_BINS,
        labels=_TENURE_LABELS,
        right=True,
        include_lowest=True,
    ).astype(str)
    return df


def add_total_services(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``total_services`` integer count of actual subscribed services (0 to 8).

    Counts the following 8 subscribed services:
    1. PhoneService == "Yes"
    2. InternetService != "No"
    3. OnlineSecurity == "Yes"
    4. OnlineBackup == "Yes"
    5. DeviceProtection == "Yes"
    6. TechSupport == "Yes"
    7. StreamingTV == "Yes"
    8. StreamingMovies == "Yes"

    MultipleLines is not counted as a separate service.

    Parameters
    ----------
    df:
        Must contain the relevant service columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``total_services`` added.
    """
    df = df.copy()

    service_checks: list[tuple[str, str, bool]] = [
        # (column, value_to_match, is_positive_match)
        ("PhoneService", "Yes", True),
        ("InternetService", "No", False),
        ("OnlineSecurity", "Yes", True),
        ("OnlineBackup", "Yes", True),
        ("DeviceProtection", "Yes", True),
        ("TechSupport", "Yes", True),
        ("StreamingTV", "Yes", True),
        ("StreamingMovies", "Yes", True),
    ]

    total = pd.Series(0, index=df.index, dtype=int)
    for col, match_val, is_positive in service_checks:
        if col not in df.columns:
            continue
        if is_positive:
            total += (df[col] == match_val).astype(int)
        else:
            # is_positive=False means "anything except match_val counts as having the service"
            total += (df[col] != match_val).astype(int)

    df["total_services"] = total
    return df


def add_is_new_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_new_customer`` flag (1 if tenure <= 12 months).

    Parameters
    ----------
    df:
        Must contain ``tenure`` as a numeric column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``is_new_customer`` (int) added.
    """
    df = df.copy()
    df["is_new_customer"] = (pd.to_numeric(df["tenure"], errors="coerce") <= 12).astype(
        int
    )
    return df


def add_is_month_to_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``is_month_to_month`` flag (1 if Contract == 'Month-to-month').

    Parameters
    ----------
    df:
        Must contain ``Contract``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``is_month_to_month`` (int) added.
    """
    df = df.copy()
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def add_has_online_security(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``has_online_security`` flag (1 if OnlineSecurity == 'Yes').

    Parameters
    ----------
    df:
        Must contain ``OnlineSecurity``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``has_online_security`` (int) added.
    """
    df = df.copy()
    df["has_online_security"] = (df["OnlineSecurity"] == "Yes").astype(int)
    return df


def add_has_tech_support(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``has_tech_support`` flag (1 if TechSupport == 'Yes').

    Parameters
    ----------
    df:
        Must contain ``TechSupport``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``has_tech_support`` (int) added.
    """
    df = df.copy()
    df["has_tech_support"] = (df["TechSupport"] == "Yes").astype(int)
    return df


def add_estimated_annual_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``estimated_annual_charges`` = MonthlyCharges × 12.

    Parameters
    ----------
    df:
        Must contain ``MonthlyCharges`` as a numeric column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``estimated_annual_charges`` (float) added.
    """
    df = df.copy()
    df["estimated_annual_charges"] = (
        pd.to_numeric(df["MonthlyCharges"], errors="coerce") * 12.0
    )
    return df


# ---------------------------------------------------------------------------
# Full feature-engineering pipeline
# ---------------------------------------------------------------------------

#: Names of all columns added by this module.
ENGINEERED_FEATURE_NAMES: list[str] = [
    "tenure_group",
    "total_services",
    "is_new_customer",
    "is_month_to_month",
    "has_online_security",
    "has_tech_support",
    "estimated_annual_charges",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature-engineering pipeline.

    Each step adds one or more columns to the DataFrame. The original
    columns are preserved; nothing is removed.

    Parameters
    ----------
    df:
        Cleaned DataFrame from :func:`src.data.cleaning.clean_raw`.
        Must be the cleaned (not raw) version so that TotalCharges is
        numeric and Churn is encoded.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with all engineered features appended.
    """
    steps = [
        add_tenure_group,
        add_total_services,
        add_is_new_customer,
        add_is_month_to_month,
        add_has_online_security,
        add_has_tech_support,
        add_estimated_annual_charges,
    ]
    for step in steps:
        df = step(df)
        logger.debug("engineer_features: Applied %s", step.__name__)

    logger.info(
        "engineer_features: Added %d features. Total columns: %d",
        len(ENGINEERED_FEATURE_NAMES),
        len(df.columns),
    )
    return df
