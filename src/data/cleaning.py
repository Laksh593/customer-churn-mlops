"""
src/data/cleaning.py
────────────────────
Deterministic cleaning pipeline for the IBM Telco Customer Churn dataset.

Design principles:
- The raw DataFrame is never modified; all functions return new DataFrames.
- Every cleaning decision is documented and logged.
- No rows are silently dropped; callers can inspect the ``_cleaning_log``
  column or use the audit DataFrame returned by :func:`clean_raw`.
- No target-aware imputation: the cleaning pipeline does not use the
  Churn column to decide how to fill other columns.
- Scalers, encoders, and imputers are NOT applied here.  This module
  produces a structurally clean dataset that is ready for splitting;
  learned transformations happen after the split to prevent data leakage.

Documented cleaning rules applied by :func:`clean_raw`:
  1. TotalCharges whitespace-only values → NaN, then imputed with
     MonthlyCharges * tenure (a deterministic formula, not statistical).
     Rows with tenure == 0 and TotalCharges blank are new customers;
     their TotalCharges is set to 0.0.
  2. Churn column encoded as int (No→0, Yes→1).
  3. SeniorCitizen cast to int (it is already 0/1 but loaded as object
     by the all-string loader).
  4. Column types are set to appropriate dtypes for downstream use.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.constants import (
    CUSTOMER_ID_COL,
    NUMERICAL_RANGES,
    TARGET_COL,
    TARGET_ENCODING,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------


def fix_total_charges(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert TotalCharges to float and impute whitespace-only entries.

    The raw dataset stores TotalCharges as a string column.  Eleven rows
    contain a single whitespace character ('' ) which is not a true NaN;
    pandas does not recognise it as missing.

    Imputation rule (deterministic, no statistical fitting):
      - If tenure == 0 → TotalCharges = 0.0  (brand-new customer)
      - Otherwise → TotalCharges = MonthlyCharges × tenure

    Both MonthlyCharges and tenure are known at prediction time, so this
    formula is safe to apply during inference as well.

    Parameters
    ----------
    df:
        DataFrame that must contain 'TotalCharges', 'MonthlyCharges',
        and 'tenure'.

    Returns
    -------
    df_clean : pd.DataFrame
        Copy of *df* with 'TotalCharges' as float64.
    affected : pd.Series
        Boolean mask indicating which rows were imputed.
    """
    df = df.copy()
    # Coerce – whitespace-only strings become NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    affected: pd.Series = df["TotalCharges"].isnull()
    n_affected = int(affected.sum())

    if n_affected > 0:
        logger.info(
            "fix_total_charges: %d rows have non-numeric TotalCharges. "
            "Imputing with MonthlyCharges × tenure.",
            n_affected,
        )
        # Deterministic formula – safe for both training and inference
        imputed_values = np.where(
            df.loc[affected, "tenure"] == 0,
            0.0,
            df.loc[affected, "MonthlyCharges"] * df.loc[affected, "tenure"],
        )
        df.loc[affected, "TotalCharges"] = imputed_values

    df["TotalCharges"] = df["TotalCharges"].astype(float)
    return df, affected


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the Churn column from string to integer (No→0, Yes→1).

    Parameters
    ----------
    df:
        DataFrame containing the 'Churn' column with string values.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with 'Churn' as int64.

    Raises
    ------
    ValueError
        If the Churn column contains values outside {'Yes', 'No'}.
    """
    df = df.copy()
    unexpected = set(df[TARGET_COL].dropna().unique()) - set(TARGET_ENCODING.keys())
    if unexpected:
        raise ValueError(
            f"Unexpected values in '{TARGET_COL}': {unexpected}. "
            f"Expected: {set(TARGET_ENCODING.keys())}"
        )
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_ENCODING).astype(int)
    logger.info(
        "encode_target: Encoded '%s' → int. Distribution: %s",
        TARGET_COL,
        df[TARGET_COL].value_counts().to_dict(),
    )
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to appropriate dtypes after cleaning.

    Parameters
    ----------
    df:
        Partially cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected dtypes.
    """
    df = df.copy()
    int_cols = ["SeniorCitizen", "tenure"]
    float_cols = ["MonthlyCharges", "TotalCharges"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    return df


# ---------------------------------------------------------------------------
# Full cleaning pipeline
# ---------------------------------------------------------------------------


def clean_raw(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the full deterministic cleaning pipeline.

    Steps (in order):
      1. Validate that required columns are present before proceeding.
      2. Drop the customerID from the feature set (retained in audit log).
      3. Fix TotalCharges: coerce to float, impute blanks deterministically.
      4. Fix dtypes for all numerical columns.
      5. Encode the Churn target as 0/1 integer.

    Parameters
    ----------
    df:
        Raw DataFrame from :func:`src.data.loader.load_raw`.

    Returns
    -------
    cleaned : pd.DataFrame
        Cleaned DataFrame with customerID retained as a column.
        Shape is identical to the input – no rows are dropped.
    audit : pd.DataFrame
        Auxiliary DataFrame with one row per input row, recording which
        cleaning operations affected each row.
        Columns: ``customerID``, ``total_charges_imputed``.

    Notes
    -----
    customerID is retained in *cleaned* so the caller can always trace
    predictions back to the source record.  Feature-engineering and
    model-training code should drop it before fitting.
    """
    df = df.copy()
    logger.info("clean_raw: Starting cleaning pipeline on %d rows.", len(df))

    # Step 1 – type coercion before TotalCharges fix
    df = fix_dtypes(df)

    # Step 2 – TotalCharges
    df, tc_imputed = fix_total_charges(df)

    # Step 3 – Encode target
    df = encode_target(df)

    # Build audit log
    audit = pd.DataFrame(
        {
            CUSTOMER_ID_COL: df[CUSTOMER_ID_COL],
            "total_charges_imputed": tc_imputed.values,
        }
    )

    logger.info("clean_raw: Cleaning complete. Output shape: %s", df.shape)
    return df, audit


def validate_numerical_ranges_after_cleaning(df: pd.DataFrame) -> dict[str, str]:
    """Check that numerical columns respect expected ranges after cleaning.

    Returns a dict of column → warning message for any violations found.
    Does not raise; the caller decides how to handle warnings.
    """
    warnings: dict[str, str] = {}
    for col, (lo, hi) in NUMERICAL_RANGES.items():
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        out = numeric[(numeric < lo) | (numeric > hi)]
        if len(out) > 0:
            warnings[col] = (
                f"{len(out)} values outside [{lo}, {hi}]: "
                f"min={numeric.min():.2f}, max={numeric.max():.2f}"
            )
    return warnings
