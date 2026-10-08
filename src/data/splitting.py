"""
src/data/splitting.py
─────────────────────
Reproducible stratified train / validation / test splitting.

Design principles:
- The split is performed AFTER cleaning but BEFORE any learned
  preprocessing (scalers, encoders, imputers). This is a hard requirement
  to prevent data leakage: if a scaler were fit on the full dataset and
  then used on the training set, the training distribution would be
  contaminated by validation and test statistics.
- Stratification ensures the class balance of Churn is approximately
  maintained in every partition.
- The random seed is sourced from constants so it is never duplicated
  across modules.

Default split ratios (configurable via parameters):
  70% training / 15% validation / 15% test
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.constants import (
    RANDOM_STATE,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSplit:
    """Container for the three dataset partitions.

    Attributes
    ----------
    train:
        Training partition (~70% of data by default).
    val:
        Validation partition (~15% of data by default).
    test:
        Test partition (~15% of data by default).  Must not be used for
        any model selection or hyperparameter tuning.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> str:
        """Return a human-readable summary of partition sizes."""
        total = len(self.train) + len(self.val) + len(self.test)
        lines = ["DataSplit summary:"]
        for name, part in [
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        ]:
            churn_rate = (
                part[TARGET_COL].mean() if TARGET_COL in part.columns else float("nan")
            )
            lines.append(
                f"  {name:<6}: {len(part):>5} rows "
                f"({len(part) / total * 100:.1f}%)  "
                f"churn rate = {churn_rate:.3f}"
            )
        return "\n".join(lines)


def split_data(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
    stratify_col: str = TARGET_COL,
) -> DataSplit:
    """Split *df* into stratified train / validation / test partitions.

    IMPORTANT – Leakage prevention
    --------------------------------
    This function must be called on the cleaned dataset BEFORE fitting any
    preprocessing transformer (e.g., StandardScaler, OrdinalEncoder).
    Learned transformers must be fit exclusively on the training partition
    and then applied to validation and test partitions.

    Parameters
    ----------
    df:
        Cleaned DataFrame with the target column encoded as int.
    train_ratio:
        Fraction of data to allocate to training.
    val_ratio:
        Fraction of data to allocate to validation.
    test_ratio:
        Fraction of data to allocate to testing.
        *train_ratio* + *val_ratio* + *test_ratio* must equal 1.0.
    random_state:
        Seed for reproducibility.
    stratify_col:
        Column to use for stratification.  Must be present in *df*.

    Returns
    -------
    DataSplit
        Frozen dataclass containing train, val, and test DataFrames.

    Raises
    ------
    ValueError
        If the ratios do not sum to 1.0 or the stratify column is missing.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    if stratify_col not in df.columns:
        raise ValueError(
            f"Stratification column '{stratify_col}' not found in DataFrame."
        )

    stratify = df[stratify_col]

    # First split: training vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    df_train, df_val_test = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=stratify,
    )

    # Second split: validation vs test (within the held-out portion)
    relative_test_ratio = test_ratio / val_test_ratio
    df_val, df_test = train_test_split(
        df_val_test,
        test_size=relative_test_ratio,
        random_state=random_state,
        stratify=df_val_test[stratify_col],
    )

    split = DataSplit(train=df_train, val=df_val, test=df_test)
    logger.info("split_data: %s", split.summary())
    return split
