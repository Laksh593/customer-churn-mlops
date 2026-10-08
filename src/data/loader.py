"""
src/data/loader.py
──────────────────
Reusable data-loading utilities for the Telco Customer Churn dataset.

Design principles:
- All paths are resolved relative to the project root using pathlib so the
  module works on any OS without hard-coded absolute paths.
- The raw DataFrame is returned as-is; callers are responsible for any
  mutation so the loader never silently changes data.
- Descriptive errors are raised when the file is missing or unreadable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.constants import REQUIRED_COLUMNS


def _project_root() -> Path:
    """Return the project root directory.

    The project root is the parent of the ``src`` directory that contains
    this file, which makes the resolution portable across machines.
    """
    return Path(__file__).resolve().parent.parent.parent


def raw_data_path() -> Path:
    """Return the canonical path to the raw CSV dataset.

    Returns
    -------
    Path
        Absolute path to ``data/raw/Telco-Customer-Churn.csv``.
    """
    return _project_root() / "data" / "raw" / "Telco-Customer-Churn.csv"


def processed_data_dir() -> Path:
    """Return the directory for processed datasets, creating it if needed.

    Returns
    -------
    Path
        Absolute path to ``data/processed/``.
    """
    path = _project_root() / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV into a DataFrame.

    The returned DataFrame is a faithful representation of the CSV file.
    No columns are added, removed, or modified.

    Parameters
    ----------
    path:
        Path to the CSV file. Defaults to the canonical raw-data path
        returned by :func:`raw_data_path`. Pass an explicit path during
        testing to avoid depending on the filesystem layout.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns and dtypes.

    Raises
    ------
    FileNotFoundError
        When the CSV file does not exist at *path*.
    ValueError
        When the file exists but cannot be parsed as a CSV.
    """
    csv_path = path or raw_data_path()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {csv_path}\n"
            "Download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and place it at data/raw/Telco-Customer-Churn.csv"
        )

    try:
        df = pd.read_csv(
            csv_path, dtype=str
        )  # Read everything as str to preserve raw values
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV at {csv_path}: {exc}") from exc

    # Cast columns that pandas can trivially handle to their natural types
    # while keeping TotalCharges as string (it has whitespace-only entries).
    _NATURAL_INT_COLS = ["SeniorCitizen", "tenure"]
    _NATURAL_FLOAT_COLS = ["MonthlyCharges"]

    for col in _NATURAL_INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in _NATURAL_FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_processed(filename: str) -> pd.DataFrame:
    """Load a processed dataset from ``data/processed/``.

    Parameters
    ----------
    filename:
        Filename (with extension) inside ``data/processed/``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        When the processed file does not exist.
    """
    path = processed_data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}\n"
            "Run the cleaning pipeline first to generate this file."
        )
    return pd.read_csv(path)


def describe_dataframe(df: pd.DataFrame) -> None:
    """Print a concise human-readable summary of a DataFrame.

    Intended for notebook and REPL use, not for production code paths.

    Parameters
    ----------
    df:
        Any pandas DataFrame.
    """
    print(f"Shape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage     : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"Columns          : {list(df.columns)}")
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values   :")
        print(missing[missing > 0].to_string())
    else:
        print("Missing values   : none")
    dupes = df.duplicated().sum()
    print(f"Duplicate rows   : {dupes:,}")


def get_required_columns() -> list[str]:
    """Return the list of columns required in the raw dataset.

    Returns
    -------
    list[str]
        Column names from the IBM Telco Customer Churn schema.
    """
    return list(REQUIRED_COLUMNS)
