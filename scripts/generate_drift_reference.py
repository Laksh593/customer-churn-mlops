"""scripts/generate_drift_reference.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate reproducible, compact baseline reference profile for data drift
and model monitoring (Milestone 11 - Step 1).

Design Principles:
1. Feature baseline distributions: Derived exclusively from the training split
   (``X_train``, 70%), using the project's fixed random state (42).
2. Prediction probability baseline: Derived from out-of-sample predictions on
   the validation split (``X_val``, 15%) to avoid in-sample overconfidence bias.
3. Test set holdout: ``X_test`` (15%) is strictly excluded and never accessed.
4. Data minimization: Stores only compact quantile bin boundaries, empirical
   proportions, categorical frequencies, and metadata. No customer IDs,
   individual sample rows, or raw customer records are retained.
5. Robustness: Gracefully handles duplicate quantile boundaries, constant features,
   missing values, and computes actual empirical bin proportions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path when invoked directly as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.ml.loader import load_registered_model  # noqa: E402
from src.data.cleaning import clean_raw  # noqa: E402
from src.data.constants import (  # noqa: E402
    CUSTOMER_ID_COL,
    RANDOM_STATE,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data.features import engineer_features  # noqa: E402
from src.data.loader import load_raw  # noqa: E402
from src.data.splitting import split_data  # noqa: E402
from src.training.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH: Path = PROJECT_ROOT / "configs" / "drift_reference_profile.json"


def compute_numerical_reference(
    series: pd.Series,
    num_quantiles: int = 10,
) -> dict[str, Any]:
    """Compute compact reference distribution statistics for a numerical feature.

    Parameters
    ----------
    series:
        Pandas Series representing a single numerical feature from the training set.
    num_quantiles:
        Target number of quantile intervals (default: 10 for deciles).

    Returns
    -------
    dict[str, Any]
        Dictionary containing sample count, missing count, mean, std, min, max,
        deduplicated bin edges, and empirical bin proportions.
    """
    valid_series = series.dropna()
    count = int(len(valid_series))
    missing_count = int(series.isna().sum())

    if count == 0:
        raise ValueError(
            f"Numerical feature '{series.name}' contains no valid non-null values."
        )

    mean_val = float(valid_series.mean())
    std_val = float(valid_series.std(ddof=1)) if count > 1 else 0.0
    min_val = float(valid_series.min())
    max_val = float(valid_series.max())

    if min_val == max_val:
        # Constant feature: create a single bounding interval around the constant value
        bin_edges = [float(min_val - 0.5), float(max_val + 0.5)]
        proportions = [1.0]
    else:
        # Compute quantile boundaries
        quantiles = np.quantile(
            valid_series,
            np.linspace(0.0, 1.0, num_quantiles + 1),
        )
        deduped_edges = np.unique(quantiles)

        # If deduplication collapsed edges to fewer than 2 distinct boundaries
        if len(deduped_edges) < 2:
            deduped_edges = np.array([deduped_edges[0] - 0.5, deduped_edges[0] + 0.5])
        elif len(deduped_edges) == 2 and valid_series.nunique() == 2:
            # Low-cardinality binary split: divide at midpoint to separate the two values
            midpoint = (deduped_edges[0] + deduped_edges[1]) / 2.0
            deduped_edges = np.array([deduped_edges[0], midpoint, deduped_edges[1]])

        # Cut into bins with include_lowest=True so min value is enclosed
        cut_bins = pd.cut(valid_series, bins=deduped_edges, include_lowest=True)
        counts = pd.Series(cut_bins).value_counts(sort=False).values
        bin_edges = [float(round(b, 6)) for b in deduped_edges]
        proportions = [float(round(c / count, 8)) for c in counts]

    return {
        "count": count,
        "missing_count": missing_count,
        "mean": float(round(mean_val, 6)),
        "std": float(round(std_val, 6)),
        "min": float(round(min_val, 6)),
        "max": float(round(max_val, 6)),
        "bin_edges": bin_edges,
        "proportions": proportions,
    }


def compute_categorical_reference(series: pd.Series) -> dict[str, Any]:
    """Compute compact reference frequency distribution for a categorical feature.

    Parameters
    ----------
    series:
        Pandas Series representing a single categorical feature from the training set.

    Returns
    -------
    dict[str, Any]
        Dictionary containing sample count, missing count, observed categories,
        and normalized frequency proportions.
    """
    valid_series = series.dropna()
    count = int(len(valid_series))
    missing_count = int(series.isna().sum())

    if count == 0:
        raise ValueError(
            f"Categorical feature '{series.name}' contains no valid non-null values."
        )

    val_counts = valid_series.value_counts(sort=True)
    categories = [str(cat) for cat in val_counts.index.tolist()]
    frequencies = {
        str(cat): float(round(cnt / count, 8)) for cat, cnt in val_counts.items()
    }

    return {
        "count": count,
        "missing_count": missing_count,
        "categories": categories,
        "frequencies": frequencies,
    }


def compute_prediction_reference(
    probabilities: np.ndarray | pd.Series,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute reference prediction-probability distribution on out-of-sample validation data.

    Parameters
    ----------
    probabilities:
        Array of predicted probabilities for positive class (Churn=1) on X_val.
    num_bins:
        Number of equal-width probability bins between 0.0 and 1.0 (default: 10).

    Returns
    -------
    dict[str, Any]
        Dictionary containing count, mean, std, median, min, max, bin boundaries,
        and observed empirical proportions across the probability spectrum.
    """
    arr = np.asarray(probabilities, dtype=float)
    valid_arr = arr[~np.isnan(arr)]
    count = int(len(valid_arr))

    if count == 0:
        raise ValueError("No valid prediction probabilities provided for reference.")

    mean_val = float(np.mean(valid_arr))
    std_val = float(np.std(valid_arr, ddof=1)) if count > 1 else 0.0
    median_val = float(np.median(valid_arr))
    min_val = float(np.min(valid_arr))
    max_val = float(np.max(valid_arr))

    # Standard fixed deciles for probabilities from 0.0 to 1.0
    bin_edges = [float(round(b, 4)) for b in np.linspace(0.0, 1.0, num_bins + 1)]

    # Compute actual empirical proportions in each decile
    cut_bins = pd.cut(valid_arr, bins=bin_edges, include_lowest=True)
    counts = pd.Series(cut_bins).value_counts(sort=False).values
    proportions = [float(round(c / count, 8)) for c in counts]

    return {
        "evaluation_split": "validation",
        "count": count,
        "mean": float(round(mean_val, 6)),
        "std": float(round(std_val, 6)),
        "median": float(round(median_val, 6)),
        "min": float(round(min_val, 6)),
        "max": float(round(max_val, 6)),
        "bin_edges": bin_edges,
        "proportions": proportions,
    }


def generate_reference_profile(
    raw_df: pd.DataFrame | None = None,
    model: Any | None = None,
    model_name: str | None = None,
    model_version: str | int | None = None,
    tracking_uri: str | None = None,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Generate the complete leakage-free drift reference profile.

    Parameters
    ----------
    raw_df:
        Optional raw DataFrame. If None, loaded via :func:`src.data.loader.load_raw`.
    model:
        Optional fitted Scikit-learn Pipeline. If None, loaded via :func:`app.ml.loader.load_registered_model`.
    model_name:
        Optional model name string.
    model_version:
        Optional model version identifier.
    tracking_uri:
        Optional MLflow tracking URI.
    random_state:
        Seed for reproducible splitting (default: 42).

    Returns
    -------
    dict[str, Any]
        Structured reference profile dictionary.
    """
    logger.info("Generating drift reference profile (random_state=%d)...", random_state)

    # 1. Load raw data if not provided
    if raw_df is None:
        try:
            raw_df = load_raw()
        except Exception as exc:
            raise FileNotFoundError(f"Failed to load raw dataset: {exc}") from exc

    # 2. Clean data using deterministic project pipeline
    cleaned_df, _ = clean_raw(raw_df)

    # 3. Engineer features using deterministic project pipeline
    featured_df = engineer_features(cleaned_df)

    # 4. Perform stratified split using project ratios (70% train / 15% val / 15% test)
    split = split_data(featured_df, random_state=random_state)

    # 5. Extract feature partitions, strictly dropping identifiers and targets
    cols_to_drop = [CUSTOMER_ID_COL, TARGET_COL]
    X_train = split.train.drop(columns=cols_to_drop)
    X_val = split.val.drop(columns=cols_to_drop)

    # Strictly guarantee that identifiers and targets are never retained in feature sets
    assert CUSTOMER_ID_COL not in X_train.columns
    assert TARGET_COL not in X_train.columns
    assert CUSTOMER_ID_COL not in X_val.columns
    assert TARGET_COL not in X_val.columns

    # 6. Load winning model if not provided
    if model is None:
        try:
            model = load_registered_model(
                model_name=model_name,
                model_version=model_version,
                tracking_uri=tracking_uri,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load model artifact: {exc}") from exc

    # 7. Compute out-of-sample predictions on X_val only (never X_train, never X_test)
    try:
        val_probabilities = model.predict_proba(X_val)[:, 1]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compute validation prediction probabilities: {exc}"
        ) from exc

    # 8. Compute numerical feature reference profiles (X_train only)
    numerical_features_profile: dict[str, Any] = {}
    for col in NUMERICAL_FEATURES:
        if col not in X_train.columns:
            raise ValueError(
                f"Required numerical feature '{col}' missing from X_train."
            )
        numerical_features_profile[col] = compute_numerical_reference(X_train[col])

    # 9. Compute categorical feature reference profiles (X_train only)
    categorical_features_profile: dict[str, Any] = {}
    for col in CATEGORICAL_FEATURES:
        if col not in X_train.columns:
            raise ValueError(
                f"Required categorical feature '{col}' missing from X_train."
            )
        categorical_features_profile[col] = compute_categorical_reference(X_train[col])

    # 10. Compute prediction probability baseline (X_val only)
    prediction_probability_profile = compute_prediction_reference(val_probabilities)

    # Extract model metadata
    actual_model_name = model_name or getattr(model, "model_name", "churn-predictor")
    actual_model_version = str(model_version or getattr(model, "model_version", "1"))
    model_type = model.__class__.__name__

    # 11. Assemble final reference profile
    profile: dict[str, Any] = {
        "metadata": {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": "Telco-Customer-Churn.csv",
            "random_state": random_state,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "num_train_samples": int(len(X_train)),
            "num_val_samples": int(len(X_val)),
            "features_split": "train",
            "prediction_probabilities_split": "validation",
            "model_name": actual_model_name,
            "model_version": actual_model_version,
            "model_type": model_type,
        },
        "numerical_features": numerical_features_profile,
        "categorical_features": categorical_features_profile,
        "prediction_probability": prediction_probability_profile,
    }

    # Verify no customer IDs or identifiers leak into profile
    assert CUSTOMER_ID_COL not in profile["numerical_features"]
    assert CUSTOMER_ID_COL not in profile["categorical_features"]
    assert TARGET_COL not in profile["numerical_features"]
    assert TARGET_COL not in profile["categorical_features"]

    return profile


def save_reference_profile(
    profile: dict[str, Any],
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Save the reference profile to a JSON file.

    Parameters
    ----------
    profile:
        Dictionary containing the reference profile.
    output_path:
        Target file path for the JSON artifact.

    Returns
    -------
    Path
        Resolved path to the saved JSON file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    logger.info("Saved drift reference profile to: %s", path)
    return path


def main() -> None:
    """CLI entrypoint for drift reference profile generation."""
    parser = argparse.ArgumentParser(
        description="Generate leakage-free drift reference profile for customer churn monitoring."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Target path for reference profile JSON.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="MLflow registered model name.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="MLflow registered model version.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducible dataset split.",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    profile = generate_reference_profile(
        model_name=args.model_name,
        model_version=args.model_version,
        tracking_uri=args.tracking_uri,
        random_state=args.random_state,
    )
    saved_path = save_reference_profile(profile, output_path=args.output)
    print(f"Drift reference profile successfully generated at: {saved_path}")


if __name__ == "__main__":
    main()
