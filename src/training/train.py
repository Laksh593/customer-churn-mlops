"""
src/training/train.py
─────────────────────
End-to-end training and benchmarking pipeline for Milestone 3.

Executes:
1. Load raw data.
2. Validate raw data.
3. Clean raw data.
4. Engineer features.
5. Split data into train (70%), validation (15%), test (15%).
6. Separate features X and target y, strictly excluding customerID and Churn from X.
7. Build candidate pipelines (Logistic Regression, Random Forest, XGBoost).
8. Train each pipeline on training data only.
9. Evaluate each model on validation data.
10. Generate comparison table.
11. Select best model based on validation ROC-AUC.
12. Evaluate the selected model ONCE on the untouched test set.
13. Print a concise final benchmarking report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.cleaning import clean_raw
from src.data.constants import CUSTOMER_ID_COL, RANDOM_STATE, TARGET_COL
from src.data.features import engineer_features
from src.data.loader import load_raw
from src.data.splitting import split_data
from src.data.validation import validate_raw
from src.training.evaluation import EvaluationResult, compare_models, evaluate_model
from src.training.models import create_model_pipelines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "training.yaml"
)


def load_training_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load training configuration YAML file with fallback to defaults.

    Parameters
    ----------
    config_path:
        Optional path to YAML config. Defaults to ``configs/training.yaml``.

    Returns
    -------
    dict[str, Any]
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                return cfg
    logger.warning("Config file %s not found. Using defaults.", path)
    return {
        "random_state": RANDOM_STATE,
        "validation_metric": "roc_auc",
        "models": {},
    }


def prepare_data(
    random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Run data pipeline and return train, validation, test feature/target splits.

    Guarantees:
    - customerID is excluded from feature matrices.
    - Churn is excluded from feature matrices.
    - Preprocessing is NOT fitted here.

    Parameters
    ----------
    random_state:
        Seed for stratified train/val/test splitting.

    Returns
    -------
    tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("Step 1: Loading raw dataset...")
    raw_df = load_raw()

    logger.info("Step 2: Validating raw dataset...")
    raw_validation_report = validate_raw(raw_df)
    logger.info(
        "Raw validation check passed=%s (total checks=%d)",
        raw_validation_report.passed,
        len(raw_validation_report.results),
    )

    if not raw_validation_report.passed:
        for failure in raw_validation_report.failures:
            logger.warning(
                "Raw validation failure - [%s]: %s",
                failure.name,
                failure.message,
            )

    # Treat known raw TotalCharges whitespace issue as expected pre-cleaning data-quality issue
    expected_raw_failures = {"numerical_types"}
    actual_raw_failures = {f.name for f in raw_validation_report.failures}
    unexpected_raw_failures = actual_raw_failures - expected_raw_failures
    if unexpected_raw_failures:
        logger.error("Unexpected raw validation failures: %s", unexpected_raw_failures)
        raw_validation_report.raise_on_failure()

    logger.info(
        "Known raw TotalCharges whitespace issue in %s is an expected "
        "pre-cleaning data-quality issue; continuing to cleaning.",
        list(actual_raw_failures),
    )

    logger.info("Step 3: Cleaning raw dataset...")
    cleaned_df, _ = clean_raw(raw_df)

    logger.info("Step 3b: Validating cleaned dataset...")
    cleaned_validation_report = validate_raw(cleaned_df)
    cleaned_validation_report.raise_on_failure()
    logger.info("Cleaned dataset validation passed all checks.")

    logger.info("Step 4: Engineering features...")
    featured_df = engineer_features(cleaned_df)

    logger.info("Step 5: Splitting data (70%% train / 15%% val / 15%% test)...")
    split = split_data(featured_df, random_state=random_state)

    logger.info("Step 6: Separating features (X) and target (y)...")
    cols_to_drop = [CUSTOMER_ID_COL, TARGET_COL]

    X_train = split.train.drop(columns=cols_to_drop)
    y_train = split.train[TARGET_COL]

    X_val = split.val.drop(columns=cols_to_drop)
    y_val = split.val[TARGET_COL]

    X_test = split.test.drop(columns=cols_to_drop)
    y_test = split.test[TARGET_COL]

    assert CUSTOMER_ID_COL not in X_train.columns
    assert TARGET_COL not in X_train.columns

    logger.info(
        "Data ready: X_train=%s, X_val=%s, X_test=%s (features=%d)",
        X_train.shape,
        X_val.shape,
        X_test.shape,
        X_train.shape[1],
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_and_benchmark(
    config_path: Path | str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, str, EvaluationResult]:
    """Execute model training, validation benchmarking, and final test evaluation.

    Parameters
    ----------
    config_path:
        Optional path to training configuration YAML.

    Returns
    -------
    tuple of:
        - trained_pipelines: dict[str, Pipeline]
        - val_comparison_df: pd.DataFrame
        - best_model_name: str
        - test_result: EvaluationResult
    """
    config = load_training_config(config_path)
    random_state = int(config.get("random_state", RANDOM_STATE))
    model_configs = config.get("models", {})
    metric_to_select = str(config.get("validation_metric", "roc_auc"))

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        random_state=random_state
    )

    logger.info("Step 7: Building candidate model pipelines...")
    pipelines = create_model_pipelines(random_state=random_state, config=model_configs)

    logger.info("Step 8: Training candidate models on X_train only...")
    val_results: dict[str, EvaluationResult] = {}
    for name, pipe in pipelines.items():
        logger.info("Training pipeline '%s'...", name)
        pipe.fit(X_train, y_train)
        logger.info("Evaluating '%s' on validation set...", name)
        val_results[name] = evaluate_model(pipe, X_val, y_val)
        logger.info("  %s -> %s", name, val_results[name].summary())

    logger.info("Step 9: Benchmarking validation performance...")
    val_df = compare_models(val_results)

    logger.info(
        "Step 10: Selecting best model based on validation %s...", metric_to_select
    )
    best_model_name = str(val_df[metric_to_select].idxmax())
    best_val_score = float(val_df.loc[best_model_name, metric_to_select])
    logger.info(
        "Winner: '%s' with validation %s = %.4f",
        best_model_name,
        metric_to_select,
        best_val_score,
    )

    logger.info(
        "Step 11: Evaluating winner '%s' ONCE on untouched test set...", best_model_name
    )
    best_pipeline = pipelines[best_model_name]
    test_result = evaluate_model(best_pipeline, X_test, y_test)
    logger.info("Final Test Performance: %s", test_result.summary())

    return pipelines, val_df, best_model_name, test_result


def main() -> None:
    """CLI entrypoint for python -m src.training.train."""
    print("=" * 70)
    print("Milestone 3 — Model Training & Benchmarking Pipeline")
    print("=" * 70)

    _, val_df, best_model_name, test_result = train_and_benchmark()

    print("\n" + "=" * 70)
    print("VALIDATION BENCHMARK RESULTS")
    print("=" * 70)
    formatted_val = val_df.copy()
    for col in formatted_val.columns:
        formatted_val[col] = formatted_val[col].map(lambda v: f"{v:.4f}")
    print(formatted_val.to_string())

    print("\n" + "=" * 70)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(
        f"Criterion: Validation ROC-AUC = {val_df.loc[best_model_name, 'roc_auc']:.4f}"
    )
    print("=" * 70)

    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION (Evaluated ONCE on untouched test set)")
    print("=" * 70)
    test_dict = test_result.to_dict()
    for metric, score in test_dict.items():
        print(f"  {metric:<15}: {score:.4f}")

    cm = test_result.confusion_matrix
    print("\nConfusion Matrix (Test Set):")
    print(f"  True Negatives  (TN): {cm['tn']}")
    print(f"  False Positives (FP): {cm['fp']}")
    print(f"  False Negatives (FN): {cm['fn']}")
    print(f"  True Positives  (TP): {cm['tp']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
