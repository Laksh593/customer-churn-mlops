"""
src/training/train.py
─────────────────────
End-to-end training, benchmarking, and MLflow tracking pipeline (Milestone 4).

Executes:
1. Load raw data.
2. Validate raw data (log warnings for known pre-cleaning TotalCharges issue).
3. Clean raw data.
4. Validate cleaned data with raise_on_failure().
5. Engineer features.
6. Split data into train (70%), validation (15%), test (15%).
7. Separate features X and target y, strictly excluding customerID and Churn from X.
8. Build candidate pipelines (Logistic Regression, Random Forest, XGBoost).
9. Train each pipeline on training data only.
10. Evaluate each candidate on validation data.
11. Log candidate runs to MLflow with parameters, validation metrics, and fitted pipeline.
12. Generate comparison table and select winner based on validation ROC-AUC.
13. Evaluate the winner ONCE on untouched test set and log test metrics to its MLflow run.
14. Register the winning pipeline into the MLflow Model Registry under the configured name.
15. Print a concise final benchmarking report.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from src.data.cleaning import clean_raw
from src.data.constants import CUSTOMER_ID_COL, RANDOM_STATE, TARGET_COL
from src.data.features import engineer_features
from src.data.loader import load_raw
from src.data.splitting import split_data
from src.data.validation import validate_raw
from src.training.evaluation import EvaluationResult, compare_models, evaluate_model
from src.training.models import create_model_pipelines

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "training.yaml"
)

DEFAULT_MLFLOW_CONFIG: dict[str, str] = {
    "experiment_name": "customer-churn-benchmarking",
    "tracking_uri": "sqlite:///mlflow.db",
    "artifact_location": "mlartifacts",
    "registered_model_name": "churn-predictor",
}


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
    config: dict[str, Any] = {
        "random_state": RANDOM_STATE,
        "validation_metric": "roc_auc",
        "mlflow": dict(DEFAULT_MLFLOW_CONFIG),
        "models": {},
    }
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                config.update(cfg)
                # Ensure mlflow section has all default keys if partial
                mlflow_sec = dict(DEFAULT_MLFLOW_CONFIG)
                mlflow_sec.update(cfg.get("mlflow", {}))
                config["mlflow"] = mlflow_sec
    else:
        logger.warning("Config file %s not found. Using defaults.", path)

    # Allow environment variable overrides for MLflow settings
    env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_tracking_uri:
        config["mlflow"]["tracking_uri"] = env_tracking_uri
    env_artifact_location = os.environ.get("MLFLOW_ARTIFACT_LOCATION")
    if env_artifact_location:
        config["mlflow"]["artifact_location"] = env_artifact_location
    env_experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if env_experiment_name:
        config["mlflow"]["experiment_name"] = env_experiment_name

    return config


def setup_mlflow(mlflow_config: dict[str, Any]) -> str:
    """Configure MLflow tracking URI and ensure the experiment exists.

    Parameters
    ----------
    mlflow_config:
        Dictionary containing tracking_uri, experiment_name, artifact_location.

    Returns
    -------
    str
        Experiment ID.
    """
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        str(mlflow_config.get("tracking_uri", "sqlite:///mlflow.db")),
    )
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME",
        str(mlflow_config.get("experiment_name", "customer-churn-benchmarking")),
    )
    artifact_location = os.environ.get(
        "MLFLOW_ARTIFACT_LOCATION",
        mlflow_config.get("artifact_location"),
    )

    if tracking_uri.startswith("sqlite:///"):
        db_path_str = tracking_uri.replace("sqlite:///", "")
        if db_path_str:
            db_path = Path(db_path_str)
            if db_path.parent:
                db_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)

    # Protect root logger from being mutated by MLflow/Alembic migrations,
    # which can attach a StreamHandler(sys.stderr) and cause mutual recursion with Airflow's StreamLogWriter.
    import logging.config

    root = logging.getLogger()
    pre_handlers = list(root.handlers)
    orig_file_config = getattr(logging.config, "fileConfig", None)
    if orig_file_config is not None:
        logging.config.fileConfig = lambda *args, **kwargs: None
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
    finally:
        if orig_file_config is not None:
            logging.config.fileConfig = orig_file_config
        for h in list(root.handlers):
            if h not in pre_handlers:
                root.removeHandler(h)

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        if artifact_location:
            art_path = Path(artifact_location)
            if art_path.is_absolute():
                art_path.mkdir(parents=True, exist_ok=True)
                artifact_uri = art_path.as_uri()
            else:
                artifact_uri = str(artifact_location)
            exp_id = client.create_experiment(
                name=experiment_name,
                artifact_location=artifact_uri,
            )
        else:
            exp_id = client.create_experiment(name=experiment_name)
    else:
        exp_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow initialized: URI='%s', experiment='%s' (id=%s)",
        tracking_uri,
        experiment_name,
        exp_id,
    )
    return exp_id


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
    config_override: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, str, EvaluationResult]:
    """Execute model training, validation benchmarking, and MLflow tracking.

    Parameters
    ----------
    config_path:
        Optional path to training configuration YAML.
    config_override:
        Optional dictionary of overrides for config settings (useful for tests).

    Returns
    -------
    tuple of:
        - trained_pipelines: dict[str, Pipeline]
        - val_comparison_df: pd.DataFrame
        - best_model_name: str
        - test_result: EvaluationResult
    """
    config = load_training_config(config_path)
    if config_override:
        config.update(config_override)
        if "mlflow" in config_override:
            config["mlflow"] = {
                **load_training_config(config_path).get("mlflow", {}),
                **config_override["mlflow"],
            }

    random_state = int(config.get("random_state", RANDOM_STATE))
    model_configs = config.get("models", {})
    metric_to_select = str(config.get("validation_metric", "roc_auc"))
    mlflow_cfg = config.get("mlflow", DEFAULT_MLFLOW_CONFIG)
    registered_model_name = str(
        mlflow_cfg.get("registered_model_name", "churn-predictor")
    )

    # Initialize MLflow experiment tracking
    setup_mlflow(mlflow_cfg)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        random_state=random_state
    )

    logger.info("Step 7: Building candidate model pipelines...")
    pipelines = create_model_pipelines(random_state=random_state, config=model_configs)

    logger.info(
        "Step 8: Training candidate models and logging validation runs to MLflow..."
    )
    val_results: dict[str, EvaluationResult] = {}
    candidate_run_ids: dict[str, str] = {}

    for name, pipe in pipelines.items():
        logger.info("Training pipeline '%s'...", name)
        pipe.fit(X_train, y_train)

        logger.info("Evaluating '%s' on validation set...", name)
        val_res = evaluate_model(pipe, X_val, y_val)
        val_results[name] = val_res
        logger.info("  %s -> %s", name, val_res.summary())

        # Start dedicated MLflow run for candidate model
        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id
            candidate_run_ids[name] = run_id

            # Run tags
            mlflow.set_tags(
                {
                    "project": "customer-churn-mlops",
                    "model_name": name,
                    "stage": "validation",
                }
            )

            # Hyperparameters
            params: dict[str, Any] = {"random_state": random_state}
            params.update(model_configs.get(name, {}))
            mlflow.log_params(params)

            # Validation metrics
            mlflow.log_metrics(
                {
                    "validation_accuracy": val_res.accuracy,
                    "validation_precision": val_res.precision,
                    "validation_recall": val_res.recall,
                    "validation_f1": val_res.f1,
                    "validation_roc_auc": val_res.roc_auc,
                    "validation_pr_auc": val_res.pr_auc,
                }
            )

            # Log fitted Scikit-learn pipeline (preprocessor + model) as artifact
            mlflow.sklearn.log_model(
                sk_model=pipe,
                artifact_path="model",
                input_example=X_train.head(5),
            )
            logger.info("  Logged MLflow run for '%s' (run_id: %s)", name, run_id)

    logger.info("Step 9: Benchmarking validation performance...")
    val_df = compare_models(val_results)

    logger.info(
        "Step 10: Selecting best model based on validation %s...", metric_to_select
    )
    best_model_name = str(val_df[metric_to_select].idxmax())
    best_val_score = float(val_df.loc[best_model_name, metric_to_select])
    winner_run_id = candidate_run_ids[best_model_name]
    best_pipeline = pipelines[best_model_name]

    logger.info(
        "Winner: '%s' with validation %s = %.4f (run_id: %s)",
        best_model_name,
        metric_to_select,
        best_val_score,
        winner_run_id,
    )

    logger.info(
        "Step 11: Evaluating winner '%s' ONCE on untouched test set...", best_model_name
    )
    test_result = evaluate_model(best_pipeline, X_test, y_test)
    logger.info("Final Test Performance: %s", test_result.summary())

    # Log test metrics and selection tag into the winning model's MLflow run
    with mlflow.start_run(run_id=winner_run_id):
        mlflow.set_tags(
            {
                "is_best_model": "true",
                "selection_metric": metric_to_select,
                "selection_metric_value": str(round(best_val_score, 4)),
            }
        )
        mlflow.log_metrics(
            {
                "test_accuracy": test_result.accuracy,
                "test_precision": test_result.precision,
                "test_recall": test_result.recall,
                "test_f1": test_result.f1,
                "test_roc_auc": test_result.roc_auc,
                "test_pr_auc": test_result.pr_auc,
            }
        )

    # Step 12: Register the selected model in the MLflow Model Registry
    logger.info(
        "Step 12: Registering winner '%s' into MLflow Model Registry as '%s'...",
        best_model_name,
        registered_model_name,
    )
    model_uri = f"runs:/{winner_run_id}/model"
    reg_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    max_wait_seconds = 30
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        version_details = client.get_model_version(
            name=registered_model_name,
            version=reg_version.version,
        )
        if version_details.status == "READY":
            break
        if version_details.status == "FAILED_REGISTRATION":
            raise RuntimeError(
                f"Model registration failed for '{registered_model_name}' version {reg_version.version}"
            )
        time.sleep(0.5)
    else:
        raise TimeoutError(
            f"Timed out waiting for model '{registered_model_name}' version {reg_version.version} to become READY"
        )

    client.update_model_version(
        name=registered_model_name,
        version=reg_version.version,
        description=(
            f"Winning candidate '{best_model_name}' selected by validation {metric_to_select}="
            f"{best_val_score:.4f}. Test ROC-AUC={test_result.roc_auc:.4f}. Run ID: {winner_run_id}."
        ),
    )
    logger.info(
        "Model successfully registered: '%s' version %s (status=%s).",
        reg_version.name,
        reg_version.version,
        version_details.status,
    )

    return pipelines, val_df, best_model_name, test_result


def main() -> None:
    """CLI entrypoint for python -m src.training.train."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=" * 70)
    print("Milestone 4 — Model Training, Benchmarking & MLflow Registry")
    print("=" * 70)

    _, val_df, best_model_name, test_result = train_and_benchmark()

    print()
    print("=" * 70)
    print("VALIDATION BENCHMARK RESULTS")
    print("=" * 70)
    formatted_val = val_df.copy()
    for col in formatted_val.columns:
        formatted_val[col] = formatted_val[col].map(lambda v: f"{v:.4f}")
    print(formatted_val.to_string())

    print()
    print("=" * 70)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(
        f"Criterion: Validation ROC-AUC = {val_df.loc[best_model_name, 'roc_auc']:.4f}"
    )
    print("=" * 70)

    print()
    print("=" * 70)
    print("FINAL TEST SET EVALUATION (Evaluated ONCE on untouched test set)")
    print("=" * 70)
    test_dict = test_result.to_dict()
    for metric, score in test_dict.items():
        print(f"  {metric:<15}: {score:.4f}")

    cm = test_result.confusion_matrix
    print()
    print("Confusion Matrix (Test Set):")
    print(f"  True Negatives  (TN): {cm['tn']}")
    print(f"  False Positives (FP): {cm['fp']}")
    print(f"  False Negatives (FN): {cm['fn']}")
    print(f"  True Positives  (TP): {cm['tp']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
