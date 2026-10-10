"""
tests/unit/test_training.py
───────────────────────────
Unit tests for Milestone 3 ML training, preprocessing, models, and evaluation.

Verifies:
1. create_preprocessor() returns a ColumnTransformer.
2. Numerical columns are assigned correctly.
3. Categorical columns are assigned correctly.
4. customerID is excluded.
5. Churn is excluded.
6. All three model definitions exist.
7. Each model can be placed into a Pipeline with the preprocessor.
8. A small synthetic dataset can successfully fit each pipeline.
9. Evaluation returns all required metrics.
10. ROC-AUC and PR-AUC are calculated from probabilities.
11. No NaN/inf is returned for a valid evaluation dataset.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.training.evaluation import (
    EvaluationResult,
    compare_models,
    evaluate_model,
    evaluate_predictions,
)
from src.training.models import (
    create_model_pipelines,
    create_pipeline,
    get_logistic_regression,
    get_models,
    get_random_forest,
    get_xgboost,
)
from src.training.preprocessing import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    NUMERICAL_FEATURES,
    create_preprocessor,
)
from src.training.train import load_training_config


@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create a minimal synthetic dataset containing all 26 feature columns."""
    n_samples = 20
    rng = np.random.RandomState(42)

    data: dict[str, list[object]] = {
        # 10 Numerical features
        "SeniorCitizen": rng.choice([0, 1], size=n_samples).tolist(),
        "tenure": rng.randint(1, 72, size=n_samples).tolist(),
        "MonthlyCharges": rng.uniform(20.0, 110.0, size=n_samples).round(2).tolist(),
        "TotalCharges": rng.uniform(20.0, 8000.0, size=n_samples).round(2).tolist(),
        "total_services": rng.randint(0, 8, size=n_samples).tolist(),
        "is_new_customer": rng.choice([0, 1], size=n_samples).tolist(),
        "is_month_to_month": rng.choice([0, 1], size=n_samples).tolist(),
        "has_online_security": rng.choice([0, 1], size=n_samples).tolist(),
        "has_tech_support": rng.choice([0, 1], size=n_samples).tolist(),
        "estimated_annual_charges": rng.uniform(240.0, 1300.0, size=n_samples)
        .round(2)
        .tolist(),
        # 16 Categorical features
        "gender": rng.choice(["Male", "Female"], size=n_samples).tolist(),
        "Partner": rng.choice(["Yes", "No"], size=n_samples).tolist(),
        "Dependents": rng.choice(["Yes", "No"], size=n_samples).tolist(),
        "PhoneService": rng.choice(["Yes", "No"], size=n_samples).tolist(),
        "MultipleLines": rng.choice(
            ["Yes", "No", "No phone service"], size=n_samples
        ).tolist(),
        "InternetService": rng.choice(
            ["DSL", "Fiber optic", "No"], size=n_samples
        ).tolist(),
        "OnlineSecurity": rng.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ).tolist(),
        "OnlineBackup": rng.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ).tolist(),
        "DeviceProtection": rng.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ).tolist(),
        "TechSupport": rng.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ).tolist(),
        "StreamingTV": rng.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ).tolist(),
        "StreamingMovies": rng.choice(
            ["Yes", "No", "No internet service"], size=n_samples
        ).tolist(),
        "Contract": rng.choice(
            ["Month-to-month", "One year", "Two year"], size=n_samples
        ).tolist(),
        "PaperlessBilling": rng.choice(["Yes", "No"], size=n_samples).tolist(),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check"], size=n_samples
        ).tolist(),
        "tenure_group": rng.choice(
            ["New", "Developing", "Established", "Loyal", "Champion"], size=n_samples
        ).tolist(),
    }

    # Introduce a couple of np.nan to test imputation
    data["TotalCharges"][0] = np.nan
    data["PaymentMethod"][1] = np.nan

    df = pd.DataFrame(data)
    y = pd.Series(rng.choice([0, 1], size=n_samples), name="Churn")
    return df, y


class TestPreprocessing:
    """Tests for src.training.preprocessing."""

    def test_create_preprocessor_returns_column_transformer(self) -> None:
        """1. create_preprocessor() returns a ColumnTransformer."""
        preprocessor = create_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_numerical_columns_assigned_correctly(self) -> None:
        """2. Numerical columns are assigned correctly."""
        expected_num = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "total_services",
            "is_new_customer",
            "is_month_to_month",
            "has_online_security",
            "has_tech_support",
            "estimated_annual_charges",
        ]
        assert NUMERICAL_FEATURES == expected_num

        preprocessor = create_preprocessor()
        transformer_dict = {name: cols for name, _, cols in preprocessor.transformers}
        assert transformer_dict["num"] == expected_num

    def test_categorical_columns_assigned_correctly(self) -> None:
        """3. Categorical columns are assigned correctly."""
        expected_cat = [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "tenure_group",
        ]
        assert CATEGORICAL_FEATURES == expected_cat

        preprocessor = create_preprocessor()
        transformer_dict = {name: cols for name, _, cols in preprocessor.transformers}
        assert transformer_dict["cat"] == expected_cat

    def test_customer_id_excluded(self) -> None:
        """4. customerID is excluded from feature sets."""
        assert "customerID" not in NUMERICAL_FEATURES
        assert "customerID" not in CATEGORICAL_FEATURES
        assert "customerID" in EXCLUDED_FEATURES

        preprocessor = create_preprocessor()
        for _, _, cols in preprocessor.transformers:
            assert "customerID" not in cols

    def test_churn_excluded(self) -> None:
        """5. Churn is excluded from feature sets."""
        assert "Churn" not in NUMERICAL_FEATURES
        assert "Churn" not in CATEGORICAL_FEATURES
        assert "Churn" in EXCLUDED_FEATURES

        preprocessor = create_preprocessor()
        for _, _, cols in preprocessor.transformers:
            assert "Churn" not in cols

    def test_preprocessor_fit_transform_synthetic(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """Preprocessor successfully transforms input with missing values."""
        X, _ = synthetic_data
        preprocessor = create_preprocessor()
        transformed = preprocessor.fit_transform(X)
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == len(X)
        assert np.isfinite(transformed).all()


class TestModels:
    """Tests for src.training.models."""

    def test_all_three_models_exist(self) -> None:
        """6. All three model definitions exist."""
        models = get_models()
        assert "logistic_regression" in models
        assert "random_forest" in models
        assert "xgboost" in models
        assert len(models) == 3

        assert isinstance(models["logistic_regression"], LogisticRegression)
        assert isinstance(models["random_forest"], RandomForestClassifier)
        assert isinstance(models["xgboost"], XGBClassifier)

    def test_individual_factories(self) -> None:
        """Factory functions return appropriately configured models."""
        lr = get_logistic_regression()
        assert lr.max_iter == 1000
        assert lr.class_weight == "balanced"

        rf = get_random_forest()
        assert rf.n_estimators == 300
        assert rf.class_weight == "balanced"
        assert rf.n_jobs == -1

        xgb = get_xgboost()
        assert xgb.n_estimators == 300
        assert xgb.max_depth == 5
        assert xgb.learning_rate == pytest.approx(0.05)

    def test_each_model_placed_into_pipeline_with_preprocessor(self) -> None:
        """7. Each model can be placed into a Pipeline with the preprocessor."""
        pipelines = create_model_pipelines()
        assert len(pipelines) == 3

        for name, pipe in pipelines.items():
            assert isinstance(pipe, Pipeline), f"{name} is not a Pipeline"
            steps = dict(pipe.steps)
            assert "preprocessor" in steps, f"{name} missing preprocessor step"
            assert "model" in steps, f"{name} missing model step"
            assert isinstance(steps["preprocessor"], ColumnTransformer)

    def test_small_synthetic_dataset_fits_each_pipeline(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """8. A small synthetic dataset can successfully fit each pipeline."""
        X, y = synthetic_data
        pipelines = create_model_pipelines()

        for name, pipe in pipelines.items():
            pipe.fit(X, y)
            preds = pipe.predict(X)
            proba = pipe.predict_proba(X)

            assert len(preds) == len(y), f"{name} prediction length mismatch"
            assert proba.shape == (len(y), 2), f"{name} probability shape mismatch"
            assert np.all((preds == 0) | (preds == 1)), f"{name} invalid prediction"


class TestEvaluation:
    """Tests for src.training.evaluation."""

    def test_evaluation_returns_all_required_metrics(self) -> None:
        """9. Evaluation returns all required metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.8, 0.2, 0.4, 0.3, 0.9, 0.7, 0.6])

        res = evaluate_predictions(y_true, y_pred, y_proba)

        assert isinstance(res, EvaluationResult)
        d = res.to_dict(include_cm=True)

        required_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "confusion_matrix",
        }
        assert required_keys.issubset(d.keys())

        cm = d["confusion_matrix"]
        assert {"tn", "fp", "fn", "tp"}.issubset(cm.keys())
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_true)

    def test_roc_auc_and_pr_auc_use_probabilities(self) -> None:
        """10. ROC-AUC and PR-AUC are calculated from probabilities."""
        y_true = np.array([0, 0, 1, 1])
        # Perfect probability ordering -> ROC-AUC and PR-AUC should be 1.0
        y_proba_perfect = np.array([0.1, 0.2, 0.8, 0.9])
        y_pred_imperfect = np.array([0, 1, 0, 1])  # Sub-optimal discrete labels

        res = evaluate_predictions(y_true, y_pred_imperfect, y_proba_perfect)
        assert res.roc_auc == pytest.approx(1.0)
        assert res.pr_auc == pytest.approx(1.0)

        # Inverted probability ordering -> ROC-AUC should be 0.0
        y_proba_inverted = np.array([0.9, 0.8, 0.2, 0.1])
        res_inv = evaluate_predictions(y_true, y_pred_imperfect, y_proba_inverted)
        assert res_inv.roc_auc == pytest.approx(0.0)

    def test_no_nan_or_inf_for_valid_dataset(
        self, synthetic_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        """11. No NaN/inf is returned for a valid evaluation dataset."""
        X, y = synthetic_data
        pipe = create_pipeline(model=get_logistic_regression())
        pipe.fit(X, y)

        res = evaluate_model(pipe, X, y)
        d = res.to_dict()

        for metric_name, val in d.items():
            assert np.isfinite(val), f"Metric '{metric_name}' is not finite: {val}"
            assert 0.0 <= val <= 1.0, f"Metric '{metric_name}' outside [0, 1]: {val}"

    def test_compare_models_dataframe_format(self) -> None:
        """compare_models() formats results into sorted DataFrame."""
        cm = {"tn": 10, "fp": 2, "fn": 3, "tp": 5}
        results = {
            "model_b": EvaluationResult(
                accuracy=0.75,
                precision=0.71,
                recall=0.62,
                f1=0.66,
                roc_auc=0.82,
                pr_auc=0.70,
                confusion_matrix=cm,
            ),
            "model_a": EvaluationResult(
                accuracy=0.80,
                precision=0.75,
                recall=0.70,
                f1=0.72,
                roc_auc=0.88,
                pr_auc=0.78,
                confusion_matrix=cm,
            ),
        }
        df = compare_models(results)
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ["model_a", "model_b"]  # Sorted descending by roc_auc
        assert "roc_auc" in df.columns


class TestTrainingConfig:
    """Tests for configs/training.yaml."""

    def test_load_training_config_reads_yaml(self) -> None:
        """Config loader parses configs/training.yaml including mlflow."""
        cfg = load_training_config()
        assert isinstance(cfg, dict)
        assert "random_state" in cfg
        assert "validation_metric" in cfg
        assert "mlflow" in cfg
        assert "experiment_name" in cfg["mlflow"]
        assert "tracking_uri" in cfg["mlflow"]
        assert "artifact_location" in cfg["mlflow"]
        assert "registered_model_name" in cfg["mlflow"]
        assert "models" in cfg
        assert "logistic_regression" in cfg["models"]
        assert "random_forest" in cfg["models"]
        assert "xgboost" in cfg["models"]


def _make_sample_raw_df(n: int = 100) -> pd.DataFrame:
    """Create a minimal raw DataFrame matching the IBM Telco schema for testing."""
    rng = np.random.default_rng(42)
    half = n // 2
    churn_col = ["No"] * half + ["Yes"] * (n - half)

    df = pd.DataFrame(
        {
            "customerID": [f"cust-{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(0, 73, n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": rng.uniform(20.0, 120.0, n).round(2),
            "TotalCharges": [str(round(v, 2)) for v in rng.uniform(20.0, 8000.0, n)],
            "Churn": churn_col,
        }
    )
    # Simulate known raw data TotalCharges whitespace for zero-tenure rows
    df.loc[0, "TotalCharges"] = " "
    df.loc[0, "tenure"] = 0
    return df


@pytest.fixture()
def sample_raw_df() -> pd.DataFrame:
    """Return a minimal raw DataFrame fixture for testing."""
    return _make_sample_raw_df(n=100)


class TestValidationHandling:
    """Tests for raw and cleaned dataset validation in the training pipeline."""

    def test_cleaned_dataset_validation_passes(
        self, sample_raw_df: pd.DataFrame
    ) -> None:
        """After clean_raw(), validate_raw() passes and raise_on_failure() succeeds."""
        from src.data.cleaning import clean_raw
        from src.data.validation import validate_raw

        cleaned_df, _ = clean_raw(sample_raw_df)
        report = validate_raw(cleaned_df)

        assert report.passed, f"Cleaned data failed validation: {report.summary()}"
        # Must not raise
        report.raise_on_failure()

    def test_unexpected_raw_validation_failure_raises(
        self, monkeypatch: pytest.MonkeyPatch, sample_raw_df: pd.DataFrame
    ) -> None:
        """Unexpected raw validation failures (e.g. missing columns) raise an error."""
        import src.training.train as train_mod

        # Drop a required column to simulate unexpected corruption
        corrupted_df = sample_raw_df.drop(columns=["MonthlyCharges"])

        monkeypatch.setattr(train_mod, "load_raw", lambda: corrupted_df)

        with pytest.raises(ValueError, match="validation failed"):
            train_mod.prepare_data()


class TestMLflowTracking:
    """Tests for Milestone 4 MLflow experiment tracking and model registry."""

    def test_setup_mlflow_creates_experiment(self, tmp_path: Path) -> None:
        """setup_mlflow configures tracking URI and initializes experiment."""
        from src.training.train import setup_mlflow

        db_path = tmp_path / "setup_test.db"
        cfg = {
            "tracking_uri": f"sqlite:///{db_path}",
            "experiment_name": "unit-test-exp",
            "artifact_location": str(tmp_path / "artifacts"),
        }
        exp_id = setup_mlflow(cfg)
        assert exp_id is not None
        assert isinstance(exp_id, str)

    def test_experiment_creation_and_candidate_runs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sample_raw_df: pd.DataFrame,
    ) -> None:
        """Full pipeline tracks 3 candidate runs, logs metrics, and registers only winner."""
        import src.training.train as train_mod
        from src.training.train import train_and_benchmark

        monkeypatch.setattr(train_mod, "load_raw", lambda: sample_raw_df)

        db_path = tmp_path / "test_mlflow.db"
        art_path = tmp_path / "artifacts"
        tracking_uri = f"sqlite:///{db_path}"
        exp_name = "test-churn-benchmarking"
        model_name = "test-churn-predictor"

        config_override = {
            "mlflow": {
                "experiment_name": exp_name,
                "tracking_uri": tracking_uri,
                "artifact_location": str(art_path),
                "registered_model_name": model_name,
            }
        }

        pipelines, val_df, best_model, test_res = train_and_benchmark(
            config_override=config_override
        )

        client = MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(exp_name)
        assert exp is not None, "Experiment was not created"

        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert len(runs) == 3, f"Expected 3 candidate runs, found {len(runs)}"

        run_names = {r.data.tags.get("model_name") for r in runs}
        assert run_names == {"logistic_regression", "random_forest", "xgboost"}

        required_val_metrics = {
            "validation_accuracy",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_roc_auc",
            "validation_pr_auc",
        }
        for run in runs:
            metrics = run.data.metrics
            assert required_val_metrics.issubset(
                metrics.keys()
            ), f"Missing validation metrics in run {run.data.tags.get('model_name')}"
            assert "random_state" in run.data.params

        # Verify only winner run has test metrics and is_best_model tag
        winner_runs = [r for r in runs if r.data.tags.get("model_name") == best_model]
        assert len(winner_runs) == 1
        winner_run = winner_runs[0]
        assert winner_run.data.tags.get("is_best_model") == "true"
        assert "test_roc_auc" in winner_run.data.metrics
        assert "test_pr_auc" in winner_run.data.metrics

        non_winner_runs = [
            r for r in runs if r.data.tags.get("model_name") != best_model
        ]
        for nw in non_winner_runs:
            assert "is_best_model" not in nw.data.tags
            assert "test_roc_auc" not in nw.data.metrics

        # Verify model registry has only the selected model
        reg_models = client.search_registered_models()
        assert len(reg_models) == 1
        assert reg_models[0].name == model_name

        versions = client.search_model_versions(f"name='{model_name}'")
        assert len(versions) == 1
        mv = versions[0]
        assert mv.status == "READY"
        assert mv.run_id == winner_run.info.run_id

        # Verify registered model artifact can be loaded and contains preprocessor + model
        loaded_pipeline = mlflow.sklearn.load_model(
            f"models:/{model_name}/{mv.version}"
        )
        assert isinstance(loaded_pipeline, Pipeline)
        assert "preprocessor" in loaded_pipeline.named_steps
        assert "model" in loaded_pipeline.named_steps


def test_import_training_module_does_not_mutate_root_handlers() -> None:
    """Verify that importing or reloading src.training.train does not attach handlers to root logger."""
    import importlib
    import logging

    import src.training.train as train_mod

    root = logging.getLogger()
    original_handlers = list(root.handlers)

    importlib.reload(train_mod)

    assert root.handlers == original_handlers


def test_setup_mlflow_restores_fileconfig_and_cleans_handlers_on_failure() -> None:
    """Verify that if MlflowClient adds a handler and raises an exception:
    1. logging.config.fileConfig is restored to its original function.
    2. Newly added root handlers are removed.
    3. Pre-existing root logger handlers remain attached.
    4. The original exception propagates to the caller without being swallowed.
    """
    import logging.config
    from unittest.mock import patch

    import pytest

    from src.training.train import setup_mlflow

    root = logging.getLogger()
    original_file_config = getattr(logging.config, "fileConfig", None)
    pre_existing_handler = logging.NullHandler()
    root.addHandler(pre_existing_handler)
    leaked_handler = logging.StreamHandler()

    def failing_client_init(*args, **kwargs):
        # Simulate a handler leaked right before a failure occurs
        root.addHandler(leaked_handler)
        raise ConnectionError("Failed to connect to tracking store")

    try:
        with patch(
            "src.training.train.MlflowClient",
            side_effect=failing_client_init,
        ):
            with pytest.raises(
                ConnectionError,
                match="Failed to connect to tracking store",
            ):
                setup_mlflow(
                    {
                        "tracking_uri": "sqlite:///fake_test.db",
                        "experiment_name": "test-exp",
                    }
                )

        # 1. fileConfig is restored on failure
        assert logging.config.fileConfig is original_file_config
        # 2. Leaked handler was removed on failure
        assert leaked_handler not in root.handlers
        # 3. Pre-existing handler remains attached
        assert pre_existing_handler in root.handlers
    finally:
        if original_file_config is not None:
            logging.config.fileConfig = original_file_config
        root.removeHandler(pre_existing_handler)
        root.removeHandler(leaked_handler)


def test_setup_mlflow_cleans_up_new_handlers_and_preserves_existing() -> None:
    """Verify that during successful setup_mlflow initialization:
    1. Pre-existing root handlers remain attached.
    2. Any root handlers introduced during MlflowClient init are removed.
    3. logging.config.fileConfig is restored to original function.
    """
    import logging.config
    from unittest.mock import MagicMock, patch

    from src.training.train import setup_mlflow

    root = logging.getLogger()
    original_file_config = getattr(logging.config, "fileConfig", None)
    pre_existing_handler = logging.NullHandler()
    root.addHandler(pre_existing_handler)
    leaked_handler = logging.StreamHandler()

    mock_client = MagicMock()
    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp-12345"
    mock_client.get_experiment_by_name.return_value = mock_exp

    def mock_client_init(*args, **kwargs):
        # Simulate Alembic migration attaching a console handler to root
        root.addHandler(leaked_handler)
        return mock_client

    try:
        with (
            patch(
                "src.training.train.MlflowClient",
                side_effect=mock_client_init,
            ),
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.set_experiment"),
        ):
            exp_id = setup_mlflow(
                {
                    "tracking_uri": "sqlite:///fake_test.db",
                    "experiment_name": "test-exp",
                }
            )

            assert exp_id == "exp-12345"

        # 1. Pre-existing handler is preserved
        assert pre_existing_handler in root.handlers
        # 2. Leaked handler was cleaned up
        assert leaked_handler not in root.handlers
        # 3. fileConfig was restored
        assert logging.config.fileConfig is original_file_config
    finally:
        if original_file_config is not None:
            logging.config.fileConfig = original_file_config
        root.removeHandler(pre_existing_handler)
        root.removeHandler(leaked_handler)
