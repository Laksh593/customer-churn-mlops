"""
src/training/models.py
──────────────────────
Model definitions and pipeline construction for Milestone 3 benchmarking.

Provides factory functions for:
1. Logistic Regression
2. Random Forest
3. XGBoost

Every model pipeline wraps the preprocessor and estimator together:
    Pipeline([("preprocessor", preprocessor), ("model", model)])
ensuring leakage-safe execution where transformations are fitted only on X_train.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.constants import RANDOM_STATE
from src.training.preprocessing import create_preprocessor

logger = logging.getLogger(__name__)


def get_logistic_regression(
    random_state: int = RANDOM_STATE,
    **kwargs: Any,
) -> LogisticRegression:
    """Create a LogisticRegression model with baseline hyperparameters.

    Parameters
    ----------
    random_state:
        Seed for reproducibility.
    **kwargs:
        Additional hyperparameter overrides.

    Returns
    -------
    LogisticRegression
    """
    params: dict[str, Any] = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": random_state,
    }
    params.update(kwargs)
    return LogisticRegression(**params)


def get_random_forest(
    random_state: int = RANDOM_STATE,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Create a RandomForestClassifier model with baseline hyperparameters.

    Parameters
    ----------
    random_state:
        Seed for reproducibility.
    **kwargs:
        Additional hyperparameter overrides.

    Returns
    -------
    RandomForestClassifier
    """
    params: dict[str, Any] = {
        "n_estimators": 300,
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return RandomForestClassifier(**params)


def get_xgboost(
    random_state: int = RANDOM_STATE,
    **kwargs: Any,
) -> XGBClassifier:
    """Create an XGBClassifier model with baseline hyperparameters.

    Parameters
    ----------
    random_state:
        Seed for reproducibility.
    **kwargs:
        Additional hyperparameter overrides.

    Returns
    -------
    XGBClassifier
    """
    params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "eval_metric": "logloss",
    }
    params.update(kwargs)
    return XGBClassifier(**params)


def get_models(
    random_state: int = RANDOM_STATE,
    config: dict[str, Any] | None = None,
) -> dict[str, BaseEstimator]:
    """Return dictionary of the three candidate model instances.

    Parameters
    ----------
    random_state:
        Global random seed.
    config:
        Optional dictionary containing model hyperparameters.

    Returns
    -------
    dict[str, BaseEstimator]
        Mapping from model identifier to unfitted estimator.
    """
    cfg = config or {}
    lr_params = dict(cfg.get("logistic_regression", {}))
    rf_params = dict(cfg.get("random_forest", {}))
    xgb_params = dict(cfg.get("xgboost", {}))

    lr_params.setdefault("random_state", random_state)
    rf_params.setdefault("random_state", random_state)
    xgb_params.setdefault("random_state", random_state)

    return {
        "logistic_regression": get_logistic_regression(**lr_params),
        "random_forest": get_random_forest(**rf_params),
        "xgboost": get_xgboost(**xgb_params),
    }


def create_pipeline(
    model: BaseEstimator,
    preprocessor: ColumnTransformer | None = None,
) -> Pipeline:
    """Assemble a leakage-safe Pipeline wrapping preprocessor and model.

    Parameters
    ----------
    model:
        Unfitted estimator (must implement fit/predict).
    preprocessor:
        Optional preprocessor. If None, a new ColumnTransformer is constructed.

    Returns
    -------
    Pipeline
        sklearn Pipeline with steps [('preprocessor', preprocessor), ('model', model)].
    """
    prep = preprocessor if preprocessor is not None else create_preprocessor()
    return Pipeline(
        steps=[
            ("preprocessor", prep),
            ("model", model),
        ]
    )


def create_model_pipelines(
    random_state: int = RANDOM_STATE,
    config: dict[str, Any] | None = None,
) -> dict[str, Pipeline]:
    """Create independent pipelines for all candidate models.

    Each pipeline receives its own fresh ColumnTransformer instance to
    guarantee complete isolation during training.

    Parameters
    ----------
    random_state:
        Global random seed.
    config:
        Optional configuration dictionary for model hyperparameters.

    Returns
    -------
    dict[str, Pipeline]
        Mapping from model identifier to pipeline instance.
    """
    models = get_models(random_state=random_state, config=config)
    pipelines: dict[str, Pipeline] = {}
    for name, model in models.items():
        pipelines[name] = create_pipeline(
            model=model, preprocessor=create_preprocessor()
        )
        logger.debug("create_model_pipelines: Created pipeline for '%s'", name)
    return pipelines
