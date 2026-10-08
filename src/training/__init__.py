"""
src.training — Model training, preprocessing, benchmarking, and evaluation (Milestone 3).
"""

from __future__ import annotations

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

__all__ = [
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "EXCLUDED_FEATURES",
    "create_preprocessor",
    "get_logistic_regression",
    "get_random_forest",
    "get_xgboost",
    "get_models",
    "create_pipeline",
    "create_model_pipelines",
    "EvaluationResult",
    "evaluate_predictions",
    "evaluate_model",
    "compare_models",
]
