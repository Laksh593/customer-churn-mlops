"""
src/training/preprocessing.py
─────────────────────────────
Feature preprocessing pipelines for model training.

Separates numerical and categorical features according to project
specifications and constructs an sklearn ColumnTransformer.

Design principles:
- Excludes customerID (identifier) and Churn (target).
- Numerical: SimpleImputer(median) + StandardScaler.
- Categorical: SimpleImputer(most_frequent) + OneHotEncoder(ignore).
- Fits ONLY through the training Pipeline to prevent data leakage.
"""

from __future__ import annotations

import logging
from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

#: Features processed through the numerical imputation and scaling pipeline.
NUMERICAL_FEATURES: Final[list[str]] = [
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

#: Features processed through categorical imputation and one-hot encoding.
CATEGORICAL_FEATURES: Final[list[str]] = [
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

#: Explicitly excluded columns from model feature sets.
EXCLUDED_FEATURES: Final[list[str]] = [
    "customerID",
    "Churn",
]


def create_preprocessor(
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build an sklearn ColumnTransformer for tabular preprocessing.

    Applies:
    - Median imputation + StandardScaler for numerical features.
    - Most-frequent imputation + OneHotEncoder for categorical features.

    Parameters
    ----------
    numerical_features:
        List of numerical feature names. Defaults to ``NUMERICAL_FEATURES``.
    categorical_features:
        List of categorical feature names. Defaults to ``CATEGORICAL_FEATURES``.

    Returns
    -------
    ColumnTransformer
        Unfitted ColumnTransformer ready for inclusion in a training Pipeline.
    """
    num_cols = (
        numerical_features if numerical_features is not None else NUMERICAL_FEATURES
    )
    cat_cols = (
        categorical_features
        if categorical_features is not None
        else CATEGORICAL_FEATURES
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    logger.debug(
        "create_preprocessor: Configured with %d numerical and %d categorical features",
        len(num_cols),
        len(cat_cols),
    )
    return preprocessor
