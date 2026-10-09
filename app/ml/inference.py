"""
app/ml/inference.py
───────────────────
Inference pipeline for raw customer input.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.api.schemas.prediction import CustomerInput
from src.data.cleaning import fix_dtypes, fix_total_charges
from src.data.features import engineer_features
from src.training.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

logger = logging.getLogger(__name__)

ALL_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def predict_churn(
    pipeline: Any,
    customer_input: CustomerInput,
) -> tuple[int, str, float]:
    """Run model inference on a single validated customer record.

    Parameters
    ----------
    pipeline:
        Fitted Scikit-learn Pipeline loaded from MLflow.
    customer_input:
        Validated CustomerInput schema instance.

    Returns
    -------
    tuple of (prediction, churn_label, churn_probability)
        - prediction: 0 or 1
        - churn_label: "No" or "Yes"
        - churn_probability: float between 0.0 and 1.0 (for class 1)
    """
    raw_dict = customer_input.model_dump()
    df = pd.DataFrame([raw_dict])

    # 1. Deterministic cleaning (reuses Milestone 2 functions)
    df = fix_dtypes(df)
    df, _ = fix_total_charges(df)

    # 2. Feature engineering (reuses Milestone 2 functions)
    df = engineer_features(df)

    # 3. Select exactly the 26 feature columns expected by the fitted ColumnTransformer
    X = df[ALL_FEATURES]

    # 4. Predict class and probabilities
    pred_val = int(pipeline.predict(X)[0])
    probabilities = pipeline.predict_proba(X)[0]
    churn_prob = float(probabilities[1])
    churn_label = "Yes" if pred_val == 1 else "No"

    return pred_val, churn_label, churn_prob
