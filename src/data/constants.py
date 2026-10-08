"""
src/data/constants.py
─────────────────────
Centralised constants for the data pipeline.

Keeping column names, expected values, and seeds in one place prevents
silent drift when the dataset schema changes and ensures every module
uses the same source of truth.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema contract – IBM Telco Customer Churn dataset
# ---------------------------------------------------------------------------

#: Identifier column – must exist but must NOT be used as a feature.
CUSTOMER_ID_COL: str = "customerID"

#: Prediction target column.
TARGET_COL: str = "Churn"

#: Mapping from raw target strings to binary integers.
TARGET_ENCODING: dict[str, int] = {"No": 0, "Yes": 1}

#: All columns that must be present in the raw CSV (order-independent).
REQUIRED_COLUMNS: list[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

#: Numerical columns (excluding the identifier and target).
NUMERICAL_COLS: list[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

#: Expected valid ranges (inclusive) for numerical columns.
NUMERICAL_RANGES: dict[str, tuple[float, float]] = {
    "tenure": (0.0, 999.0),
    "MonthlyCharges": (0.0, 10_000.0),
    "TotalCharges": (0.0, 100_000.0),
    "SeniorCitizen": (0.0, 1.0),
}

#: Categorical columns and their known valid value sets.
#: "No internet service" / "No phone service" carry real information and are
#: treated as valid categorical values, not as missing data.
CATEGORICAL_VALID_VALUES: dict[str, set[str]] = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Global random seed used for all splits and shuffles.
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Split ratios  (must sum to 1.0)
# ---------------------------------------------------------------------------

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15
