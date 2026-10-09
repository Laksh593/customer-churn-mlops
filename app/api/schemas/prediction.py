"""
app/api/schemas/prediction.py
─────────────────────────────
Pydantic v2 schemas for customer churn prediction requests and responses.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CustomerInput(BaseModel):
    """Pydantic v2 schema for customer churn prediction input.

    Includes the 19 original customer feature fields from the IBM Telco
    Customer Churn dataset, strictly excluding customerID and Churn.
    """

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Whether customer is a senior citizen (1) or not (0)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has a partner"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has dependents"
    )
    tenure: Annotated[
        int,
        Field(
            ge=0, le=999, description="Number of months customer stayed with company"
        ),
    ]
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has a phone service"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Whether the customer has multiple lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer's internet service provider"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has online security"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has online backup"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has device protection"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has tech support"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has streaming TV"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has streaming movies"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="The contract term of the customer"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether the customer has paperless billing"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Customer payment method")
    MonthlyCharges: Annotated[
        float,
        Field(ge=0.0, le=10_000.0, description="Monthly amount charged to customer"),
    ]
    TotalCharges: float | str = Field(
        ...,
        description="Total amount charged to customer (numeric float, or numeric/blank string)",
    )

    @field_validator("TotalCharges")
    @classmethod
    def validate_total_charges(cls, v: float | str) -> float | str:
        """Validate TotalCharges according to project range and cleaning rules."""
        if isinstance(v, int | float):
            if v < 0.0 or v > 100_000.0:
                raise ValueError("TotalCharges must be between 0.0 and 100000.0")
            return float(v)
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return v  # Blank string is valid; deterministically imputed during cleaning
            try:
                numeric_val = float(stripped)
            except ValueError as err:
                raise ValueError(
                    f"TotalCharges string must be numeric or blank: {v!r}"
                ) from err
            if numeric_val < 0.0 or numeric_val > 100_000.0:
                raise ValueError("TotalCharges must be between 0.0 and 100000.0")
            return v
        raise ValueError(
            f"TotalCharges must be float or string, got {type(v).__name__}"
        )


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(..., description="Binary churn prediction (0: No, 1: Yes)")
    churn_label: Literal["No", "Yes"] = Field(
        ..., description="Human-readable churn label"
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted probability of churn (class 1)"
    )
    model_name: str = Field(..., description="Name of the registered MLflow model used")
    model_version: str = Field(
        ..., description="Version of the registered MLflow model used"
    )
