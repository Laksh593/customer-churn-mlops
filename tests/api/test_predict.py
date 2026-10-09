"""
tests/api/test_predict.py
─────────────────────────
Unit and integration tests for POST /predict endpoint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


class MockPipeline:
    """Deterministic mock pipeline for fast, isolated endpoint testing."""

    def __init__(self, pred: int = 1, proba_1: float = 0.785) -> None:
        self.pred = pred
        self.proba_1 = proba_1

    def predict(self, X: Any) -> np.ndarray:
        return np.array([self.pred])

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.array([[1.0 - self.proba_1, self.proba_1]])


class BrokenPipeline:
    """Mock pipeline that raises an exception on predict."""

    def predict(self, X: Any) -> np.ndarray:
        raise RuntimeError("Internal computation error")

    def predict_proba(self, X: Any) -> np.ndarray:
        raise RuntimeError("Internal computation error")


@pytest.fixture()
def valid_customer_payload() -> dict[str, Any]:
    """Return a complete, valid customer payload matching IBM Telco schema."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


@pytest.fixture()
def client() -> TestClient:
    """TestClient fixture with mock model loaded on app.state."""
    with TestClient(app, raise_server_exceptions=False) as c:
        c.app.state.model = MockPipeline(pred=1, proba_1=0.785)
        c.app.state.model_ready = True
        c.app.state.model_name = "churn-predictor"
        c.app.state.model_version = "1"
        yield c


def test_predict_success_churn_yes(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    """Successful prediction returning churn class 1 and Yes label."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["churn_label"] == "Yes"
    assert data["churn_probability"] == 0.785
    assert data["model_name"] == "churn-predictor"
    assert data["model_version"] == "1"


def test_predict_success_churn_no(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    """Successful prediction returning churn class 0 and No label."""
    client.app.state.model = MockPipeline(pred=0, proba_1=0.1234)

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["churn_label"] == "No"
    assert data["churn_probability"] == 0.1234


def test_predict_probability_bounds(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    """Probability returned is within [0.0, 1.0]."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_handles_whitespace_total_charges(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    """TotalCharges with whitespace string is accepted and handled by cleaning."""
    payload = dict(valid_customer_payload)
    payload["TotalCharges"] = " "

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_rejects_missing_field(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    """Missing a required field results in 422 Unprocessable Entity."""
    payload = dict(valid_customer_payload)
    del payload["Contract"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_category(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    """Invalid category value results in 422 Unprocessable Entity."""
    payload = dict(valid_customer_payload)
    payload["gender"] = "Unknown"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_numeric_range(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    """Negative numerical values out of range result in 422 Unprocessable Entity."""
    payload = dict(valid_customer_payload)
    payload["tenure"] = -1

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_unexpected_extra_field(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    """Unexpected fields (e.g. customerID, Churn) result in 422 Unprocessable Entity."""
    payload = dict(valid_customer_payload)
    payload["customerID"] = "1234-ABCD"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_model_not_ready_returns_503(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    """Returns HTTP 503 when the model is not ready or failed to load."""
    client.app.state.model_ready = False
    client.app.state.model = None

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()


def test_predict_pipeline_failure_returns_500(
    client: TestClient, valid_customer_payload: dict[str, Any]
) -> None:
    """Returns HTTP 500 without exposing internal trace when inference raises."""
    client.app.state.model = BrokenPipeline()
    client.app.state.model_ready = True

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 500
    assert "inference failed" in response.json()["detail"].lower()


def test_integration_with_real_registered_model(
    valid_customer_payload: dict[str, Any],
) -> None:
    """Integration test: load real registered model from SQLite and predict."""
    from app.ml.loader import PROJECT_ROOT, load_registered_model

    db_path = PROJECT_ROOT / "mlflow.db"
    if not db_path.exists():
        pytest.skip("Local mlflow.db does not exist; skipping integration test.")

    # Test loading real model
    try:
        real_pipeline = load_registered_model(
            model_name="churn-predictor", model_version="1"
        )
    except Exception as exc:
        pytest.skip(f"Could not load registered model: {exc}")

    with TestClient(app, raise_server_exceptions=False) as real_client:
        real_client.app.state.model = real_pipeline
        real_client.app.state.model_ready = True
        real_client.app.state.model_name = "churn-predictor"
        real_client.app.state.model_version = "1"

        response = real_client.post("/predict", json=valid_customer_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in (0, 1)
        assert data["churn_label"] in ("No", "Yes")
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["model_name"] == "churn-predictor"
        assert data["model_version"] == "1"
