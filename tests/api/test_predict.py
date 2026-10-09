"""
tests/api/test_predict.py
─────────────────────────
Unit and integration tests for POST /predict endpoint including DB persistence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import PredictionRecord
from app.db.session import get_db
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
def test_db_session() -> Session:
    """Isolated in-memory SQLite database session for API unit tests."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    session_factory = sessionmaker(
        bind=test_engine,
        autocommit=False,
        autoflush=False,
    )
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture()
def client(test_db_session: Session) -> TestClient:
    """TestClient fixture with mock model and isolated in-memory DB override."""
    # Override get_db dependency with test database session
    app.dependency_overrides[get_db] = lambda: test_db_session
    with TestClient(app, raise_server_exceptions=False) as c:
        c.app.state.model = MockPipeline(pred=1, proba_1=0.785)
        c.app.state.model_ready = True
        c.app.state.model_name = "churn-predictor"
        c.app.state.model_version = "1"
        yield c
    app.dependency_overrides.clear()


def test_predict_success_churn_yes(
    client: TestClient,
    test_db_session: Session,
    valid_customer_payload: dict[str, Any],
) -> None:
    """Successful prediction returning churn class 1 and Yes label, persisting record."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["churn_label"] == "Yes"
    assert data["churn_probability"] == 0.785
    assert data["model_name"] == "churn-predictor"
    assert data["model_version"] == "1"

    # Verify row was saved in the database
    records = test_db_session.execute(select(PredictionRecord)).scalars().all()
    assert len(records) == 1
    record = records[0]
    assert record.prediction == 1
    assert record.churn_label == "Yes"
    assert record.churn_probability == 0.785
    assert record.model_name == "churn-predictor"
    assert record.model_version == "1"
    assert record.features["gender"] == "Female"
    assert "customerID" not in record.features
    assert "Churn" not in record.features


def test_predict_success_churn_no(
    client: TestClient,
    test_db_session: Session,
    valid_customer_payload: dict[str, Any],
) -> None:
    """Successful prediction returning churn class 0 and No label, persisting record."""
    client.app.state.model = MockPipeline(pred=0, proba_1=0.1234)

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["churn_label"] == "No"
    assert data["churn_probability"] == 0.1234

    records = test_db_session.execute(select(PredictionRecord)).scalars().all()
    assert len(records) == 1
    assert records[0].prediction == 0
    assert records[0].churn_label == "No"


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


def test_predict_database_failure_returns_500(
    client: TestClient,
    test_db_session: Session,
    valid_customer_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returns HTTP 500 when database persistence fails, rolling back the session."""

    def broken_commit():
        raise RuntimeError("DB connection dropped")

    monkeypatch.setattr(test_db_session, "commit", broken_commit)

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 500
    assert (
        "failed to record prediction in database" in response.json()["detail"].lower()
    )

    # Verify no record was saved
    records = test_db_session.execute(select(PredictionRecord)).scalars().all()
    assert len(records) == 0


def test_integration_with_real_registered_model(
    client: TestClient,
    test_db_session: Session,
    valid_customer_payload: dict[str, Any],
) -> None:
    """Integration test: load real registered model from MLflow and predict with persistence."""
    from app.ml.loader import PROJECT_ROOT, load_registered_model

    db_path = PROJECT_ROOT / "mlflow.db"
    if not db_path.exists():
        pytest.skip("Local mlflow.db does not exist; skipping integration test.")

    try:
        real_pipeline = load_registered_model(
            model_name="churn-predictor", model_version="1"
        )
    except Exception as exc:
        pytest.skip(f"Could not load registered model: {exc}")

    client.app.state.model = real_pipeline
    client.app.state.model_ready = True
    client.app.state.model_name = "churn-predictor"
    client.app.state.model_version = "1"

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert data["churn_label"] in ("No", "Yes")
    assert 0.0 <= data["churn_probability"] <= 1.0

    # Verify persistence
    records = test_db_session.execute(select(PredictionRecord)).scalars().all()
    assert len(records) == 1
    assert records[0].prediction == data["prediction"]
    assert records[0].churn_probability == data["churn_probability"]
