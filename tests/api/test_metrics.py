"""
tests/api/test_metrics.py
─────────────────────────
Unit and integration tests for Prometheus metrics instrumentation.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import PredictionRecord
from app.db.session import get_db
from app.main import app


class MockPipeline:
    """Mock Scikit-learn Pipeline returning predetermined predictions."""

    def __init__(self, pred: int = 1, proba_1: float = 0.785) -> None:
        self.pred = pred
        self.proba_1 = proba_1

    def predict(self, X: Any) -> list[int]:
        return [self.pred]

    def predict_proba(self, X: Any) -> list[list[float]]:
        return [[1.0 - self.proba_1, self.proba_1]]


class BrokenPipeline:
    """Mock Pipeline that raises an exception during inference."""

    def predict(self, X: Any) -> list[int]:
        raise RuntimeError("Mock inference failure")

    def predict_proba(self, X: Any) -> list[list[float]]:
        raise RuntimeError("Mock inference failure")


@pytest.fixture()
def test_db_session() -> Session:
    """In-memory SQLite database session isolated per test."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    session_factory = sessionmaker(
        autocommit=False,
        bind=test_engine,
        expire_on_commit=False,
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
    app.dependency_overrides[get_db] = lambda: test_db_session
    with TestClient(app, raise_server_exceptions=False) as c:
        c.app.state.model = MockPipeline(pred=1, proba_1=0.785)
        c.app.state.model_ready = True
        c.app.state.model_name = "churn-predictor"
        c.app.state.model_version = "1"
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def valid_customer_payload() -> dict[str, Any]:
    """Valid IBM Telco customer payload with all 19 raw feature fields."""
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


def get_sample_value(metric_name: str, labels: dict[str, str] | None = None) -> float:
    """Safely get sample value from Prometheus default REGISTRY or 0.0 if not yet sampled."""
    val = REGISTRY.get_sample_value(metric_name, labels or {})
    return 0.0 if val is None else float(val)


def test_metrics_endpoint_returns_200_and_prometheus_format(client: TestClient) -> None:
    """GET /metrics returns HTTP 200 with standard Prometheus text content type."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "version=0.0.4" in response.headers["content-type"]
    text = response.text
    assert "churn_prediction_requests_total" in text
    assert "churn_prediction_latency_seconds" in text
    assert "churn_predictions_total" in text
    assert "churn_prediction_probability" in text
    assert "churn_prediction_errors_total" in text


def test_successful_prediction_increments_request_and_prediction_counters(
    client: TestClient,
    valid_customer_payload: dict[str, Any],
) -> None:
    """Successful prediction increments requests_total[status=success] and predictions_total."""
    req_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
        "status": "success",
    }
    pred_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
        "churn_label": "Yes",
    }

    req_before = get_sample_value("churn_prediction_requests_total", req_labels)
    pred_before = get_sample_value("churn_predictions_total", pred_labels)

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["churn_label"] == "Yes"

    req_after = get_sample_value("churn_prediction_requests_total", req_labels)
    pred_after = get_sample_value("churn_predictions_total", pred_labels)

    assert req_after - req_before == 1.0
    assert pred_after - pred_before == 1.0


def test_prediction_latency_and_probability_observed(
    client: TestClient,
    valid_customer_payload: dict[str, Any],
) -> None:
    """Successful prediction observes latency histogram and probability distribution histogram."""
    hist_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
    }

    latency_count_before = get_sample_value(
        "churn_prediction_latency_seconds_count", hist_labels
    )
    prob_count_before = get_sample_value(
        "churn_prediction_probability_count", hist_labels
    )
    prob_sum_before = get_sample_value("churn_prediction_probability_sum", hist_labels)

    client.app.state.model = MockPipeline(pred=1, proba_1=0.785)
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200

    latency_count_after = get_sample_value(
        "churn_prediction_latency_seconds_count", hist_labels
    )
    prob_count_after = get_sample_value(
        "churn_prediction_probability_count", hist_labels
    )
    prob_sum_after = get_sample_value("churn_prediction_probability_sum", hist_labels)

    assert latency_count_after - latency_count_before == 1.0
    assert prob_count_after - prob_count_before == 1.0
    assert round(prob_sum_after - prob_sum_before, 3) == 0.785


def test_failed_prediction_model_not_ready_increments_error_counter(
    client: TestClient,
    valid_customer_payload: dict[str, Any],
) -> None:
    """When model is not ready (503), error counter and request error counter are incremented."""
    err_labels = {"error_category": "model_not_ready"}
    req_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
        "status": "error",
    }
    pred_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
        "churn_label": "Yes",
    }

    err_before = get_sample_value("churn_prediction_errors_total", err_labels)
    req_before = get_sample_value("churn_prediction_requests_total", req_labels)
    pred_before = get_sample_value("churn_predictions_total", pred_labels)

    client.app.state.model_ready = False
    client.app.state.model = None

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 503

    err_after = get_sample_value("churn_prediction_errors_total", err_labels)
    req_after = get_sample_value("churn_prediction_requests_total", req_labels)
    pred_after = get_sample_value("churn_predictions_total", pred_labels)

    assert err_after - err_before == 1.0
    assert req_after - req_before == 1.0
    assert pred_after == pred_before


def test_failed_prediction_inference_error_increments_error_counter(
    client: TestClient,
    valid_customer_payload: dict[str, Any],
) -> None:
    """When inference fails (500), error counter and latency are recorded, but predictions counter is not."""
    err_labels = {"error_category": "inference_failure"}
    req_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
        "status": "error",
    }
    hist_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
    }

    err_before = get_sample_value("churn_prediction_errors_total", err_labels)
    req_before = get_sample_value("churn_prediction_requests_total", req_labels)
    latency_count_before = get_sample_value(
        "churn_prediction_latency_seconds_count", hist_labels
    )

    client.app.state.model = BrokenPipeline()
    client.app.state.model_ready = True

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 500

    err_after = get_sample_value("churn_prediction_errors_total", err_labels)
    req_after = get_sample_value("churn_prediction_requests_total", req_labels)
    latency_count_after = get_sample_value(
        "churn_prediction_latency_seconds_count", hist_labels
    )

    assert err_after - err_before == 1.0
    assert req_after - req_before == 1.0
    assert latency_count_after - latency_count_before == 1.0


def test_failed_prediction_database_error_increments_error_counter(
    client: TestClient,
    test_db_session: Session,
    valid_customer_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When DB persistence fails (500), database_failure is counted and rolled back."""
    err_labels = {"error_category": "database_failure"}
    req_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
        "status": "error",
    }

    def broken_commit():
        raise RuntimeError("DB connection failure")

    monkeypatch.setattr(test_db_session, "commit", broken_commit)

    err_before = get_sample_value("churn_prediction_errors_total", err_labels)
    req_before = get_sample_value("churn_prediction_requests_total", req_labels)

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 500

    err_after = get_sample_value("churn_prediction_errors_total", err_labels)
    req_after = get_sample_value("churn_prediction_requests_total", req_labels)

    assert err_after - err_before == 1.0
    assert req_after - req_before == 1.0


def test_repeated_requests_do_not_double_count_metrics(
    client: TestClient,
    valid_customer_payload: dict[str, Any],
) -> None:
    """Exactly N predictions increment counters and observations by exactly N."""
    req_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
        "status": "success",
    }
    pred_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
        "churn_label": "Yes",
    }
    hist_labels = {
        "model_name": "churn-predictor",
        "model_version": "1",
    }

    req_before = get_sample_value("churn_prediction_requests_total", req_labels)
    pred_before = get_sample_value("churn_predictions_total", pred_labels)
    latency_before = get_sample_value(
        "churn_prediction_latency_seconds_count", hist_labels
    )
    prob_before = get_sample_value("churn_prediction_probability_count", hist_labels)

    count = 3
    for _ in range(count):
        res = client.post("/predict", json=valid_customer_payload)
        assert res.status_code == 200

    req_after = get_sample_value("churn_prediction_requests_total", req_labels)
    pred_after = get_sample_value("churn_predictions_total", pred_labels)
    latency_after = get_sample_value(
        "churn_prediction_latency_seconds_count", hist_labels
    )
    prob_after = get_sample_value("churn_prediction_probability_count", hist_labels)

    assert req_after - req_before == float(count)
    assert pred_after - pred_before == float(count)
    assert latency_after - latency_before == float(count)
    assert prob_after - prob_before == float(count)


def test_existing_prediction_response_and_persistence_unchanged(
    client: TestClient,
    test_db_session: Session,
    valid_customer_payload: dict[str, Any],
) -> None:
    """Metrics instrumentation does not alter the response schema or DB record structure."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()

    assert set(data.keys()) == {
        "prediction",
        "churn_label",
        "churn_probability",
        "model_name",
        "model_version",
    }
    assert data["prediction"] == 1
    assert data["churn_label"] == "Yes"
    assert data["churn_probability"] == 0.785
    assert data["model_name"] == "churn-predictor"
    assert data["model_version"] == "1"

    # Confirm database persistence intact
    records = test_db_session.execute(select(PredictionRecord)).scalars().all()
    assert len(records) == 1
    assert records[0].prediction == 1
    assert records[0].churn_label == "Yes"
    assert records[0].churn_probability == 0.785


def test_predict_defaults_when_model_metadata_absent(
    client: TestClient,
    test_db_session: Session,
    valid_customer_payload: dict[str, Any],
) -> None:
    """When model metadata is absent from app.state, response defaults to churn-predictor:1 and metrics remain valid."""
    client.app.state.model = MockPipeline(pred=1, proba_1=0.785)
    client.app.state.model_ready = True
    client.app.state.model_name = None
    client.app.state.model_version = None

    req_labels = {
        "model_name": "unknown",
        "model_version": "unknown",
        "status": "success",
    }
    pred_labels = {
        "model_name": "unknown",
        "model_version": "unknown",
        "churn_label": "Yes",
    }

    req_before = get_sample_value("churn_prediction_requests_total", req_labels)
    pred_before = get_sample_value("churn_predictions_total", pred_labels)

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()

    # Verify original defaults are preserved in response
    assert data["model_name"] == "churn-predictor"
    assert data["model_version"] == "1"
    assert data["prediction"] == 1
    assert data["churn_label"] == "Yes"

    # Verify DB record preserves the original defaults
    records = test_db_session.execute(select(PredictionRecord)).scalars().all()
    assert len(records) == 1
    assert records[0].model_name == "churn-predictor"
    assert records[0].model_version == "1"

    # Verify metrics remain valid and increment under safe metric labels
    req_after = get_sample_value("churn_prediction_requests_total", req_labels)
    pred_after = get_sample_value("churn_predictions_total", pred_labels)
    assert req_after - req_before == 1.0
    assert pred_after - pred_before == 1.0

    # Verify GET /metrics succeeds and includes the sample
    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    assert (
        'churn_prediction_requests_total{model_name="unknown",model_version="unknown",status="success"}'
        in metrics_res.text
    )


def test_metric_recording_failure_does_not_mask_api_error(
    client: TestClient,
    valid_customer_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When metric observation raises an exception in finally, the original API error response is preserved."""
    from app.core import metrics as metrics_module

    def broken_observe(*args, **kwargs):
        raise RuntimeError("Simulated Prometheus failure")

    # Monkeypatch latency histogram to fail
    monkeypatch.setattr(
        metrics_module.churn_prediction_latency_seconds, "labels", broken_observe
    )

    client.app.state.model_ready = False
    client.app.state.model = None

    response = client.post("/predict", json=valid_customer_payload)
    # Must preserve the original 503 rather than raising an unhandled 500 from the finally block
    assert response.status_code == 503
    assert (
        response.json()["detail"]
        == "Model is not ready for inference. Please check service health."
    )


def test_database_rollback_failure_preserves_http_500_and_category(
    client: TestClient,
    test_db_session: Session,
    valid_customer_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both DB commit and rollback raise exceptions, HTTP 500 with original detail and database_failure is recorded."""
    err_labels = {"error_category": "database_failure"}
    err_before = get_sample_value("churn_prediction_errors_total", err_labels)

    def broken_commit():
        raise RuntimeError("DB commit failure")

    def broken_rollback():
        raise RuntimeError("DB rollback failure")

    monkeypatch.setattr(test_db_session, "commit", broken_commit)
    monkeypatch.setattr(test_db_session, "rollback", broken_rollback)

    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to record prediction in database."

    err_after = get_sample_value("churn_prediction_errors_total", err_labels)
    assert err_after - err_before == 1.0


def test_incompatible_collector_type_rejected() -> None:
    """Attempting to reuse an existing collector with an incompatible type raises TypeError."""
    from prometheus_client import CollectorRegistry, Gauge

    from app.core.metrics import get_or_create_counter

    reg = CollectorRegistry()
    Gauge("test_gauge_metric", "A gauge metric", registry=reg)

    with pytest.raises(
        TypeError, match="already exists in registry as Gauge, not Counter"
    ):
        get_or_create_counter("test_gauge_metric", "A counter metric", registry=reg)


def test_incompatible_collector_labels_rejected() -> None:
    """Attempting to reuse an existing counter with mismatched labels raises ValueError."""
    from prometheus_client import CollectorRegistry

    from app.core.metrics import get_or_create_counter

    reg = CollectorRegistry()
    get_or_create_counter(
        "test_lbl_metric", "doc", labelnames=["model_name"], registry=reg
    )

    with pytest.raises(ValueError, match="already exists with labels"):
        get_or_create_counter(
            "test_lbl_metric", "doc", labelnames=["model_name", "status"], registry=reg
        )


def test_incompatible_histogram_buckets_rejected() -> None:
    """Attempting to reuse an existing histogram with mismatched buckets raises ValueError."""
    from prometheus_client import CollectorRegistry

    from app.core.metrics import get_or_create_histogram

    reg = CollectorRegistry()
    get_or_create_histogram(
        "test_bucket_metric", "doc", buckets=(0.1, 0.5), registry=reg
    )

    with pytest.raises(ValueError, match="already exists with buckets"):
        get_or_create_histogram(
            "test_bucket_metric", "doc", buckets=(0.2, 0.5), registry=reg
        )
