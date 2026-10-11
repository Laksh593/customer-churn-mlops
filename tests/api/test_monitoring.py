"""tests/api/test_monitoring.py
────────────────────────────
Unit and integration tests for the /monitoring/drift/readiness endpoint.

Verifies:
1. Empty database reports NOT_READY with zero records and full shortfall.
2. Below-threshold records report NOT_READY as normal monitoring status.
3. At or above threshold records report READY with zero shortfall.
4. Custom window and threshold parameters (window_hours, min_samples, window_start, window_end).
5. Endpoint is strictly read-only: no snapshots created, no records modified, no drift evaluation triggered.
6. Database query failure raises HTTP 500 server error and is NOT misreported as NOT_READY.
7. Internal database exception text, credentials, and connection strings are strictly NOT exposed in HTTP response.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import DriftMetricsSnapshot, PredictionRecord
from app.db.session import get_db
from app.main import app


@pytest.fixture()
def test_db_session() -> Session:
    """Isolated in-memory SQLite database session."""
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
    """TestClient fixture with isolated test database."""
    app.dependency_overrides[get_db] = lambda: test_db_session
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


def _seed_prediction_records(
    session: Session,
    count: int,
    base_time: datetime | None = None,
    delta_hours: float = 0.1,
) -> None:
    """Helper to seed lightweight dummy prediction records."""
    t0 = base_time or datetime.now(UTC)
    records = [
        PredictionRecord(
            features={"SeniorCitizen": 0, "MonthlyCharges": 50.0},
            prediction=0,
            churn_label="No",
            churn_probability=0.25,
            model_name="churn-predictor",
            model_version="1",
            created_at=t0 - timedelta(hours=i * delta_hours),
        )
        for i in range(count)
    ]
    session.add_all(records)
    session.commit()


def test_readiness_empty_db(client: TestClient) -> None:
    """Empty database returns NOT_READY with zero records and shortfall=100."""
    response = client.get("/monitoring/drift/readiness")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NOT_READY"
    assert data["eligible_records"] == 0
    assert data["min_samples"] == 100
    assert data["shortfall"] == 100
    assert "window_start_utc" in data
    assert "window_end_utc" in data


def test_readiness_below_threshold(
    client: TestClient, test_db_session: Session
) -> None:
    """Database with 7 records returns NOT_READY with eligible_records=7 and shortfall=93."""
    _seed_prediction_records(test_db_session, count=7)
    response = client.get("/monitoring/drift/readiness")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NOT_READY"
    assert data["eligible_records"] == 7
    assert data["min_samples"] == 100
    assert data["shortfall"] == 93


def test_readiness_at_or_above_threshold(
    client: TestClient, test_db_session: Session
) -> None:
    """Database with 105 records returns READY with shortfall=0."""
    _seed_prediction_records(test_db_session, count=105, delta_hours=0.05)
    response = client.get("/monitoring/drift/readiness")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "READY"
    assert data["eligible_records"] == 105
    assert data["min_samples"] == 100
    assert data["shortfall"] == 0


def test_readiness_custom_window_and_threshold(
    client: TestClient, test_db_session: Session
) -> None:
    """Custom parameters (window_hours=48, min_samples=50) evaluate correctly."""
    now = datetime.now(UTC)
    # Seed 40 records 10 hours ago (inside 24h & 48h)
    _seed_prediction_records(
        test_db_session, count=40, base_time=now - timedelta(hours=10), delta_hours=0.1
    )
    # Seed 20 records 36 hours ago (inside 48h, outside 24h)
    _seed_prediction_records(
        test_db_session, count=20, base_time=now - timedelta(hours=36), delta_hours=0.1
    )

    # Under default 24h window: 40 records < 50 => NOT_READY
    resp_24h = client.get("/monitoring/drift/readiness?window_hours=24&min_samples=50")
    assert resp_24h.status_code == 200
    assert resp_24h.json()["status"] == "NOT_READY"
    assert resp_24h.json()["eligible_records"] == 40
    assert resp_24h.json()["shortfall"] == 10

    # Under 48h window: 60 records >= 50 => READY
    resp_48h = client.get("/monitoring/drift/readiness?window_hours=48&min_samples=50")
    assert resp_48h.status_code == 200
    assert resp_48h.json()["status"] == "READY"
    assert resp_48h.json()["eligible_records"] == 60
    assert resp_48h.json()["shortfall"] == 0


def test_readiness_strictly_read_only(
    client: TestClient, test_db_session: Session
) -> None:
    """Readiness endpoint must not write snapshots, alter records, or invoke drift evaluation."""
    _seed_prediction_records(test_db_session, count=10)

    # Initial state
    snapshots_before = test_db_session.scalar(select(DriftMetricsSnapshot.id))
    assert snapshots_before is None

    with patch("scripts.run_drift_check.run_drift_evaluation") as mock_eval:
        response = client.get("/monitoring/drift/readiness")
        assert response.status_code == 200
        # Drift evaluation engine must NOT be triggered
        mock_eval.assert_not_called()

    # Verify no snapshots written
    snapshot_count = len(test_db_session.scalars(select(DriftMetricsSnapshot)).all())
    assert snapshot_count == 0

    # Verify prediction record count unchanged
    record_count = len(test_db_session.scalars(select(PredictionRecord)).all())
    assert record_count == 10


def test_readiness_database_error_handling_sanitized(client: TestClient) -> None:
    """Database query failure must return generic HTTP 500 detail and hide raw exception text."""
    sensitive_error_msg = "connection to server at postgres:5432 failed: password authentication failed for user churn_admin"
    with patch(
        "app.api.routes.monitoring.check_drift_readiness",
        side_effect=RuntimeError(sensitive_error_msg),
    ):
        response = client.get("/monitoring/drift/readiness")
        assert response.status_code == 500
        data = response.json()
        assert data["detail"] == "Database query error during readiness check"
        # Confirm internal sensitive error text is NOT leaked
        assert sensitive_error_msg not in response.text
        assert "churn_admin" not in response.text
        assert "password" not in response.text
