"""tests/unit/test_drift_readiness.py
───────────────────────────────────
Unit tests for the lightweight drift evaluation readiness check.

Verifies:
- 0 eligible records returns NOT_READY with shortfall=100
- 7 eligible records with min_samples=100 returns NOT_READY with shortfall=93
- Exactly 100 eligible records returns READY with shortfall=0
- Explicit UTC window boundaries filter accurately
- No snapshot writes or prediction record modifications occur
- Database query failures are safely handled and do not report NOT_READY
- Existing full evaluation behavior remains unchanged when --check-only is omitted
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from scripts.run_drift_check import (
    check_drift_readiness,
    main,
    parse_args,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import DriftMetricsSnapshot, PredictionRecord


@pytest.fixture
def sqlite_engine():
    """Create an isolated in-memory SQLite engine with project models."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(sqlite_engine) -> Session:
    """Create an isolated test session bound to in-memory SQLite engine."""
    session_maker = sessionmaker(bind=sqlite_engine)
    session = session_maker()
    yield session
    session.rollback()
    session.close()


def _create_sample_record(created_at: datetime) -> PredictionRecord:
    """Helper to construct a valid prediction record."""
    return PredictionRecord(
        created_at=created_at,
        features={
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
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
            "MonthlyCharges": 50.0,
            "TotalCharges": 600.0,
        },
        prediction=1,
        churn_label="Yes",
        churn_probability=0.82,
        model_name="churn-predictor",
        model_version="1",
    )


def test_readiness_zero_eligible_records(db_session: Session) -> None:
    """Verify readiness check with an empty database."""
    result = check_drift_readiness(
        window_hours=24.0,
        min_samples=100,
        session=db_session,
    )

    assert result["status"] == "NOT_READY"
    assert result["eligible_records"] == 0
    assert result["min_samples"] == 100
    assert result["shortfall"] == 100
    assert "window_start_utc" in result
    assert "window_end_utc" in result

    # Assert no snapshots were written
    snapshot_count = db_session.query(DriftMetricsSnapshot).count()
    assert snapshot_count == 0


def test_readiness_seven_eligible_records(db_session: Session) -> None:
    """Verify readiness check with 7 records when minimum required is 100."""
    now = datetime.now(UTC)
    for i in range(7):
        db_session.add(_create_sample_record(now - timedelta(hours=i)))
    db_session.commit()

    result = check_drift_readiness(
        window_hours=24.0,
        min_samples=100,
        session=db_session,
    )

    assert result["status"] == "NOT_READY"
    assert result["eligible_records"] == 7
    assert result["min_samples"] == 100
    assert result["shortfall"] == 93

    # Assert no snapshots were written
    snapshot_count = db_session.query(DriftMetricsSnapshot).count()
    assert snapshot_count == 0


def test_readiness_exactly_one_hundred_eligible_records(db_session: Session) -> None:
    """Verify readiness check returns READY when exactly 100 records exist."""
    now = datetime.now(UTC)
    for i in range(100):
        db_session.add(_create_sample_record(now - timedelta(minutes=i * 5)))
    db_session.commit()

    result = check_drift_readiness(
        window_hours=24.0,
        min_samples=100,
        session=db_session,
    )

    assert result["status"] == "READY"
    assert result["eligible_records"] == 100
    assert result["min_samples"] == 100
    assert result["shortfall"] == 0

    # Assert no snapshots were written
    snapshot_count = db_session.query(DriftMetricsSnapshot).count()
    assert snapshot_count == 0


def test_readiness_explicit_utc_window_boundaries(db_session: Session) -> None:
    """Verify explicit UTC ISO-8601 window boundaries accurately filter records."""
    t1 = datetime(2026, 10, 9, 12, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 10, 11, 12, 0, 0, tzinfo=UTC)

    db_session.add(_create_sample_record(t1))
    db_session.add(_create_sample_record(t2))
    db_session.add(_create_sample_record(t3))
    db_session.commit()

    result = check_drift_readiness(
        window_start="2026-10-10T00:00:00Z",
        window_end="2026-10-10T23:59:59Z",
        min_samples=100,
        session=db_session,
    )

    assert result["eligible_records"] == 1
    assert result["status"] == "NOT_READY"
    assert result["shortfall"] == 99
    assert result["window_start_utc"] == "2026-10-10T00:00:00+00:00"
    assert result["window_end_utc"] == "2026-10-10T23:59:59+00:00"


def test_readiness_no_writes_or_modifications(db_session: Session) -> None:
    """Confirm check-only mode performs zero writes to prediction or snapshot tables."""
    now = datetime.now(UTC)
    for _ in range(5):
        db_session.add(_create_sample_record(now))
    db_session.commit()

    initial_preds = db_session.query(PredictionRecord).count()
    initial_snapshots = db_session.query(DriftMetricsSnapshot).count()

    check_drift_readiness(
        window_hours=24.0,
        min_samples=100,
        session=db_session,
    )

    assert db_session.query(PredictionRecord).count() == initial_preds
    assert db_session.query(DriftMetricsSnapshot).count() == initial_snapshots


def test_readiness_database_query_failure(db_session: Session) -> None:
    """Ensure database query failures raise cleanly and are not reported as NOT_READY."""
    mock_session = MagicMock(spec=Session)
    mock_session.scalar.side_effect = RuntimeError("Database connection lost")

    with pytest.raises(RuntimeError, match="Database connection lost"):
        check_drift_readiness(session=mock_session)

    # In main(), verify failure returns exit code 2 and does not print NOT_READY
    with patch(
        "scripts.run_drift_check.check_drift_readiness",
        side_effect=RuntimeError("DB query failed"),
    ):
        exit_code = main(["--check-only"])
        assert exit_code == 2


def test_main_cli_exit_codes(
    db_session: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify main() exit code 0 when READY, 1 when NOT_READY."""
    now = datetime.now(UTC)
    for _ in range(10):
        db_session.add(_create_sample_record(now))
    db_session.commit()

    # Case 1: Threshold = 100 -> NOT_READY -> Exit 1
    with patch(
        "scripts.run_drift_check.get_session_factory", return_value=lambda: db_session
    ):
        exit_code_not_ready = main(["--check-only", "--min-samples", "100"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert exit_code_not_ready == 1
        assert data["status"] == "NOT_READY"
        assert data["eligible_records"] == 10
        assert data["shortfall"] == 90

    # Case 2: Threshold = 10 -> READY -> Exit 0
    with patch(
        "scripts.run_drift_check.get_session_factory", return_value=lambda: db_session
    ):
        exit_code_ready = main(["--check-only", "--min-samples", "10"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert exit_code_ready == 0
        assert data["status"] == "READY"
        assert data["eligible_records"] == 10
        assert data["shortfall"] == 0


def test_full_evaluation_preserved_when_check_only_omitted() -> None:
    """Verify that omitting --check-only delegates to run_drift_evaluation."""
    args = parse_args([])
    assert args.check_only is False

    mock_snapshot = MagicMock()
    mock_snapshot.id = 42
    mock_snapshot.status = "evaluated"
    mock_snapshot.records_evaluated = 150
    mock_snapshot.num_drifted_features = 0
    mock_snapshot.prediction_drift_detected = False

    with patch(
        "scripts.run_drift_check.run_drift_evaluation",
        return_value=(mock_snapshot, True),
    ) as mock_eval:
        exit_code = main([])
        assert exit_code == 0
        mock_eval.assert_called_once()
