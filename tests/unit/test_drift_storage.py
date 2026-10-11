"""tests/unit/test_drift_storage.py
───────────────────────────────────
Unit tests for persistent drift snapshots, Alembic migration consistency,
drift check runner, fingerprint deduplication, and Prometheus metrics exposition.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from prometheus_client import CollectorRegistry
from scripts.run_drift_check import (
    compute_record_fingerprint,
    determine_model_identity,
    resolve_window,
    run_drift_evaluation,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.api.schemas.prediction import CustomerInput
from app.core.metrics import (
    churn_data_drift_detected,
    churn_data_drift_feature_status,
    churn_data_drift_psi,
    churn_drift_evaluation_status,
    churn_drift_last_evaluation_timestamp_seconds,
    churn_prediction_drift_detected,
    churn_prediction_drift_psi,
    churn_prediction_drift_status,
    get_or_create_gauge,
    reset_drift_sync_cache,
    sync_drift_gauges_from_db,
)
from app.db.base import Base
from app.db.models import DriftMetricsSnapshot, PredictionRecord


@pytest.fixture
def sqlite_engine():
    """Create an isolated in-memory SQLite engine with the project models."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(sqlite_engine) -> Session:
    """Create an isolated test session bound to the in-memory SQLite engine."""
    session_maker = sessionmaker(bind=sqlite_engine)
    session = session_maker()
    yield session
    session.close()


@pytest.fixture
def sample_raw_customer_dict() -> dict[str, Any]:
    """Return valid 19 raw features matching CustomerInput schema."""
    return CustomerInput(
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=12,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="DSL",
        OnlineSecurity="Yes",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Mailed check",
        MonthlyCharges=50.0,
        TotalCharges=600.0,
    ).model_dump()


# ---------------------------------------------------------------------------
# 1. Model & Migration Consistency Tests
# ---------------------------------------------------------------------------


def test_drift_snapshot_model_fields_and_repr(db_session: Session) -> None:
    """DriftMetricsSnapshot stores required fields, constraints, and formats __repr__ correctly."""
    now = datetime.now(UTC)
    snapshot = DriftMetricsSnapshot(
        created_at=now,
        window_start=now - timedelta(days=1),
        window_end=now,
        records_evaluated=150,
        profile_version="1.0.0",
        model_name="churn-predictor",
        model_version="1",
        status="evaluated",
        num_features_evaluated=3,
        num_drifted_features=1,
        prediction_drift_detected=False,
        prediction_psi=0.045,
        prediction_p_value=0.231,
        prediction_status="evaluated",
        feature_metrics={"tenure": {"psi": 0.12, "drift_detected": True}},
        summary_metadata={"notes": "test snapshot"},
    )
    db_session.add(snapshot)
    db_session.commit()
    db_session.refresh(snapshot)

    assert snapshot.id is not None
    assert snapshot.records_evaluated == 150
    assert snapshot.num_drifted_features == 1
    assert "DriftMetricsSnapshot" in repr(snapshot)
    assert "drifted_features=1" in repr(snapshot)


def test_alembic_migration_revisions_chain_is_consistent() -> None:
    """Verify that Alembic script directory has a continuous, linear revision chain."""
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()

    assert len(heads) == 1, f"Expected exactly 1 migration head, got: {heads}"
    assert heads[0] == "2026_10_11_0002"

    head_rev = script.get_revision(heads[0])
    assert head_rev.down_revision == "2026_10_09_0001"

    base_rev = script.get_revision("2026_10_09_0001")
    assert base_rev.down_revision is None


# ---------------------------------------------------------------------------
# 2. Runner & Preparation Tests (F-02, F-03)
# ---------------------------------------------------------------------------


def test_runner_window_resolution() -> None:
    """Window resolution respects ISO strings, hours fallback, and validates ordering."""
    start_str = "2026-10-10T12:00:00+00:00"
    end_str = "2026-10-11T12:00:00+00:00"
    s, e = resolve_window(
        window_hours=24.0, window_start_str=start_str, window_end_str=end_str
    )
    assert s == datetime(2026, 10, 10, 12, 0, 0, tzinfo=UTC)
    assert e == datetime(2026, 10, 11, 12, 0, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="must be earlier than window_end"):
        resolve_window(
            window_hours=24.0, window_start_str=end_str, window_end_str=start_str
        )


def test_runner_model_identity_extraction() -> None:
    """Model identity extracted correctly when uniform, mixed, or empty."""
    assert determine_model_identity([]) == ("unknown", "unknown")

    rec1 = PredictionRecord(model_name="churn-v1", model_version="1")
    rec2 = PredictionRecord(model_name="churn-v1", model_version="1")
    assert determine_model_identity([rec1, rec2]) == ("churn-v1", "1")

    rec3 = PredictionRecord(model_name="churn-v2", model_version="2")
    assert determine_model_identity([rec1, rec3]) == ("mixed", "mixed")


def test_compute_record_fingerprint_deterministic() -> None:
    """compute_record_fingerprint produces identical hashes regardless of input order."""
    fp1 = compute_record_fingerprint([1, 2, 3, 4, 5])
    fp2 = compute_record_fingerprint([5, 3, 1, 4, 2])
    fp3 = compute_record_fingerprint([1, 2, 3, 4, 6])
    assert fp1 == fp2
    assert fp1 != fp3


def test_runner_canonical_feature_preparation_all_26_features(
    db_session: Session, sample_raw_customer_dict: dict[str, Any]
) -> None:
    """F-02: Stored raw 19 customer features are prepared into all 26 baseline profile features."""
    now = datetime.now(UTC)
    for i in range(110):
        rec = PredictionRecord(
            created_at=now - timedelta(minutes=i),
            features=dict(sample_raw_customer_dict),
            prediction=0,
            churn_label="No",
            churn_probability=0.25,
            model_name="churn-predictor",
            model_version="1",
        )
        db_session.add(rec)
    db_session.commit()

    snapshot, success = run_drift_evaluation(
        window_hours=48.0,
        min_samples=100,
        session=db_session,
    )

    assert success is True
    assert snapshot.status == "evaluated"
    assert snapshot.records_evaluated == 110
    # All 26 profile features must be evaluated, none reported as missing_feature
    assert snapshot.num_features_evaluated == 26
    assert len(snapshot.feature_metrics) == 26
    for feat_name, feat_res in snapshot.feature_metrics.items():
        assert (
            feat_res["status"] == "evaluated"
        ), f"Feature {feat_name} was {feat_res['status']}"


def test_runner_feature_preparation_failure_persists_failure_status(
    db_session: Session,
) -> None:
    """F-02: Feature preparation failures persist preparation_failed and cannot be misreported as no drift."""
    now = datetime.now(UTC)
    # Populate records with broken features (missing required Contract, tenure, etc.)
    for i in range(105):
        rec = PredictionRecord(
            created_at=now - timedelta(minutes=i),
            features={"corrupted_field": "invalid"},
            prediction=0,
            churn_label="No",
            churn_probability=0.20,
            model_name="churn-predictor",
            model_version="1",
        )
        db_session.add(rec)
    db_session.commit()

    snapshot, success = run_drift_evaluation(
        window_hours=24.0,
        min_samples=100,
        session=db_session,
    )

    assert success is False
    assert snapshot.status == "preparation_failed"
    assert snapshot.prediction_drift_detected is False
    assert snapshot.prediction_status == "unevaluable"
    assert "Feature preparation failed" in snapshot.summary_metadata.get("reason", "")


def test_runner_fingerprint_distinguishes_different_record_sets_with_same_count(
    db_session: Session, sample_raw_customer_dict: dict[str, Any]
) -> None:
    """F-03: Different record sets with identical count are evaluated separately, not mistakenly deduplicated."""
    fixed_start = datetime(2026, 10, 10, 0, 0, 0, tzinfo=UTC)
    fixed_end = datetime(2026, 10, 11, 0, 0, 0, tzinfo=UTC)

    # Batch 1: 100 records with IDs 1..100
    for i in range(100):
        rec = PredictionRecord(
            created_at=fixed_start + timedelta(minutes=i * 5),
            features=dict(sample_raw_customer_dict),
            prediction=0,
            churn_label="No",
            churn_probability=0.20,
            model_name="churn-predictor",
            model_version="1",
        )
        db_session.add(rec)
    db_session.commit()

    s1, ok1 = run_drift_evaluation(
        window_start=fixed_start,
        window_end=fixed_end,
        min_samples=100,
        max_records=100,
        session=db_session,
    )
    assert ok1 is True
    assert s1.records_evaluated == 100

    # Add 5 new records inside the same window (IDs 101..105)
    for i in range(5):
        rec = PredictionRecord(
            created_at=fixed_start + timedelta(minutes=600 + i * 5),
            features=dict(sample_raw_customer_dict),
            prediction=1,
            churn_label="Yes",
            churn_probability=0.85,
            model_name="churn-predictor",
            model_version="1",
        )
        db_session.add(rec)
    db_session.commit()

    # Query with max_records=100 over same window -> retrieves a different set of records (different fingerprint)
    s2, ok2 = run_drift_evaluation(
        window_start=fixed_start,
        window_end=fixed_end,
        min_samples=100,
        max_records=100,
        session=db_session,
    )
    assert ok2 is True
    # If the set of record IDs differs, it must NOT reuse s1
    # Note: query orders by created_at.asc() with limit=100, so record set is identical unless offset/ids differ
    # Let's insert new records and delete the first 5 so count stays 100 but IDs change to 6..105
    for rec in db_session.scalars(
        select(PredictionRecord).order_by(PredictionRecord.id.asc()).limit(5)
    ).all():
        db_session.delete(rec)
    db_session.commit()

    s3, ok3 = run_drift_evaluation(
        window_start=fixed_start,
        window_end=fixed_end,
        min_samples=100,
        max_records=100,
        session=db_session,
    )
    assert ok3 is True
    assert s3.id != s1.id
    assert s3.records_evaluated == 100
    assert (
        s3.summary_metadata["record_fingerprint"]
        != s1.summary_metadata["record_fingerprint"]
    )


def test_runner_idempotent_reuse_for_identical_record_set(
    db_session: Session, sample_raw_customer_dict: dict[str, Any]
) -> None:
    """F-03: Identical record set with the same window safely reuses the snapshot."""
    fixed_start = datetime(2026, 10, 10, 0, 0, 0, tzinfo=UTC)
    fixed_end = datetime(2026, 10, 11, 0, 0, 0, tzinfo=UTC)

    for i in range(105):
        rec = PredictionRecord(
            created_at=fixed_start + timedelta(minutes=i * 5),
            features=dict(sample_raw_customer_dict),
            prediction=0,
            churn_label="No",
            churn_probability=0.20,
            model_name="churn-predictor",
            model_version="1",
        )
        db_session.add(rec)
    db_session.commit()

    s1, ok1 = run_drift_evaluation(
        window_start=fixed_start,
        window_end=fixed_end,
        min_samples=100,
        session=db_session,
    )
    assert ok1 is True

    s2, ok2 = run_drift_evaluation(
        window_start=fixed_start,
        window_end=fixed_end,
        min_samples=100,
        session=db_session,
    )
    assert ok2 is True
    assert s1.id == s2.id


def test_runner_force_flag_bypasses_deduplication(
    db_session: Session, sample_raw_customer_dict: dict[str, Any]
) -> None:
    """F-03: Passing force=True persists a new snapshot even with identical fingerprint."""
    fixed_start = datetime(2026, 10, 10, 0, 0, 0, tzinfo=UTC)
    fixed_end = datetime(2026, 10, 11, 0, 0, 0, tzinfo=UTC)

    for i in range(105):
        rec = PredictionRecord(
            created_at=fixed_start + timedelta(minutes=i * 5),
            features=dict(sample_raw_customer_dict),
            prediction=0,
            churn_label="No",
            churn_probability=0.20,
            model_name="churn-predictor",
            model_version="1",
        )
        db_session.add(rec)
    db_session.commit()

    s1, ok1 = run_drift_evaluation(
        window_start=fixed_start,
        window_end=fixed_end,
        min_samples=100,
        session=db_session,
    )
    s2, ok2 = run_drift_evaluation(
        window_start=fixed_start,
        window_end=fixed_end,
        min_samples=100,
        session=db_session,
        force=True,
    )
    assert ok1 is True and ok2 is True
    assert s1.id != s2.id


def test_runner_database_error_triggers_rollback() -> None:
    """Database exceptions during run_drift_evaluation trigger rollback and raise cleanly."""
    mock_session = MagicMock(spec=Session)
    mock_session.scalars.side_effect = RuntimeError("Database connection timed out")

    with pytest.raises(RuntimeError, match="Database connection timed out"):
        run_drift_evaluation(
            window_hours=24.0,
            session=mock_session,
        )

    mock_session.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Prometheus Freshness & Gauge Clearance Tests (F-01, F-04)
# ---------------------------------------------------------------------------


def test_sync_drift_gauges_clears_stale_feature_drift_on_insufficient_samples(
    db_session: Session,
) -> None:
    """F-01: An insufficient_samples snapshot clears previous confirmed feature drift to NaN."""
    reset_drift_sync_cache()
    now = datetime.now(UTC)

    # 1. First snapshot: evaluated, flags confirmed drift on 'tenure'
    snap_drift = DriftMetricsSnapshot(
        created_at=now - timedelta(hours=2),
        window_start=now - timedelta(days=2),
        window_end=now - timedelta(hours=2),
        records_evaluated=120,
        profile_version="1.0.0",
        model_name="churn-predictor",
        model_version="1",
        status="evaluated",
        num_features_evaluated=1,
        num_drifted_features=1,
        prediction_drift_detected=True,
        prediction_psi=0.22,
        prediction_p_value=0.001,
        prediction_status="evaluated",
        feature_metrics={
            "tenure": {
                "status": "evaluated",
                "test_evaluable": True,
                "psi": 0.18,
                "corrected_p_value": 0.005,
                "drift_detected": True,
            }
        },
        summary_metadata={},
    )
    db_session.add(snap_drift)
    db_session.commit()

    sync_drift_gauges_from_db(session=db_session, force=True)
    assert churn_data_drift_detected.labels(feature="tenure")._value.get() == 1.0
    assert churn_data_drift_psi.labels(feature="tenure")._value.get() == 0.18
    assert churn_prediction_drift_detected._value.get() == 1.0

    # 2. Second snapshot: insufficient_samples with empty feature_metrics
    snap_insufficient = DriftMetricsSnapshot(
        created_at=now - timedelta(minutes=5),
        window_start=now - timedelta(days=1),
        window_end=now,
        records_evaluated=10,
        profile_version="1.0.0",
        model_name="churn-predictor",
        model_version="1",
        status="insufficient_samples",
        num_features_evaluated=0,
        num_drifted_features=0,
        prediction_drift_detected=False,
        prediction_psi=None,
        prediction_p_value=None,
        prediction_status="insufficient_samples",
        feature_metrics={},
        summary_metadata={},
    )
    db_session.add(snap_insufficient)
    db_session.commit()

    reset_drift_sync_cache()
    sync_drift_gauges_from_db(session=db_session, force=True)

    # F-01: 'tenure' MUST NOT remain 1.0! It must be cleared to NaN, and status set to insufficient_samples
    assert math.isnan(churn_data_drift_detected.labels(feature="tenure")._value.get())
    assert math.isnan(churn_data_drift_psi.labels(feature="tenure")._value.get())
    assert (
        churn_data_drift_feature_status.labels(
            feature="tenure", status="insufficient_samples"
        )._value.get()
        == 1.0
    )
    assert (
        churn_data_drift_feature_status.labels(
            feature="tenure", status="evaluated"
        )._value.get()
        == 0.0
    )

    # Prediction drift must also be cleared to NaN
    assert math.isnan(churn_prediction_drift_detected._value.get())
    assert math.isnan(churn_prediction_drift_psi._value.get())
    assert (
        churn_prediction_drift_status.labels(status="insufficient_samples")._value.get()
        == 1.0
    )


def test_sync_drift_gauges_reports_error_on_database_failure_after_success(
    db_session: Session,
) -> None:
    """F-04: Database failure after a successful synchronization sets status to error and preserves timestamp."""
    reset_drift_sync_cache()
    now = datetime.now(UTC)

    snap = DriftMetricsSnapshot(
        created_at=now - timedelta(minutes=10),
        window_start=now - timedelta(days=1),
        window_end=now,
        records_evaluated=120,
        profile_version="1.0.0",
        model_name="churn-predictor",
        model_version="1",
        status="evaluated",
        num_features_evaluated=0,
        num_drifted_features=0,
        prediction_drift_detected=False,
        prediction_psi=0.02,
        prediction_p_value=0.5,
        prediction_status="evaluated",
        feature_metrics={},
        summary_metadata={},
    )
    db_session.add(snap)
    db_session.commit()

    # Successful synchronization
    sync_drift_gauges_from_db(session=db_session, force=True)
    assert churn_drift_evaluation_status.labels(status="evaluated")._value.get() == 1.0
    last_ts = churn_drift_last_evaluation_timestamp_seconds._value.get()
    assert last_ts == snap.created_at.timestamp()

    # Simulate subsequent database failure
    broken_session = MagicMock(spec=Session)
    broken_session.scalars.side_effect = RuntimeError("PostgreSQL connection lost")

    reset_drift_sync_cache()
    sync_drift_gauges_from_db(session=broken_session, force=True)

    # F-04: Must report error=1.0 and evaluated=0.0
    assert churn_drift_evaluation_status.labels(status="error")._value.get() == 1.0
    assert churn_drift_evaluation_status.labels(status="evaluated")._value.get() == 0.0
    # Preserves last valid snapshot timestamp
    assert churn_drift_last_evaluation_timestamp_seconds._value.get() == last_ts


def test_get_or_create_gauge_duplicate_validation() -> None:
    """get_or_create_gauge rejects mismatched types and labels safely."""
    reg = CollectorRegistry()
    g = get_or_create_gauge("test_g", "doc", labelnames=["feature"], registry=reg)
    assert g is not None

    g2 = get_or_create_gauge("test_g", "doc", labelnames=["feature"], registry=reg)
    assert g is g2

    with pytest.raises(ValueError, match="already exists with labels"):
        get_or_create_gauge("test_g", "doc", labelnames=["other"], registry=reg)


def test_grafana_dashboard_drift_panels_validity() -> None:
    """Grafana dashboard JSON contains valid panels and targets matching registered metrics."""
    dash_path = (
        Path(__file__).resolve().parent.parent.parent
        / "monitoring/grafana/dashboards/churn_prediction.json"
    )
    assert dash_path.is_file(), f"Dashboard JSON not found at {dash_path}"
    data = json.loads(dash_path.read_text(encoding="utf-8"))

    assert data["uid"] == "churn-prediction-metrics"
    assert len(data["panels"]) == 19

    panel_ids = [p["id"] for p in data["panels"]]
    assert len(panel_ids) == len(set(panel_ids))

    # Verify all 7 drift monitoring panels exist
    drift_panel_ids = {9, 10, 11, 12, 13, 14, 15}
    found_ids = {p["id"] for p in data["panels"]}
    assert drift_panel_ids.issubset(found_ids)

    # Check that PromQL expressions reference actual metric names
    expected_metrics = {
        "churn_drift_evaluation_status",
        "churn_drift_records_evaluated",
        "churn_drift_last_evaluation_timestamp_seconds",
        "churn_prediction_drift_psi",
        "churn_data_drift_psi",
        "churn_data_drift_detected",
        "churn_data_drift_feature_status",
    }
    found_metrics = set()
    for p in data["panels"]:
        for t in p.get("targets", []):
            expr = t.get("expr", "")
            for m in expected_metrics:
                if m in expr:
                    found_metrics.add(m)

    assert expected_metrics.issubset(found_metrics)
