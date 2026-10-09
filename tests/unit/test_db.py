"""
tests/unit/test_db.py
─────────────────────
Unit tests for database models, session management, and transaction handling.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import PredictionRecord
from app.db.session import get_db


@pytest.fixture()
def sqlite_engine():
    """Create an isolated in-memory SQLite engine with StaticPool for unit tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session(sqlite_engine) -> Session:
    """Create a new database session bound to the in-memory engine."""
    session_factory = sessionmaker(
        bind=sqlite_engine,
        autocommit=False,
        autoflush=False,
    )
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_prediction_record_creation(db_session: Session) -> None:
    """PredictionRecord correctly stores features JSON, prediction, and metadata."""
    sample_features = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "tenure": 12,
        "MonthlyCharges": 65.5,
    }
    record = PredictionRecord(
        features=sample_features,
        prediction=1,
        churn_label="Yes",
        churn_probability=0.7421,
        model_name="churn-predictor",
        model_version="1",
    )
    db_session.add(record)
    db_session.commit()
    db_session.refresh(record)

    assert record.id is not None
    assert record.id > 0
    assert record.created_at is not None
    assert record.features == sample_features
    assert record.prediction == 1
    assert record.churn_label == "Yes"
    assert record.churn_probability == 0.7421
    assert record.model_name == "churn-predictor"
    assert record.model_version == "1"

    # Query back
    stmt = select(PredictionRecord).where(PredictionRecord.id == record.id)
    retrieved = db_session.execute(stmt).scalar_one()
    assert retrieved.id == record.id
    assert repr(retrieved).startswith("<PredictionRecord")


def test_transaction_rollback_on_failure(db_session: Session) -> None:
    """Failed transactions roll back and do not commit partial data."""
    record = PredictionRecord(
        features={"tenure": 5},
        prediction=0,
        churn_label="No",
        churn_probability=0.2,
        model_name="churn-predictor",
        model_version="1",
    )
    db_session.add(record)

    # Simulate an error causing rollback
    db_session.rollback()

    stmt = select(PredictionRecord)
    results = db_session.execute(stmt).scalars().all()
    assert len(results) == 0


def test_session_generator_cleanup(
    sqlite_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_db generator properly closes sessions after yielding."""
    session_factory = sessionmaker(
        bind=sqlite_engine,
        autocommit=False,
        autoflush=False,
    )
    monkeypatch.setattr("app.db.session.get_session_factory", lambda: session_factory)

    gen = get_db()
    session = next(gen)
    assert isinstance(session, Session)

    with patch.object(session, "close", wraps=session.close) as mock_close:
        with pytest.raises(StopIteration):
            next(gen)
        mock_close.assert_called_once()
