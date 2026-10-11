"""
app/db/models.py
────────────────
SQLAlchemy ORM models for prediction persistence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db.base import Base


class PredictionRecord(Base):
    """Record storing an individual customer churn prediction and feature snapshot."""

    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )
    features: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    churn_label: Mapped[str] = mapped_column(String(10), nullable=False)
    churn_probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<PredictionRecord(id={self.id}, prediction={self.prediction}, "
            f"probability={self.churn_probability:.4f}, model='{self.model_name}:{self.model_version}')>"
        )


class DriftMetricsSnapshot(Base):
    """Record storing an aggregate drift evaluation snapshot."""

    __tablename__ = "drift_metrics_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    window_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    records_evaluated: Mapped[int] = mapped_column(Integer, nullable=False)
    profile_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    num_features_evaluated: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    num_drifted_features: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    prediction_drift_detected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    prediction_psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="not_evaluated"
    )
    feature_metrics: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    summary_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )

    def __repr__(self) -> str:
        return (
            f"<DriftMetricsSnapshot(id={self.id}, status='{self.status}', "
            f"records={self.records_evaluated}, drifted_features={self.num_drifted_features}, "
            f"window='{self.window_start.isoformat()}..{self.window_end.isoformat()}')>"
        )
