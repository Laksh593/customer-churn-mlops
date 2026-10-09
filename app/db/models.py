"""
app/db/models.py
────────────────
SQLAlchemy ORM models for prediction persistence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String
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
