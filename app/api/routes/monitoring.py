"""app/api/routes/monitoring.py
───────────────────────────────
Model and data drift monitoring readiness endpoints.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from scripts.run_drift_check import check_drift_readiness
from sqlalchemy.orm import Session

from app.db.session import get_db

logger = logging.getLogger("churn.api.monitoring")
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


class DriftReadinessResponse(BaseModel):
    """Response schema for drift evaluation readiness check."""

    status: str = Field(..., description="Readiness status: READY or NOT_READY")
    eligible_records: int = Field(
        ..., description="Number of eligible prediction records in window"
    )
    min_samples: int = Field(
        ..., description="Minimum samples required for drift evaluation"
    )
    shortfall: int = Field(
        ..., description="Shortfall in sample count below min_samples"
    )
    window_start_utc: str = Field(
        ..., description="ISO-8601 UTC window start timestamp"
    )
    window_end_utc: str = Field(..., description="ISO-8601 UTC window end timestamp")


@router.get(
    "/drift/readiness",
    response_model=DriftReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Check drift evaluation readiness",
    description="Check whether enough prediction records exist in the evaluation window for statistical drift evaluation.",
)
def get_drift_readiness(
    db: Annotated[Session, Depends(get_db)],
    window_hours: float = Query(
        24.0, ge=0.1, le=8760.0, description="Window duration in hours"
    ),
    window_start: str | None = Query(
        None, description="Optional ISO-8601 UTC window start timestamp"
    ),
    window_end: str | None = Query(
        None, description="Optional ISO-8601 UTC window end timestamp"
    ),
    min_samples: int = Query(
        100, ge=1, le=1000000, description="Minimum records threshold"
    ),
) -> dict[str, Any]:
    """Execute a lightweight, index-only readiness count query against PostgreSQL.

    Does not retrieve feature payloads, compute drift statistics, write snapshots,
    or alter Prometheus gauges.
    """
    try:
        result = check_drift_readiness(
            window_hours=window_hours,
            window_start=window_start,
            window_end=window_end,
            min_samples=min_samples,
            session=db,
        )
        return result
    except Exception as exc:
        logger.error(
            "Drift readiness evaluation failed with database error: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database query error during readiness check",
        ) from exc
