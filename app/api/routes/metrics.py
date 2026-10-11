"""app/api/routes/metrics.py
───────────────────────────────
Prometheus metrics exposition endpoint.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy.orm import Session

from app.core.metrics import sync_drift_gauges_from_db
from app.db.session import get_db

logger = logging.getLogger("churn.api.metrics")
router = APIRouter(tags=["Monitoring"])


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Expose Prometheus-formatted application and drift monitoring metrics.",
    include_in_schema=False,
)
def metrics(db: Annotated[Session, Depends(get_db)]) -> Response:
    """Expose Prometheus application metrics in standard text exposition format.

    Synchronizes the latest drift snapshot gauges with a small TTL cache.
    Database errors are caught and logged so the metrics endpoint remains highly available.
    """
    try:
        sync_drift_gauges_from_db(session=db)
    except Exception as exc:
        logger.warning("Failed to sync drift gauges during metrics scrape: %s", exc)

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
