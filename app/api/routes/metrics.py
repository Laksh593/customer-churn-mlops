"""
app/api/routes/metrics.py
─────────────────────────
Prometheus metrics exposition endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["Monitoring"])


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Expose Prometheus-formatted application metrics.",
    include_in_schema=False,
)
def metrics() -> Response:
    """Expose Prometheus application metrics in standard text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
