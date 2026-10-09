"""
app/api/routes/health.py
────────────────────────
Health and readiness check endpoints.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health status response schema."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["healthy"] = Field(
        ..., description="Application operational status"
    )
    model_status: Literal["ready", "not_ready"] = Field(
        ..., description="Loaded model readiness status"
    )
    model_name: str | None = Field(None, description="Name of loaded model")
    model_version: str | None = Field(None, description="Version of loaded model")
    environment: str = Field(..., description="Current deployment environment")


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """Check application health and model readiness status."""
    is_ready = bool(getattr(request.app.state, "model_ready", False))
    return HealthResponse(
        status="healthy",
        model_status="ready" if is_ready else "not_ready",
        model_name=getattr(request.app.state, "model_name", None),
        model_version=getattr(request.app.state, "model_version", None),
        environment=getattr(request.app.state, "environment", "development"),
    )


@router.get("/", response_model=HealthResponse)
def root(request: Request) -> HealthResponse:
    """Root endpoint delegating to health check."""
    return get_health(request)
