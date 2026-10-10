"""
app/main.py
───────────
FastAPI application entrypoint with lifespan model loading.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.predict import router as predict_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ml.loader import load_registered_model

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup and shutdown events."""
    settings = get_settings()
    configure_logging(level=settings.log_level)
    logger.info(
        "Starting Customer Churn Prediction Service in '%s' environment...",
        settings.environment,
    )

    app.state.environment = settings.environment
    app.state.model_name = settings.model_name
    app.state.model_version = str(settings.model_version)
    app.state.model = None
    app.state.model_ready = False

    try:
        model = load_registered_model(
            model_name=settings.model_name,
            model_version=settings.model_version,
            tracking_uri=settings.mlflow_tracking_uri,
        )
        app.state.model = model
        app.state.model_ready = True
        logger.info(
            "Model '%s' (version %s) loaded and ready for inference.",
            settings.model_name,
            settings.model_version,
        )
    except Exception as exc:
        logger.error(
            "Failed to load model '%s' during application startup: %s",
            settings.model_name,
            exc,
            exc_info=True,
        )

    yield

    logger.info("Shutting down Customer Churn Prediction Service...")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready model-serving API for predicting IBM Telco customer churn.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(metrics_router)
