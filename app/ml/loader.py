"""
app/ml/loader.py
────────────────
Loads the trained model from the MLflow Model Registry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """Resolve the MLflow tracking URI, ensuring SQLite paths are absolute.

    Parameters
    ----------
    tracking_uri:
        Explicit tracking URI, or None to read from settings or project default.

    Returns
    -------
    str
        Standardised MLflow tracking URI.
    """
    if not tracking_uri:
        settings: Settings = get_settings()
        tracking_uri = settings.mlflow_tracking_uri

    if not tracking_uri:
        # Default to local SQLite database in project root
        db_path = (PROJECT_ROOT / "mlflow.db").resolve()
        return f"sqlite:///{db_path.as_posix()}"

    if tracking_uri.startswith("sqlite:///"):
        sub = tracking_uri[len("sqlite:///") :]
        p = Path(sub)
        if not p.is_absolute():
            abs_p = (PROJECT_ROOT / p).resolve()
            return f"sqlite:///{abs_p.as_posix()}"

    return tracking_uri


def load_registered_model(
    model_name: str | None = None,
    model_version: str | int | None = None,
    tracking_uri: str | None = None,
) -> Any:
    """Load a model from the MLflow Model Registry.

    Parameters
    ----------
    model_name:
        Registered model name (e.g. 'churn-predictor').
    model_version:
        Model version (e.g. '1' or 1).
    tracking_uri:
        Optional tracking URI.

    Returns
    -------
    Any
        The loaded Scikit-learn Pipeline.

    Raises
    ------
    RuntimeError
        If model loading fails.
    """
    settings = get_settings()
    name = model_name or settings.model_name
    version = str(model_version or settings.model_version)
    resolved_uri = resolve_tracking_uri(tracking_uri)

    logger.info(
        "Loading model '%s' version '%s' from MLflow tracking URI '%s'...",
        name,
        version,
        resolved_uri,
    )
    mlflow.set_tracking_uri(resolved_uri)
    model_uri = f"models:/{name}/{version}"

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Successfully loaded model from '%s'", model_uri)
        return model
    except Exception as exc:
        logger.error("Failed to load model from '%s': %s", model_uri, exc)
        raise RuntimeError(f"Failed to load model from {model_uri}: {exc}") from exc
