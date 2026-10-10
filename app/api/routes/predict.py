"""
app/api/routes/predict.py
─────────────────────────
Inference and persistence endpoints instrumented with Prometheus metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.schemas.prediction import CustomerInput, PredictionResponse
from app.core.metrics import (
    churn_prediction_errors_total,
    churn_prediction_latency_seconds,
    churn_prediction_probability,
    churn_prediction_requests_total,
    churn_predictions_total,
)
from app.db.models import PredictionRecord
from app.db.session import get_db
from app.ml.inference import predict_churn

logger = logging.getLogger("churn.api")

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict customer churn",
    description="Predict churn probability and classification for a single customer and store the prediction record in PostgreSQL.",
)
def predict(
    customer_input: CustomerInput,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> PredictionResponse:
    """Predict customer churn for a single customer and persist record in database."""
    start_time = time.perf_counter()
    outcome_status = "error"
    error_category: str | None = None

    raw_model_name = getattr(request.app.state, "model_name", None)
    raw_model_version = getattr(request.app.state, "model_version", None)

    # Response and database record metadata preserve original fallback defaults
    model_name = (
        str(raw_model_name) if raw_model_name is not None else "churn-predictor"
    )
    model_version = str(raw_model_version) if raw_model_version is not None else "1"

    # Metric labels safely identify missing metadata without altering prediction response
    metric_model_name = str(raw_model_name) if raw_model_name is not None else "unknown"
    metric_model_version = (
        str(raw_model_version) if raw_model_version is not None else "unknown"
    )

    try:
        model_ready = bool(getattr(request.app.state, "model_ready", False))
        pipeline = getattr(request.app.state, "model", None)

        if not model_ready or pipeline is None:
            error_category = "model_not_ready"
            logger.warning("Predict request rejected: model is not ready.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready for inference. Please check service health.",
            )

        try:
            prediction, churn_label, churn_probability = predict_churn(
                pipeline=pipeline,
                customer_input=customer_input,
            )
        except Exception as exc:
            error_category = "inference_failure"
            logger.error("Inference execution failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference failed while processing customer record.",
            ) from exc

        # Persist prediction record in database before returning response
        try:
            record = PredictionRecord(
                features=customer_input.model_dump(),
                prediction=prediction,
                churn_label=churn_label,
                churn_probability=round(churn_probability, 4),
                model_name=model_name,
                model_version=model_version,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            logger.info(
                "Persisted prediction record (id=%s) for model '%s:%s'",
                record.id,
                model_name,
                model_version,
            )
        except Exception as db_exc:
            error_category = "database_failure"
            try:
                db.rollback()
            except Exception as rb_exc:
                logger.warning("Database rollback failed: %s", rb_exc)
            logger.error("Database persistence failed: %s", db_exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to record prediction in database.",
            ) from db_exc

        # Record successful prediction metrics (isolated from response delivery)
        outcome_status = "success"
        try:
            churn_predictions_total.labels(
                model_name=metric_model_name,
                model_version=metric_model_version,
                churn_label=churn_label,
            ).inc()
        except Exception as metric_exc:
            logger.warning("Failed to record churn predictions counter: %s", metric_exc)

        try:
            churn_prediction_probability.labels(
                model_name=metric_model_name,
                model_version=metric_model_version,
            ).observe(churn_probability)
        except Exception as metric_exc:
            logger.warning(
                "Failed to record churn prediction probability: %s", metric_exc
            )

        return PredictionResponse(
            prediction=prediction,
            churn_label=churn_label,
            churn_probability=round(churn_probability, 4),
            model_name=model_name,
            model_version=model_version,
        )

    except HTTPException:
        if error_category is None:
            error_category = "unexpected_error"
        raise
    except Exception:
        if error_category is None:
            error_category = "unexpected_error"
        raise
    finally:
        latency = time.perf_counter() - start_time
        try:
            churn_prediction_latency_seconds.labels(
                model_name=metric_model_name,
                model_version=metric_model_version,
            ).observe(latency)
        except Exception as metric_exc:
            logger.warning("Failed to record prediction latency metric: %s", metric_exc)

        try:
            churn_prediction_requests_total.labels(
                model_name=metric_model_name,
                model_version=metric_model_version,
                status=outcome_status,
            ).inc()
        except Exception as metric_exc:
            logger.warning(
                "Failed to record prediction request counter metric: %s", metric_exc
            )

        if error_category is not None:
            try:
                churn_prediction_errors_total.labels(
                    error_category=error_category,
                ).inc()
            except Exception as metric_exc:
                logger.warning(
                    "Failed to record prediction error counter metric: %s", metric_exc
                )
