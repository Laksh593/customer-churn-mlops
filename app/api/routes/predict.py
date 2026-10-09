"""
app/api/routes/predict.py
─────────────────────────
Inference and persistence endpoints.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.schemas.prediction import CustomerInput, PredictionResponse
from app.db.models import PredictionRecord
from app.db.session import get_db
from app.ml.inference import predict_churn

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict customer churn",
    description="Predict churn probability and classification for a single customer.",
)
def predict(
    request: Request,
    customer_input: CustomerInput,
    db: Annotated[Session, Depends(get_db)],
) -> PredictionResponse:
    """Predict customer churn for a single customer and persist record in database."""
    model_ready = bool(getattr(request.app.state, "model_ready", False))
    pipeline = getattr(request.app.state, "model", None)

    if not model_ready or pipeline is None:
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
        logger.error("Inference execution failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed while processing customer record.",
        ) from exc

    model_name = str(getattr(request.app.state, "model_name", "churn-predictor"))
    model_version = str(getattr(request.app.state, "model_version", "1"))

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
        db.rollback()
        logger.error("Database persistence failed: %s", db_exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record prediction in database.",
        ) from db_exc

    return PredictionResponse(
        prediction=prediction,
        churn_label=churn_label,
        churn_probability=round(churn_probability, 4),
        model_name=model_name,
        model_version=model_version,
    )
