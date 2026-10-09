"""
app/api/routes/predict.py
─────────────────────────
Inference endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.api.schemas.prediction import CustomerInput, PredictionResponse
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
def predict(request: Request, customer_input: CustomerInput) -> PredictionResponse:
    """Predict customer churn for a single customer."""
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

    return PredictionResponse(
        prediction=prediction,
        churn_label=churn_label,
        churn_probability=round(churn_probability, 4),
        model_name=str(getattr(request.app.state, "model_name", "churn-predictor")),
        model_version=str(getattr(request.app.state, "model_version", "1")),
    )
