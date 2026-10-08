"""
src/training/evaluation.py
──────────────────────────
Evaluation metrics and benchmarking utilities for binary classification models.

Calculates:
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC (using predicted probabilities)
- PR-AUC (Average Precision score using predicted probabilities)
- Confusion Matrix (TN, FP, FN, TP)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Structured container for binary classification performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: dict[str, int]

    def to_dict(self, include_cm: bool = False) -> dict[str, Any]:
        """Convert metrics to a flat dictionary suitable for DataFrame creation.

        Parameters
        ----------
        include_cm:
            If True, includes confusion matrix dictionary.

        Returns
        -------
        dict[str, Any]
        """
        data: dict[str, Any] = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
        }
        if include_cm:
            data["confusion_matrix"] = self.confusion_matrix
        return data

    def summary(self) -> str:
        """Format metrics into a concise human-readable string."""
        cm = self.confusion_matrix
        return (
            f"ROC-AUC: {self.roc_auc:.4f} | PR-AUC: {self.pr_auc:.4f} | "
            f"F1: {self.f1:.4f} | Prec: {self.precision:.4f} | Rec: {self.recall:.4f} | "
            f"Acc: {self.accuracy:.4f} | CM [TN={cm['tn']}, FP={cm['fp']}, "
            f"FN={cm['fn']}, TP={cm['tp']}]"
        )


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
) -> EvaluationResult:
    """Calculate binary classification metrics from predictions and probabilities.

    Parameters
    ----------
    y_true:
        True binary target values (0 or 1).
    y_pred:
        Predicted binary labels (0 or 1).
    y_proba:
        Predicted probabilities for the positive class (1).

    Returns
    -------
    EvaluationResult
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # ROC-AUC and PR-AUC require at least 2 distinct classes in y_true
    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) > 1:
        roc_auc = float(roc_auc_score(y_true_arr, y_proba_arr))
        pr_auc = float(average_precision_score(y_true_arr, y_proba_arr))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    cm_raw = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    cm_dict = {
        "tn": int(cm_raw[0, 0]),
        "fp": int(cm_raw[0, 1]),
        "fn": int(cm_raw[1, 0]),
        "tp": int(cm_raw[1, 1]),
    }

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm_dict,
    )


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> EvaluationResult:
    """Evaluate a trained model or pipeline on a feature matrix and target.

    Uses ``predict()`` for discrete labels and ``predict_proba()`` for
    positive-class probabilities.

    Parameters
    ----------
    model:
        Trained estimator or pipeline implementing ``predict`` and ``predict_proba``.
    X:
        Features matrix or DataFrame.
    y:
        True target vector.

    Returns
    -------
    EvaluationResult
    """
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_proba = proba[:, 1]
        else:
            y_proba = proba.ravel()
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X)
    else:
        y_proba = y_pred

    return evaluate_predictions(y_true=y, y_pred=y_pred, y_proba=y_proba)


def compare_models(results: dict[str, EvaluationResult]) -> pd.DataFrame:
    """Convert a dictionary of model evaluation results to a summary DataFrame.

    Parameters
    ----------
    results:
        Dictionary mapping model names to EvaluationResult instances.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by model name with metric columns sorted descending by roc_auc.
    """
    rows: list[dict[str, Any]] = []
    for name, res in results.items():
        row = {"model": name, **res.to_dict()}
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    if "roc_auc" in df.columns:
        df = df.sort_values(by="roc_auc", ascending=False)
    return df
