"""app/core/metrics.py
───────────────────────
Prometheus application and model drift metrics for the churn prediction service.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Sequence

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger("churn.metrics")


def get_or_create_counter(
    name: str,
    documentation: str,
    labelnames: Sequence[str] = (),
    registry: CollectorRegistry = REGISTRY,
) -> Counter:
    """Get existing counter from registry or create a new one to avoid duplicate registration."""
    if name in registry._names_to_collectors:
        collector = registry._names_to_collectors[name]
        if not isinstance(collector, Counter):
            raise TypeError(
                f"Metric '{name}' already exists in registry as {type(collector).__name__}, not Counter"
            )
        existing_labels = tuple(getattr(collector, "_labelnames", ()))
        expected_labels = tuple(labelnames)
        if existing_labels != expected_labels:
            raise ValueError(
                f"Counter '{name}' already exists with labels {existing_labels}, incompatible with requested {expected_labels}"
            )
        return collector
    return Counter(name, documentation, labelnames=labelnames, registry=registry)


def get_or_create_histogram(
    name: str,
    documentation: str,
    labelnames: Sequence[str] = (),
    buckets: Sequence[float] = Histogram.DEFAULT_BUCKETS,
    registry: CollectorRegistry = REGISTRY,
) -> Histogram:
    """Get existing histogram from registry or create a new one to avoid duplicate registration."""
    if name in registry._names_to_collectors:
        collector = registry._names_to_collectors[name]
        if not isinstance(collector, Histogram):
            raise TypeError(
                f"Metric '{name}' already exists in registry as {type(collector).__name__}, not Histogram"
            )
        existing_labels = tuple(getattr(collector, "_labelnames", ()))
        expected_labels = tuple(labelnames)
        if existing_labels != expected_labels:
            raise ValueError(
                f"Histogram '{name}' already exists with labels {existing_labels}, incompatible with requested {expected_labels}"
            )
        existing_bounds = [float(b) for b in getattr(collector, "_upper_bounds", [])]
        expected_bounds = [float(b) for b in buckets]
        if not expected_bounds or expected_bounds[-1] != float("inf"):
            expected_bounds.append(float("inf"))
        if existing_bounds != expected_bounds:
            raise ValueError(
                f"Histogram '{name}' already exists with buckets {existing_bounds}, incompatible with requested {expected_bounds}"
            )
        return collector
    return Histogram(
        name, documentation, labelnames=labelnames, buckets=buckets, registry=registry
    )


def get_or_create_gauge(
    name: str,
    documentation: str,
    labelnames: Sequence[str] = (),
    registry: CollectorRegistry = REGISTRY,
) -> Gauge:
    """Get existing gauge from registry or create a new one to avoid duplicate registration."""
    if name in registry._names_to_collectors:
        collector = registry._names_to_collectors[name]
        if not isinstance(collector, Gauge):
            raise TypeError(
                f"Metric '{name}' already exists in registry as {type(collector).__name__}, not Gauge"
            )
        existing_labels = tuple(getattr(collector, "_labelnames", ()))
        expected_labels = tuple(labelnames)
        if existing_labels != expected_labels:
            raise ValueError(
                f"Gauge '{name}' already exists with labels {existing_labels}, incompatible with requested {expected_labels}"
            )
        return collector
    return Gauge(name, documentation, labelnames=labelnames, registry=registry)


# ---------------------------------------------------------------------------
# 1. Serving & Prediction Metrics (Preserved intact)
# ---------------------------------------------------------------------------

churn_prediction_requests_total = get_or_create_counter(
    name="churn_prediction_requests_total",
    documentation="Total count of customer churn prediction requests.",
    labelnames=["model_name", "model_version", "status"],
)

churn_prediction_latency_seconds = get_or_create_histogram(
    name="churn_prediction_latency_seconds",
    documentation="Latency of customer churn prediction requests in seconds.",
    labelnames=["model_name", "model_version"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)

churn_predictions_total = get_or_create_counter(
    name="churn_predictions_total",
    documentation="Total count of successful churn predictions by outcome label.",
    labelnames=["model_name", "model_version", "churn_label"],
)

churn_prediction_probability = get_or_create_histogram(
    name="churn_prediction_probability",
    documentation="Predicted customer churn probability distribution.",
    labelnames=["model_name", "model_version"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

churn_prediction_errors_total = get_or_create_counter(
    name="churn_prediction_errors_total",
    documentation="Total count of customer churn prediction errors by category.",
    labelnames=["error_category"],
)


# ---------------------------------------------------------------------------
# 2. Persistent Drift Metrics (Milestone 11 - Step 3)
# ---------------------------------------------------------------------------

churn_data_drift_psi = get_or_create_gauge(
    name="churn_data_drift_psi",
    documentation="Population Stability Index (PSI) score for input features against training baseline.",
    labelnames=["feature"],
)

churn_data_drift_detected = get_or_create_gauge(
    name="churn_data_drift_detected",
    documentation=(
        "Covariate drift flag for input feature (1.0 = confirmed drift, 0.0 = no detected drift, "
        "NaN = non-evaluable/insufficient samples)."
    ),
    labelnames=["feature"],
)

churn_data_drift_p_value = get_or_create_gauge(
    name="churn_data_drift_p_value",
    documentation="Bonferroni-corrected p-value for feature statistical drift hypothesis test (NaN if non-evaluable).",
    labelnames=["feature"],
)

churn_data_drift_feature_status = get_or_create_gauge(
    name="churn_data_drift_feature_status",
    documentation=(
        "Evaluation status flag per feature (1.0 for active status, 0.0 otherwise). "
        "Statuses: evaluated, insufficient_samples, missing_feature, unevaluable, preparation_failed."
    ),
    labelnames=["feature", "status"],
)

churn_prediction_drift_psi = get_or_create_gauge(
    name="churn_prediction_drift_psi",
    documentation="Population Stability Index (PSI) for model prediction probability distribution.",
    labelnames=[],
)

churn_prediction_drift_detected = get_or_create_gauge(
    name="churn_prediction_drift_detected",
    documentation=(
        "Prediction probability drift flag (1.0 = confirmed drift, 0.0 = no detected drift, "
        "NaN = non-evaluable/insufficient samples)."
    ),
    labelnames=[],
)

churn_prediction_drift_status = get_or_create_gauge(
    name="churn_prediction_drift_status",
    documentation=(
        "Evaluation status flag for prediction probability distribution (1.0 for active status). "
        "Statuses: evaluated, insufficient_samples, not_evaluated, unevaluable."
    ),
    labelnames=["status"],
)

churn_drift_records_evaluated = get_or_create_gauge(
    name="churn_drift_records_evaluated",
    documentation="Number of prediction records evaluated in the latest persistent drift snapshot.",
    labelnames=[],
)

churn_drift_last_evaluation_timestamp_seconds = get_or_create_gauge(
    name="churn_drift_last_evaluation_timestamp_seconds",
    documentation="UTC epoch timestamp in seconds of the latest persisted drift snapshot.",
    labelnames=[],
)

churn_drift_evaluation_status = get_or_create_gauge(
    name="churn_drift_evaluation_status",
    documentation=(
        "Overall status of latest drift evaluation snapshot (1.0 for active status, 0.0 otherwise). "
        "Statuses: evaluated, insufficient_samples, preparation_failed, no_snapshot, error."
    ),
    labelnames=["status"],
)


# Bounded status enumerations
KNOWN_OVERALL_STATUSES: tuple[str, ...] = (
    "evaluated",
    "insufficient_samples",
    "preparation_failed",
    "no_snapshot",
    "error",
)
KNOWN_PRED_STATUSES: tuple[str, ...] = (
    "evaluated",
    "insufficient_samples",
    "not_evaluated",
    "unevaluable",
)
KNOWN_FEATURE_STATUSES: tuple[str, ...] = (
    "evaluated",
    "insufficient_samples",
    "missing_feature",
    "unevaluable",
    "preparation_failed",
)

# Canonical 26 feature names from reference baseline profile
CANONICAL_FEATURE_NAMES: tuple[str, ...] = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "tenure_group",
    "total_services",
    "is_new_customer",
    "is_month_to_month",
    "has_online_security",
    "has_tech_support",
    "estimated_annual_charges",
)
_REGISTERED_FEATURE_NAMES: set[str] = set(CANONICAL_FEATURE_NAMES)

# TTL cache state for asynchronous exposition
_LAST_DRIFT_SYNC_TIME: float = 0.0
_DRIFT_SYNC_LOCK = threading.Lock()


def reset_drift_sync_cache() -> None:
    """Reset the internal TTL cache timestamp (useful for testing)."""
    global _LAST_DRIFT_SYNC_TIME
    with _DRIFT_SYNC_LOCK:
        _LAST_DRIFT_SYNC_TIME = 0.0


def sync_drift_gauges_from_db(
    session: Session | None = None,
    ttl_seconds: float = 60.0,
    force: bool = False,
) -> None:
    """Synchronize Prometheus gauges from the latest persisted DriftMetricsSnapshot.

    Uses a TTL cache to avoid hitting the database on every Prometheus scrape.
    Catches and logs database exceptions to prevent breaking the /metrics endpoint.
    Guarantees that a newer insufficient-sample or failed snapshot clears previous drift flags.
    """
    global _LAST_DRIFT_SYNC_TIME

    # Check TTL cache before querying database
    if not force:
        now_mono = time.monotonic()
        if (now_mono - _LAST_DRIFT_SYNC_TIME) < ttl_seconds:
            return

    with _DRIFT_SYNC_LOCK:
        if not force:
            now_mono = time.monotonic()
            if (now_mono - _LAST_DRIFT_SYNC_TIME) < ttl_seconds:
                return

        from app.db.models import DriftMetricsSnapshot
        from app.db.session import get_session_factory

        owns_session = session is None
        db: Session | None = None
        try:
            db = get_session_factory()() if owns_session else session
            assert db is not None

            stmt = (
                select(DriftMetricsSnapshot)
                .order_by(
                    DriftMetricsSnapshot.created_at.desc(),
                    DriftMetricsSnapshot.id.desc(),
                )
                .limit(1)
            )
            snapshot = db.scalars(stmt).first()

            if snapshot is None:
                # No snapshot exists yet in PostgreSQL
                churn_drift_records_evaluated.set(0)
                churn_drift_last_evaluation_timestamp_seconds.set(0.0)
                for st in KNOWN_OVERALL_STATUSES:
                    churn_drift_evaluation_status.labels(status=st).set(
                        1.0 if st == "no_snapshot" else 0.0
                    )
                for st in KNOWN_PRED_STATUSES:
                    churn_prediction_drift_status.labels(status=st).set(
                        1.0 if st == "not_evaluated" else 0.0
                    )
                churn_prediction_drift_psi.set(math.nan)
                churn_prediction_drift_detected.set(math.nan)

                # Clear all feature gauges to NaN so no stale drift signals appear
                for feat_name in _REGISTERED_FEATURE_NAMES:
                    churn_data_drift_psi.labels(feature=feat_name).set(math.nan)
                    churn_data_drift_detected.labels(feature=feat_name).set(math.nan)
                    churn_data_drift_p_value.labels(feature=feat_name).set(math.nan)
                    for st in KNOWN_FEATURE_STATUSES:
                        churn_data_drift_feature_status.labels(
                            feature=feat_name, status=st
                        ).set(1.0 if st == "unevaluable" else 0.0)

                _LAST_DRIFT_SYNC_TIME = time.monotonic()
                return

            # Publish overall snapshot metrics
            active_overall = (
                snapshot.status
                if snapshot.status in KNOWN_OVERALL_STATUSES
                else "unevaluable"
            )
            for st in KNOWN_OVERALL_STATUSES:
                churn_drift_evaluation_status.labels(status=st).set(
                    1.0 if st == active_overall else 0.0
                )

            churn_drift_records_evaluated.set(snapshot.records_evaluated)
            ts = snapshot.created_at.timestamp() if snapshot.created_at else 0.0
            churn_drift_last_evaluation_timestamp_seconds.set(ts)

            # Publish prediction probability drift metrics
            raw_pred_status = snapshot.prediction_status
            active_pred_status = (
                raw_pred_status
                if raw_pred_status in KNOWN_PRED_STATUSES
                else "unevaluable"
            )
            for st in KNOWN_PRED_STATUSES:
                churn_prediction_drift_status.labels(status=st).set(
                    1.0 if st == active_pred_status else 0.0
                )

            if active_overall == "evaluated" and active_pred_status == "evaluated":
                psi_val = (
                    snapshot.prediction_psi
                    if snapshot.prediction_psi is not None
                    else math.nan
                )
                churn_prediction_drift_psi.set(psi_val)
                churn_prediction_drift_detected.set(
                    1.0 if snapshot.prediction_drift_detected else 0.0
                )
            else:
                churn_prediction_drift_psi.set(math.nan)
                churn_prediction_drift_detected.set(math.nan)

            # Publish per-feature drift metrics with bounded labels
            feature_data = snapshot.feature_metrics or {}
            _REGISTERED_FEATURE_NAMES.update(feature_data.keys())

            if active_overall != "evaluated":
                # F-01: When snapshot lacks valid evaluation, invalidate previous feature drift values
                fallback_feat_st = (
                    active_overall
                    if active_overall in KNOWN_FEATURE_STATUSES
                    else "unevaluable"
                )
                for feat_name in _REGISTERED_FEATURE_NAMES:
                    churn_data_drift_psi.labels(feature=feat_name).set(math.nan)
                    churn_data_drift_detected.labels(feature=feat_name).set(math.nan)
                    churn_data_drift_p_value.labels(feature=feat_name).set(math.nan)
                    for st in KNOWN_FEATURE_STATUSES:
                        churn_data_drift_feature_status.labels(
                            feature=feat_name, status=st
                        ).set(1.0 if st == fallback_feat_st else 0.0)
            else:
                for feat_name in _REGISTERED_FEATURE_NAMES:
                    if feat_name in feature_data and isinstance(
                        feature_data[feat_name], dict
                    ):
                        feat_metrics = feature_data[feat_name]
                        raw_feat_status = feat_metrics.get("status", "unevaluable")
                        test_evaluable = bool(feat_metrics.get("test_evaluable", False))
                        drift_flag = bool(feat_metrics.get("drift_detected", False))
                        psi_num = feat_metrics.get("psi")
                        p_val = feat_metrics.get("corrected_p_value")
                        if p_val is None:
                            p_val = feat_metrics.get("raw_p_value")

                        if raw_feat_status == "evaluated" and not test_evaluable:
                            active_feat_status = "unevaluable"
                        elif raw_feat_status in KNOWN_FEATURE_STATUSES:
                            active_feat_status = raw_feat_status
                        else:
                            active_feat_status = "unevaluable"

                        for st in KNOWN_FEATURE_STATUSES:
                            churn_data_drift_feature_status.labels(
                                feature=feat_name, status=st
                            ).set(1.0 if st == active_feat_status else 0.0)

                        if active_feat_status == "evaluated":
                            churn_data_drift_psi.labels(feature=feat_name).set(
                                float(psi_num) if psi_num is not None else math.nan
                            )
                            churn_data_drift_detected.labels(feature=feat_name).set(
                                1.0 if drift_flag else 0.0
                            )
                            churn_data_drift_p_value.labels(feature=feat_name).set(
                                float(p_val) if p_val is not None else math.nan
                            )
                        else:
                            churn_data_drift_psi.labels(feature=feat_name).set(
                                float(psi_num)
                                if (
                                    psi_num is not None
                                    and active_feat_status != "insufficient_samples"
                                )
                                else math.nan
                            )
                            churn_data_drift_detected.labels(feature=feat_name).set(
                                math.nan
                            )
                            churn_data_drift_p_value.labels(feature=feat_name).set(
                                math.nan
                            )
                    else:
                        # Known feature absent from evaluated snapshot
                        for st in KNOWN_FEATURE_STATUSES:
                            churn_data_drift_feature_status.labels(
                                feature=feat_name, status=st
                            ).set(1.0 if st == "missing_feature" else 0.0)
                        churn_data_drift_psi.labels(feature=feat_name).set(math.nan)
                        churn_data_drift_detected.labels(feature=feat_name).set(
                            math.nan
                        )
                        churn_data_drift_p_value.labels(feature=feat_name).set(math.nan)

            _LAST_DRIFT_SYNC_TIME = time.monotonic()

        except Exception as exc:
            # F-04: Ensure PostgreSQL sync failures unconditionally update status to error
            logger.warning("Failed to synchronize drift metrics from database: %s", exc)
            for st in KNOWN_OVERALL_STATUSES:
                churn_drift_evaluation_status.labels(status=st).set(
                    1.0 if st == "error" else 0.0
                )
        finally:
            if owns_session and db is not None:
                db.close()
