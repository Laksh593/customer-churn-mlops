"""
app/core/metrics.py
───────────────────
Prometheus application metrics for the churn prediction service.
"""

from __future__ import annotations

from collections.abc import Sequence

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Histogram


def get_or_create_counter(
    name: str,
    documentation: str,
    labelnames: Sequence[str] = (),
    registry: CollectorRegistry = REGISTRY,
) -> Counter:
    """Get existing counter from registry or create a new one to avoid duplicate registration.

    Note: prometheus-client CollectorRegistry does not expose a public lookup API
    for registered collectors. We check internal `registry._names_to_collectors` and
    validate `_labelnames` strictly to ensure safe idempotent reuse across module
    reloads without duplicate-registration errors or silent configuration mismatches.
    """
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
    """Get existing histogram from registry or create a new one to avoid duplicate registration.

    Note: prometheus-client CollectorRegistry does not expose a public lookup API
    for registered collectors. We check internal `registry._names_to_collectors` and
    validate `_labelnames` and normalized `_upper_bounds` strictly to ensure safe
    idempotent reuse across module reloads without duplicate-registration errors.
    """
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


# 1. Total prediction requests (status: 'success', 'error')
churn_prediction_requests_total = get_or_create_counter(
    name="churn_prediction_requests_total",
    documentation="Total count of customer churn prediction requests.",
    labelnames=["model_name", "model_version", "status"],
)

# 2. Prediction request processing latency in seconds
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

# 3. Successful predictions by label
churn_predictions_total = get_or_create_counter(
    name="churn_predictions_total",
    documentation="Total count of successful churn predictions by outcome label.",
    labelnames=["model_name", "model_version", "churn_label"],
)

# 4. Predicted churn probability distribution
churn_prediction_probability = get_or_create_histogram(
    name="churn_prediction_probability",
    documentation="Predicted customer churn probability distribution.",
    labelnames=["model_name", "model_version"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# 5. Prediction errors by bounded category
churn_prediction_errors_total = get_or_create_counter(
    name="churn_prediction_errors_total",
    documentation="Total count of customer churn prediction errors by category.",
    labelnames=["error_category"],
)
