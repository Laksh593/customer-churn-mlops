"""src/monitoring/drift.py
~~~~~~~~~~~~~~~~~~~~~~~
Statistical drift calculation engine (Milestone 11 - Step 2).

Provides deterministic, isolated functions to compare incoming production
inference batches against the precomputed reference baseline profile.

Capabilities:
1. Numerical Features:
   - Evaluates Population Stability Index (PSI) using reference bin boundaries.
   - Handles values below or above reference boundaries without data loss.
   - Tracks missingness changes separately.
   - Safely handles duplicate boundaries, constant features, non-finite values,
     and empty inputs.
   - Evaluates Chi-square goodness-of-fit across binned counts when Cochran's
     statistical assumptions are satisfied.

2. Categorical Features:
   - Evaluates PSI across reference categories with Laplace smoothing.
   - Explicitly pools unseen categories into a dedicated bucket without dropping them.
   - Tracks missingness and unseen category rates.
   - Applies Chi-square goodness-of-fit only when Cochran's assumptions hold.

3. Prediction Probabilities:
   - Compares predicted churn probabilities against validation baseline deciles.
   - Validates that probabilities fall strictly within [0.0, 1.0].
   - Evaluates probability PSI and binned Chi-square goodness-of-fit.

4. Multi-Feature Batch Evaluation:
   - Applies family-wise Bonferroni correction across all evaluable statistical tests.
   - Enforces dual-condition drift detection: drift is flagged if and only if
     both the PSI exceeds the threshold AND the corrected p-value is significant.
   - Explicitly distinguishes non-evaluable states (insufficient_samples,
     invalid_input, empty_input) from genuine findings of no drift.
   - Completely decoupled from FastAPI, PostgreSQL, Prometheus, and MLflow.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare

logger = logging.getLogger(__name__)

DEFAULT_PSI_THRESHOLD: float = 0.10
DEFAULT_ALPHA_THRESHOLD: float = 0.05
DEFAULT_MIN_SAMPLES: int = 30
DEFAULT_EPSILON: float = 1e-4


@dataclass(frozen=True)
class FeatureDriftResult:
    """Individual feature drift assessment result.

    Attributes
    ----------
    feature_name:
        Name of the evaluated feature.
    feature_type:
        Type of the feature ('numerical', 'categorical', 'prediction_probability').
    status:
        Evaluation status ('evaluated', 'insufficient_samples', 'empty_input',
        'invalid_input', 'all_missing', 'missing_feature').
    sample_count:
        Number of valid (non-null) samples evaluated.
    missing_count:
        Number of missing / null samples in the current batch.
    missing_rate:
        Proportion of current batch that was missing.
    missing_rate_delta:
        Difference between current and reference missing rate.
    psi:
        Population Stability Index score, or None if non-evaluable.
    raw_p_value:
        Raw statistical test p-value, or None if assumptions not met.
    corrected_p_value:
        Bonferroni-adjusted p-value across batch comparisons, or None.
    test_statistic:
        Statistical test statistic (e.g. Chi-square), or None.
    test_method:
        Name of statistical test used ('chisquare_binned', 'chisquare_categorical', None).
    test_evaluable:
        Whether statistical hypothesis test assumptions were satisfied.
    drift_detected:
        Boolean indicating whether both PSI and corrected p-value indicate drift.
    unseen_categories:
        List of previously unseen categorical values observed.
    unseen_rate:
        Proportion of observations belonging to unseen categories.
    notes:
        Explanatory notes, assumptions check details, or reason for non-evaluation.
    """

    feature_name: str
    feature_type: str
    status: str
    sample_count: int
    missing_count: int
    missing_rate: float
    missing_rate_delta: float
    psi: float | None
    raw_p_value: float | None
    corrected_p_value: float | None
    test_statistic: float | None
    test_method: str | None
    test_evaluable: bool
    drift_detected: bool
    unseen_categories: list[str] = field(default_factory=list)
    unseen_rate: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a plain serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class DriftEvaluationReport:
    """Consolidated drift report across an inference batch.

    Attributes
    ----------
    status:
        Overall evaluation status ('evaluated', 'insufficient_samples', 'empty_input').
    total_samples:
        Total rows evaluated in the input batch.
    num_features_evaluated:
        Number of individual features evaluated.
    num_drifted_features:
        Count of features where covariate drift was flagged.
    prediction_drift_detected:
        Whether model prediction probability distribution has drifted.
    psi_threshold:
        Configured PSI drift threshold used.
    alpha_threshold:
        Configured significance threshold used.
    num_statistical_tests:
        Number of evaluable statistical tests included in Bonferroni correction.
    feature_results:
        Dictionary mapping feature names to FeatureDriftResult instances.
    prediction_result:
        FeatureDriftResult for model prediction probabilities, or None if not provided.
    metadata:
        Diagnostic and configuration metadata.
    """

    status: str
    total_samples: int
    num_features_evaluated: int
    num_drifted_features: int
    prediction_drift_detected: bool
    psi_threshold: float
    alpha_threshold: float
    num_statistical_tests: int
    feature_results: dict[str, FeatureDriftResult]
    prediction_result: FeatureDriftResult | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a plain serializable dictionary."""
        return {
            "status": self.status,
            "total_samples": self.total_samples,
            "num_features_evaluated": self.num_features_evaluated,
            "num_drifted_features": self.num_drifted_features,
            "prediction_drift_detected": self.prediction_drift_detected,
            "psi_threshold": self.psi_threshold,
            "alpha_threshold": self.alpha_threshold,
            "num_statistical_tests": self.num_statistical_tests,
            "feature_results": {
                k: v.to_dict() for k, v in self.feature_results.items()
            },
            "prediction_result": (
                self.prediction_result.to_dict() if self.prediction_result else None
            ),
            "metadata": self.metadata,
        }


def calculate_psi(
    reference_proportions: list[float] | np.ndarray,
    current_proportions: list[float] | np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Calculate Population Stability Index (PSI) with epsilon smoothing.

    Parameters
    ----------
    reference_proportions:
        Baseline frequency proportions across bins (summing to ~1.0).
    current_proportions:
        Current observed frequency proportions across the same bins (summing to ~1.0).
    epsilon:
        Smoothing parameter to avoid division by zero or log of zero (default: 1e-4).

    Returns
    -------
    float
        Non-negative Population Stability Index score.
    """
    ref = np.asarray(reference_proportions, dtype=float)
    curr = np.asarray(current_proportions, dtype=float)

    if len(ref) != len(curr):
        raise ValueError(
            f"Reference length ({len(ref)}) and current length ({len(curr)}) must match."
        )
    if len(ref) == 0:
        return 0.0

    # Smooth zero or negative proportions and re-normalize
    ref_adj = np.maximum(ref, epsilon)
    ref_norm = ref_adj / np.sum(ref_adj)

    curr_adj = np.maximum(curr, epsilon)
    curr_norm = curr_adj / np.sum(curr_adj)

    psi_val = np.sum((curr_norm - ref_norm) * np.log(curr_norm / ref_norm))
    return float(max(0.0, psi_val))


def _check_cochran_criteria(
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> tuple[bool, str]:
    """Check Cochran (1954) assumptions for valid Chi-square goodness-of-fit approximation.

    Rules:
    1. Degrees of freedom >= 1 (at least 2 categories).
    2. Total sample size >= min_samples.
    3. All expected cell frequencies >= 1.0.
    4. At least 80% of expected cell frequencies >= 5.0.

    Returns
    -------
    tuple of (is_valid: bool, reason: str)
    """
    if len(observed_counts) < 2:
        return False, "Fewer than 2 bins/categories (df < 1)"

    total_obs = np.sum(observed_counts)
    if total_obs < min_samples:
        return (
            False,
            f"Sample size ({int(total_obs)}) below minimum threshold ({min_samples})",
        )

    if np.any(expected_counts < 1.0):
        return False, "At least one expected frequency is below 1.0"

    prop_ge_5 = float(np.mean(expected_counts >= 5.0))
    if prop_ge_5 < 0.8:
        return (
            False,
            f"Only {prop_ge_5:.1%} of expected frequencies >= 5.0 (requires >= 80%)",
        )

    return True, ""


def evaluate_numerical_drift(
    current_series: pd.Series,
    reference_info: dict[str, Any],
    feature_name: str = "numerical_feature",
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    epsilon: float = DEFAULT_EPSILON,
) -> FeatureDriftResult:
    """Evaluate drift for a single numerical feature against its reference profile."""
    total_n = len(current_series)
    if total_n == 0:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            status="empty_input",
            sample_count=0,
            missing_count=0,
            missing_rate=0.0,
            missing_rate_delta=0.0,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes="Input series is empty",
        )

    # Missing value accounting
    missing_count = int(current_series.isna().sum())
    missing_rate = missing_count / total_n
    ref_missing = int(reference_info.get("missing_count", 0))
    ref_count = int(reference_info.get("count", 1))
    ref_missing_rate = (
        ref_missing / (ref_count + ref_missing)
        if (ref_count + ref_missing) > 0
        else 0.0
    )
    missing_delta = missing_rate - ref_missing_rate

    # Filter out NaNs and non-finite values safely
    valid_series = current_series.dropna()
    valid_numeric = pd.to_numeric(valid_series, errors="coerce").dropna()
    valid_finite = valid_numeric[np.isfinite(valid_numeric)]
    valid_n = len(valid_finite)

    if valid_n == 0:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            status="all_missing",
            sample_count=0,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=missing_delta,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes="All values are missing or non-finite",
        )

    if valid_n < min_samples:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            status="insufficient_samples",
            sample_count=valid_n,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=missing_delta,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes=f"Valid samples ({valid_n}) below minimum threshold ({min_samples})",
        )

    ref_edges = reference_info.get("bin_edges", [])
    ref_props = reference_info.get("proportions", [])

    if len(ref_edges) < 2 or len(ref_props) == 0:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            status="invalid_input",
            sample_count=valid_n,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=missing_delta,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes="Reference profile has invalid bin configuration",
        )

    # Binning with boundary extension to capture out-of-range values
    num_bins = len(ref_props)
    if num_bins == 1:
        # Constant reference feature
        curr_props = np.array([1.0])
        curr_counts = np.array([valid_n])
        psi_val = 0.0
        # If reference was constant, check if current values deviate from reference constant
        ref_min = float(reference_info.get("min", 0.0))
        ref_max = float(reference_info.get("max", 0.0))
        curr_min = float(valid_finite.min())
        curr_max = float(valid_finite.max())
        if curr_min != ref_min or curr_max != ref_max:
            # Current values differ from reference constant
            psi_val = 1.0  # Significant constant shift

        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="numerical",
            status="evaluated",
            sample_count=valid_n,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=missing_delta,
            psi=float(round(psi_val, 6)),
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=bool(psi_val >= psi_threshold),
            notes="Constant feature evaluated via boundary check (Chi-square df=0 not evaluable)",
        )

    # Standard multi-bin numerical feature: extend outer edges to -inf and +inf
    eval_bins = [-np.inf] + [float(b) for b in ref_edges[1:-1]] + [np.inf]
    cut_series = pd.cut(valid_finite, bins=eval_bins, include_lowest=True)
    curr_counts = pd.Series(cut_series).value_counts(sort=False).values
    curr_props = curr_counts / float(valid_n)

    psi_val = calculate_psi(ref_props, curr_props, epsilon=epsilon)

    # Chi-square goodness-of-fit test across bins
    expected_counts = np.asarray(ref_props, dtype=float) * float(valid_n)
    is_valid_chi2, chi2_reason = _check_cochran_criteria(
        curr_counts, expected_counts, min_samples=min_samples
    )

    raw_p: float | None = None
    test_stat: float | None = None
    test_method: str | None = None

    if is_valid_chi2:
        res = chisquare(f_obs=curr_counts, f_exp=expected_counts)
        test_stat = float(round(float(res.statistic), 6))
        raw_p = float(round(float(res.pvalue), 8))
        test_method = "chisquare_binned"

    return FeatureDriftResult(
        feature_name=feature_name,
        feature_type="numerical",
        status="evaluated",
        sample_count=valid_n,
        missing_count=missing_count,
        missing_rate=float(round(missing_rate, 6)),
        missing_rate_delta=float(round(missing_delta, 6)),
        psi=float(round(psi_val, 6)),
        raw_p_value=raw_p,
        corrected_p_value=raw_p,
        test_statistic=test_stat,
        test_method=test_method,
        test_evaluable=is_valid_chi2,
        drift_detected=False,
        notes="" if is_valid_chi2 else f"Chi-square not evaluable: {chi2_reason}",
    )


def evaluate_categorical_drift(
    current_series: pd.Series,
    reference_info: dict[str, Any],
    feature_name: str = "categorical_feature",
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    epsilon: float = DEFAULT_EPSILON,
) -> FeatureDriftResult:
    """Evaluate drift for a single categorical feature against its reference profile."""
    total_n = len(current_series)
    if total_n == 0:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            status="empty_input",
            sample_count=0,
            missing_count=0,
            missing_rate=0.0,
            missing_rate_delta=0.0,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes="Input series is empty",
        )

    # Missing value accounting
    missing_count = int(current_series.isna().sum())
    missing_rate = missing_count / total_n
    ref_missing = int(reference_info.get("missing_count", 0))
    ref_count = int(reference_info.get("count", 1))
    ref_missing_rate = (
        ref_missing / (ref_count + ref_missing)
        if (ref_count + ref_missing) > 0
        else 0.0
    )
    missing_delta = missing_rate - ref_missing_rate

    valid_series = current_series.dropna().astype(str)
    valid_n = len(valid_series)

    if valid_n == 0:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            status="all_missing",
            sample_count=0,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=missing_delta,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes="All categorical values are missing",
        )

    if valid_n < min_samples:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            status="insufficient_samples",
            sample_count=valid_n,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=missing_delta,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes=f"Valid samples ({valid_n}) below minimum threshold ({min_samples})",
        )

    ref_freqs = reference_info.get("frequencies", {})
    ref_categories = list(ref_freqs.keys())

    if len(ref_categories) == 0:
        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type="categorical",
            status="invalid_input",
            sample_count=valid_n,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=missing_delta,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes="Reference profile contains no categories",
        )

    # Detect unseen categories
    curr_counts_map = valid_series.value_counts().to_dict()
    unseen_cats = sorted([cat for cat in curr_counts_map if cat not in ref_categories])
    unseen_count = sum(curr_counts_map[c] for c in unseen_cats)
    unseen_rate = unseen_count / float(valid_n)

    # Build aligned category evaluation arrays (including explicit unseen bucket)
    ref_prop_list = [float(ref_freqs.get(c, 0.0)) for c in ref_categories] + [0.0]
    curr_counts_list = [int(curr_counts_map.get(c, 0)) for c in ref_categories] + [
        unseen_count
    ]
    curr_prop_list = [cnt / float(valid_n) for cnt in curr_counts_list]

    # Compute PSI with smoothing
    psi_val = calculate_psi(ref_prop_list, curr_prop_list, epsilon=epsilon)

    # Chi-square goodness-of-fit comparison
    raw_p: float | None = None
    test_stat: float | None = None
    test_method: str | None = None
    is_valid_chi2 = False
    chi2_reason = ""

    if unseen_count > 0:
        is_valid_chi2 = False
        chi2_reason = f"Unseen categories ({len(unseen_cats)}) have 0 expected reference frequency"
    else:
        obs_counts = np.array(
            [curr_counts_map.get(c, 0) for c in ref_categories], dtype=float
        )
        exp_counts = np.array(
            [ref_freqs[c] * float(valid_n) for c in ref_categories], dtype=float
        )
        is_valid_chi2, chi2_reason = _check_cochran_criteria(
            obs_counts, exp_counts, min_samples=min_samples
        )

        if is_valid_chi2:
            res = chisquare(f_obs=obs_counts, f_exp=exp_counts)
            test_stat = float(round(float(res.statistic), 6))
            raw_p = float(round(float(res.pvalue), 8))
            test_method = "chisquare_categorical"

    notes_msg = ""
    if unseen_count > 0:
        notes_msg = (
            f"Observed {len(unseen_cats)} unseen categories ({unseen_rate:.1%}). "
        )
    if not is_valid_chi2:
        notes_msg += f"Chi-square not evaluable: {chi2_reason}"

    return FeatureDriftResult(
        feature_name=feature_name,
        feature_type="categorical",
        status="evaluated",
        sample_count=valid_n,
        missing_count=missing_count,
        missing_rate=float(round(missing_rate, 6)),
        missing_rate_delta=float(round(missing_delta, 6)),
        psi=float(round(psi_val, 6)),
        raw_p_value=raw_p,
        corrected_p_value=raw_p,
        test_statistic=test_stat,
        test_method=test_method,
        test_evaluable=is_valid_chi2,
        drift_detected=False,
        unseen_categories=unseen_cats,
        unseen_rate=float(round(unseen_rate, 6)),
        notes=notes_msg.strip(),
    )


def evaluate_prediction_drift(
    current_probabilities: np.ndarray | pd.Series | list[float],
    reference_info: dict[str, Any],
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    epsilon: float = DEFAULT_EPSILON,
) -> FeatureDriftResult:
    """Evaluate drift in predicted churn probabilities against validation baseline."""
    arr = np.asarray(current_probabilities, dtype=float)
    total_n = len(arr)

    if total_n == 0:
        return FeatureDriftResult(
            feature_name="prediction_probability",
            feature_type="prediction_probability",
            status="empty_input",
            sample_count=0,
            missing_count=0,
            missing_rate=0.0,
            missing_rate_delta=0.0,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes="Probability batch is empty",
        )

    missing_count = int(np.isnan(arr).sum())
    missing_rate = missing_count / total_n
    valid_arr = arr[~np.isnan(arr)]
    valid_n = len(valid_arr)

    if valid_n == 0:
        return FeatureDriftResult(
            feature_name="prediction_probability",
            feature_type="prediction_probability",
            status="all_missing",
            sample_count=0,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=missing_rate,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes="All probability values are NaN",
        )

    # Validate probability range [0.0, 1.0]
    if np.any(valid_arr < 0.0) or np.any(valid_arr > 1.0):
        return FeatureDriftResult(
            feature_name="prediction_probability",
            feature_type="prediction_probability",
            status="invalid_input",
            sample_count=valid_n,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=0.0,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes="Probability values outside valid range [0.0, 1.0]",
        )

    if valid_n < min_samples:
        return FeatureDriftResult(
            feature_name="prediction_probability",
            feature_type="prediction_probability",
            status="insufficient_samples",
            sample_count=valid_n,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=0.0,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes=f"Valid samples ({valid_n}) below minimum threshold ({min_samples})",
        )

    ref_edges = reference_info.get(
        "bin_edges", [round(b, 4) for b in np.linspace(0.0, 1.0, 11)]
    )
    ref_props = reference_info.get("proportions", [])

    if len(ref_props) == 0:
        return FeatureDriftResult(
            feature_name="prediction_probability",
            feature_type="prediction_probability",
            status="invalid_input",
            sample_count=valid_n,
            missing_count=missing_count,
            missing_rate=missing_rate,
            missing_rate_delta=0.0,
            psi=None,
            raw_p_value=None,
            corrected_p_value=None,
            test_statistic=None,
            test_method=None,
            test_evaluable=False,
            drift_detected=False,
            notes="Reference probability profile has no bins",
        )

    cut_bins = pd.cut(valid_arr, bins=ref_edges, include_lowest=True)
    curr_counts = pd.Series(cut_bins).value_counts(sort=False).values
    curr_props = curr_counts / float(valid_n)

    psi_val = calculate_psi(ref_props, curr_props, epsilon=epsilon)

    expected_counts = np.asarray(ref_props, dtype=float) * float(valid_n)
    is_valid_chi2, chi2_reason = _check_cochran_criteria(
        curr_counts, expected_counts, min_samples=min_samples
    )

    raw_p: float | None = None
    test_stat: float | None = None
    test_method: str | None = None

    if is_valid_chi2:
        res = chisquare(f_obs=curr_counts, f_exp=expected_counts)
        test_stat = float(round(float(res.statistic), 6))
        raw_p = float(round(float(res.pvalue), 8))
        test_method = "chisquare_binned"

    return FeatureDriftResult(
        feature_name="prediction_probability",
        feature_type="prediction_probability",
        status="evaluated",
        sample_count=valid_n,
        missing_count=missing_count,
        missing_rate=float(round(missing_rate, 6)),
        missing_rate_delta=0.0,
        psi=float(round(psi_val, 6)),
        raw_p_value=raw_p,
        corrected_p_value=raw_p,
        test_statistic=test_stat,
        test_method=test_method,
        test_evaluable=is_valid_chi2,
        drift_detected=False,
        notes="" if is_valid_chi2 else f"Chi-square not evaluable: {chi2_reason}",
    )


def evaluate_batch_drift(
    current_df: pd.DataFrame,
    reference_profile: dict[str, Any],
    current_probabilities: np.ndarray | pd.Series | list[float] | None = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    alpha_threshold: float = DEFAULT_ALPHA_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    epsilon: float = DEFAULT_EPSILON,
) -> DriftEvaluationReport:
    """Evaluate full batch drift across all features with Bonferroni correction."""
    total_samples = len(current_df)

    if total_samples == 0:
        return DriftEvaluationReport(
            status="empty_input",
            total_samples=0,
            num_features_evaluated=0,
            num_drifted_features=0,
            prediction_drift_detected=False,
            psi_threshold=psi_threshold,
            alpha_threshold=alpha_threshold,
            num_statistical_tests=0,
            feature_results={},
            prediction_result=None,
            metadata={"reason": "Input DataFrame is empty"},
        )

    if total_samples < min_samples:
        report_status = "insufficient_samples"
    else:
        report_status = "evaluated"

    feature_results: dict[str, FeatureDriftResult] = {}

    # 1. Evaluate numerical features
    num_refs = reference_profile.get("numerical_features", {})
    for col, ref_info in num_refs.items():
        if col in current_df.columns:
            res = evaluate_numerical_drift(
                current_series=current_df[col],
                reference_info=ref_info,
                feature_name=col,
                psi_threshold=psi_threshold,
                alpha_threshold=alpha_threshold,
                min_samples=min_samples,
                epsilon=epsilon,
            )
        else:
            res = FeatureDriftResult(
                feature_name=col,
                feature_type="numerical",
                status="missing_feature",
                sample_count=0,
                missing_count=total_samples,
                missing_rate=1.0,
                missing_rate_delta=1.0,
                psi=None,
                raw_p_value=None,
                corrected_p_value=None,
                test_statistic=None,
                test_method=None,
                test_evaluable=False,
                drift_detected=False,
                notes="Feature missing from input DataFrame",
            )
        feature_results[col] = res

    # 2. Evaluate categorical features
    cat_refs = reference_profile.get("categorical_features", {})
    for col, ref_info in cat_refs.items():
        if col in current_df.columns:
            res = evaluate_categorical_drift(
                current_series=current_df[col],
                reference_info=ref_info,
                feature_name=col,
                psi_threshold=psi_threshold,
                alpha_threshold=alpha_threshold,
                min_samples=min_samples,
                epsilon=epsilon,
            )
        else:
            res = FeatureDriftResult(
                feature_name=col,
                feature_type="categorical",
                status="missing_feature",
                sample_count=0,
                missing_count=total_samples,
                missing_rate=1.0,
                missing_rate_delta=1.0,
                psi=None,
                raw_p_value=None,
                corrected_p_value=None,
                test_statistic=None,
                test_method=None,
                test_evaluable=False,
                drift_detected=False,
                notes="Feature missing from input DataFrame",
            )
        feature_results[col] = res

    # 3. Evaluate prediction probability if supplied
    pred_res: FeatureDriftResult | None = None
    pred_ref = reference_profile.get("prediction_probability", {})
    if current_probabilities is not None and len(pred_ref) > 0:
        pred_res = evaluate_prediction_drift(
            current_probabilities=current_probabilities,
            reference_info=pred_ref,
            psi_threshold=psi_threshold,
            alpha_threshold=alpha_threshold,
            min_samples=min_samples,
            epsilon=epsilon,
        )

    # 4. Multi-Comparison Bonferroni Correction
    evaluable_keys: list[str] = [
        k
        for k, v in feature_results.items()
        if v.test_evaluable and v.raw_p_value is not None
    ]
    include_pred_test = (
        pred_res is not None
        and pred_res.test_evaluable
        and pred_res.raw_p_value is not None
    )
    m_tests = len(evaluable_keys) + (1 if include_pred_test else 0)

    # Apply Bonferroni correction and dual-condition drift decision
    updated_feature_results: dict[str, FeatureDriftResult] = {}
    for col, res in feature_results.items():
        if res.status != "evaluated":
            updated_feature_results[col] = res
            continue

        if res.test_evaluable and res.raw_p_value is not None and m_tests > 0:
            corrected_p = float(round(min(1.0, res.raw_p_value * m_tests), 8))
            drift_flag = bool(
                (res.psi is not None and res.psi >= psi_threshold)
                and (corrected_p < alpha_threshold)
            )
        elif res.psi is not None and not res.test_evaluable:
            corrected_p = None
            drift_flag = False
        else:
            corrected_p = None
            drift_flag = False

        updated_res = FeatureDriftResult(
            feature_name=res.feature_name,
            feature_type=res.feature_type,
            status=res.status,
            sample_count=res.sample_count,
            missing_count=res.missing_count,
            missing_rate=res.missing_rate,
            missing_rate_delta=res.missing_rate_delta,
            psi=res.psi,
            raw_p_value=res.raw_p_value,
            corrected_p_value=corrected_p,
            test_statistic=res.test_statistic,
            test_method=res.test_method,
            test_evaluable=res.test_evaluable,
            drift_detected=drift_flag,
            unseen_categories=res.unseen_categories,
            unseen_rate=res.unseen_rate,
            notes=res.notes,
        )
        updated_feature_results[col] = updated_res

    updated_pred_res: FeatureDriftResult | None = None
    if pred_res is not None:
        if (
            pred_res.status == "evaluated"
            and pred_res.test_evaluable
            and pred_res.raw_p_value is not None
            and m_tests > 0
        ):
            corr_p = float(round(min(1.0, pred_res.raw_p_value * m_tests), 8))
            p_drift = bool(
                (pred_res.psi is not None and pred_res.psi >= psi_threshold)
                and (corr_p < alpha_threshold)
            )
        else:
            corr_p = None
            p_drift = False

        updated_pred_res = FeatureDriftResult(
            feature_name=pred_res.feature_name,
            feature_type=pred_res.feature_type,
            status=pred_res.status,
            sample_count=pred_res.sample_count,
            missing_count=pred_res.missing_count,
            missing_rate=pred_res.missing_rate,
            missing_rate_delta=pred_res.missing_rate_delta,
            psi=pred_res.psi,
            raw_p_value=pred_res.raw_p_value,
            corrected_p_value=corr_p,
            test_statistic=pred_res.test_statistic,
            test_method=pred_res.test_method,
            test_evaluable=pred_res.test_evaluable,
            drift_detected=p_drift,
            unseen_categories=pred_res.unseen_categories,
            unseen_rate=pred_res.unseen_rate,
            notes=pred_res.notes,
        )

    num_drifted = sum(
        1 for res in updated_feature_results.values() if res.drift_detected
    )
    pred_drift_detected = updated_pred_res.drift_detected if updated_pred_res else False

    return DriftEvaluationReport(
        status=report_status,
        total_samples=total_samples,
        num_features_evaluated=len(updated_feature_results),
        num_drifted_features=num_drifted,
        prediction_drift_detected=pred_drift_detected,
        psi_threshold=psi_threshold,
        alpha_threshold=alpha_threshold,
        num_statistical_tests=m_tests,
        feature_results=updated_feature_results,
        prediction_result=updated_pred_res,
        metadata={
            "bonferroni_multiplier": m_tests,
            "disclaimer": "Distribution drift detects feature or prediction shift. It does not prove model performance degradation without ground-truth labels.",
        },
    )
