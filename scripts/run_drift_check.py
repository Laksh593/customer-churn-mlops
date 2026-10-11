"""scripts/run_drift_check.py
───────────────────────────────
Standalone, repeatable data drift and prediction drift evaluation runner.

Queries recent prediction records from PostgreSQL over a configurable time window,
evaluates distribution drift against the saved baseline profile, and persists
a structured DriftMetricsSnapshot to PostgreSQL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path when invoked directly as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db.models import DriftMetricsSnapshot, PredictionRecord  # noqa: E402
from app.db.session import get_session_factory  # noqa: E402
from src.data.cleaning import fix_dtypes, fix_total_charges  # noqa: E402
from src.data.features import engineer_features  # noqa: E402
from src.monitoring.drift import evaluate_batch_drift  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_drift_check")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for drift evaluation."""
    parser = argparse.ArgumentParser(
        description="Run data and prediction probability drift evaluation against PostgreSQL prediction records.",
    )
    parser.add_argument(
        "--profile-path",
        type=str,
        default="configs/drift_reference_profile.json",
        help="Path to the saved reference baseline JSON profile (default: configs/drift_reference_profile.json).",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Evaluation window duration in hours prior to window-end (default: 24.0).",
    )
    parser.add_argument(
        "--window-start",
        type=str,
        default=None,
        help="Optional ISO-8601 UTC start timestamp (overrides --window-hours).",
    )
    parser.add_argument(
        "--window-end",
        type=str,
        default=None,
        help="Optional ISO-8601 UTC end timestamp (default: current UTC time).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum number of prediction records required to evaluate drift (default: 100).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=10000,
        help="Maximum prediction records to load into memory (default: 10000).",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.1,
        help="PSI drift threshold (default: 0.1).",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.05,
        help="Statistical significance threshold for Bonferroni-corrected tests (default: 0.05).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Persist a new snapshot even if an identical evaluation already exists.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check whether enough eligible prediction records exist for drift evaluation without computing drift or writing snapshots.",
    )
    return parser.parse_args(args)


def load_reference_profile(profile_path: Path | str) -> dict[str, Any]:
    """Load and validate reference profile JSON from disk."""
    path = Path(profile_path)
    if not path.is_file():
        raise FileNotFoundError(f"Reference profile not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Reference profile at {path} must be a JSON object")
    return data


def resolve_window(
    window_hours: float,
    window_start_str: str | None,
    window_end_str: str | None,
) -> tuple[datetime, datetime]:
    """Resolve and validate timezone-aware UTC window start and end timestamps."""
    if window_end_str:
        end_dt = datetime.fromisoformat(window_end_str)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=UTC)
        else:
            end_dt = end_dt.astimezone(UTC)
    else:
        end_dt = datetime.now(UTC)

    if window_start_str:
        start_dt = datetime.fromisoformat(window_start_str)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=UTC)
        else:
            start_dt = start_dt.astimezone(UTC)
    else:
        start_dt = end_dt - timedelta(hours=window_hours)

    if start_dt >= end_dt:
        raise ValueError(
            f"Invalid window: window_start ({start_dt.isoformat()}) must be earlier than window_end ({end_dt.isoformat()})"
        )

    return start_dt, end_dt


def determine_model_identity(records: list[PredictionRecord]) -> tuple[str, str]:
    """Extract model name and version safely from prediction records."""
    if not records:
        return "unknown", "unknown"

    names = {r.model_name for r in records if r.model_name}
    versions = {r.model_version for r in records if r.model_version}

    model_name = (
        names.pop() if len(names) == 1 else ("mixed" if len(names) > 1 else "unknown")
    )
    model_version = (
        versions.pop()
        if len(versions) == 1
        else ("mixed" if len(versions) > 1 else "unknown")
    )

    return model_name, model_version


def compute_record_fingerprint(record_ids: list[int]) -> str:
    """Compute deterministic SHA-256 fingerprint from ordered record IDs (F-03)."""
    if not record_ids:
        return "empty"
    sorted_ids = sorted(record_ids)
    id_str = ",".join(str(rid) for rid in sorted_ids)
    return hashlib.sha256(id_str.encode("utf-8")).hexdigest()[:16]


def check_drift_readiness(
    window_hours: float = 24.0,
    window_start: str | None = None,
    window_end: str | None = None,
    min_samples: int = 100,
    session: Session | None = None,
) -> dict[str, Any]:
    """Check whether enough prediction records exist in the window for drift evaluation.

    Executes an index-only count query without loading prediction rows, feature JSON,
    building DataFrames, or persisting snapshots.
    """
    start_dt, end_dt = resolve_window(window_hours, window_start, window_end)
    logger.info(
        "Checking drift readiness across window [%s to %s] (min_samples=%d)",
        start_dt.isoformat(),
        end_dt.isoformat(),
        min_samples,
    )
    owns_session = session is None
    db: Session | None = None
    try:
        db = get_session_factory()() if owns_session else session
        assert db is not None

        stmt = select(func.count(PredictionRecord.id)).where(
            PredictionRecord.created_at >= start_dt,
            PredictionRecord.created_at <= end_dt,
        )
        eligible_records = int(db.scalar(stmt) or 0)
        shortfall = max(0, min_samples - eligible_records)
        status = "READY" if eligible_records >= min_samples else "NOT_READY"

        return {
            "status": status,
            "eligible_records": eligible_records,
            "min_samples": min_samples,
            "shortfall": shortfall,
            "window_start_utc": start_dt.isoformat(),
            "window_end_utc": end_dt.isoformat(),
        }
    finally:
        if owns_session and db is not None:
            db.close()


def run_drift_evaluation(
    window_hours: float = 24.0,
    window_start: datetime | str | None = None,
    window_end: datetime | str | None = None,
    min_samples: int = 100,
    max_records: int = 10000,
    profile_path: Path | str = "configs/drift_reference_profile.json",
    psi_threshold: float = 0.1,
    alpha_threshold: float = 0.05,
    session: Session | None = None,
    force: bool = False,
) -> tuple[DriftMetricsSnapshot, bool]:
    """Execute drift check against PostgreSQL prediction records.

    Parameters
    ----------
    window_hours:
        Window duration in hours if window_start is not specified.
    window_start:
        Start of the evaluation window (datetime or ISO string).
    window_end:
        End of the evaluation window (datetime or ISO string).
    min_samples:
        Minimum number of records required for full statistical drift evaluation.
    max_records:
        Maximum records to load to prevent memory unbounded growth.
    profile_path:
        Path to reference baseline profile JSON.
    psi_threshold:
        PSI threshold.
    alpha_threshold:
        Alpha significance threshold.
    session:
        Optional SQLAlchemy Session (uses get_session_factory() if None).
    force:
        If True, persists snapshot even if an identical snapshot exists.

    Returns
    -------
    tuple[DriftMetricsSnapshot, bool]:
        The saved snapshot and a boolean indicating whether full evaluation succeeded.
    """
    profile = load_reference_profile(profile_path)
    metadata = profile.get("metadata", {})
    profile_version = str(metadata.get("schema_version", "1.0.0"))

    start_str = (
        window_start.isoformat() if isinstance(window_start, datetime) else window_start
    )
    end_str = window_end.isoformat() if isinstance(window_end, datetime) else window_end
    start_dt, end_dt = resolve_window(window_hours, start_str, end_str)

    logger.info(
        "Evaluating drift across window [%s to %s] (min_samples=%d, max_records=%d)",
        start_dt.isoformat(),
        end_dt.isoformat(),
        min_samples,
        max_records,
    )

    owns_session = session is None
    db = get_session_factory()() if owns_session else session

    try:
        stmt = (
            select(PredictionRecord)
            .where(
                PredictionRecord.created_at >= start_dt,
                PredictionRecord.created_at <= end_dt,
            )
            .order_by(PredictionRecord.created_at.asc())
            .limit(max_records)
        )
        records = list(db.scalars(stmt).all())
        record_count = len(records)
        record_ids = [r.id for r in records if r.id is not None]
        record_fingerprint = compute_record_fingerprint(record_ids)
        logger.info(
            "Retrieved %d prediction records from database (fingerprint=%s)",
            record_count,
            record_fingerprint,
        )

        model_name, model_version = determine_model_identity(records)

        # Handle insufficient samples without misreporting as no drift
        if record_count < min_samples:
            logger.warning(
                "Insufficient records for statistical drift evaluation: found %d, minimum required is %d",
                record_count,
                min_samples,
            )
            snapshot = DriftMetricsSnapshot(
                window_start=start_dt,
                window_end=end_dt,
                records_evaluated=record_count,
                profile_version=profile_version,
                model_name=model_name,
                model_version=model_version,
                status="insufficient_samples",
                num_features_evaluated=0,
                num_drifted_features=0,
                prediction_drift_detected=False,
                prediction_psi=None,
                prediction_p_value=None,
                prediction_status="insufficient_samples",
                feature_metrics={},
                summary_metadata={
                    "reason": f"Insufficient sample count: found {record_count}, minimum required is {min_samples}",
                    "record_fingerprint": record_fingerprint,
                    "min_samples": min_samples,
                    "psi_threshold": psi_threshold,
                    "alpha_threshold": alpha_threshold,
                },
            )
            db.add(snapshot)
            db.commit()
            db.refresh(snapshot)
            logger.info("Persisted insufficient_samples snapshot (id=%d)", snapshot.id)
            return snapshot, False

        # F-03: Deduplication check compares deterministic record fingerprint
        if not force:
            candidate_snapshots = list(
                db.scalars(
                    select(DriftMetricsSnapshot)
                    .where(
                        DriftMetricsSnapshot.window_start == start_dt,
                        DriftMetricsSnapshot.window_end == end_dt,
                        DriftMetricsSnapshot.records_evaluated == record_count,
                        DriftMetricsSnapshot.model_name == model_name,
                        DriftMetricsSnapshot.model_version == model_version,
                        DriftMetricsSnapshot.status == "evaluated",
                    )
                    .order_by(DriftMetricsSnapshot.id.desc())
                ).all()
            )
            for existing in candidate_snapshots:
                cand_meta = existing.summary_metadata or {}
                if cand_meta.get("record_fingerprint") == record_fingerprint:
                    logger.info(
                        "Identical evaluation snapshot already exists (id=%d, fingerprint=%s). Skipping redundant insert.",
                        existing.id,
                        record_fingerprint,
                    )
                    return existing, True

        # Extract raw features and probabilities safely
        features_list: list[dict[str, Any]] = []
        probabilities_list: list[float] = []

        for rec in records:
            feat = rec.features if isinstance(rec.features, dict) else {}
            features_list.append(feat)
            prob = rec.churn_probability
            probabilities_list.append(float(prob) if prob is not None else float("nan"))

        current_df = pd.DataFrame(features_list)
        current_probs = pd.Series(probabilities_list, dtype=float)

        # F-02: Apply canonical cleaning and deterministic feature engineering
        try:
            current_df = fix_dtypes(current_df)
            current_df, _ = fix_total_charges(current_df)
            current_df = engineer_features(current_df)

            # Validate that reference features are present
            expected_features = set(profile.get("numerical_features", {}).keys()) | set(
                profile.get("categorical_features", {}).keys()
            )
            missing_expected = expected_features - set(current_df.columns)
            if missing_expected:
                raise ValueError(
                    f"Required reference features missing after preparation: {sorted(missing_expected)}"
                )
        except Exception as prep_exc:
            logger.error(
                "Feature preparation failed on inference batch: %s",
                prep_exc,
                exc_info=True,
            )
            snapshot = DriftMetricsSnapshot(
                window_start=start_dt,
                window_end=end_dt,
                records_evaluated=record_count,
                profile_version=profile_version,
                model_name=model_name,
                model_version=model_version,
                status="preparation_failed",
                num_features_evaluated=0,
                num_drifted_features=0,
                prediction_drift_detected=False,
                prediction_psi=None,
                prediction_p_value=None,
                prediction_status="unevaluable",
                feature_metrics={},
                summary_metadata={
                    "reason": f"Feature preparation failed: {prep_exc}",
                    "record_fingerprint": record_fingerprint,
                    "min_samples": min_samples,
                    "psi_threshold": psi_threshold,
                    "alpha_threshold": alpha_threshold,
                },
            )
            db.add(snapshot)
            db.commit()
            db.refresh(snapshot)
            return snapshot, False

        # Run statistical drift engine on the complete prepared feature set
        report = evaluate_batch_drift(
            current_df=current_df,
            reference_profile=profile,
            current_probabilities=current_probs,
            psi_threshold=psi_threshold,
            alpha_threshold=alpha_threshold,
            min_samples=min_samples,
        )

        pred_res = report.prediction_result
        pred_psi = pred_res.psi if pred_res else None
        pred_p = (
            pred_res.corrected_p_value
            if pred_res and pred_res.corrected_p_value is not None
            else (pred_res.raw_p_value if pred_res else None)
        )
        pred_status = pred_res.status if pred_res else "not_evaluated"

        snapshot = DriftMetricsSnapshot(
            window_start=start_dt,
            window_end=end_dt,
            records_evaluated=record_count,
            profile_version=profile_version,
            model_name=model_name,
            model_version=model_version,
            status=report.status,
            num_features_evaluated=report.num_features_evaluated,
            num_drifted_features=report.num_drifted_features,
            prediction_drift_detected=report.prediction_drift_detected,
            prediction_psi=pred_psi,
            prediction_p_value=pred_p,
            prediction_status=pred_status,
            feature_metrics={k: v.to_dict() for k, v in report.feature_results.items()},
            summary_metadata={
                "record_fingerprint": record_fingerprint,
                "psi_threshold": psi_threshold,
                "alpha_threshold": alpha_threshold,
                "num_statistical_tests": report.num_statistical_tests,
                "prediction_result": pred_res.to_dict() if pred_res else None,
                "metadata": report.metadata,
            },
        )
        db.add(snapshot)
        db.commit()
        db.refresh(snapshot)
        logger.info(
            "Persisted drift snapshot (id=%d, status='%s', drifted_features=%d, prediction_drift=%s)",
            snapshot.id,
            snapshot.status,
            snapshot.num_drifted_features,
            snapshot.prediction_drift_detected,
        )
        return snapshot, True

    except Exception:
        db.rollback()
        raise
    finally:
        if owns_session:
            db.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for drift evaluation or readiness check."""
    args = parse_args(argv)
    if args.check_only:
        try:
            res = check_drift_readiness(
                window_hours=args.window_hours,
                window_start=args.window_start,
                window_end=args.window_end,
                min_samples=args.min_samples,
            )
            print(json.dumps(res, indent=2))
            if res["status"] == "READY":
                logger.info(
                    "Drift readiness check PASSED: %d eligible records found (minimum required: %d).",
                    res["eligible_records"],
                    res["min_samples"],
                )
                return 0
            logger.warning(
                "Drift readiness check FAILED: %d eligible records found (minimum required: %d, shortfall: %d).",
                res["eligible_records"],
                res["min_samples"],
                res["shortfall"],
            )
            return 1
        except Exception as exc:
            logger.error(
                "Drift readiness check failed with database error: %s",
                exc,
                exc_info=True,
            )
            return 2

    try:
        snapshot, success = run_drift_evaluation(
            window_hours=args.window_hours,
            window_start=args.window_start,
            window_end=args.window_end,
            min_samples=args.min_samples,
            max_records=args.max_records,
            profile_path=args.profile_path,
            psi_threshold=args.psi_threshold,
            alpha_threshold=args.alpha_threshold,
            force=args.force,
        )
        if not success:
            logger.error(
                "Drift check completed with incomplete status: %s (records: %d, min_required: %d)",
                snapshot.status,
                snapshot.records_evaluated,
                args.min_samples,
            )
            return 1

        print(
            f"Drift check succeeded: Snapshot ID={snapshot.id}, Status={snapshot.status}, "
            f"Records={snapshot.records_evaluated}, Drifted Features={snapshot.num_drifted_features}, "
            f"Prediction Drift={snapshot.prediction_drift_detected}"
        )
        return 0

    except Exception as exc:
        logger.error("Drift check failed with error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
