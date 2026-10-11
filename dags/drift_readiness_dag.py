"""dags/drift_readiness_dag.py
──────────────────────────
Scheduled Airflow DAG for monitoring customer churn inference traffic drift readiness.

Queries the FastAPI monitoring service via HTTP (GET /monitoring/drift/readiness)
to evaluate whether enough eligible prediction records exist in the evaluation window
for statistical drift detection.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger("airflow.task.drift_readiness")

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except (ImportError, AttributeError):

    class _BaseOperatorShim:
        def __init__(
            self,
            task_id: str,
            python_callable: Any = None,
            op_kwargs: dict[str, Any] | None = None,
            retries: int = 0,
            retry_delay: timedelta | None = None,
            dag: Any = None,
            **kwargs: Any,
        ) -> None:
            self.task_id = task_id
            self.python_callable = python_callable
            self.op_kwargs = op_kwargs or {}
            self.retries = retries
            self.retry_delay = retry_delay
            self.upstream_list: list[_BaseOperatorShim] = []
            self.downstream_list: list[_BaseOperatorShim] = []
            self.dag = dag
            if dag is not None:
                dag.tasks.append(self)
                dag.task_dict[task_id] = self

        @property
        def upstream_task_ids(self) -> set[str]:
            return {t.task_id for t in self.upstream_list}

        @property
        def downstream_task_ids(self) -> set[str]:
            return {t.task_id for t in self.downstream_list}

        def execute(self, context: Any = None) -> Any:
            if self.python_callable:
                return self.python_callable(**self.op_kwargs)
            return None

    class _PythonOperatorShim(_BaseOperatorShim):
        pass

    class _DAGShim:
        def __init__(
            self,
            dag_id: str,
            default_args: dict[str, Any] | None = None,
            schedule: Any = None,
            schedule_interval: Any = None,
            start_date: datetime | None = None,
            catchup: bool = False,
            max_active_runs: int = 1,
            tags: list[str] | None = None,
            **kwargs: Any,
        ) -> None:
            self.dag_id = dag_id
            self.default_args = default_args or {}
            self.schedule = schedule
            self.schedule_interval = schedule_interval
            self.start_date = start_date
            self.catchup = catchup
            self.max_active_runs = max_active_runs
            self.tags = tags or []
            self.tasks: list[_BaseOperatorShim] = []
            self.task_dict: dict[str, _BaseOperatorShim] = {}

        @property
        def task_ids(self) -> list[str]:
            return list(self.task_dict.keys())

        def __enter__(self) -> _DAGShim:
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

    DAG = _DAGShim  # type: ignore[misc,assignment]
    PythonOperator = _PythonOperatorShim  # type: ignore[misc,assignment]


DAG_ID = "customer_churn_drift_readiness_monitoring"

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _is_exact_int(val: Any) -> bool:
    """Check if value is an integer and strictly not a boolean."""
    return isinstance(val, int) and not isinstance(val, bool)


def check_drift_readiness_task(
    window_hours: float = 24.0,
    min_samples: int = 100,
    api_url: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Execute the drift readiness check via HTTP request to the FastAPI monitoring service.

    Queries `GET /monitoring/drift/readiness`.
    Treats `READY` and `NOT_READY` as normal successful monitoring outcomes.
    Raises RuntimeError on HTTP errors, timeouts, malformed responses, invalid types,
    and API/database failures.

    Returns
    -------
    dict[str, Any]
        Structured readiness report containing readiness_status, eligible_records,
        shortfall, and window timestamps.
    """
    base_url = (api_url or os.getenv("CHURN_API_URL") or "http://api:8000").rstrip("/")
    query_params = urllib.parse.urlencode(
        {
            "window_hours": window_hours,
            "min_samples": min_samples,
        }
    )
    target_url = f"{base_url}/monitoring/drift/readiness?{query_params}"
    logger.info("Requesting drift readiness status from API: %s", target_url)

    req = urllib.request.Request(
        target_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Airflow-Drift-Readiness/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            status_code = resp.status if hasattr(resp, "status") else resp.getcode()
            if status_code != 200:
                raise RuntimeError(
                    f"Unexpected HTTP status {status_code} from readiness endpoint"
                )
            raw_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as http_err:
        err_body = ""
        try:
            err_body = http_err.read().decode("utf-8")
        except Exception:
            pass
        logger.error(
            "API returned HTTP error %d: %s (response: %s)",
            http_err.code,
            http_err.reason,
            err_body,
        )
        raise RuntimeError(
            f"Readiness check failed with HTTP {http_err.code}: {http_err.reason}. Response: {err_body}"
        ) from http_err
    except urllib.error.URLError as url_err:
        logger.error("Failed to reach API endpoint %s: %s", target_url, url_err.reason)
        raise RuntimeError(f"API connection error: {url_err.reason}") from url_err
    except TimeoutError as timeout_err:
        logger.error(
            "Request to %s timed out after %.1f seconds", target_url, timeout_seconds
        )
        raise RuntimeError(
            f"API request timed out after {timeout_seconds} seconds"
        ) from timeout_err
    except Exception as exc:
        logger.error(
            "Unexpected error contacting readiness endpoint: %s", exc, exc_info=True
        )
        raise RuntimeError(f"Readiness check request failed: {exc}") from exc

    try:
        data = json.loads(raw_body)
        if not isinstance(data, dict):
            raise ValueError("Response JSON must be an object")
    except Exception as json_exc:
        logger.error(
            "Failed to parse API response JSON: %s (body: %s)", json_exc, raw_body
        )
        raise RuntimeError(
            f"Malformed JSON from readiness API: {json_exc}"
        ) from json_exc

    # 1. Require status to be exactly READY or NOT_READY
    status = data.get("status")
    if status not in ("READY", "NOT_READY"):
        logger.error(
            "Invalid readiness status returned: %s (payload: %s)", status, data
        )
        raise RuntimeError(f"Invalid readiness status from API: {status!r}")

    # 2. Require eligible_records to be a valid non-negative integer
    if "eligible_records" not in data or not _is_exact_int(data["eligible_records"]):
        raise RuntimeError(
            f"Missing or invalid eligible_records in API response: {data.get('eligible_records')!r}"
        )
    eligible_records = data["eligible_records"]
    if eligible_records < 0:
        raise RuntimeError(
            f"eligible_records must be non-negative, got: {eligible_records}"
        )

    # 3. Require min_samples to be a positive integer matching requested threshold
    if "min_samples" not in data or not _is_exact_int(data["min_samples"]):
        raise RuntimeError(
            f"Missing or invalid min_samples in API response: {data.get('min_samples')!r}"
        )
    returned_min_samples = data["min_samples"]
    if returned_min_samples <= 0:
        raise RuntimeError(f"min_samples must be positive, got: {returned_min_samples}")
    if returned_min_samples != min_samples:
        raise RuntimeError(
            f"Inconsistent min_samples: API returned {returned_min_samples}, expected {min_samples}"
        )

    # 4. Require shortfall to be a non-negative integer equal to max(0, min_samples - eligible_records)
    if "shortfall" not in data or not _is_exact_int(data["shortfall"]):
        raise RuntimeError(
            f"Missing or invalid shortfall in API response: {data.get('shortfall')!r}"
        )
    shortfall = data["shortfall"]
    expected_shortfall = max(0, min_samples - eligible_records)
    if shortfall != expected_shortfall:
        raise RuntimeError(
            f"Inconsistent shortfall in API response: expected {expected_shortfall}, got {shortfall}"
        )

    # 5. Require both window timestamps to be present, valid ISO-8601 with timezone, and end > start
    win_start_raw = data.get("window_start_utc")
    win_end_raw = data.get("window_end_utc")
    if not isinstance(win_start_raw, str) or not win_start_raw.strip():
        raise RuntimeError(
            f"Missing or invalid window_start_utc in API response: {win_start_raw!r}"
        )
    if not isinstance(win_end_raw, str) or not win_end_raw.strip():
        raise RuntimeError(
            f"Missing or invalid window_end_utc in API response: {win_end_raw!r}"
        )

    try:
        start_dt = datetime.fromisoformat(win_start_raw)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid ISO-8601 format for window_start_utc ({win_start_raw!r}): {exc}"
        ) from exc

    try:
        end_dt = datetime.fromisoformat(win_end_raw)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid ISO-8601 format for window_end_utc ({win_end_raw!r}): {exc}"
        ) from exc

    if start_dt.tzinfo is None:
        raise RuntimeError(
            f"window_start_utc must include timezone information: {win_start_raw!r}"
        )
    if end_dt.tzinfo is None:
        raise RuntimeError(
            f"window_end_utc must include timezone information: {win_end_raw!r}"
        )

    if end_dt <= start_dt:
        raise RuntimeError(
            f"window_end_utc ({win_end_raw}) must be later than window_start_utc ({win_start_raw})"
        )

    # 6. Reject inconsistent status values
    if status == "READY" and eligible_records < min_samples:
        raise RuntimeError(
            f"Inconsistent response: status is READY but eligible_records ({eligible_records}) < min_samples ({min_samples})"
        )
    if status == "NOT_READY" and eligible_records >= min_samples:
        raise RuntimeError(
            f"Inconsistent response: status is NOT_READY but eligible_records ({eligible_records}) >= min_samples ({min_samples})"
        )

    summary: dict[str, Any] = {
        "readiness_status": status,
        "eligible_records": eligible_records,
        "min_samples": min_samples,
        "shortfall": shortfall,
        "window_start_utc": win_start_raw,
        "window_end_utc": win_end_raw,
        "checked_at_utc": datetime.now(UTC).isoformat(),
    }

    if status == "READY":
        logger.info(
            "DRIFT READINESS: READY. Eligible records: %d >= %d. Window: [%s to %s].",
            eligible_records,
            min_samples,
            win_start_raw,
            win_end_raw,
        )
    else:
        logger.warning(
            "DRIFT READINESS: NOT_READY. Eligible records: %d < %d (shortfall: %d). Window: [%s to %s]. Normal monitoring outcome; no drift evaluation performed.",
            eligible_records,
            min_samples,
            shortfall,
            win_start_raw,
            win_end_raw,
        )

    return summary


with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    description="Customer churn data drift readiness monitoring (daily inference traffic evaluation)",
    schedule="0 6 * * *",  # Configurable daily schedule at 06:00 UTC
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "churn", "monitoring", "drift_readiness"],
) as dag:
    check_readiness = PythonOperator(
        task_id="check_drift_readiness",
        python_callable=check_drift_readiness_task,
        dag=dag,
    )
