"""tests/unit/test_drift_readiness_dag.py
───────────────────────────────────────
Unit tests for the scheduled data drift readiness monitoring Airflow DAG.

Tests:
1. DAG import, metadata, schedule, and timezone conventions.
2. Expected DAG ID and task IDs.
3. Task execution handling when API returns READY (HTTP 200).
4. Task execution handling when API returns NOT_READY (HTTP 200) as normal monitoring outcome.
5. Task execution handling of custom window_hours and min_samples parameters.
6. Task execution handling when API returns HTTP errors (e.g. 500 Internal Server Error).
7. Task execution handling when API connection times out.
8. Task execution handling when API returns connection errors (e.g. URLError).
9. Task execution handling when API returns malformed or invalid JSON.
10. Task execution handling when API returns an invalid status string.
11. Confirmation that readiness task does not trigger drift evaluation.
12. Strict validation: missing and null window timestamps.
13. Strict validation: invalid or naive timestamp formats.
14. Strict validation: window end is not later than window start.
15. Strict validation: negative eligible_records or shortfall.
16. Strict validation: non-integer numeric types (booleans, strings, floats).
17. Strict validation: invalid min_samples (<= 0 or inconsistent with requested).
18. Strict validation: inconsistent shortfall value.
19. Strict validation: inconsistent status/count combinations (READY < threshold or NOT_READY >= threshold).
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from dags.drift_readiness_dag import (
    DAG_ID,
    check_drift_readiness_task,
    dag,
)


def _make_http_response(payload: dict, status_code: int = 200) -> MagicMock:
    """Create a mock HTTP response object compatible with urllib.request.urlopen."""
    body_bytes = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.status = status_code
    mock_resp.getcode.return_value = status_code
    mock_resp.read.return_value = body_bytes
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


def _valid_ready_payload() -> dict:
    return {
        "status": "READY",
        "eligible_records": 125,
        "min_samples": 100,
        "shortfall": 0,
        "window_start_utc": "2026-10-10T06:00:00+00:00",
        "window_end_utc": "2026-10-11T06:00:00+00:00",
    }


def _valid_not_ready_payload() -> dict:
    return {
        "status": "NOT_READY",
        "eligible_records": 7,
        "min_samples": 100,
        "shortfall": 93,
        "window_start_utc": "2026-10-10T06:00:00+00:00",
        "window_end_utc": "2026-10-11T06:00:00+00:00",
    }


def test_dag_import_and_metadata() -> None:
    """Verify that the readiness monitoring DAG imports with expected configuration."""
    assert dag is not None
    assert dag.dag_id == DAG_ID
    assert dag.dag_id == "customer_churn_drift_readiness_monitoring"
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert dag.schedule == "0 6 * * *" or dag.schedule_interval == "0 6 * * *"
    assert "drift_readiness" in dag.tags
    assert "monitoring" in dag.tags


def test_dag_task_ids() -> None:
    """Verify the expected task IDs exist in the DAG."""
    expected_task_ids = {"check_drift_readiness"}
    actual_task_ids = set(dag.task_dict.keys())
    assert actual_task_ids == expected_task_ids
    assert len(dag.tasks) == 1


def test_task_execution_ready() -> None:
    """Verify task execution when API reports READY."""
    mock_payload = _valid_ready_payload()
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        summary = check_drift_readiness_task(
            window_hours=24.0,
            min_samples=100,
            api_url="http://api:8000",
        )

        assert summary["readiness_status"] == "READY"
        assert summary["eligible_records"] == 125
        assert summary["min_samples"] == 100
        assert summary["shortfall"] == 0
        assert summary["window_start_utc"] == "2026-10-10T06:00:00+00:00"
        assert summary["window_end_utc"] == "2026-10-11T06:00:00+00:00"


def test_task_execution_not_ready_as_normal_monitoring_outcome() -> None:
    """Verify NOT_READY is treated as a normal monitoring outcome without raising errors."""
    mock_payload = _valid_not_ready_payload()
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        summary = check_drift_readiness_task(
            window_hours=24.0,
            min_samples=100,
            api_url="http://api:8000",
        )

        assert summary["readiness_status"] == "NOT_READY"
        assert summary["eligible_records"] == 7
        assert summary["min_samples"] == 100
        assert summary["shortfall"] == 93


def test_task_execution_zero_records_not_ready() -> None:
    """Verify zero records outcome is handled cleanly as NOT_READY."""
    mock_payload = {
        "status": "NOT_READY",
        "eligible_records": 0,
        "min_samples": 100,
        "shortfall": 100,
        "window_start_utc": "2026-10-10T06:00:00+00:00",
        "window_end_utc": "2026-10-11T06:00:00+00:00",
    }
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        summary = check_drift_readiness_task(
            window_hours=24.0,
            min_samples=100,
        )

        assert summary["readiness_status"] == "NOT_READY"
        assert summary["eligible_records"] == 0
        assert summary["shortfall"] == 100


def test_task_execution_custom_parameters_url_construction() -> None:
    """Verify correct URL parameters are queried when custom window and samples are passed."""
    mock_payload = {
        "status": "READY",
        "eligible_records": 50,
        "min_samples": 50,
        "shortfall": 0,
        "window_start_utc": "2026-10-09T06:00:00+00:00",
        "window_end_utc": "2026-10-11T06:00:00+00:00",
    }
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        check_drift_readiness_task(
            window_hours=48.0,
            min_samples=50,
            api_url="http://custom-api:9000",
        )

        called_req = mock_urlopen.call_args[0][0]
        assert (
            "http://custom-api:9000/monitoring/drift/readiness?" in called_req.full_url
        )
        assert "window_hours=48.0" in called_req.full_url
        assert "min_samples=50" in called_req.full_url


def test_task_execution_http_500_error_raises_runtime_error() -> None:
    """Verify task raises RuntimeError on HTTP 500 error from API/database."""
    http_error = urllib.error.HTTPError(
        url="http://api:8000/monitoring/drift/readiness",
        code=500,
        msg="Internal Server Error",
        hdrs=MagicMock(),
        fp=io.BytesIO(b'{"detail": "Database query error during readiness check"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(RuntimeError, match="Readiness check failed with HTTP 500"):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_timeout_raises_runtime_error() -> None:
    """Verify task raises RuntimeError on request timeout."""
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
        with pytest.raises(RuntimeError, match="timed out"):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_connection_refused_raises_runtime_error() -> None:
    """Verify task raises RuntimeError on network/connection failure."""
    url_error = urllib.error.URLError("Connection refused")
    with patch("urllib.request.urlopen", side_effect=url_error):
        with pytest.raises(RuntimeError, match="API connection error"):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_invalid_json_raises_runtime_error() -> None:
    """Verify task raises RuntimeError when API returns invalid non-JSON body."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = b"<html>502 Bad Gateway</html>"
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Malformed JSON from readiness API"):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_invalid_status_value_raises_runtime_error() -> None:
    """Verify task raises RuntimeError when API returns unexpected status field."""
    mock_payload = _valid_ready_payload()
    mock_payload["status"] = "UNKNOWN_STATUS"
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(
            RuntimeError, match="Invalid readiness status from API: 'UNKNOWN_STATUS'"
        ):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_does_not_invoke_drift_evaluation() -> None:
    """Verify task execution is read-only and never calls full drift evaluation."""
    mock_payload = _valid_ready_payload()
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with patch("scripts.run_drift_check.run_drift_evaluation") as mock_eval:
            summary = check_drift_readiness_task(api_url="http://api:8000")
            assert summary["readiness_status"] == "READY"
            mock_eval.assert_not_called()


# ── Strict Validation Hardening Tests ──────────────────────────────────────────


@pytest.mark.parametrize("missing_field", ["window_start_utc", "window_end_utc"])
def test_task_execution_missing_timestamp_raises(missing_field: str) -> None:
    """Verify task raises RuntimeError when required window timestamp fields are missing or null."""
    mock_payload = _valid_ready_payload()
    mock_payload[missing_field] = None
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(RuntimeError, match=f"Missing or invalid {missing_field}"):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_invalid_timestamp_format_raises() -> None:
    """Verify task raises RuntimeError when timestamp format cannot be parsed."""
    mock_payload = _valid_ready_payload()
    mock_payload["window_start_utc"] = "not-a-valid-date"
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(
            RuntimeError, match="Invalid ISO-8601 format for window_start_utc"
        ):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_naive_timestamp_raises() -> None:
    """Verify task raises RuntimeError when timestamp lacks timezone information."""
    mock_payload = _valid_ready_payload()
    mock_payload["window_start_utc"] = "2026-10-10T06:00:00"  # Naive, no tz
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(
            RuntimeError, match="window_start_utc must include timezone information"
        ):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_window_end_before_start_raises() -> None:
    """Verify task raises RuntimeError when window end is earlier than or equal to window start."""
    mock_payload = _valid_ready_payload()
    mock_payload["window_start_utc"] = "2026-10-11T06:00:00+00:00"
    mock_payload["window_end_utc"] = "2026-10-10T06:00:00+00:00"  # End is before start
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(
            RuntimeError, match="window_end_utc .* must be later than window_start_utc"
        ):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_negative_eligible_records_raises() -> None:
    """Verify task raises RuntimeError when eligible_records is negative."""
    mock_payload = _valid_not_ready_payload()
    mock_payload["eligible_records"] = -5
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="eligible_records must be non-negative"):
            check_drift_readiness_task(api_url="http://api:8000")


@pytest.mark.parametrize("invalid_val", [True, False, "100", 100.5, None])
def test_task_execution_invalid_numeric_type_for_eligible_records_raises(
    invalid_val,
) -> None:
    """Verify booleans, strings, floats, and nulls for eligible_records are rejected."""
    mock_payload = _valid_ready_payload()
    mock_payload["eligible_records"] = invalid_val
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(RuntimeError, match="Missing or invalid eligible_records"):
            check_drift_readiness_task(api_url="http://api:8000")


@pytest.mark.parametrize("invalid_min_samples", [0, -10, True, "100", 50.0])
def test_task_execution_invalid_min_samples_type_or_value_raises(
    invalid_min_samples,
) -> None:
    """Verify non-positive or non-integer min_samples values are rejected."""
    mock_payload = _valid_ready_payload()
    mock_payload["min_samples"] = invalid_min_samples
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(
            RuntimeError,
            match="(Missing or invalid min_samples|min_samples must be positive)",
        ):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_inconsistent_min_samples_threshold_raises() -> None:
    """Verify task raises RuntimeError when API returns min_samples inconsistent with requested."""
    mock_payload = _valid_ready_payload()
    mock_payload["min_samples"] = 50  # Request was for 100
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(
            RuntimeError,
            match="Inconsistent min_samples: API returned 50, expected 100",
        ):
            check_drift_readiness_task(min_samples=100, api_url="http://api:8000")


def test_task_execution_incorrect_shortfall_calculation_raises() -> None:
    """Verify task raises RuntimeError when returned shortfall does not equal max(0, min_samples - eligible_records)."""
    mock_payload = (
        _valid_not_ready_payload()
    )  # min_samples=100, eligible_records=7 -> expected shortfall=93
    mock_payload["shortfall"] = 50  # incorrect shortfall
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(
            RuntimeError,
            match="Inconsistent shortfall in API response: expected 93, got 50",
        ):
            check_drift_readiness_task(api_url="http://api:8000")


def test_task_execution_inconsistent_ready_with_low_count_raises() -> None:
    """Verify task raises RuntimeError when status is READY but eligible_records < min_samples."""
    mock_payload = _valid_ready_payload()
    mock_payload["status"] = "READY"
    mock_payload["eligible_records"] = 50  # 50 < 100
    mock_payload["shortfall"] = 50
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(
            RuntimeError,
            match="Inconsistent response: status is READY but eligible_records",
        ):
            check_drift_readiness_task(min_samples=100, api_url="http://api:8000")


def test_task_execution_inconsistent_not_ready_with_high_count_raises() -> None:
    """Verify task raises RuntimeError when status is NOT_READY but eligible_records >= min_samples."""
    mock_payload = _valid_not_ready_payload()
    mock_payload["status"] = "NOT_READY"
    mock_payload["eligible_records"] = 150  # 150 >= 100
    mock_payload["shortfall"] = 0
    mock_resp = _make_http_response(mock_payload, 200)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(
            RuntimeError,
            match="Inconsistent response: status is NOT_READY but eligible_records",
        ):
            check_drift_readiness_task(min_samples=100, api_url="http://api:8000")
