"""
tests/unit/test_docker_mlflow.py
────────────────────────────────
Unit tests for Docker MLflow database URI normalization script.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import mlflow.sklearn
import pytest
from scripts.normalize_mlflow_db import (
    normalize_database,
    normalize_uri,
    uri_to_path,
)


def test_normalize_uri_windows_path() -> None:
    """Correctly rewrites Windows file URI to target container base dir."""
    win_uri = "file:///C:/Users/laksh/customer-churn-mlops/mlartifacts/0decfba7/artifacts/model"
    normalized = normalize_uri(win_uri, target_base_dir="/app")
    assert normalized == "file:///app/mlartifacts/0decfba7/artifacts/model"


def test_normalize_uri_backslashes() -> None:
    """Normalizes backslashes to forward slashes."""
    win_uri = "file:///C:\\Users\\test\\mlartifacts\\run123\\model"
    normalized = normalize_uri(win_uri, target_base_dir="/custom_app")
    assert normalized == "file:///custom_app/mlartifacts/run123/model"


def test_normalize_uri_mlruns() -> None:
    """Normalizes mlruns URIs."""
    exp_uri = "file:///C:/Users/test/mlruns/0"
    normalized = normalize_uri(exp_uri, target_base_dir="/app")
    assert normalized == "file:///app/mlruns/0"


def test_normalize_uri_unrelated() -> None:
    """Preserves URIs that do not contain mlartifacts or mlruns."""
    http_uri = "http://remote-server:5000"
    assert normalize_uri(http_uri, target_base_dir="/app") == http_uri


def test_uri_to_path() -> None:
    """Converts file URI to a Path object."""
    path = uri_to_path("file:///app/mlartifacts/model")
    assert isinstance(path, Path)


def test_normalize_database_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normalizes all tables in a temporary SQLite database, verifies artifacts, and tests load."""
    db_file = tmp_path / "test_mlflow.db"
    artifacts_dir = tmp_path / "mlartifacts" / "test_run" / "model"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "MLmodel").write_text("model metadata", encoding="utf-8")
    (artifacts_dir / "model.pkl").write_text("dummy pickle", encoding="utf-8")

    # Set up mock sqlite db
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE model_versions (name TEXT, version TEXT, source TEXT, storage_location TEXT)"
    )
    cur.execute("CREATE TABLE runs (run_uuid TEXT, artifact_uri TEXT)")
    cur.execute("CREATE TABLE experiments (experiment_id TEXT, artifact_location TEXT)")

    cur.execute(
        "INSERT INTO model_versions VALUES (?, ?, ?, ?)",
        (
            "churn-predictor",
            "1",
            "file:///C:/Users/dummy/mlartifacts/test_run/model",
            "file:///C:/Users/dummy/mlartifacts/test_run/model",
        ),
    )
    cur.execute(
        "INSERT INTO runs VALUES (?, ?)",
        ("test_run", "file:///C:/Users/dummy/mlartifacts/test_run"),
    )
    cur.execute(
        "INSERT INTO experiments VALUES (?, ?)",
        ("1", "file:///C:/Users/dummy/mlartifacts"),
    )
    conn.commit()
    conn.close()

    # Simulate mlflow.sklearn.load_model successfully returning a mock model
    monkeypatch.setattr(
        mlflow.sklearn, "load_model", lambda uri: "mock_fitted_pipeline"
    )

    # Run normalization targeting tmp_path as the base directory
    target_base = tmp_path.as_posix()
    normalize_database(
        db_path=db_file,
        target_base_dir=target_base,
        model_name="churn-predictor",
        model_version="1",
    )

    # Verify rows in DB were updated
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    source_row = cur.execute(
        "SELECT source, storage_location FROM model_versions WHERE name='churn-predictor' AND version='1'"
    ).fetchone()
    assert source_row is not None
    assert source_row[0] == f"file://{target_base}/mlartifacts/test_run/model"
    assert source_row[1] == f"file://{target_base}/mlartifacts/test_run/model"

    run_row = cur.execute(
        "SELECT artifact_uri FROM runs WHERE run_uuid='test_run'"
    ).fetchone()
    assert run_row is not None
    assert run_row[0] == f"file://{target_base}/mlartifacts/test_run"

    exp_row = cur.execute(
        "SELECT artifact_location FROM experiments WHERE experiment_id='1'"
    ).fetchone()
    assert exp_row is not None
    assert exp_row[0] == f"file://{target_base}/mlartifacts"
    conn.close()


def test_normalize_database_strict_load_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicitly simulates mlflow.sklearn.load_model failing, asserting RuntimeError is raised."""
    db_file = tmp_path / "test_mlflow.db"
    artifacts_dir = tmp_path / "mlartifacts" / "test_run" / "model"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "MLmodel").write_text("flavor: sklearn", encoding="utf-8")
    (artifacts_dir / "model.pkl").write_text("model weights", encoding="utf-8")

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE model_versions (name TEXT, version TEXT, source TEXT, storage_location TEXT)"
    )
    cur.execute("CREATE TABLE runs (run_uuid TEXT, artifact_uri TEXT)")
    cur.execute("CREATE TABLE experiments (experiment_id TEXT, artifact_location TEXT)")
    cur.execute(
        "INSERT INTO model_versions VALUES (?, ?, ?, ?)",
        (
            "churn-predictor",
            "1",
            "file:///C:/Users/dummy/mlartifacts/test_run/model",
            "file:///C:/Users/dummy/mlartifacts/test_run/model",
        ),
    )
    conn.commit()
    conn.close()

    def mock_fail(uri: str):
        raise RuntimeError("Simulated artifact corruption during load_model")

    monkeypatch.setattr(mlflow.sklearn, "load_model", mock_fail)

    target_base = tmp_path.as_posix()
    with pytest.raises(
        RuntimeError, match="Strict MLflow model load verification failed"
    ):
        normalize_database(
            db_path=db_file,
            target_base_dir=target_base,
            model_name="churn-predictor",
            model_version="1",
        )


def test_normalize_database_missing_model_raises(tmp_path: Path) -> None:
    """Raises RuntimeError when expected model version is not found in database."""
    db_file = tmp_path / "empty_mlflow.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE model_versions (name TEXT, version TEXT, source TEXT, storage_location TEXT)"
    )
    cur.execute("CREATE TABLE runs (run_uuid TEXT, artifact_uri TEXT)")
    cur.execute("CREATE TABLE experiments (experiment_id TEXT, artifact_location TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(
        RuntimeError, match="Required registered model 'churn-predictor'"
    ):
        normalize_database(
            db_path=db_file,
            target_base_dir=str(tmp_path),
            model_name="churn-predictor",
            model_version="1",
        )


def test_normalize_database_missing_artifacts_raises(tmp_path: Path) -> None:
    """Raises FileNotFoundError when physical artifact files do not exist."""
    db_file = tmp_path / "test_mlflow.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE model_versions (name TEXT, version TEXT, source TEXT, storage_location TEXT)"
    )
    cur.execute("CREATE TABLE runs (run_uuid TEXT, artifact_uri TEXT)")
    cur.execute("CREATE TABLE experiments (experiment_id TEXT, artifact_location TEXT)")
    cur.execute(
        "INSERT INTO model_versions VALUES ('churn-predictor', '1', 'file:///C:/Users/dummy/mlartifacts/missing/model', 'file:///C:/Users/dummy/mlartifacts/missing/model')"
    )
    conn.commit()
    conn.close()

    with pytest.raises(FileNotFoundError):
        normalize_database(
            db_path=db_file,
            target_base_dir=str(tmp_path),
            model_name="churn-predictor",
            model_version="1",
        )
