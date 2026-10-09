"""
scripts/normalize_mlflow_db.py
──────────────────────────────
Normalizes MLflow tracking database file URIs for container portability.

During image build, converts host-specific file URIs (e.g., Windows file:///C:/...)
in the copied SQLite database to target container paths (e.g., file:///app/mlartifacts/...),
validates that physical model artifacts exist on disk, and strictly enforces that
the registered model can be loaded into memory.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def normalize_uri(uri: str, target_base_dir: str) -> str:
    """Rewrite a host-specific file URI to point to target_base_dir.

    Extracts the relative path starting from 'mlartifacts' or 'mlruns'
    and prefixes with target_base_dir.

    Parameters
    ----------
    uri:
        Original file URI (e.g. 'file:///C:/Users/.../mlartifacts/run/artifacts/model').
    target_base_dir:
        Container base directory (e.g. '/app').

    Returns
    -------
    str
        Normalized file URI (e.g. 'file:///app/mlartifacts/run/artifacts/model').
    """
    normalized_base = target_base_dir.rstrip("/").replace("\\", "/")
    for marker in ("mlartifacts", "mlruns"):
        if marker in uri:
            idx = uri.find(marker)
            rel_path = uri[idx:].replace("\\", "/").lstrip("/")
            return f"file://{normalized_base}/{rel_path}"
    return uri


def uri_to_path(uri: str) -> Path:
    """Convert a file URI to a local Path object."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        local_path = url2pathname(parsed.path)
        return Path(local_path)
    return Path(uri)


def normalize_database(
    db_path: Path,
    target_base_dir: str = "/app",
    model_name: str = "churn-predictor",
    model_version: str = "1",
) -> None:
    """Normalize SQLite database URIs, verify artifacts, and strictly test model loading.

    Parameters
    ----------
    db_path:
        Path to SQLite database file.
    target_base_dir:
        Base directory inside the container.
    model_name:
        Expected registered model name.
    model_version:
        Expected registered model version.

    Raises
    ------
    FileNotFoundError:
        If database or model artifacts are missing.
    RuntimeError:
        If the registered model is not found in the database or fails to load.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"MLflow database not found at '{db_path}'.")

    logger.info("Connecting to MLflow database at '%s'...", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 1. Update model_versions table (source and storage_location)
        cursor.execute(
            "SELECT name, version, source, storage_location FROM model_versions"
        )
        mv_rows = cursor.fetchall()
        for name, version, source, storage_location in mv_rows:
            new_source = normalize_uri(source, target_base_dir) if source else source
            new_storage = (
                normalize_uri(storage_location, target_base_dir)
                if storage_location
                else new_source
            )
            cursor.execute(
                "UPDATE model_versions SET source = ?, storage_location = ? WHERE name = ? AND version = ?",
                (new_source, new_storage, name, version),
            )
            logger.info(
                "model_versions: '%s' v%s: source='%s', storage_location='%s'",
                name,
                version,
                new_source,
                new_storage,
            )

        # 2. Update runs table
        cursor.execute("SELECT run_uuid, artifact_uri FROM runs")
        run_rows = cursor.fetchall()
        for run_uuid, artifact_uri in run_rows:
            if artifact_uri:
                new_uri = normalize_uri(artifact_uri, target_base_dir)
                cursor.execute(
                    "UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
                    (new_uri, run_uuid),
                )

        # 3. Update experiments table
        cursor.execute("SELECT experiment_id, artifact_location FROM experiments")
        exp_rows = cursor.fetchall()
        for exp_id, artifact_loc in exp_rows:
            if artifact_loc:
                new_loc = normalize_uri(artifact_loc, target_base_dir)
                cursor.execute(
                    "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
                    (new_loc, exp_id),
                )

        conn.commit()
        logger.info("Database URI normalization committed successfully.")

        # 4. Validate registered model and physical artifacts
        cursor.execute(
            "SELECT source, storage_location FROM model_versions WHERE name = ? AND version = ?",
            (model_name, str(model_version)),
        )
        row = cursor.fetchone()
        if not row:
            raise RuntimeError(
                f"Required registered model '{model_name}' version '{model_version}' "
                f"not found in '{db_path}'."
            )

        norm_source_uri = row[0]
        model_dir = uri_to_path(norm_source_uri)
        logger.info(
            "Verifying model directory at '%s' (from URI '%s')...",
            model_dir,
            norm_source_uri,
        )

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Normalized model directory does not exist on disk: '{model_dir}'."
            )

        mlmodel_file = model_dir / "MLmodel"
        model_pkl = model_dir / "model.pkl"
        if not mlmodel_file.exists():
            raise FileNotFoundError(f"Missing MLmodel file in '{model_dir}'.")
        if not model_pkl.exists():
            raise FileNotFoundError(f"Missing model.pkl file in '{model_dir}'.")

        logger.info(
            "Artifacts verified on disk: Model '%s' v%s files present at '%s'.",
            model_name,
            model_version,
            model_dir,
        )

    finally:
        conn.close()

    # 5. Strict verification of model loading
    import mlflow.sklearn

    tracking_uri = f"sqlite:///{db_path.resolve().as_posix()}"
    model_uri = f"models:/{model_name}/{model_version}"
    logger.info(
        "Performing strict MLflow model load verification from tracking URI '%s'...",
        tracking_uri,
    )
    mlflow.set_tracking_uri(tracking_uri)
    try:
        loaded_model = mlflow.sklearn.load_model(model_uri)
        logger.info(
            "Strict verification passed: successfully loaded model '%s' (%s).",
            model_uri,
            type(loaded_model).__name__,
        )
    except Exception as exc:
        logger.error(
            "Strict verification failed: could not load model '%s': %s",
            model_uri,
            exc,
        )
        raise RuntimeError(
            f"Strict MLflow model load verification failed for '{model_uri}': {exc}"
        ) from exc


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Normalize MLflow SQLite database URIs for container deployment."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("/app/mlflow.db"),
        help="Path to the MLflow SQLite database file.",
    )
    parser.add_argument(
        "--target-base-dir",
        type=str,
        default="/app",
        help="Target base directory in the container (e.g. /app).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="churn-predictor",
        help="Expected registered model name.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="1",
        help="Expected registered model version.",
    )

    args = parser.parse_args()
    try:
        normalize_database(
            db_path=args.db_path,
            target_base_dir=args.target_base_dir,
            model_name=args.model_name,
            model_version=args.model_version,
        )
    except Exception as exc:
        logger.error("Normalization failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
