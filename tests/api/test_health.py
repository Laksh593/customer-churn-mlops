"""
tests/api/test_health.py
────────────────────────
Unit tests for the GET /health and GET / endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    """Create a FastAPI test client without running external lifespan side effects."""
    # We do not use the lifespan context manager here so unit tests don't load external models
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_when_model_ready(client: TestClient) -> None:
    """GET /health reports model_status as ready when model is loaded."""
    client.app.state.model_ready = True
    client.app.state.model_name = "churn-predictor"
    client.app.state.model_version = "1"
    client.app.state.environment = "development"

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_status"] == "ready"
    assert data["model_name"] == "churn-predictor"
    assert data["model_version"] == "1"
    assert data["environment"] == "development"


def test_health_when_model_not_ready(client: TestClient) -> None:
    """GET /health reports model_status as not_ready when model is absent."""
    client.app.state.model_ready = False
    client.app.state.model_name = None
    client.app.state.model_version = None

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_status"] == "not_ready"
    assert data["model_name"] is None
    assert data["model_version"] is None


def test_root_delegates_to_health(client: TestClient) -> None:
    """GET / returns the health status response."""
    client.app.state.model_ready = True
    client.app.state.model_name = "churn-predictor"
    client.app.state.model_version = "1"

    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_status"] == "ready"
