"""
tests/unit/test_config.py
─────────────────────────
Smoke tests for the application configuration system.

These tests verify that:
- ``Settings`` can be instantiated with default / test values.
- The ``get_settings()`` singleton returns a ``Settings`` object.
- Known field types and constraints are respected.
- The ``is_production`` property works correctly.

No external services, environment files, or real secrets are required.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


class TestSettingsDefaults:
    """Settings can be constructed without any environment variables."""

    def test_instantiation_with_defaults(self) -> None:
        settings = Settings()
        assert settings is not None

    def test_default_environment(self) -> None:
        settings = Settings()
        assert settings.environment == "development"

    def test_default_model_name(self) -> None:
        settings = Settings()
        assert settings.model_name == "churn-predictor"

    def test_default_api_port(self) -> None:
        settings = Settings()
        assert settings.api_port == 8000

    def test_default_log_level(self) -> None:
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_database_url_defaults_to_empty_string(self) -> None:
        settings = Settings()
        assert settings.database_url == ""

    def test_mlflow_tracking_uri_defaults_to_empty_string(self) -> None:
        settings = Settings()
        assert settings.mlflow_tracking_uri == ""


class TestSettingsOverrides:
    """Settings correctly accept values from the environment."""

    def test_environment_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        settings = Settings()
        assert settings.environment == "production"

    def test_model_name_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_NAME", "my-custom-model")
        settings = Settings()
        assert settings.model_name == "my-custom-model"

    def test_api_port_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_PORT", "9000")
        settings = Settings()
        assert settings.api_port == 9000

    def test_invalid_environment_raises(self) -> None:
        with pytest.raises(ValidationError):
            Settings(environment="invalid-env")  # type: ignore[arg-type]


class TestIsProductionProperty:
    """The ``is_production`` convenience property works correctly."""

    def test_is_false_by_default(self) -> None:
        settings = Settings()
        assert settings.is_production is False

    def test_is_true_for_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        settings = Settings()
        assert settings.is_production is True

    def test_is_false_for_staging(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "staging")
        settings = Settings()
        assert settings.is_production is False


class TestGetSettings:
    """The ``get_settings`` cached factory returns the correct type."""

    def test_returns_settings_instance(self) -> None:
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_returns_same_object_on_repeated_calls(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
