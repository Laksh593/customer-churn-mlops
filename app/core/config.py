"""
app/core/config.py
──────────────────
Centralised application configuration loaded from environment variables.

Uses pydantic-settings so every setting can be overridden via a .env file
or shell environment without touching source code.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings.

    All fields are read from environment variables (case-insensitive).
    Defaults are safe for local development; they should be overridden in
    staging/production via the runtime environment or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Suppress Pydantic's warning about fields whose names start with
        # "model_" (e.g. model_name, model_version). These are application
        # domain fields, not Pydantic internals.
        protected_namespaces=(),
    )

    # ── Runtime environment ───────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = ""

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = ""

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "churn-predictor"
    model_version: str = "1"

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        """Return True when running in the production environment."""
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    Using lru_cache ensures the .env file is read only once per process,
    avoiding repeated I/O on every settings access.
    """
    return Settings()
