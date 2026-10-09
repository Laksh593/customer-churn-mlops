"""
app/db/session.py
─────────────────
Database engine, session factory, and FastAPI dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the global SQLAlchemy Engine, initializing if needed."""
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.database_url.strip()
        if not db_url:
            raise RuntimeError(
                "DATABASE_URL is not configured. Set DATABASE_URL in environment or .env."
            )
        connect_args: dict[str, Any] = {}
        if db_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        safe_url = db_url.split("@")[-1] if "@" in db_url else db_url
        logger.info("Initialized SQLAlchemy engine for: %s", safe_url)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the global sessionmaker, initializing if needed."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine,
        )
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    """FastAPI request-scoped dependency providing a database session.

    Guarantees session cleanup upon request completion.
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
    finally:
        session.close()
