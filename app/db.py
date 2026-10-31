# app/db.py
"""
DB engine/session setup.

The engine and SessionLocal are built lazily, on first use, not at
import time. The previous version read os.environ["DATABASE_URL"] at
module scope, which raised KeyError the instant anything imported
app.db - including app.main at FastAPI startup, and any test module
that transitively imports app.models - so CI and local test runs
needed DATABASE_URL set even for tests that never touch the DB. This
version only requires it once get_engine()/get_db() is actually called,
and app/config.py's Settings is the single place env vars are read.
"""

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True)


@lru_cache
def get_session_factory() -> sessionmaker:
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_db():
    """FastAPI dependency - yields a DB session and closes it after use."""
    SessionLocal = get_session_factory()
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
