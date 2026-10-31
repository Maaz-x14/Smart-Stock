# tests/conftest.py
"""
Shared pytest fixtures.

- `db_session`: a real SQLAlchemy session against a fresh SQLite
  file DB per test, with app.models.Base's tables created. SQLite
  stands in for Postgres in unit tests that don't need
  Postgres-only features (UUID server defaults, GENERATED ALWAYS
  columns, JSONB) - those are covered by CI's test-backend job
  running real migrations against the Postgres 15 service container,
  not by this fixture. uuid.uuid4() Python-side defaults on each
  model (see app/models.py's _uuid_pk) work identically on SQLite.
- `client`: a FastAPI TestClient with get_db overridden to use
  db_session, so every request in a test shares one transaction-like
  session instead of hitting whatever DATABASE_URL happens to be set.
- `mock_groq_batch` / `mock_groq_fallback`: patch the two real Groq
  call sites (food_classifier._call_groq_batch, llm_fallback's HTTP
  call) so no test ever makes a network call or needs GROQ_API_KEY.
- `mock_paddle_ocr`: patches ml_service.ocr.model.run_ocr so no test
  ever loads real PaddleOCR model weights or needs paddleocr/
  paddlepaddle installed - this is what lets CI run on
  requirements-ci.txt.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("JWT_SECRET", "test-secret-do-not-use-in-prod")

from app.db import get_db
from app.main import app
from app.models import Base


@pytest.fixture()
def db_session(tmp_path):
    db_path = tmp_path / f"test_{uuid.uuid4().hex}.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # SQLite does not enforce foreign keys unless told to per-connection
    # (unlike Postgres, which always does) - without this, the
    # delete-blocked-by-waste-history behavior (app/services/inventory.
    # delete_item's IntegrityError catch) silently never fires in tests,
    # masking what would be a real constraint violation on Postgres.
    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def client(db_session):
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def mock_groq_batch():
    """Patches the Stage 2 batched is_food/unit/brand call. Yields the
    mock so a test can set .return_value per-case."""
    with patch("ml_service.item_extraction.food_classifier._call_groq_batch") as mock:
        yield mock


@pytest.fixture()
def mock_groq_fallback():
    """Patches Stage 3's Pass 3 LLM fallback HTTP call. pass3_llm()
    has no separate _call_groq function - the httpx.post call is
    inline - so this patches httpx.post at the module level instead.
    Set mock.return_value to an object with .raise_for_status() and
    .json() returning Groq's chat-completion shape."""
    with patch("ml_service.normalization.llm_fallback.httpx.post") as mock:
        yield mock


@pytest.fixture()
def mock_paddle_ocr():
    """Patches Stage 1's OCR entry point - never loads real PaddleOCR
    weights, never requires paddleocr/paddlepaddle to be installed."""
    with patch("ml_service.ocr.model.run_ocr") as mock:
        mock.return_value = {
            "rec_texts": [],
            "rec_scores": [],
            "rec_boxes": [],
            "rec_polys": [],
        }
        yield mock
