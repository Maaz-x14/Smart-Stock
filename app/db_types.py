# app/db_types.py
"""
A UUID column type that works on both Postgres (native UUID) and
SQLite (CHAR(32) hex, for tests - see tests/conftest.py's db_session
fixture). SQLAlchemy's own docs recommend exactly this pattern
(TypeDecorator over CHAR) for dialect-portable UUID columns; it is
not reinventing postgresql.UUID; it is picking its storage based on
the dialect in use, Postgres native UUID in production, CHAR(32) in
SQLite tests, always surfacing a Python uuid.UUID to the app.
"""

from __future__ import annotations

import uuid

from sqlalchemy import CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.types import TypeDecorator


class GUID(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return str(value)
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        return value.hex

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(value)
        return value
