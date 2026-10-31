# app/security.py
"""
Password hashing (bcrypt) and JWT access/refresh tokens.

Access token: 30 min, returned in the login/refresh response body
(API_Spec.md §1), carried by the client as `Authorization: Bearer`.
Refresh token: 7 days, set as an HttpOnly, SameSite=lax cookie - never
readable by JS, never sent in a response body.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
from jose import JWTError, jwt

from app.config import get_settings

REFRESH_COOKIE_NAME = "refresh_token"


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def _create_token(subject: str, expires_delta: timedelta, token_type: str) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: str) -> tuple[str, int]:
    settings = get_settings()
    expires_delta = timedelta(minutes=settings.access_token_expire_minutes)
    token = _create_token(user_id, expires_delta, "access")
    return token, int(expires_delta.total_seconds())


def create_refresh_token(user_id: str) -> tuple[str, timedelta]:
    settings = get_settings()
    expires_delta = timedelta(days=settings.refresh_token_expire_days)
    token = _create_token(user_id, expires_delta, "refresh")
    return token, expires_delta


def decode_token(token: str, expected_type: str) -> str:
    """Returns the subject (user id) if valid; raises JWTError otherwise."""
    settings = get_settings()
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    if payload.get("type") != expected_type:
        raise JWTError(f"Expected a {expected_type} token.")
    return payload["sub"]
