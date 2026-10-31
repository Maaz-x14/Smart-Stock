# app/deps.py
"""
Shared FastAPI dependencies: current-user auth, and a simple in-memory
per-user rate limiter (API_Spec.md §8).

Rate limiter is intentionally in-memory (a dict of sliding windows) -
documented in the plan as not spanning multiple instances or
surviving a restart; fine for a single-instance deployment, and noted
as a known limit rather than quietly built on the assumption it
scales.

No `from __future__ import annotations` here, deliberately: FastAPI
resolves a route FUNCTION's string annotations fine via
typing.get_type_hints(), but RateLimiter.__call__ is a bound method on
a class instance used as a dependency, and under postponed evaluation
its `request: Request` parameter was not recognized as the special
Request injection - FastAPI treated it as a query param named
`request` instead, surfacing as a spurious 422 "Field required" on
every route using RateLimiter. Confirmed by reproducing and fixing
this exact failure while building the test suite.
"""

import time
from collections import defaultdict

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.orm import Session

from app.db import get_db
from app.errors import rate_limited, unauthorized
from app.models import User
from app.security import decode_token

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_current_user(
    token: str | None = Depends(_oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    if token is None:
        raise unauthorized()
    try:
        user_id = decode_token(token, expected_type="access")
    except JWTError:
        raise unauthorized("Access token expired or invalid.") from None

    user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
    if user is None:
        raise unauthorized("User not found or deactivated.")
    return user


# -- Rate limiting -------------------------------------------------------------

_WINDOW_SECONDS = 3600

# bucket key -> list[timestamp]; swept lazily on each check.
_buckets: dict[str, list[float]] = defaultdict(list)

_LIMITS = {
    "upload": 10,
    "recipes": 50,
    "default": 200,
}


def _bucket_key(user_id: str, scope: str) -> str:
    return f"{scope}:{user_id}"


class RateLimiter:
    """Use as a route dependency: `Depends(RateLimiter("upload"))`.
    Sets X-RateLimit-* response headers per API_Spec.md §8 and raises
    429 (RATE_LIMITED) once the scope's hourly limit is exceeded."""

    def __init__(self, scope: str = "default"):
        self.scope = scope
        self.limit = _LIMITS.get(scope, _LIMITS["default"])

    def __call__(self, request: Request, user: User = Depends(get_current_user)) -> None:
        now = time.time()
        key = _bucket_key(str(user.id), self.scope)
        window_start = now - _WINDOW_SECONDS

        hits = [t for t in _buckets[key] if t > window_start]
        remaining = self.limit - len(hits)
        reset_at = int((hits[0] if hits else now) + _WINDOW_SECONDS)

        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(remaining - 1, 0)),
            "X-RateLimit-Reset": str(reset_at),
        }

        if remaining <= 0:
            raise rate_limited(f"Rate limit exceeded for '{self.scope}'. Try again later.")

        hits.append(now)
        _buckets[key] = hits
