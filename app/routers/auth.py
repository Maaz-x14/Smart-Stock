# app/routers/auth.py
"""Register / login / refresh / logout (API_Spec.md §1)."""

from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, Response
from jose import JWTError
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.errors import AppError, unauthorized
from app.models import User
from app.schemas_auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.security import (
    REFRESH_COOKIE_NAME,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_refresh_cookie(response: Response, user_id: str) -> None:
    from app.config import get_settings

    token, expires_delta = create_refresh_token(user_id)
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        # Secure (HTTPS-only) in every environment except local dev -
        # a plain http:// TestClient/dev server never gets the cookie
        # back otherwise, which is a real footgun, not just a test
        # inconvenience (caught while building the test suite).
        secure=get_settings().environment != "development",
        max_age=int(expires_delta.total_seconds()),
        # Must match the mounted prefix (app/main.py mounts this
        # router under /api/v1), not just this router's own /auth
        # prefix - otherwise the browser never sends the cookie back
        # on POST /api/v1/auth/refresh (caught by test_refresh_rotates_
        # access_token, which was a real bug, not a test issue).
        path="/api/v1/auth",
    )


@router.post("/register", response_model=UserResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> User:
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing is not None:
        raise AppError("EMAIL_ALREADY_REGISTERED", "Email already registered.", 400)

    user = User(
        email=payload.email,
        name=payload.name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise unauthorized("Invalid credentials.")
    if not user.is_active:
        raise unauthorized("Account is deactivated.")

    access_token, expires_in = create_access_token(str(user.id))
    _set_refresh_cookie(response, str(user.id))
    return TokenResponse(access_token=access_token, expires_in=expires_in)


@router.post("/refresh", response_model=TokenResponse)
def refresh(
    response: Response,
    db: Session = Depends(get_db),
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE_NAME),
) -> TokenResponse:
    if refresh_token is None:
        raise unauthorized("Refresh token expired or invalid.")
    try:
        user_id = decode_token(refresh_token, expected_type="refresh")
    except JWTError:
        raise unauthorized("Refresh token expired or invalid.") from None

    user = db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
    if user is None:
        raise unauthorized("Refresh token expired or invalid.")

    access_token, expires_in = create_access_token(str(user.id))
    _set_refresh_cookie(response, str(user.id))  # rotate refresh token on each use
    return TokenResponse(access_token=access_token, expires_in=expires_in)


@router.post("/logout", status_code=204)
def logout(response: Response, user: User = Depends(get_current_user)) -> None:
    response.delete_cookie(REFRESH_COOKIE_NAME, path="/api/v1/auth")
