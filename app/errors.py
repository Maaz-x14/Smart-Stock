# app/errors.py
"""
Error response format per API_Spec.md §7:

    {"error": {"code": "ITEM_NOT_FOUND", "message": "...", "status": 404}}

AppError is the one exception type routes/services raise for any
expected failure. FastAPI's generic validation errors (422 from
Pydantic) and uncaught exceptions are also normalized into this same
shape by the handlers registered in register_error_handlers(), so no
route can leak a raw FastAPI/Starlette default error body.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    """Raise this anywhere in a route or service for an expected,
    user-facing error. `code` must be one of API_Spec.md §7's codes
    (or a route-specific extension of that vocabulary - keep it
    SCREAMING_SNAKE_CASE and add new codes to that table when used)."""

    def __init__(self, code: str, message: str, status_code: int):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


# -- Convenience constructors for the standard codes (API_Spec.md §7) --


def validation_error(message: str) -> AppError:
    return AppError("VALIDATION_ERROR", message, status.HTTP_422_UNPROCESSABLE_ENTITY)


def unauthorized(message: str = "Missing or invalid token.") -> AppError:
    return AppError("UNAUTHORIZED", message, status.HTTP_401_UNAUTHORIZED)


def forbidden(message: str = "Resource not owned by requesting user.") -> AppError:
    return AppError("FORBIDDEN", message, status.HTTP_403_FORBIDDEN)


def not_found(message: str = "Resource does not exist.") -> AppError:
    return AppError("NOT_FOUND", message, status.HTTP_404_NOT_FOUND)


def ocr_failure(message: str = "ML pipeline could not extract text.") -> AppError:
    return AppError("OCR_FAILURE", message, status.HTTP_422_UNPROCESSABLE_ENTITY)


def upstream_error(message: str = "External API unavailable.") -> AppError:
    return AppError("UPSTREAM_ERROR", message, status.HTTP_503_SERVICE_UNAVAILABLE)


def rate_limited(message: str = "Too many requests.") -> AppError:
    return AppError("RATE_LIMITED", message, status.HTTP_429_TOO_MANY_REQUESTS)


def receipt_already_confirmed(message: str = "Receipt already confirmed.") -> AppError:
    return AppError("RECEIPT_ALREADY_CONFIRMED", message, status.HTTP_410_GONE)


def _error_body(code: str, message: str, status_code: int) -> dict:
    return {"error": {"code": code, "message": message, "status": status_code}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message, exc.status_code),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body(
                "VALIDATION_ERROR",
                "Request body failed validation.",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Catches FastAPI's own raised HTTPExceptions (e.g. 404 on an
        # unmatched route) and normalizes them to the same shape.
        code = {
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            413: "PAYLOAD_TOO_LARGE",
            429: "RATE_LIMITED",
        }.get(exc.status_code, "ERROR")
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(code, str(exc.detail), exc.status_code),
        )

    @app.exception_handler(Exception)
    async def _handle_uncaught(_: Request, exc: Exception) -> JSONResponse:
        # Last resort: never leak a stack trace or a bare 500 with no
        # body. Logged by uvicorn's own error logging; this handler
        # only shapes the response.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(
                "INTERNAL_ERROR",
                "An unexpected error occurred.",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ),
        )
