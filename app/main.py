# app/main.py
"""App factory. Mounts every router, registers error handlers, and
starts/stops the daily-alert scheduler with the app's lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.errors import register_error_handlers
from app.routers import alerts, auth, inventory, receipts, recipes, waste
from app.scheduler import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="Smart-Stock API", version="1.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)

    @app.middleware("http")
    async def _attach_rate_limit_headers(request: Request, call_next):
        """Copies the headers a RateLimiter dependency staged on
        request.state (app/deps.py) onto the actual response, since a
        FastAPI dependency can't set response headers directly."""
        response = await call_next(request)
        headers = getattr(request.state, "rate_limit_headers", None)
        if headers:
            for key, value in headers.items():
                response.headers[key] = value
        return response

    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(receipts.router, prefix="/api/v1")
    app.include_router(inventory.router, prefix="/api/v1")
    app.include_router(alerts.router, prefix="/api/v1")
    app.include_router(recipes.router, prefix="/api/v1")
    app.include_router(waste.router, prefix="/api/v1")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
