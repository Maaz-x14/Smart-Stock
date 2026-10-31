# app/config.py
"""
Single source of truth for environment configuration. Every other
module reads settings via get_settings() - nothing else calls
os.environ directly, so tests can override settings without env vars
leaking between them, and importing any module never requires a real
environment to be configured.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(..., alias="DATABASE_URL")

    jwt_secret: str = Field(..., alias="JWT_SECRET")
    jwt_algorithm: str = Field("HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(7, alias="REFRESH_TOKEN_EXPIRE_DAYS")

    groq_api_key: str | None = Field(None, alias="GROQ_API_KEY")
    spoonacular_api_key: str | None = Field(None, alias="SPOONACULAR_API_KEY")

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"], alias="CORS_ORIGINS")

    max_upload_mb: int = Field(10, alias="MAX_UPLOAD_MB")

    environment: str = Field("development", alias="ENVIRONMENT")


@lru_cache
def get_settings() -> Settings:
    return Settings()
