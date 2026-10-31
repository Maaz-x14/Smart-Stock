# app/services/recipes.py
"""
GET /recipes (API_Spec.md §5) with a Postgres-backed cache.

Spoonacular's free tier is 150 requests/day, shared by this single
instance across all users - caching by ingredient set is what keeps
repeat lookups (the same few items near their expiry, looked up by
several users or the same user refreshing the page) from burning that
quota. Cache entries expire after 6 hours: recipes don't change, but a
fixed TTL bounds how stale "fresh" results can look without needing
any invalidation logic.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy.orm import Session

from app.config import get_settings
from app.errors import upstream_error
from app.models import RecipeCache

_CACHE_TTL = timedelta(hours=6)
_SPOONACULAR_BASE = "https://api.spoonacular.com/recipes"


def _cache_key(ingredients: list[str], limit: int) -> str:
    normalized = sorted(i.strip().lower() for i in ingredients if i.strip())
    raw = json.dumps({"ingredients": normalized, "limit": limit}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:64]


def _read_cache(db: Session, key: str) -> dict | None:
    row = db.query(RecipeCache).filter(RecipeCache.cache_key == key).first()
    if row is None:
        return None
    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        # SQLite (used in tests - see tests/conftest.py) does not
        # preserve tz-awareness through DateTime(timezone=True); it
        # round-trips naive. Postgres does. Treat a naive value as UTC
        # (everything this app writes is UTC) rather than letting the
        # comparison below raise.
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at < datetime.now(UTC):
        return None
    return row.response_json


def _write_cache(db: Session, key: str, response_json: dict) -> None:
    existing = db.query(RecipeCache).filter(RecipeCache.cache_key == key).first()
    expires_at = datetime.now(UTC) + _CACHE_TTL
    if existing:
        existing.response_json = response_json
        existing.expires_at = expires_at
    else:
        db.add(RecipeCache(cache_key=key, response_json=response_json, expires_at=expires_at))
    db.commit()


def _call_spoonacular(ingredients: list[str], limit: int) -> list[dict]:
    settings = get_settings()
    if not settings.spoonacular_api_key:
        raise upstream_error("Recipe lookup is not configured (no Spoonacular API key).")

    try:
        response = httpx.get(
            f"{_SPOONACULAR_BASE}/findByIngredients",
            params={
                "ingredients": ",".join(ingredients),
                "number": limit,
                "ranking": 1,
                "apiKey": settings.spoonacular_api_key,
            },
            timeout=8.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise upstream_error(f"Spoonacular API unavailable: {e}") from e

    return response.json()


def _to_recipe_result(raw: dict) -> dict:
    used = [i["name"] for i in raw.get("usedIngredients", [])]
    missed = [i["name"] for i in raw.get("missedIngredients", [])]
    return {
        "id": str(raw["id"]),
        "title": raw["title"],
        "image_url": raw.get("image"),
        "ready_in_minutes": raw.get("readyInMinutes"),
        "servings": raw.get("servings"),
        "source_url": raw.get("sourceUrl"),
        "matched_ingredients": used,
        "matched_count": len(used),
        "missing_ingredients": missed,
    }


def get_recipes(db: Session, ingredients: list[str], limit: int) -> tuple[list[dict], bool]:
    key = _cache_key(ingredients, limit)

    cached = _read_cache(db, key)
    if cached is not None:
        return cached["recipes"], True

    raw_results = _call_spoonacular(ingredients, limit)
    recipes = [_to_recipe_result(r) for r in raw_results]
    _write_cache(db, key, {"recipes": recipes})
    return recipes, False
