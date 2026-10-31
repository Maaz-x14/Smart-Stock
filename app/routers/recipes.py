# app/routers/recipes.py
"""GET /recipes (API_Spec.md §5)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import RateLimiter, get_current_user
from app.errors import validation_error
from app.models import User
from app.schemas_inventory import RecipeListResponse, RecipeResult
from app.services import recipes as recipes_service

router = APIRouter(prefix="/recipes", tags=["recipes"])


@router.get("", response_model=RecipeListResponse)
def get_recipes(
    ingredients: str = Query(..., description="Comma-separated ingredient names."),
    limit: int = Query(5, ge=1, le=10),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _rate_limit: None = Depends(RateLimiter("recipes")),
) -> RecipeListResponse:
    ingredient_list = [i for i in ingredients.split(",") if i.strip()]
    if not ingredient_list:
        raise validation_error("No ingredients provided.")

    recipes, cached = recipes_service.get_recipes(db, ingredient_list, limit)
    return RecipeListResponse(recipes=[RecipeResult(**r) for r in recipes], cached=cached)
