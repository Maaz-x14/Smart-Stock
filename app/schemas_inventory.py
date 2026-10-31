# app/schemas_inventory.py
"""Pydantic schemas for inventory, alerts, waste-log, recipes
(API_Spec.md §3-6)."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

VALID_STORAGE_CONTEXTS = {"fridge", "freezer", "pantry"}
VALID_CATEGORIES = {"Produce", "Dairy", "Meat", "Pantry", "Frozen"}


def _lower_storage_context(v: str | None) -> str | None:
    if v is None:
        return v
    v = v.lower()
    if v not in VALID_STORAGE_CONTEXTS:
        raise ValueError(f"storage_context must be one of {sorted(VALID_STORAGE_CONTEXTS)}")
    return v


class InventoryItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    canonical_name: str
    brand: str | None
    quantity: float
    unit: str | None
    category: str
    purchase_date: date
    predicted_expiry_date: date
    days_remaining: int
    urgency_tier: str
    storage_context: str
    status: str
    created_at: datetime


class InventoryListResponse(BaseModel):
    items: list[InventoryItemResponse]
    total: int
    page: int
    limit: int


class InventoryCreateRequest(BaseModel):
    canonical_name: str = Field(..., min_length=1, max_length=255)
    brand: str | None = None
    quantity: float = Field(..., gt=0)
    unit: str | None = None
    category: str
    purchase_date: date
    predicted_expiry_date: date
    storage_context: str = "fridge"

    _normalize_storage = field_validator("storage_context")(_lower_storage_context)


class InventoryUpdateRequest(BaseModel):
    canonical_name: str | None = Field(None, min_length=1, max_length=255)
    brand: str | None = None
    quantity: float | None = Field(None, gt=0)
    predicted_expiry_date: date | None = None
    storage_context: str | None = None
    status: str | None = None

    _normalize_storage = field_validator("storage_context")(_lower_storage_context)


class BulkConsumeRequest(BaseModel):
    item_ids: list[uuid.UUID] = Field(..., min_length=1)
    consumed_at: datetime | None = None
    recipe_id: str | None = None


class BulkConsumeResponse(BaseModel):
    consumed_count: int
    waste_log_ids: list[uuid.UUID]


# -- Alerts --------------------------------------------------------------------


class AtRiskItem(BaseModel):
    item_id: uuid.UUID
    name: str
    expiry_date: date
    days_remaining: int


class AlertResponse(BaseModel):
    id: uuid.UUID
    triggered_at: datetime
    expiry_threshold_hours: int
    status: str
    at_risk_items: list[AtRiskItem]


class AlertListResponse(BaseModel):
    alerts: list[AlertResponse]


class AlertDismissResponse(BaseModel):
    id: uuid.UUID
    status: str
    dismissed_at: datetime


# -- Waste log ------------------------------------------------------------------


class WasteLogEntry(BaseModel):
    id: uuid.UUID
    item_name: str
    outcome: str
    recorded_at: datetime
    recipe_used: str | None


class WasteSummary(BaseModel):
    total_items: int
    consumed: int
    wasted: int
    waste_rate_percent: float


class WasteLogResponse(BaseModel):
    summary: WasteSummary
    entries: list[WasteLogEntry]


# -- Recipes --------------------------------------------------------------------


class RecipeResult(BaseModel):
    id: str
    title: str
    image_url: str | None
    ready_in_minutes: int | None
    servings: int | None
    source_url: str | None
    matched_ingredients: list[str]
    matched_count: int
    missing_ingredients: list[str]


class RecipeListResponse(BaseModel):
    recipes: list[RecipeResult]
    cached: bool
