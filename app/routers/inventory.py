# app/routers/inventory.py
"""Inventory CRUD + bulk-consume (API_Spec.md §3)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import RateLimiter, get_current_user
from app.models import InventoryItem, User
from app.schemas_inventory import (
    BulkConsumeRequest,
    BulkConsumeResponse,
    InventoryCreateRequest,
    InventoryItemResponse,
    InventoryListResponse,
    InventoryUpdateRequest,
)
from app.services import inventory as inventory_service

router = APIRouter(prefix="/inventory", tags=["inventory"])


def _urgency_tier(days_remaining: int) -> str:
    # Computed here, not in the DB: Postgres GENERATED ALWAYS columns
    # must be immutable, and CURRENT_DATE isn't (the original
    # DB_Schema.md design used a GENERATED ALWAYS AS (... CURRENT_DATE
    # ...) STORED column for urgency_tier - this fails at migration
    # time with "generation expression is not immutable", caught by
    # actually running `alembic upgrade head` against real Postgres,
    # not just SQLite. Computing it here, from the same
    # predicted_expiry_date -> days_remaining value already used for
    # the API response, is simpler and always correct for "today",
    # which a stored column computed at write time could never be.
    if days_remaining < 0:
        return "expired"
    if days_remaining <= 2:
        return "red"
    if days_remaining <= 5:
        return "yellow"
    return "green"


def _to_response(item: InventoryItem) -> InventoryItemResponse:
    from datetime import date

    days_remaining = (item.predicted_expiry_date - date.today()).days
    return InventoryItemResponse(
        id=item.id,
        canonical_name=item.canonical_name,
        brand=item.brand,
        quantity=float(item.quantity),
        unit=item.unit,
        category=item.category,
        purchase_date=item.purchase_date,
        predicted_expiry_date=item.predicted_expiry_date,
        days_remaining=days_remaining,
        urgency_tier=_urgency_tier(days_remaining),
        storage_context=item.storage_context,
        status=item.status,
        created_at=item.created_at,
    )


@router.get("", response_model=InventoryListResponse)
def list_inventory(
    category: str | None = None,
    status: str = "ACTIVE",
    sort_by: str = "expiry_date",
    order: str = "asc",
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _rate_limit: None = Depends(RateLimiter("default")),
) -> InventoryListResponse:
    items, total = inventory_service.list_inventory(db, user, category, status, sort_by, order, page, limit)
    return InventoryListResponse(items=[_to_response(i) for i in items], total=total, page=page, limit=limit)


@router.post("", response_model=InventoryItemResponse, status_code=201)
def create_inventory_item(
    payload: InventoryCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _rate_limit: None = Depends(RateLimiter("default")),
) -> InventoryItemResponse:
    item = inventory_service.create_item(db, user, payload)
    return _to_response(item)


@router.get("/{item_id}", response_model=InventoryItemResponse)
def get_inventory_item(
    item_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> InventoryItemResponse:
    item = inventory_service.get_item(db, user, item_id)
    return _to_response(item)


@router.patch("/{item_id}", response_model=InventoryItemResponse)
def update_inventory_item(
    item_id: uuid.UUID,
    payload: InventoryUpdateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> InventoryItemResponse:
    item = inventory_service.update_item(db, user, item_id, payload)
    return _to_response(item)


@router.delete("/{item_id}", status_code=204)
def delete_inventory_item(
    item_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    inventory_service.delete_item(db, user, item_id)


@router.post("/bulk-consume", response_model=BulkConsumeResponse)
def bulk_consume(
    payload: BulkConsumeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> BulkConsumeResponse:
    consumed_count, waste_log_ids = inventory_service.bulk_consume(db, user, payload)
    return BulkConsumeResponse(consumed_count=consumed_count, waste_log_ids=waste_log_ids)
