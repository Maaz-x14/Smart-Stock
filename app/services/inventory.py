# app/services/inventory.py
"""Inventory CRUD, bulk-consume, and the waste-log write that
bulk-consume triggers (API_Spec.md §3, §6)."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

from sqlalchemy import asc, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.errors import AppError, forbidden, not_found
from app.models import InventoryItem, User, WasteLog
from app.schemas_inventory import BulkConsumeRequest, InventoryCreateRequest, InventoryUpdateRequest

_SORT_COLUMNS = {
    "expiry_date": InventoryItem.predicted_expiry_date,
    "name": InventoryItem.canonical_name,
    "category": InventoryItem.category,
}


def _days_remaining(predicted_expiry_date: date) -> int:
    return (predicted_expiry_date - date.today()).days


def list_inventory(
    db: Session,
    user: User,
    category: str | None,
    status: str,
    sort_by: str,
    order: str,
    page: int,
    limit: int,
) -> tuple[list[InventoryItem], int]:
    query = db.query(InventoryItem).filter(InventoryItem.user_id == user.id, InventoryItem.status == status)
    if category:
        query = query.filter(InventoryItem.category == category)

    total = query.count()

    sort_col = _SORT_COLUMNS.get(sort_by, InventoryItem.predicted_expiry_date)
    query = query.order_by(asc(sort_col) if order == "asc" else desc(sort_col))

    items = query.offset((page - 1) * limit).limit(limit).all()
    return items, total


def get_item(db: Session, user: User, item_id: uuid.UUID) -> InventoryItem:
    item = db.query(InventoryItem).filter(InventoryItem.id == item_id).first()
    if item is None:
        raise not_found(f"Inventory item with id '{item_id}' not found.")
    if item.user_id != user.id:
        raise forbidden()
    return item


def create_item(db: Session, user: User, payload: InventoryCreateRequest) -> InventoryItem:
    item = InventoryItem(
        user_id=user.id,
        canonical_name=payload.canonical_name,
        brand=payload.brand,
        category=payload.category,
        quantity=payload.quantity,
        unit=payload.unit,
        storage_context=payload.storage_context,
        purchase_date=payload.purchase_date,
        predicted_expiry_date=payload.predicted_expiry_date,
        source="manual",
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def update_item(
    db: Session, user: User, item_id: uuid.UUID, payload: InventoryUpdateRequest
) -> InventoryItem:
    item = get_item(db, user, item_id)
    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(item, field, value)
    db.commit()
    db.refresh(item)
    return item


def delete_item(db: Session, user: User, item_id: uuid.UUID) -> None:
    item = get_item(db, user, item_id)
    try:
        db.delete(item)
        db.commit()
    except IntegrityError as e:
        # waste_log keeps item_id with no cascade (DB_Schema.md: "kept
        # even if item deleted" means the FK has no ON DELETE clause -
        # a row with waste history cannot be hard-deleted).
        db.rollback()
        raise AppError(
            "ITEM_HAS_WASTE_HISTORY",
            "This item has waste-log history and cannot be deleted. Mark it consumed/wasted instead.",
            409,
        ) from e


def bulk_consume(db: Session, user: User, payload: BulkConsumeRequest) -> tuple[int, list[uuid.UUID]]:
    items = (
        db.query(InventoryItem)
        .filter(InventoryItem.id.in_(payload.item_ids), InventoryItem.user_id == user.id)
        .all()
    )
    found_ids = {item.id for item in items}
    missing = set(payload.item_ids) - found_ids
    if missing:
        raise not_found(f"Inventory item(s) not found: {sorted(str(i) for i in missing)}")

    consumed_at = payload.consumed_at or datetime.now(UTC)
    waste_log_ids: list[uuid.UUID] = []

    for item in items:
        item.status = "CONSUMED"
        log_entry = WasteLog(
            user_id=user.id,
            item_id=item.id,
            item_name=item.canonical_name,
            outcome="CONSUMED",
            recipe_used=payload.recipe_id,
            recorded_at=consumed_at,
        )
        db.add(log_entry)
        db.flush()
        waste_log_ids.append(log_entry.id)

    db.commit()
    return len(items), waste_log_ids
