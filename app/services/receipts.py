# app/services/receipts.py
"""
Receipt upload and confirm flow.

Maps ml_service.pipeline.process_receipt()'s ExtractedItem dataclasses
to the API/DB shapes, and owns the storage-context casing boundary:
the API and DB use lowercase ('fridge'/'freezer'/'pantry' per
DB_Schema.md), the ML predictor uses capitalized ('Fridge'/'Freezer'/
'Pantry', see ml_service/expiry/data/category_default_storage.json
and db/seeds/shelf_life_seed.py). Nothing on the ML side is touched -
this module is the single place the two conventions meet.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, date, datetime

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.errors import AppError, not_found, ocr_failure, receipt_already_confirmed, validation_error
from app.models import InventoryItem, Receipt, User
from app.schemas_receipts import ConfirmItem
from ml_service.pipeline import (
    OCRError,
    RowParserError,
    RowReconstructionError,
    process_receipt,
)

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "application/pdf"}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB, API_Spec.md §2

_TO_PIPELINE_CASING = {"fridge": "Fridge", "freezer": "Freezer", "pantry": "Pantry"}


def _to_pipeline_storage_context(storage_context: str) -> str:
    """API/DB lowercase -> the casing predict_expiry()'s
    CATEGORY_DEFAULT_STORAGE / ShelfLifeReference rows use."""
    return _TO_PIPELINE_CASING.get(storage_context.lower(), "Fridge")


def _to_api_storage_context(value: str | None) -> str | None:
    """Pipeline's capitalized casing (or None) -> API/DB lowercase."""
    return value.lower() if value else None


async def upload_receipt(
    file: UploadFile,
    storage_context: str,
    db: Session,
    user: User,
) -> Receipt:
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise AppError(
            "UNSUPPORTED_FILE_TYPE",
            f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or PDF.",
            400,
        )

    image_bytes = await file.read()
    if len(image_bytes) > _MAX_UPLOAD_BYTES:
        raise AppError("FILE_TOO_LARGE", "File exceeds the 10MB limit.", 413)

    storage_context = storage_context.lower()
    if storage_context not in _TO_PIPELINE_CASING:
        raise validation_error(f"storage_context must be one of {sorted(_TO_PIPELINE_CASING)}")

    started = datetime.now(UTC)
    try:
        extracted = await process_receipt(
            image_bytes=image_bytes,
            db=db,
            storage_context=_to_pipeline_storage_context(storage_context),
            purchase_date=date.today(),
        )
    except (OCRError, RowReconstructionError, RowParserError) as e:
        raise ocr_failure(f"Could not read this receipt: {e}") from e

    elapsed_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)

    items_json = []
    for item in extracted:
        d = dataclasses.asdict(item)
        d["storage_context"] = _to_api_storage_context(d["storage_context"])
        if d["predicted_expiry_date"] is not None:
            d["predicted_expiry_date"] = d["predicted_expiry_date"].isoformat()
        items_json.append(d)

    receipt = Receipt(
        user_id=user.id,
        extracted_items=items_json,
        storage_context=storage_context,
        purchase_date=date.today(),
        status="pending",
        processing_time_ms=elapsed_ms,
    )
    db.add(receipt)
    db.commit()
    db.refresh(receipt)
    return receipt


def confirm_receipt(
    receipt_id: uuid.UUID,
    items: list[ConfirmItem],
    db: Session,
    user: User,
) -> list[uuid.UUID]:
    receipt = db.query(Receipt).filter(Receipt.id == receipt_id, Receipt.user_id == user.id).first()
    if receipt is None:
        raise not_found(f"Receipt with id '{receipt_id}' not found.")
    if receipt.status == "confirmed":
        raise receipt_already_confirmed()

    created_ids: list[uuid.UUID] = []
    for item in items:
        inv = InventoryItem(
            user_id=user.id,
            canonical_name=item.canonical_name,
            brand=item.brand,
            category=item.category,
            quantity=item.quantity,
            unit=item.unit,
            storage_context=item.storage_context,
            purchase_date=date.today(),
            predicted_expiry_date=item.predicted_expiry_date,
            source="receipt",
        )
        db.add(inv)
        db.flush()  # populate inv.id without committing yet
        created_ids.append(inv.id)

    receipt.status = "confirmed"
    receipt.confirmed_at = datetime.now(UTC)
    db.commit()
    return created_ids
