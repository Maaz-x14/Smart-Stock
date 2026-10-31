# app/schemas_receipts.py
"""Pydantic schemas for the receipts endpoints (API_Spec.md §2)."""

from __future__ import annotations

import uuid
from datetime import date

from pydantic import BaseModel, Field, field_validator

from app.schemas import ExtractedItem
from app.schemas_inventory import _lower_storage_context


class ReceiptUploadResponse(BaseModel):
    receipt_id: uuid.UUID
    extracted_items: list[ExtractedItem]
    total_items_extracted: int
    processing_time_ms: int


class ConfirmItem(BaseModel):
    """One user-approved item from the confirmation modal. Mirrors
    API_Spec.md §2's /confirm request shape - the user may have
    edited canonical_name/quantity/predicted_expiry_date from what
    /upload returned."""

    canonical_name: str = Field(..., min_length=1, max_length=255)
    brand: str | None = None
    quantity: float = Field(..., gt=0)
    unit: str | None = None
    category: str
    predicted_expiry_date: date
    storage_context: str = "fridge"

    _normalize_storage = field_validator("storage_context")(_lower_storage_context)


class ConfirmRequest(BaseModel):
    receipt_id: uuid.UUID
    items: list[ConfirmItem]


class ConfirmResponse(BaseModel):
    created_items: int
    inventory_ids: list[uuid.UUID]
