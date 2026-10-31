# app/routers/receipts.py
"""POST /receipts/upload, POST /receipts/confirm (API_Spec.md §2)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import RateLimiter, get_current_user
from app.models import User
from app.schemas_receipts import ConfirmRequest, ConfirmResponse, ReceiptUploadResponse
from app.services import receipts as receipts_service

router = APIRouter(prefix="/receipts", tags=["receipts"])


@router.post("/upload", response_model=ReceiptUploadResponse)
async def upload(
    file: UploadFile,
    storage_context: str = Form("fridge"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    _rate_limit: None = Depends(RateLimiter("upload")),
) -> ReceiptUploadResponse:
    receipt = await receipts_service.upload_receipt(file, storage_context, db, user)
    return ReceiptUploadResponse(
        receipt_id=receipt.id,
        extracted_items=receipt.extracted_items,
        total_items_extracted=len(receipt.extracted_items),
        processing_time_ms=receipt.processing_time_ms,
    )


@router.post("/confirm", response_model=ConfirmResponse, status_code=201)
def confirm(
    payload: ConfirmRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ConfirmResponse:
    inventory_ids = receipts_service.confirm_receipt(payload.receipt_id, payload.items, db, user)
    return ConfirmResponse(created_items=len(inventory_ids), inventory_ids=inventory_ids)
