# app/routers/waste.py
"""GET /waste-log (API_Spec.md §6)."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import User
from app.schemas_inventory import WasteLogEntry, WasteLogResponse, WasteSummary
from app.services import waste as waste_service

router = APIRouter(prefix="/waste-log", tags=["waste-log"])


@router.get("", response_model=WasteLogResponse)
def get_waste_log(
    outcome: str | None = None,
    from_date: date | None = None,
    to_date: date | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> WasteLogResponse:
    entries, total, consumed, wasted = waste_service.get_waste_log(db, user, outcome, from_date, to_date)
    waste_rate = round((wasted / total) * 100, 1) if total else 0.0

    return WasteLogResponse(
        summary=WasteSummary(
            total_items=total, consumed=consumed, wasted=wasted, waste_rate_percent=waste_rate
        ),
        entries=[
            WasteLogEntry(
                id=e.id,
                item_name=e.item_name,
                outcome=e.outcome,
                recorded_at=e.recorded_at,
                recipe_used=e.recipe_used,
            )
            for e in entries
        ],
    )
