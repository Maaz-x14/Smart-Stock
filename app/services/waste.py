# app/services/waste.py
"""GET /waste-log (API_Spec.md §6)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from app.models import User, WasteLog


def get_waste_log(
    db: Session,
    user: User,
    outcome: str | None,
    from_date: date | None,
    to_date: date | None,
) -> tuple[list[WasteLog], int, int, int]:
    query = db.query(WasteLog).filter(WasteLog.user_id == user.id)
    if outcome:
        query = query.filter(WasteLog.outcome == outcome)
    if from_date:
        query = query.filter(
            WasteLog.recorded_at >= datetime.combine(from_date, datetime.min.time(), tzinfo=UTC)
        )
    if to_date:
        query = query.filter(
            WasteLog.recorded_at <= datetime.combine(to_date, datetime.max.time(), tzinfo=UTC)
        )

    entries = query.order_by(WasteLog.recorded_at.desc()).all()

    consumed = sum(1 for e in entries if e.outcome == "CONSUMED")
    wasted = sum(1 for e in entries if e.outcome == "WASTED")
    total = consumed + wasted
    return entries, total, consumed, wasted
