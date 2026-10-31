# app/services/alerts.py
"""Alerts: GET /alerts, PATCH /alerts/{id}/dismiss (API_Spec.md §4),
plus the daily scan that creates them (called by app/scheduler.py)."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

from sqlalchemy.orm import Session

from app.errors import forbidden, not_found
from app.models import Alert, AlertItem, InventoryItem, User

DEFAULT_THRESHOLD_HOURS = 48


def list_alerts(db: Session, user: User, status: str) -> list[Alert]:
    return (
        db.query(Alert)
        .filter(Alert.user_id == user.id, Alert.status == status)
        .order_by(Alert.triggered_at.desc())
        .all()
    )


def dismiss_alert(db: Session, user: User, alert_id: uuid.UUID) -> Alert:
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert is None:
        raise not_found(f"Alert with id '{alert_id}' not found.")
    if alert.user_id != user.id:
        raise forbidden()

    alert.status = "DISMISSED"
    alert.dismissed_at = datetime.now(UTC)
    db.commit()
    db.refresh(alert)
    return alert


def run_daily_expiry_scan(db: Session, threshold_hours: int = DEFAULT_THRESHOLD_HOURS) -> int:
    """
    DB_Schema.md §7's "at-risk items" query, grouped per user into one
    Alert row with its matching AlertItem rows. Idempotency: items
    already covered by an existing ACTIVE alert for that user are not
    re-alerted (checked via a join on alert_items), so re-running the
    scan within the same day does not spam duplicate alerts.

    Returns the number of alerts created.
    """
    cutoff = date.today() + timedelta(hours=threshold_hours)

    at_risk_items = (
        db.query(InventoryItem)
        .filter(InventoryItem.status == "ACTIVE", InventoryItem.predicted_expiry_date <= cutoff)
        .order_by(InventoryItem.user_id, InventoryItem.predicted_expiry_date.asc())
        .all()
    )

    already_alerted_item_ids = {
        row.item_id
        for row in (
            db.query(AlertItem)
            .join(Alert, Alert.id == AlertItem.alert_id)
            .filter(Alert.status == "ACTIVE")
            .all()
        )
    }

    by_user: dict[uuid.UUID, list[InventoryItem]] = {}
    for item in at_risk_items:
        if item.id in already_alerted_item_ids:
            continue
        by_user.setdefault(item.user_id, []).append(item)

    created = 0
    for user_id, items in by_user.items():
        alert = Alert(user_id=user_id, expiry_threshold_hours=threshold_hours, status="ACTIVE")
        db.add(alert)
        db.flush()
        for item in items:
            db.add(AlertItem(alert_id=alert.id, item_id=item.id))
        created += 1

    db.commit()
    return created
