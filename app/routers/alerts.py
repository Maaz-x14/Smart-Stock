# app/routers/alerts.py
"""GET /alerts, PATCH /alerts/{id}/dismiss (API_Spec.md §4)."""

from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import InventoryItem, User
from app.schemas_inventory import AlertDismissResponse, AlertListResponse, AlertResponse, AtRiskItem
from app.services import alerts as alerts_service

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=AlertListResponse)
def list_alerts(
    status: str = "ACTIVE",
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AlertListResponse:
    alerts = alerts_service.list_alerts(db, user, status)

    results = []
    for alert in alerts:
        at_risk = []
        for alert_item in alert.alert_items:
            item: InventoryItem = alert_item.item
            at_risk.append(
                AtRiskItem(
                    item_id=item.id,
                    name=item.canonical_name,
                    expiry_date=item.predicted_expiry_date,
                    days_remaining=(item.predicted_expiry_date - date.today()).days,
                )
            )
        results.append(
            AlertResponse(
                id=alert.id,
                triggered_at=alert.triggered_at,
                expiry_threshold_hours=alert.expiry_threshold_hours,
                status=alert.status,
                at_risk_items=at_risk,
            )
        )
    return AlertListResponse(alerts=results)


@router.patch("/{alert_id}/dismiss", response_model=AlertDismissResponse)
def dismiss_alert(
    alert_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> AlertDismissResponse:
    alert = alerts_service.dismiss_alert(db, user, alert_id)
    return AlertDismissResponse(id=alert.id, status=alert.status, dismissed_at=alert.dismissed_at)
