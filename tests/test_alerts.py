# tests/test_alerts.py
"""The daily expiry scan (app/services/alerts.run_daily_expiry_scan)
and the alerts API surface."""

from __future__ import annotations

from datetime import date, timedelta

from app.models import InventoryItem, User
from app.security import hash_password
from app.services.alerts import run_daily_expiry_scan


def _make_user(db_session, email="scan@example.com"):
    user = User(email=email, name="Scan", hashed_password=hash_password("x"))
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def _make_item(db_session, user, days_until_expiry):
    item = InventoryItem(
        user_id=user.id,
        canonical_name="Yogurt",
        category="Dairy",
        quantity=1,
        storage_context="fridge",
        purchase_date=date.today(),
        predicted_expiry_date=date.today() + timedelta(days=days_until_expiry),
        source="manual",
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)
    return item


def test_scan_creates_alert_for_at_risk_item(db_session):
    user = _make_user(db_session)
    _make_item(db_session, user, days_until_expiry=1)  # within the 48h default threshold

    created = run_daily_expiry_scan(db_session)
    assert created == 1


def test_scan_skips_items_outside_threshold(db_session):
    user = _make_user(db_session, "far@example.com")
    _make_item(db_session, user, days_until_expiry=10)

    created = run_daily_expiry_scan(db_session)
    assert created == 0


def test_scan_is_idempotent_within_active_alert_window(db_session):
    """Re-running the scan for an item already covered by an ACTIVE
    alert must not create a second alert for the same item."""
    user = _make_user(db_session, "idempotent@example.com")
    _make_item(db_session, user, days_until_expiry=1)

    first_run = run_daily_expiry_scan(db_session)
    second_run = run_daily_expiry_scan(db_session)

    assert first_run == 1
    assert second_run == 0


def test_dismiss_alert_via_api(client, db_session):
    client.post(
        "/api/v1/auth/register",
        json={"email": "dismiss@example.com", "password": "correct-password-1", "name": "D"},
    )
    login = client.post(
        "/api/v1/auth/login", json={"email": "dismiss@example.com", "password": "correct-password-1"}
    )
    token = login.json()["access_token"]

    user = db_session.query(User).filter(User.email == "dismiss@example.com").first()
    _make_item(db_session, user, days_until_expiry=0)
    run_daily_expiry_scan(db_session)

    list_resp = client.get("/api/v1/alerts", headers={"Authorization": f"Bearer {token}"})
    alerts = list_resp.json()["alerts"]
    assert len(alerts) == 1
    alert_id = alerts[0]["id"]
    assert len(alerts[0]["at_risk_items"]) == 1

    dismiss_resp = client.patch(
        f"/api/v1/alerts/{alert_id}/dismiss", headers={"Authorization": f"Bearer {token}"}
    )
    assert dismiss_resp.status_code == 200
    assert dismiss_resp.json()["status"] == "DISMISSED"
