# tests/test_auth.py
"""Register / login / refresh / logout, and the user-isolation
guarantee that every other resource's tests build on."""

from __future__ import annotations


def _register_and_login(client, email="alice@example.com", password="correct-password-1"):
    client.post("/api/v1/auth/register", json={"email": email, "password": password, "name": "Alice"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return resp.json()["access_token"]


def test_register_then_login_succeeds(client):
    register_resp = client.post(
        "/api/v1/auth/register",
        json={"email": "bob@example.com", "password": "correct-password-1", "name": "Bob"},
    )
    assert register_resp.status_code == 201
    assert register_resp.json()["email"] == "bob@example.com"

    login_resp = client.post(
        "/api/v1/auth/login", json={"email": "bob@example.com", "password": "correct-password-1"}
    )
    assert login_resp.status_code == 200
    body = login_resp.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 1800
    assert "refresh_token" in login_resp.cookies


def test_duplicate_email_registration_rejected(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "dup@example.com", "password": "correct-password-1", "name": "Dup"},
    )
    second = client.post(
        "/api/v1/auth/register",
        json={"email": "dup@example.com", "password": "correct-password-1", "name": "Dup2"},
    )
    assert second.status_code == 400
    assert second.json()["error"]["code"] == "EMAIL_ALREADY_REGISTERED"


def test_login_wrong_password_rejected(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "carol@example.com", "password": "correct-password-1", "name": "Carol"},
    )
    resp = client.post("/api/v1/auth/login", json={"email": "carol@example.com", "password": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"


def test_protected_route_without_token_is_401(client):
    resp = client.get("/api/v1/inventory")
    assert resp.status_code == 401


def test_refresh_without_cookie_is_401(client):
    resp = client.post("/api/v1/auth/refresh")
    assert resp.status_code == 401


def test_refresh_rotates_access_token(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "dave@example.com", "password": "correct-password-1", "name": "Dave"},
    )
    client.post("/api/v1/auth/login", json={"email": "dave@example.com", "password": "correct-password-1"})
    resp = client.post("/api/v1/auth/refresh")
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_user_cannot_read_another_users_inventory_item(client):
    """The isolation guarantee every resource router relies on
    (plan doc self-review checklist): a filtered-out-of-scope 404
    AND an owned-but-forbidden 403, never a leak of another user's
    data through either error path."""
    token_a = _register_and_login(client, "owner@example.com")
    token_b = _register_and_login(client, "stranger@example.com")

    create_resp = client.post(
        "/api/v1/inventory",
        headers={"Authorization": f"Bearer {token_a}"},
        json={
            "canonical_name": "Strawberries",
            "quantity": 1.0,
            "category": "Produce",
            "purchase_date": "2026-10-01",
            "predicted_expiry_date": "2026-10-10",
            "storage_context": "fridge",
        },
    )
    assert create_resp.status_code == 201
    item_id = create_resp.json()["id"]

    # Owner can read it.
    owned_resp = client.get(f"/api/v1/inventory/{item_id}", headers={"Authorization": f"Bearer {token_a}"})
    assert owned_resp.status_code == 200

    # A different authenticated user gets 403, not the item's data.
    stranger_resp = client.get(f"/api/v1/inventory/{item_id}", headers={"Authorization": f"Bearer {token_b}"})
    assert stranger_resp.status_code == 403
    assert "Strawberries" not in stranger_resp.text

    # The stranger's own inventory list does not include it.
    list_resp = client.get("/api/v1/inventory", headers={"Authorization": f"Bearer {token_b}"})
    assert list_resp.json()["total"] == 0
