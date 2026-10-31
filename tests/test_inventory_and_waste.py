# tests/test_inventory_and_waste.py
"""Inventory CRUD, bulk-consume -> waste_log write, and the
delete-blocked-by-waste-history guard."""

from __future__ import annotations


def _register_and_login(client, email="inv@example.com"):
    client.post("/api/v1/auth/register", json={"email": email, "password": "correct-password-1", "name": "I"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "correct-password-1"})
    return resp.json()["access_token"]


def _create_item(client, token, **overrides):
    body = {
        "canonical_name": "Whole Milk",
        "quantity": 1.0,
        "unit": "gal",
        "category": "Dairy",
        "purchase_date": "2026-10-20",
        "predicted_expiry_date": "2026-11-01",
        "storage_context": "fridge",
    }
    body.update(overrides)
    resp = client.post("/api/v1/inventory", headers={"Authorization": f"Bearer {token}"}, json=body)
    assert resp.status_code == 201
    return resp.json()


def test_create_list_update_delete_item(client):
    token = _register_and_login(client)
    item = _create_item(client, token)

    list_resp = client.get("/api/v1/inventory", headers={"Authorization": f"Bearer {token}"})
    assert list_resp.json()["total"] == 1

    patch_resp = client.patch(
        f"/api/v1/inventory/{item['id']}",
        headers={"Authorization": f"Bearer {token}"},
        json={"quantity": 0.5},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["quantity"] == 0.5

    delete_resp = client.delete(
        f"/api/v1/inventory/{item['id']}", headers={"Authorization": f"Bearer {token}"}
    )
    assert delete_resp.status_code == 204


def test_rejects_invalid_storage_context(client):
    token = _register_and_login(client, "badstorage@example.com")
    resp = client.post(
        "/api/v1/inventory",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "canonical_name": "Milk",
            "quantity": 1.0,
            "category": "Dairy",
            "purchase_date": "2026-10-20",
            "predicted_expiry_date": "2026-11-01",
            "storage_context": "Refrigerator",  # not one of fridge/freezer/pantry
        },
    )
    assert resp.status_code == 422


def test_bulk_consume_writes_waste_log_and_blocks_hard_delete(client):
    token = _register_and_login(client, "consume@example.com")
    item = _create_item(client, token)

    consume_resp = client.post(
        "/api/v1/inventory/bulk-consume",
        headers={"Authorization": f"Bearer {token}"},
        json={"item_ids": [item["id"]]},
    )
    assert consume_resp.status_code == 200
    assert consume_resp.json()["consumed_count"] == 1
    assert len(consume_resp.json()["waste_log_ids"]) == 1

    waste_resp = client.get("/api/v1/waste-log", headers={"Authorization": f"Bearer {token}"})
    body = waste_resp.json()
    assert body["summary"]["total_items"] == 1
    assert body["summary"]["consumed"] == 1
    assert body["entries"][0]["item_name"] == "Whole Milk"

    # DB_Schema.md: waste_log keeps item_id even if the item is later
    # deleted, so a hard delete of an item with waste history must be
    # blocked, not silently orphan the waste_log row.
    delete_resp = client.delete(
        f"/api/v1/inventory/{item['id']}", headers={"Authorization": f"Bearer {token}"}
    )
    assert delete_resp.status_code == 409
    assert delete_resp.json()["error"]["code"] == "ITEM_HAS_WASTE_HISTORY"


def test_bulk_consume_unknown_item_id_is_404(client):
    token = _register_and_login(client, "missingconsume@example.com")
    resp = client.post(
        "/api/v1/inventory/bulk-consume",
        headers={"Authorization": f"Bearer {token}"},
        json={"item_ids": ["00000000-0000-0000-0000-000000000000"]},
    )
    assert resp.status_code == 404
