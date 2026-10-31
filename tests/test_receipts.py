# tests/test_receipts.py
"""Upload (pipeline mocked end-to-end) + confirm, including the
404/410 idempotency guarantee API_Spec.md §2 requires."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch


def _register_and_login(client, email="receipts@example.com"):
    client.post("/api/v1/auth/register", json={"email": email, "password": "correct-password-1", "name": "R"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "correct-password-1"})
    return resp.json()["access_token"]


def _fake_extracted_item():
    from datetime import date

    from ml_service.pipeline import ExtractedItem

    return ExtractedItem(
        raw_token="Strawberries 1lb",
        canonical_name="Strawberries",
        brand=None,
        is_food=True,
        quantity=1.0,
        unit="lb",
        category="Produce",
        predicted_expiry_date=date(2026, 11, 5),
        shelf_life_days=7,
        confidence=0.91,
        storage_context="Fridge",  # pipeline casing - service must lowercase at the boundary
        normalization_pass=1,
    )


def test_upload_lowercases_storage_context_in_response(client):
    token = _register_and_login(client)

    with patch("app.services.receipts.process_receipt", new=AsyncMock(return_value=[_fake_extracted_item()])):
        resp = client.post(
            "/api/v1/receipts/upload",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("receipt.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
            data={"storage_context": "fridge"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_items_extracted"] == 1
    item = body["extracted_items"][0]
    assert item["storage_context"] == "fridge"  # lowercase, not the pipeline's "Fridge"
    assert item["canonical_name"] == "Strawberries"


def test_confirm_then_reconfirm_is_410(client):
    token = _register_and_login(client, "confirm@example.com")

    with patch("app.services.receipts.process_receipt", new=AsyncMock(return_value=[_fake_extracted_item()])):
        upload_resp = client.post(
            "/api/v1/receipts/upload",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("receipt.jpg", io.BytesIO(b"fake-image-bytes"), "image/jpeg")},
            data={"storage_context": "fridge"},
        )
    receipt_id = upload_resp.json()["receipt_id"]

    confirm_body = {
        "receipt_id": receipt_id,
        "items": [
            {
                "canonical_name": "Strawberries",
                "quantity": 1.0,
                "unit": "lb",
                "category": "Produce",
                "predicted_expiry_date": "2026-11-05",
                "storage_context": "fridge",
            }
        ],
    }

    first = client.post(
        "/api/v1/receipts/confirm", headers={"Authorization": f"Bearer {token}"}, json=confirm_body
    )
    assert first.status_code == 201
    assert first.json()["created_items"] == 1

    second = client.post(
        "/api/v1/receipts/confirm", headers={"Authorization": f"Bearer {token}"}, json=confirm_body
    )
    assert second.status_code == 410
    assert second.json()["error"]["code"] == "RECEIPT_ALREADY_CONFIRMED"


def test_confirm_unknown_receipt_id_is_404(client):
    token = _register_and_login(client, "unknown@example.com")
    resp = client.post(
        "/api/v1/receipts/confirm",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "receipt_id": "00000000-0000-0000-0000-000000000000",
            "items": [],
        },
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"


def test_upload_rejects_unsupported_file_type(client):
    token = _register_and_login(client, "badtype@example.com")
    resp = client.post(
        "/api/v1/receipts/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("receipt.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"
