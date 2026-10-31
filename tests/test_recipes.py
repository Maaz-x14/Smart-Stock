# tests/test_recipes.py
"""Recipe caching: a second identical lookup must not re-call
Spoonacular, and a missing API key degrades to a clear 503 rather
than a bare 500."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _register_and_login(client, email="recipe@example.com"):
    client.post("/api/v1/auth/register", json={"email": email, "password": "correct-password-1", "name": "R"})
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "correct-password-1"})
    return resp.json()["access_token"]


def _fake_spoonacular_response():
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {
            "id": 12345,
            "title": "Strawberry Smoothie",
            "image": "https://spoonacular.com/img.jpg",
            "readyInMinutes": 5,
            "servings": 2,
            "sourceUrl": "https://example.com/recipe",
            "usedIngredients": [{"name": "strawberries"}, {"name": "milk"}],
            "missedIngredients": [{"name": "banana"}],
        }
    ]
    return mock_resp


def test_recipes_second_identical_call_is_cached(client, monkeypatch):
    token = _register_and_login(client)
    monkeypatch.setenv("SPOONACULAR_API_KEY", "test-key")

    with (
        patch("app.services.recipes.get_settings") as mock_settings,
        patch("app.services.recipes.httpx.get", return_value=_fake_spoonacular_response()) as mock_get,
    ):
        mock_settings.return_value.spoonacular_api_key = "test-key"

        first = client.get(
            "/api/v1/recipes",
            headers={"Authorization": f"Bearer {token}"},
            params={"ingredients": "strawberries,milk"},
        )
        second = client.get(
            "/api/v1/recipes",
            headers={"Authorization": f"Bearer {token}"},
            params={"ingredients": "milk,strawberries"},  # different order, same set
        )

    assert first.status_code == 200
    assert first.json()["cached"] is False
    assert second.status_code == 200
    assert second.json()["cached"] is True
    mock_get.assert_called_once()  # Spoonacular hit only once across both calls


def test_recipes_no_api_key_returns_503_not_500(client):
    token = _register_and_login(client, "nokey@example.com")

    with patch("app.services.recipes.get_settings") as mock_settings:
        mock_settings.return_value.spoonacular_api_key = None
        resp = client.get(
            "/api/v1/recipes",
            headers={"Authorization": f"Bearer {token}"},
            params={"ingredients": "strawberries"},
        )

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "UPSTREAM_ERROR"


def test_recipes_empty_ingredients_is_422(client):
    token = _register_and_login(client, "empty@example.com")
    resp = client.get(
        "/api/v1/recipes", headers={"Authorization": f"Bearer {token}"}, params={"ingredients": " , "}
    )
    assert resp.status_code == 422
