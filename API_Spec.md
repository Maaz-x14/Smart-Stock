# API_Spec.md — API Specification
## Smart-Stock REST API

**Version:** 1.0  
**Base URL:** `https://api.smart-stock.app/api/v1`  
**Auth:** JWT Bearer Token (all endpoints except `/auth/*`)

---

## Authentication

All protected endpoints require the header:
```
Authorization: Bearer <access_token>
```

Tokens are obtained via `/auth/login`. Access tokens expire in 30 minutes. Use `/auth/refresh` with the HttpOnly refresh cookie to obtain new access tokens.

---

## 1. Auth Endpoints

### POST `/auth/register`
Register a new user.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "name": "Maaz Khan"
}
```

**Response `201`:**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "name": "Maaz Khan",
  "created_at": "2025-01-01T10:00:00Z"
}
```

**Errors:** `400` Email already registered | `422` Validation error

---

### POST `/auth/login`
Authenticate and receive access token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response `200`:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```
Sets `refresh_token` as HttpOnly cookie.

**Errors:** `401` Invalid credentials

---

### POST `/auth/refresh`
Refresh access token using cookie.

**Request:** No body. Requires `refresh_token` HttpOnly cookie.

**Response `200`:**
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Errors:** `401` Refresh token expired or invalid

---

### POST `/auth/logout`
Invalidate refresh token.

**Response `204`:** No content. Clears refresh cookie.

---

## 2. Receipt Endpoints

### POST `/receipts/upload`
Upload a receipt image and run the ML pipeline.

**Auth:** Required  
**Content-Type:** `multipart/form-data`

**Request Form Fields:**
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` | Yes | JPEG/PNG/PDF receipt image (max 10MB) |
| `storage_context` | `string` | No | Default storage: `fridge`, `freezer`, `pantry` (default: `fridge`) |

**Response `200`:**
```json
{
  "receipt_id": "uuid",
  "extracted_items": [
    {
      "raw_token": "ORG STRWBRY 1LB",
      "canonical_name": "Strawberries",
      "quantity": 1.0,
      "unit": "lb",
      "category": "Produce",
      "predicted_expiry_date": "2025-01-08",
      "shelf_life_days": 7,
      "confidence": 0.91,
      "storage_context": "fridge"
    },
    {
      "raw_token": "WHOLE MILK 1GAL",
      "canonical_name": "Whole Milk",
      "quantity": 1.0,
      "unit": "gal",
      "category": "Dairy",
      "predicted_expiry_date": "2025-01-15",
      "shelf_life_days": 14,
      "confidence": 0.96,
      "storage_context": "fridge"
    }
  ],
  "total_items_extracted": 2,
  "processing_time_ms": 1842
}
```

**Errors:** `400` Unsupported file type | `413` File too large | `422` OCR returned no readable text | `500` ML pipeline failure

---

### POST `/receipts/confirm`
Confirm extracted items and save to inventory.

**Auth:** Required

**Request Body:**
```json
{
  "receipt_id": "uuid",
  "items": [
    {
      "canonical_name": "Strawberries",
      "quantity": 1.0,
      "unit": "lb",
      "category": "Produce",
      "predicted_expiry_date": "2025-01-08",
      "storage_context": "fridge"
    }
  ]
}
```
*User may edit `canonical_name`, `quantity`, `predicted_expiry_date` before confirming.*

**Response `201`:**
```json
{
  "created_items": 2,
  "inventory_ids": ["uuid1", "uuid2"]
}
```

**Errors:** `404` Receipt ID not found | `410` Receipt already confirmed

---

## 3. Inventory Endpoints

### GET `/inventory`
List all inventory items for the authenticated user.

**Auth:** Required

**Query Parameters:**
| Param | Type | Description |
|---|---|---|
| `category` | `string` | Filter by category: `Produce`, `Dairy`, `Meat`, `Pantry`, `Frozen` |
| `status` | `string` | Filter by status: `ACTIVE`, `CONSUMED`, `WASTED` (default: `ACTIVE`) |
| `sort_by` | `string` | `expiry_date`, `name`, `category` (default: `expiry_date`) |
| `order` | `string` | `asc`, `desc` (default: `asc`) |
| `page` | `int` | Page number (default: 1) |
| `limit` | `int` | Items per page (default: 20, max: 100) |

**Response `200`:**
```json
{
  "items": [
    {
      "id": "uuid",
      "canonical_name": "Strawberries",
      "quantity": 1.0,
      "unit": "lb",
      "category": "Produce",
      "purchase_date": "2025-01-01",
      "predicted_expiry_date": "2025-01-08",
      "days_remaining": 7,
      "urgency_tier": "green",
      "storage_context": "fridge",
      "status": "ACTIVE",
      "created_at": "2025-01-01T10:00:00Z"
    }
  ],
  "total": 14,
  "page": 1,
  "limit": 20
}
```

`urgency_tier` values: `green` (>5 days), `yellow` (2–5 days), `red` (<2 days or expired)

---

### POST `/inventory`
Manually add a single inventory item.

**Auth:** Required

**Request Body:**
```json
{
  "canonical_name": "Greek Yogurt",
  "quantity": 2.0,
  "unit": "container",
  "category": "Dairy",
  "purchase_date": "2025-01-01",
  "predicted_expiry_date": "2025-01-14",
  "storage_context": "fridge"
}
```

**Response `201`:**
```json
{
  "id": "uuid",
  "canonical_name": "Greek Yogurt",
  "quantity": 2.0,
  "unit": "container",
  "category": "Dairy",
  "purchase_date": "2025-01-01",
  "predicted_expiry_date": "2025-01-14",
  "days_remaining": 13,
  "urgency_tier": "green",
  "storage_context": "fridge",
  "status": "ACTIVE",
  "created_at": "2025-01-01T10:00:00Z"
}
```

---

### GET `/inventory/{item_id}`
Get a single inventory item.

**Auth:** Required

**Response `200`:** Single item object (same shape as list item above).

**Errors:** `404` Item not found | `403` Not owned by user

---

### PATCH `/inventory/{item_id}`
Update an inventory item.

**Auth:** Required

**Request Body (partial update, all fields optional):**
```json
{
  "canonical_name": "Strawberries",
  "quantity": 0.5,
  "predicted_expiry_date": "2025-01-07",
  "storage_context": "fridge",
  "status": "ACTIVE"
}
```

**Response `200`:** Updated item object.

**Errors:** `404` Not found | `403` Not owned by user

---

### DELETE `/inventory/{item_id}`
Delete an inventory item (hard delete).

**Auth:** Required

**Response `204`:** No content.

**Errors:** `404` Not found | `403` Not owned by user

---

### POST `/inventory/bulk-consume`
Mark multiple items as consumed (after cooking).

**Auth:** Required

**Request Body:**
```json
{
  "item_ids": ["uuid1", "uuid2"],
  "consumed_at": "2025-01-06T19:00:00Z",
  "recipe_id": "spoonacular_recipe_id_optional"
}
```

**Response `200`:**
```json
{
  "consumed_count": 2,
  "waste_log_ids": ["uuid3", "uuid4"]
}
```

---

## 4. Alerts Endpoints

### GET `/alerts`
Get all active alerts for the authenticated user.

**Auth:** Required

**Query Parameters:**
| Param | Type | Description |
|---|---|---|
| `status` | `string` | `ACTIVE`, `DISMISSED` (default: `ACTIVE`) |

**Response `200`:**
```json
{
  "alerts": [
    {
      "id": "uuid",
      "triggered_at": "2025-01-06T08:00:00Z",
      "expiry_threshold_hours": 48,
      "status": "ACTIVE",
      "at_risk_items": [
        {
          "item_id": "uuid",
          "name": "Strawberries",
          "expiry_date": "2025-01-08",
          "days_remaining": 2
        }
      ]
    }
  ]
}
```

---

### PATCH `/alerts/{alert_id}/dismiss`
Dismiss an alert.

**Auth:** Required

**Response `200`:**
```json
{
  "id": "uuid",
  "status": "DISMISSED",
  "dismissed_at": "2025-01-06T09:00:00Z"
}
```

**Errors:** `404` Alert not found | `403` Not owned by user

---

## 5. Recipes Endpoints

### GET `/recipes`
Fetch recipe suggestions for a list of ingredients.

**Auth:** Required

**Query Parameters:**
| Param | Type | Required | Description |
|---|---|---|---|
| `ingredients` | `string` | Yes | Comma-separated ingredient names: `strawberries,milk,eggs` |
| `limit` | `int` | No | Max recipes to return (default: 5, max: 10) |

**Response `200`:**
```json
{
  "recipes": [
    {
      "id": "spoonacular_id",
      "title": "Strawberry Smoothie",
      "image_url": "https://spoonacular.com/...",
      "ready_in_minutes": 5,
      "servings": 2,
      "source_url": "https://...",
      "matched_ingredients": ["strawberries", "milk"],
      "matched_count": 2,
      "missing_ingredients": ["banana"]
    }
  ],
  "cached": false
}
```

`cached: true` means response served from cache, not a live Spoonacular call.

**Errors:** `400` No ingredients provided | `503` Spoonacular API unavailable

---

## 6. Waste Log Endpoints

### GET `/waste-log`
Get waste log entries for the authenticated user.

**Auth:** Required

**Query Parameters:**
| Param | Type | Description |
|---|---|---|
| `outcome` | `string` | `CONSUMED`, `WASTED` |
| `from_date` | `date` | Start date filter (ISO 8601) |
| `to_date` | `date` | End date filter (ISO 8601) |

**Response `200`:**
```json
{
  "summary": {
    "total_items": 45,
    "consumed": 38,
    "wasted": 7,
    "waste_rate_percent": 15.6
  },
  "entries": [
    {
      "id": "uuid",
      "item_name": "Strawberries",
      "outcome": "CONSUMED",
      "recorded_at": "2025-01-06T19:00:00Z",
      "recipe_used": "Strawberry Smoothie"
    }
  ]
}
```

---

## 7. Error Response Format

All errors follow a consistent shape:

```json
{
  "error": {
    "code": "ITEM_NOT_FOUND",
    "message": "Inventory item with id 'abc-123' not found.",
    "status": 404
  }
}
```

**Standard Error Codes:**
| Code | HTTP Status | Meaning |
|---|---|---|
| `VALIDATION_ERROR` | 422 | Request body failed Pydantic validation |
| `UNAUTHORIZED` | 401 | Missing or invalid token |
| `FORBIDDEN` | 403 | Resource not owned by requesting user |
| `NOT_FOUND` | 404 | Resource does not exist |
| `OCR_FAILURE` | 422 | ML pipeline could not extract text |
| `UPSTREAM_ERROR` | 503 | External API (Spoonacular) unavailable |
| `RATE_LIMITED` | 429 | Too many requests |

---

## 8. Rate Limiting

| Endpoint | Limit |
|---|---|
| `POST /receipts/upload` | 10 requests / hour / user |
| `GET /recipes` | 50 requests / hour / user |
| All other endpoints | 200 requests / hour / user |

Rate limit headers returned on all responses:
```
X-RateLimit-Limit: 200
X-RateLimit-Remaining: 147
X-RateLimit-Reset: 1704499200
```
