# Architecture.md — High-Level System Design

## Smart-Stock

**Version:** 2.1 — application layer (backend/frontend/CI/Docker) built around the already-validated ML pipeline. v2.0's NER retirement (row-parser + item field extraction) is unchanged; see ML_Pipeline.md for that rationale.

**Note:** this document describes the target/high-level design. Where the as-built code differs in a material way, that's called out inline rather than silently updating the diagrams — see §3.2–§3.4 and §6–§7 in particular. For a precise file-by-file account of what was actually built, see `HANDOFF.md`.

---

## 1. System Overview

Smart-Stock is a full-stack web application with an embedded ML pipeline. The system is composed of four primary layers:

1. **React Frontend** — User-facing dashboard and upload interface
2. **FastAPI Backend** — Business logic, orchestration, scheduling
3. **ML Pipeline** — OCR → Row Reconstruction → Prefilter → Row Parser → Item Field Extraction → Normalization → Expiry Prediction
4. **PostgreSQL Database** — Persistent storage for inventory, shelf-life data, alerts

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                               │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │              React + TypeScript (Vite)                   │      │
│   │                                                          │      │
│   │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐   │      │
│   │  │  Receipt   │  │   Virtual    │  │   Waste / Alert │   │      │
│   │  │  Upload UI │  │  Fridge View │  │   Dashboard     │   │      │
│   │  └────────────┘  └──────────────┘  └─────────────────┘   │      │
│   └──────────────────────────┬───────────────────────────────┘      │
└──────────────────────────────│──────────────────────────────────────┘
                               │  HTTPS / REST + WebSocket
┌──────────────────────────────▼──────────────────────────────────────┐
│                         API LAYER                                   │
│                                                                     │
│              FastAPI (Python 3.12) + Uvicorn                        │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │  /receipts   │  │  /inventory  │  │  /recipes  │  /alerts    │   │
│  │  (upload,    │  │  (CRUD,      │  │  (fetch,   │  (schedule, │   │
│  │   process)   │  │   list)      │  │   suggest) │   dismiss)  │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬─────┴──────┬──────┘   │
│         │                 │                 │            │          │
└─────────│─────────────────│─────────────────│────────────│──────────┘
          │                 └─────────────────│            │
┌─────────▼───────────────────────────────────│────────────│─────────┐
│                    ML PIPELINE LAYER        │            │         │
│                                             │            │         │
│  ┌──────────────────────────────────────┐   │            │         │
│  │            Receipt Image             │   │            │         │
│  │                  │                   │   │            │         │
│  │         ┌────────▼────────┐          │   │            │         │
│  │         │  PaddleOCR      │          │   │            │         │
│  │         │  (pretrained)   │          │   │            │         │
│  │         └────────┬────────┘          │   │            │         │
│  │                  │  text + boxes     │   │            │         │
│  │         ┌────────▼────────┐          │   │            │         │
│  │         │  Row Reconstr.  │          │   │            │         │
│  │         │ (deskew+cluster)│          │   │            │         │
│  │         └────────┬────────┘          │   │            │         │
│  │                  │  logical rows     │   │            │         │
│  │         ┌────────▼────────┐          │   │            │         │
│  │         │  Prefilter      │          │   │            │         │
│  │         │  (drop metadata)│          │   │            │         │
│  │         └────────┬────────┘          │   │            │         │
│  │                  │  item rows        │   │            │         │
│  │         ┌────────▼────────┐          │   │            │         │
│  │         │  Row Parser     │          │   │            │         │
│  │         │  (header-driven)│          │   │            │         │
│  │         └────────┬────────┘          │   │            │         │
│  │                  │  name,qty,price.. │   │            │         │
│  │         ┌────────▼────────┐          │   │            │         │
│  │         │  Item Field     │          │   │            │         │
│  │         │  Extraction     │          │   │            │         │
│  │         │  (unit/brand/   │          │   │            │         │
│  │         │   is_food)      │          │   │            │         │
│  │         └────────┬────────┘          │   │            │         │
│  │                  │  is_food=true only│   │            │         │
│  │         ┌────────▼────────┐          │   │            │         │
│  │         │  Normalization  │          │   │            │         │
│  │         │  Layer          │          │   │            │         │
│  │         └────────┬────────┘          │   │            │         │
│  │                  │  canonical items  │   │            │         │
│  │         ┌────────▼────────┐          │   │            │         │
│  │         │  Expiry Engine  │          │   │            │         │
│  │         │  (rule + ML)    │          │   │            │         │
│  │         └────────┬────────┘          │   │            │         │
│  │                  │  items + expiry   │   │            │         │
│  └──────────────────│───────────────────┘   │            │         │
│                     │                       │            │         │
└─────────────────────│───────────────────────│────────────│─────────┘
                      │                       │            │
┌─────────────────────▼───────────────────────▼────────────▼─────────┐
│                       DATA LAYER                                   │
│                                                                    │
│                     PostgreSQL (via SQLAlchemy)                    │
│                                                                    │
│   ┌──────────┐  ┌──────────────────┐  ┌────────────────────────┐   │
│   │  users   │  │ inventory_items  │  │  shelf_life_reference  │   │
│   └──────────┘  └──────────────────┘  └────────────────────────┘   │
│   ┌──────────┐  ┌──────────────────┐  ┌────────────────────────┐   │
│   │  alerts  │  │  waste_log       │  │  receipts              │   │
│   └──────────┘  └──────────────────┘  └────────────────────────┘   │
│   ┌──────────────┐  ┌──────────────────┐                           │
│   │  alert_items │  │  recipe_cache    │                           │
│   └──────────────┘  └──────────────────┘                           │
└────────────────────────────────────────────────────────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │        External Services        │
               │                                 │
               │  Spoonacular API (recipes)      │
               │  LLM API (Groq) — is_food gate, │
               │  Normalization Pass 3           │
               └─────────────────────────────────┘
```

As built, there are 8 tables, not 5 (`receipts` holds pending/unconfirmed extractions so `/confirm` can be idempotent; `alert_items` joins alerts to the inventory items they cover; `recipe_cache` backs the Spoonacular cache). There's no WebSocket or SMTP/push channel in the built backend — alerts are polled via `GET /alerts`, not pushed. See §6 for the actual auth flow and §4 for the actual upload flow.

<img src="assets/svgs/smart_stock_pipeline.svg" alt="Pipeline" width="500" />

---

## 3. Module Breakdown

### 3.1 Frontend (React + TypeScript)

As built, under `frontend/src/`:

| Module                          | Responsibility                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| `features/receipts/UploadPage` + `ScanProgress` + `ConfirmModal` | File selection, upload, animated scan progress, mandatory confirm modal before items hit inventory |
| `features/fridge/FridgePage`    | Inventory grid grouped by storage context/category, urgency tiers (word + shape, never color alone) |
| `features/alerts/AlertsPage`    | Active expiry alerts, dismiss action, links to recipes                                   |
| `features/recipes/RecipesPage`  | Fetched recipes for at-risk ingredients                                                  |
| `features/waste/WastePage`      | CONSUMED vs WASTED visualization over time (Recharts)                                    |
| `features/auth/useAuth` + `LoginPage`/`RegisterPage` | Login/register forms, calls into `authStore`                                 |
| `lib/authStore.ts`              | Zustand: access token in memory only (never localStorage)                               |
| `lib/api.ts`                    | Fetch wrapper with 401 → refresh → retry-once logic                                      |
| `components/UrgencyBadge`, `ItemTag`, `OfflineBanner`, `AppLayout`, `ProtectedRoute` | Shared UI/layout                                            |

**State Management:** TanStack Query for server state, Zustand for UI state only (upload/undo, auth token).

**Confirmation modal note:** items with `is_food: false` are surfaced (not silently dropped) with a "detected but excluded — not a food item" indicator, so the user can see what was filtered and why. See API_Spec.md §2.

### 3.2 Backend (FastAPI)

As built, under `app/`:

| Module                       | Responsibility                                                     |
| ------------------------------ | -------------------------------------------------------------------- |
| `routers/receipts.py`        | `/upload` (runs ML pipeline, stores pending `Receipt`) + `/confirm` (writes `InventoryItem` rows) |
| `routers/inventory.py`       | Full CRUD + `/bulk-consume` for inventory items                     |
| `routers/recipes.py`         | Proxies Spoonacular with a Postgres-backed cache                    |
| `routers/alerts.py`          | Alert fetch + dismiss                                                |
| `routers/auth.py`            | register/login/refresh/logout — JWT access + HttpOnly refresh cookie |
| `routers/waste.py`           | Waste log read + summary stats                                       |
| `services/receipts.py`       | Upload/confirm business logic; storage-context casing boundary (API lowercase ↔ ML pipeline capitalized) |
| `services/inventory.py`, `services/alerts.py`, `services/recipes.py`, `services/waste.py` | Business logic per domain, called by routers |
| `scheduler.py`               | APScheduler daily 08:00 UTC job: scans expiry, creates alerts (idempotent) |
| `security.py`                | bcrypt hashing, JWT create/decode                                    |
| `deps.py`                    | `get_current_user`, rate-limiting dependency                         |
| `models.py`                  | SQLAlchemy ORM models (single module, not a package)                 |
| `schemas*.py`                | Pydantic request/response schemas, split by domain                   |
| `errors.py`                  | Unified `{"error": {...}}` response shape + exception handlers       |

### 3.3 ML Pipeline

Detailed in `ML_Pipeline.md`. Summary:

```
Receipt Image
     │
     ▼
PaddleOCR (pretrained PP-OCRv6)  — text + bounding boxes
     │
     ▼
Row Reconstruction — deskew + cluster boxes into logical rows
     │
     ▼
Prefilter — drops metadata rows (GST#, Invoice#, Payments, etc.)
     │
     ▼
Row Parser — header-driven column mapping → {item_name, quantity, price, discount, total}
     │
     ▼
Item Field Extraction — regex unit + brand lexicon + LLM is_food gate
     │  is_food = false → item surfaced to user, skips remaining stages
     ▼
Normalization Layer (fuzzy match + lookup + LLM canonical naming) — is_food = true only
     │  Canonical: {item: "Strawberries", qty: 1, unit: "lb"}
     ▼
Expiry Prediction Engine
     │  {item, storage_type} → predicted_expiry_date, confidence
     ▼
Structured Output → API Response
```

**Retired:** fine-tuned DistilBERT NER. It was trained to tag FOOD/QTY/UNIT/PRICE across full single-line receipt text — a task that no longer exists once Row Parser extracts qty/price/discount/total structurally via detected column headers. See NER_Training.md for the retirement note and preserved historical record.

### 3.4 Database (PostgreSQL)

Detailed in `DB_Schema.md`. As built (reconciled in migration `f1b2c3d4e5a6`), 8 tables, all with UUID primary keys:

- `users` — authentication and profile
- `receipts` — pending/unconfirmed extractions from `/upload`, confirmed via `/confirm`
- `inventory_items` — live inventory with expiry metadata; nullable ML-only fields for manually-added items; DB-generated `urgency_tier`
- `shelf_life_reference` — canonical shelf-life lookup per food category
- `alerts` / `alert_items` — expiry alert records per user, joined to the inventory items they cover
- `waste_log` — terminal state events (CONSUMED / WASTED), FK to `inventory_items` with no cascade delete
- `recipe_cache` — Postgres-backed cache for Spoonacular lookups

---

## 4. Data Flow: Receipt Upload

```
User uploads receipt image
        │
        ▼
POST /api/v1/receipts/upload  (multipart, rate-limited: 10/h)
        │
        ▼
ml_service.pipeline.run_pipeline(image_bytes)
    ├── PaddleOCR extracts text + bounding boxes
    ├── Row reconstruction clusters boxes into logical rows (deskew-corrected)
    ├── Prefilter drops metadata rows
    ├── Row parser maps qty/price/discount/total via detected headers
    ├── Item field extraction: unit (regex), brand (fuzzy lexicon), is_food (LLM gate)
    ├── Normalization maps is_food=true items to canonical names (skipped for is_food=false)
    └── Expiry engine predicts best-before dates (skipped for is_food=false)
        │
        ▼
A `Receipt` row is written with status=pending and the extracted items as JSON
        │
        ▼
Returns: ReceiptUploadResponse (receipt_id + List[ExtractedItem], includes
         is_food=false items flagged as excluded — not silently dropped)
        │
        ▼
Frontend shows confirmation modal
        │
User confirms / edits / removes items
        │
        ▼
POST /api/v1/receipts/confirm  (receipt_id + edited item list)
        │
        ▼
Receipt marked confirmed (idempotent — repeat calls get 410, not a duplicate write);
items saved to inventory_items with lowercase storage_context
```

---

## 5. Data Flow: Expiry Alert Cycle

```
APScheduler — runs daily @ 08:00 UTC (BackgroundScheduler, in-process)
        │
        ▼
Query: SELECT * FROM inventory_items
       WHERE expiry_date <= NOW() + INTERVAL '48 hours'
        │
        ▼
For each user with at-risk items not already covered by an ACTIVE alert:
    Create alert + alert_items rows (idempotent — won't duplicate-alert the same item)
        │
        ▼
User polls GET /api/v1/alerts → sees alert → clicks "Get Recipes"
        │
        ▼
GET /api/v1/recipes?ingredients=[list]&limit=N  (rate-limited: 50/h)
        │
        ▼
Check recipe_cache (sha256 of sorted ingredients) → if miss → call Spoonacular API
        │
        ▼
Return ranked recipes (graceful 503, not 500, if Spoonacular key missing)
        │
User marks recipe as "Cooked"
        │
        ▼
PATCH /api/v1/inventory/bulk-consume
    → Writes waste_log rows with outcome=CONSUMED
```

There's no WebSocket or push channel in the built backend — alerts are pulled by the frontend via polling `GET /alerts`, not pushed server-side. That's a scope decision, not an oversight; see `HANDOFF.md` "Next steps" if real-time push is wanted later.

---

## 6. Authentication Flow

- **Method:** JWT (JSON Web Tokens) via `python-jose`, password hashing via `bcrypt`
- Access token: 30-minute expiry, returned in the response body, held in memory on the frontend (never localStorage)
- Refresh token: 7-day expiry, HttpOnly cookie scoped to `/api/v1/auth`, `secure` flag on outside local dev
- All `/api/v1/*` routes except `/auth/login` and `/auth/register` require `Authorization: Bearer <token>`
- Frontend's `lib/api.ts` auto-refreshes on a 401 and retries the request once before giving up

---

## 7. Deployment Architecture (Target)

```
┌──────────────────────────────────────────────────────────┐
│                        Render / Railway                  │
│                                                          │
│   ┌─────────────────────┐   ┌──────────────────────┐     │
│   │  FastAPI + ML Models│   │  PostgreSQL (managed)│     │
│   │  (Docker container) │   │                      │     │
│   └─────────────────────┘   └──────────────────────┘     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                       Vercel                             │
│              React Frontend (Static Build)               │
└──────────────────────────────────────────────────────────┘
```

**ML Model Hosting:** PaddleOCR runs through its own CPU runtime. No other trained model in the pipeline as of v2.0 — item field extraction and normalization use regex, fuzzy lexicon matching, and LLM API calls (no local model to serve or export). Inference is synchronous for MVP; async job queue (Celery + Redis) added post-MVP for scale.

---

## 8. Key Technical Decisions

| Decision              | Choice                              | Rationale                                                    |
| --------------------- | ------------------------------------ | -------------------------------------------------------------- |
| ML framework          | PaddleOCR + regex/fuzzy/LLM          | PaddleOCR handles OCR; no other trained model needed after row-parser redesign |
| Item field extraction | Regex (unit) + fuzzy lexicon (brand) + LLM (is_food) | Cheaper, no training data required, generalizes better to novel item text than a small fine-tuned classifier — see ML_Pipeline.md §3 for the full analysis |
| API framework          | FastAPI                             | Native async, Pydantic validation, Python for ML co-location |
| Frontend state          | React Query + Zustand               | Server state and UI state separated cleanly                  |
| DB ORM                 | SQLAlchemy 2.0                      | Type-safe, async-compatible                                  |
| Model serialization    | PaddleOCR runtime only              | No fine-tuned model left to export/serve as of v2.0 (DistilBERT NER retired) |
| Receipt image storage  | Temp only (deleted post-processing) | Privacy; no long-term image retention                        |

**Superseded decision (v1.0):** "DistilBERT fine-tuned NER, ONNX-exported for inference" — retired. See ML_Pipeline.md §3 for why, and NER_Training.md for the preserved historical record.
