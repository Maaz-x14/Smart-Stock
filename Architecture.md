# Architecture.md — High-Level System Design
## Smart-Stock

**Version:** 1.0

---

## 1. System Overview

Smart-Stock is a full-stack web application with an embedded ML pipeline. The system is composed of four primary layers:

1. **React Frontend** — User-facing dashboard and upload interface
2. **FastAPI Backend** — Business logic, orchestration, scheduling
3. **ML Pipeline** — OCR → NER → Normalization → Expiry Prediction
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
│   │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │      │
│   │  │  Receipt   │  │   Virtual    │  │   Waste / Alert │  │      │
│   │  │  Upload UI │  │  Fridge View │  │   Dashboard     │  │      │
│   │  └────────────┘  └──────────────┘  └─────────────────┘  │      │
│   └──────────────────────────┬───────────────────────────────┘      │
└─────────────────────────────-│─────────────────────────────────────┘
                               │  HTTPS / REST + WebSocket
┌──────────────────────────────▼──────────────────────────────────────┐
│                         API LAYER                                   │
│                                                                     │
│              FastAPI (Python 3.11) + Uvicorn                        │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  /receipts   │  │  /inventory  │  │  /recipes  │  /alerts    │  │
│  │  (upload,    │  │  (CRUD,      │  │  (fetch,   │  (schedule, │  │
│  │   process)   │  │   list)      │  │   suggest) │   dismiss)  │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┴─────┬──────┘  │
│         │                 │                 │            │          │
└─────────│─────────────────│─────────────────│────────────│──────────┘
          │                 │                 │            │
┌─────────▼─────────────────│─────────────────│────────────│──────────┐
│                    ML PIPELINE LAYER          │            │         │
│                                              │            │         │
│  ┌──────────────────────────────────────┐   │            │         │
│  │            Receipt Image             │   │            │         │
│  │                  │                   │   │            │         │
│  │         ┌────────▼────────┐          │   │            │         │
│  │         │  TrOCR (fine-   │          │   │            │         │
│  │         │  tuned)         │          │   │            │         │
│  │         └────────┬────────┘          │   │            │         │
│  │                  │  raw text         │   │            │         │
│  │         ┌────────▼────────┐          │   │            │         │
│  │         │  DistilBERT NER │          │   │            │         │
│  │         │  (fine-tuned)   │          │   │            │         │
│  │         └────────┬────────┘          │   │            │         │
│  │                  │  entities         │   │            │         │
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
│                       DATA LAYER                                    │
│                                                                     │
│                     PostgreSQL (via SQLAlchemy)                     │
│                                                                     │
│   ┌──────────┐  ┌──────────────────┐  ┌────────────────────────┐   │
│   │  users   │  │ inventory_items  │  │  shelf_life_reference  │   │
│   └──────────┘  └──────────────────┘  └────────────────────────┘   │
│   ┌──────────┐  ┌──────────────────┐                               │
│   │  alerts  │  │  waste_log       │                               │
│   └──────────┘  └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │        External Services         │
               │                                  │
               │  Spoonacular API (recipes)        │
               │  SMTP / Push (notifications)      │
               └──────────────────────────────────┘
```

---

## 3. Module Breakdown

### 3.1 Frontend (React + TypeScript)

| Module | Responsibility |
|---|---|
| `ReceiptUpload` | Handles file selection, preview, upload POST, and item confirmation modal |
| `FridgeView` | Renders inventory grid with category grouping, expiry color-coding, sort/filter controls |
| `ItemCard` | Individual item display: name, qty, expiry bar, CRUD actions, "Cook with this" button |
| `AlertPanel` | Displays active expiry alerts, links to recipe view |
| `RecipeModal` | Shows fetched recipes for at-risk ingredients; "Mark as Cooked" action |
| `WasteTracker` | Visualizes CONSUMED vs WASTED ratios over time (Recharts) |
| `AuthContext` | JWT token storage, login/logout, route guards |

**State Management:** React Query for server state, Zustand for local UI state.

### 3.2 Backend (FastAPI)

| Module | Responsibility |
|---|---|
| `routers/receipts.py` | Receipt upload endpoint; orchestrates ML pipeline call |
| `routers/inventory.py` | Full CRUD for inventory items |
| `routers/recipes.py` | Proxies Spoonacular API with caching layer |
| `routers/alerts.py` | Alert fetch, creation, and dismissal |
| `routers/auth.py` | JWT-based authentication |
| `services/ml_service.py` | Loads models, runs OCR → NER → Normalization → Expiry pipeline |
| `services/scheduler.py` | APScheduler daily job: scans expiry, creates alerts |
| `services/spoonacular.py` | Spoonacular API client with Redis/in-memory cache |
| `models/` | SQLAlchemy ORM models |
| `schemas/` | Pydantic request/response schemas |

### 3.3 ML Pipeline

Detailed in `ML_Pipeline.md`. Summary:

```
Receipt Image
     │
     ▼
TrOCR (fine-tuned on SROIE/CORD)
     │  Raw extracted text
     ▼
DistilBERT NER (fine-tuned on annotated receipt corpus)
     │  Entities: FOOD_ITEM, QUANTITY, UNIT, BRAND
     ▼
Normalization Layer (fuzzy match + lookup table)
     │  Canonical: {item: "Strawberries", qty: 1, unit: "lb"}
     ▼
Expiry Prediction Engine
     │  {item, storage_type} → predicted_expiry_date, confidence
     ▼
Structured Output → API Response
```

### 3.4 Database (PostgreSQL)

Detailed in `DB_Schema.md`. Five core tables:
- `users` — authentication and profile
- `inventory_items` — live inventory with expiry metadata
- `shelf_life_reference` — canonical shelf-life lookup per food category
- `alerts` — expiry alert records per user
- `waste_log` — terminal state events (CONSUMED / WASTED)

---

## 4. Data Flow: Receipt Upload

```
User uploads receipt image
        │
        ▼
POST /api/receipts/upload
        │
        ▼
FastAPI saves image to temp storage
        │
        ▼
ml_service.run_pipeline(image_path)
    ├── TrOCR extracts raw text
    ├── NER model extracts entities
    ├── Normalization maps to canonical items
    └── Expiry engine predicts best-before dates
        │
        ▼
Returns: List[ExtractedItem] (unconfirmed)
        │
        ▼
Frontend shows confirmation modal
        │
User confirms / edits / removes items
        │
        ▼
POST /api/inventory/batch-create
        │
        ▼
Items saved to inventory_items table
        │
Image deleted from temp storage
```

---

## 5. Data Flow: Expiry Alert Cycle

```
APScheduler — runs daily @ 08:00 UTC
        │
        ▼
Query: SELECT * FROM inventory_items
       WHERE expiry_date <= NOW() + INTERVAL '48 hours'
       AND status = 'ACTIVE'
        │
        ▼
For each user with at-risk items:
    Create alert record in `alerts` table
    Send in-app notification (WebSocket push)
        │
        ▼
User sees alert → clicks "Get Recipes"
        │
        ▼
GET /api/recipes?ingredients=[list]
        │
        ▼
Check cache → if miss → call Spoonacular API
        │
        ▼
Return ranked recipes
        │
User marks recipe as "Cooked"
        │
        ▼
PATCH /api/inventory/bulk-consume
    → Sets item status = 'CONSUMED' in waste_log
```

---

## 6. Authentication Flow

- **Method:** JWT (JSON Web Tokens) via `python-jose`
- Access token: 30-minute expiry
- Refresh token: 7-day expiry, stored in HttpOnly cookie
- All `/api/*` routes except `/auth/login` and `/auth/register` require `Authorization: Bearer <token>`

---

## 7. Deployment Architecture (Target)

```
┌──────────────────────────────────────────────────────────┐
│                        Render / Railway                  │
│                                                          │
│   ┌─────────────────────┐   ┌──────────────────────┐    │
│   │  FastAPI + ML Models │   │  PostgreSQL (managed) │    │
│   │  (Docker container)  │   │                      │    │
│   └─────────────────────┘   └──────────────────────┘    │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                       Vercel                             │
│              React Frontend (Static Build)               │
└──────────────────────────────────────────────────────────┘
```

**ML Model Hosting:** Models serialized to ONNX or TorchScript, loaded at startup in the FastAPI container. Inference is synchronous for MVP; async job queue (Celery + Redis) added post-MVP for scale.

---

## 8. Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| ML framework | PyTorch + HuggingFace Transformers | Best ecosystem for TrOCR + DistilBERT fine-tuning |
| API framework | FastAPI | Native async, Pydantic validation, Python for ML co-location |
| Frontend state | React Query + Zustand | Server state and UI state separated cleanly |
| DB ORM | SQLAlchemy 2.0 | Type-safe, async-compatible |
| Model serialization | ONNX | Faster CPU inference for deployment without GPU |
| Receipt image storage | Temp only (deleted post-processing) | Privacy; no long-term image retention |
