# PRD.md — Product Requirements Document
## Smart-Stock: AI-Powered Inventory & Waste Reduction System

**Version:** 1.0  
**Status:** MVP Definition  
**Author:** Smart-Stock Team  

---

## 1. Problem Statement

Households waste approximately 30% of purchased food annually. Two root causes drive this:

- **Memory Gap** — Users at grocery stores have no visibility into current inventory, leading to duplicate purchases and overstocking.
- **Expiration Oversight** — Items pushed to the back of shelves or fridge are forgotten until they are unsafe for consumption.

Existing solutions (manual tracking apps, smart fridges) fail due to high friction — users do not maintain manual logs consistently, and smart hardware is expensive and non-portable.

---

## 2. Solution Framing

Smart-Stock creates a **Digital Twin** of the user's refrigerator/pantry by extracting inventory data automatically from grocery receipts using a trained ML pipeline (OCR + NER), and maintaining a live expiry-aware inventory with proactive alerts and recipe suggestions.

**Core differentiator:** The OCR and Named Entity Recognition (NER) pipeline is trained in-house — not a third-party API wrapper. This gives the system high accuracy on messy real-world receipt formats and constitutes the primary technical artifact.

---

## 3. Target Users

| Persona | Description |
|---|---|
| Primary | Households of 1–4 people who regularly grocery shop |
| Secondary | Meal-prep focused individuals tracking perishable stock |
| Out of scope | Restaurants, commercial kitchens, B2B inventory management |

---

## 4. MVP Scope

### 4.1 In Scope (MVP)

| Feature | Description | Priority |
|---|---|---|
| Receipt Upload & OCR | User uploads receipt image; system extracts text via trained TrOCR model | P0 |
| Food Entity Extraction | NER model maps raw receipt tokens to canonical food items with quantity/unit | P0 |
| Expiry Prediction | Shelf-life engine assigns "Best Before" date per extracted item | P0 |
| Virtual Fridge Dashboard | React UI showing all inventory items, quantities, expiry countdowns, urgency tiers | P0 |
| Manual CRUD | User can add, edit, or delete inventory items manually | P0 |
| Expiry Alerts | Push/in-app notification triggered 48 hours before expiry | P1 |
| At-Risk Recipes | Fetches recipe suggestions based on items expiring within 48 hours | P1 |
| Waste Tracker | Logs whether items were cooked or expired; displays waste stats | P2 |

### 4.2 Out of Scope (MVP)

- Barcode scanning
- Multi-user household syncing
- Mobile app (React Native — post-MVP)
- Price tracking / budget features
- Integration with grocery delivery APIs

---

## 5. Feature Requirements

### 5.1 Receipt Scanner

- User uploads a JPEG/PNG/PDF image of a grocery receipt.
- System runs the image through the trained TrOCR model to extract raw text.
- Raw text is passed through the DistilBERT NER model to extract food entities.
- Normalization layer maps tokens like `"ORG STRWBRY 1LB"` → `{item: "Strawberries", quantity: 1, unit: "lb"}`.
- Extracted items are presented to the user for confirmation before saving to inventory.
- **Accuracy requirement:** ≥ 85% item-level extraction accuracy on standard retail receipts.

### 5.2 Virtual Fridge Dashboard

- Displays all inventory items grouped by category (Produce, Dairy, Meat, Pantry, Frozen).
- Each item card shows: name, quantity, unit, purchase date, predicted expiry date, days remaining.
- Color-coded urgency system:
  - 🟢 Green — > 5 days remaining
  - 🟡 Yellow — 2–5 days remaining
  - 🔴 Red — < 2 days remaining / expired
- Items are sortable by expiry date, category, or name.
- Pagination or virtual scroll for large inventories.

### 5.3 Expiry Prediction Engine

- Assigns shelf-life estimates based on item category and storage method.
- Uses a hybrid approach: rule-based baseline + learned adjustments from `shelf_life_reference` table.
- Storage context input: Fridge, Freezer, Pantry (user-selectable per item, defaulted by category).
- Confidence score exposed via API for transparency.

### 5.4 Smart Alerts

- Scheduler runs daily at 08:00 UTC.
- Identifies all inventory items with expiry ≤ 48 hours.
- Sends in-app notification with item list.
- Each alert links directly to recipe suggestions for those items.
- Alerts dismissed by user are not re-triggered for the same expiry window.

### 5.5 Waste-Free Recipes

- Triggered by: (a) alert click, (b) manual "What can I cook?" button on item card.
- Sends at-risk ingredient list to Spoonacular API.
- Returns up to 5 ranked recipes sorted by number of matching at-risk ingredients used.
- User can mark a recipe as "Cooked" — this removes used ingredients from inventory.

### 5.6 Waste Tracker

- Tracks two terminal states per item: `CONSUMED` (cooked/eaten) or `WASTED` (expired/discarded).
- Dashboard widget shows weekly/monthly waste ratio.
- Success metric: reduction in `WASTED` events over time correlates with engagement.

---

## 6. UX Expectations

- **Upload flow** must complete (upload → extract → confirm) in under 10 seconds on a standard connection.
- **Dashboard** must load initial inventory in under 2 seconds.
- **Confirmation step** after OCR extraction is mandatory — user must review and approve items before they are saved. No silent auto-save.
- Mobile-responsive design required (web-first, but must be usable on a phone browser).
- Offline state: display cached inventory if backend unreachable; disable mutation operations with a clear status banner.

---

## 7. Success Metrics

| Metric | Definition | Target |
|---|---|---|
| OCR Item Accuracy | % of receipt line items correctly identified | ≥ 85% |
| NER F1 Score | F1 on food entity extraction (test set) | ≥ 0.88 |
| Expiry Prediction MAE | Mean Absolute Error in days vs. actual shelf life | ≤ 1.5 days |
| Alert Engagement Rate | % of alerts that result in a recipe view | ≥ 40% |
| Waste Reduction Proxy | Ratio of CONSUMED to WASTED events per user | > 3:1 |
| Dashboard Load Time | P95 load time for inventory dashboard | < 2s |

---

## 8. Constraints

- Training infrastructure: Google Colab / Kaggle (no dedicated GPU cluster).
- No PII collection beyond email for authentication.
- Receipt images must be deleted from server within 24 hours of processing (privacy).
- Spoonacular API free tier: 150 requests/day — implement caching layer to stay within limits.
