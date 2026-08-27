# PRD.md — Product Requirements Document
## Smart-Stock: AI-Powered Inventory & Waste Reduction System

**Version:** 1.1 — updated after removing fine-tuned NER; pipeline now uses header-driven row parsing + regex/lexicon/LLM item field extraction
**Status:** MVP Definition
**Author:** Maaz Ahmad

---

## 1. Problem Statement

Households waste approximately 30% of purchased food annually. Two root causes drive this:

- **Memory Gap** — Users at grocery stores have no visibility into current inventory, leading to duplicate purchases and overstocking.
- **Expiration Oversight** — Items pushed to the back of shelves or fridge are forgotten until they are unsafe for consumption.

Existing solutions (manual tracking apps, smart fridges) fail due to high friction — users do not maintain manual logs consistently, and smart hardware is expensive and non-portable.

---

## 2. Solution Framing

Smart-Stock creates a **Digital Twin** of the user's refrigerator/pantry by extracting inventory data automatically from grocery receipts using an owned ML pipeline (OCR + structural row parsing + lightweight item field extraction), and maintaining a live expiry-aware inventory with proactive alerts and recipe suggestions.

**Core differentiator:** The pipeline is owned and evaluated end-to-end, not a black-box third-party API wrapper. OCR uses PaddleOCR (pretrained) — chosen after benchmarking against an in-house fine-tuned TrOCR model and measurably outperforming it on both accuracy and CPU latency (see OCR_Training.md). Item field extraction (unit, brand, is_food) deliberately does **not** use a fine-tuned model — after testing against real Pakistani retail receipts, it became clear that no two stores share a receipt header format, so a header-driven row parser (reading each receipt's own column layout) handles quantity/price/discount/total, and a combination of regex, fuzzy lexicon matching, and an LLM gate handles what's left in the item name. A previously fine-tuned NER model (DistilBERT, F1 0.907) was retired once this restructuring made its task obsolete — see ML_Pipeline.md §0 and NER_Training.md for the full record. Full ownership of the pipeline gives measurable accuracy targets and the ability to swap components on evidence, including reversing prior work when the evidence calls for it.

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
| Receipt Upload & OCR | User uploads receipt image; system extracts text + layout via PaddleOCR | P0 |
| Row Reconstruction & Parsing | Deskew-corrected row clustering, then header-driven column mapping to extract quantity/price/discount/total per item | P0 |
| Item Field Extraction | Regex unit extraction, fuzzy brand lexicon match, LLM is_food gate on each item name | P0 |
| Expiry Prediction | Shelf-life engine assigns "Best Before" date per extracted item (food items only) | P0 |
| Virtual Fridge Dashboard | React UI showing all inventory items, quantities, expiry countdowns, urgency tiers | P0 |
| Manual CRUD | User can add, edit, or delete inventory items manually | P0 |
| Expiry Alerts | Push/in-app notification triggered 48 hours before expiry | P1 |
| At-Risk Recipes | Fetches recipe suggestions based on items expiring within 48 hours | P1 |
| Waste Tracker | Logs whether items were cooked or expired; displays waste stats | P2 |

### 4.2 Out of Scope (MVP)

- Barcode scanning as a primary input method (barcode text embedded in item names is stripped during Item Field Extraction, not scanned directly)
- Multi-user household syncing
- Mobile app (React Native — post-MVP)
- Price tracking / budget features (price is extracted internally by the row parser but not persisted or exposed — see DB_Schema.md, API_Spec.md)
- Integration with grocery delivery APIs
- Receipt formats with multi-line or non-standard headers (e.g. Tax(%) column receipts) — deferred, see ML_Pipeline.md §5

---

## 5. Feature Requirements

### 5.1 Receipt Scanner

- User uploads a JPEG/PNG/PDF image of a grocery receipt.
- System runs the image through PaddleOCR to extract raw text and bounding boxes.
- Text boxes are deskew-corrected and clustered into logical rows.
- Metadata rows (store info, GST numbers, totals, etc.) are dropped by a rule-based prefilter.
- Remaining rows are parsed via each receipt's own detected column headers into `{item_name, quantity, price, discount, total}`.
- Each `item_name` passes through Item Field Extraction: regex unit parsing, fuzzy brand lexicon matching, and an LLM gate classifying whether the item is food.
- Food items are normalized (`"ORG STRWBRY 1LB"` → `{item: "Strawberries", quantity: 1, unit: "lb"}`) and assigned a predicted expiry date.
- Non-food items (e.g. pharmacy products on a grocery receipt) are surfaced to the user flagged as excluded, not silently dropped.
- Extracted items are presented to the user for confirmation before saving to inventory.
- **Accuracy requirement:** ≥ 85% item-level extraction accuracy on standard retail receipts.

### 5.2 Virtual Fridge Dashboard

- Displays all inventory items grouped by category (Produce, Dairy, Meat, Pantry, Frozen).
- Each item card shows: name, brand (if identified), quantity, unit, purchase date, predicted expiry date, days remaining.
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
- Only runs for items classified as food by Item Field Extraction — non-food items never reach this stage.

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
- **Confirmation step** after OCR extraction is mandatory — user must review and approve items before they are saved. No silent auto-save. Non-food items are shown, flagged, and excluded by default rather than hidden entirely.
- Mobile-responsive design required (web-first, but must be usable on a phone browser).
- Offline state: display cached inventory if backend unreachable; disable mutation operations with a clear status banner.

---

## 7. Success Metrics

| Metric | Definition | Target |
|---|---|---|
| OCR Item Accuracy | % of receipt line items correctly identified | ≥ 85% |
| Row Parser Field Accuracy | % of quantity/price/discount/total fields correctly extracted vs. ground truth | Validated 100% on initial 4-receipt sample; not yet measured at scale |
| is_food Classification Accuracy | % of items correctly classified as food/not-food | Not yet measured — no labeled sample built |
| Expiry Prediction MAE | Mean Absolute Error in days vs. actual shelf life | ≤ 1.5 days |
| Alert Engagement Rate | % of alerts that result in a recipe view | ≥ 40% |
| Waste Reduction Proxy | Ratio of CONSUMED to WASTED events per user | > 3:1 |
| Dashboard Load Time | P95 load time for inventory dashboard | < 2s |

**Retired:** NER F1 Score target — no trained model in the pipeline as of v1.1; nothing to score. Historical DistilBERT result (F1 0.907) preserved in NER_Training.md.

---

## 8. Constraints

- Training infrastructure: Google Colab / Kaggle (no dedicated GPU cluster) — no longer load-bearing for the current pipeline (no model requires training), but retained as a constraint should future components need it.
- No PII collection beyond email for authentication.
- Receipt images must be deleted from server within 24 hours of processing (privacy).
- Spoonacular API free tier: 150 requests/day — implement caching layer to stay within limits.
- Item Field Extraction's is_food gate and Normalization's Pass 3 both depend on LLM API availability — a fallback behavior for LLM API downtime is not yet defined (open item).
