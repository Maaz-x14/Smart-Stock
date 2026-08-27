# SmartStock — AI-Powered Inventory & Waste Reduction

Households waste ~30% of purchased food annually. SmartStock eliminates that by turning a grocery receipt photo into a live, expiry-aware inventory — with zero manual data entry.

---

## What it does

1. You photograph your grocery receipt
2. SmartStock extracts every item via a custom-built OCR + row-parsing + item-field-extraction pipeline (no third-party Vision API)
3. Each item gets a predicted expiry date based on category and storage context
4. A virtual fridge dashboard tracks everything, color-coded by urgency
5. 48 hours before something expires, you get a push notification with a recipe that uses it

---

## Why the ML is built in-house

Most receipt apps wrap Google Cloud Vision or AWS Textract as a black-box API. SmartStock's pipeline is evaluated and owned end-to-end, including reversing earlier decisions when real-world testing called for it:

- **OCR:** a fine-tuned TrOCR model was built in-house first (see OCR_Training.md); pretrained PaddleOCR was evaluated against it and won on both accuracy and CPU latency, so it's used as-is — a deliberate, measured choice, not a fallback to an unowned API.
- **Row parsing:** testing against real Pakistani retail receipts found that no two stores share a receipt header format (`Quantity | Price | Discount | Total` vs `Qty | Item | Rate | Amount` vs merged/multi-line variants). A header-driven parser reads each receipt's own column layout instead of assuming one fixed format.
- **Item field extraction:** a fine-tuned DistilBERT NER model was built and evaluated first (F1 0.907 — see NER_Training.md), then **retired** once row parsing made its original task (tagging quantity/unit/price within a line) obsolete. What's left — unit, brand, is_food — is solved with regex, a fuzzy lexicon, and an LLM gate, none of which need training data or a served model.
- Full control over the pipeline means measurable accuracy targets, swappable components on evidence, and the willingness to undo prior work when testing shows it's the wrong tool.

---

## ML Pipeline

```
Receipt Image
     │
     ▼
┌─────────────────────────────────┐
│  Stage 1: OCR                   │
│  PaddleOCR (pretrained PP-OCRv6)│
│  → text + bounding boxes        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 1.5: Row Reconstruction  │
│  Deskew + y-position clustering │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 1.6: Prefilter           │
│  Drops metadata rows            │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 1.7: Row Parser          │
│  Header-driven column mapping   │
│  → qty, price, discount, total  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 2: Item Field Extraction │
│  Regex unit + brand lexicon     │
│  + LLM is_food gate             │
└────────────────┬────────────────┘
     is_food=false → surfaced, stops here
     is_food=true  │
                 ▼
┌─────────────────────────────────┐
│  Stage 3: Normalization         │
│  Lookup → Fuzzy → LLM fallback  │
│  "ORG STRWBRY" → "Strawberries" │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 4: Expiry Prediction     │
│  Rule-based + shelf-life DB     │
│  Target: MAE ≤ 1.5 days         │
└────────────────┬────────────────┘
                 │
                 ▼
         Inventory Item
  { name, brand, qty, unit, expiry, confidence }
```

---

## Current ML Status

### Stage 1 — OCR ✅ Complete

| Model | PaddleOCR (PP-OCRv6, pretrained — no fine-tuning) |
|-------|------------------------------------------|
| Config | Small det/rec models; doc-orientation, unwarping, textline-orientation disabled |
| CPU latency | ~5-6s/receipt |
| Note | Fine-tuned TrOCR (CER 0.0631) was tried first — see OCR_Training.md — but pretrained PaddleOCR beat it on both accuracy and speed |

### Stage 1.5–1.7 — Row Reconstruction, Prefilter, Row Parser ✅ Complete

| Component | Status |
|-----------|--------|
| Row reconstruction | Deskew correction + y-clustering, validated on 4 real receipts |
| Prefilter | Rule-based metadata drop, relocated from `ner/` to `parsing/` |
| Row parser | Fuzzy header detection + x-position column mapping, output matches ground truth on all fields across the 4-receipt sample |

One receipt format (multi-line header with a Tax(%) column) remains unsupported — deferred, tracked separately.

### Stage 2 — Item Field Extraction ✅ Complete

| Component | Approach |
|-----------|----------|
| Unit | Regex extraction from item_name, including known OCR corruption patterns |
| Brand | Fuzzy lexicon match (rapidfuzz), same infra as Stage 3 |
| is_food | LLM gate — binary classification, no training data needed |

**Retired:** fine-tuned DistilBERT NER (F1 0.907). See NER_Training.md — real training work, preserved as historical record, but solving a task (full-line entity tagging) that no longer exists once Row Parser handles quantity/price/discount/total structurally.

### Stage 3 — Normalization

Three-pass: direct abbreviation lookup (~800 entries, handles ~65% of cases) → fuzzy match via rapidfuzz → LLM fallback for unresolved tokens. Now only runs for items Stage 2 classified as food. Not yet re-tested against real Stage 1.5–2 output.

### Stage 4 — Expiry Prediction

Rule-based lookup against a shelf-life reference database with 180+ items. Confidence score exposed per item — items below 0.60 are flagged for user review in the confirmation modal. Not yet re-tested against real pipeline output.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML training | PyTorch, HuggingFace Transformers, PEFT (LoRA), Kaggle T4 — used historically for OCR (TrOCR, retired) and NER (DistilBERT, retired). No component in the current pipeline requires training. |
| Backend | FastAPI (Python), PostgreSQL, SQLAlchemy 2.0, Alembic migrations |
| ML service | FastAPI (Python) — PaddleOCR (own runtime) for OCR; pure Python for row reconstruction/parsing; LLM API calls for is_food gate and Normalization Pass 3 |
| Frontend | React (web-first, mobile-responsive) |
| Recipe suggestions | Spoonacular API |
| Notifications | Push via scheduled daily job |

---

## Repo Structure

```
Smart-Stock/
├── app/                    # Frontend (React)
├── ml_service/             # FastAPI ML inference service
│   ├── ocr/                # PaddleOCR loader + row reconstruction (Stage 1, 1.5)
│   ├── parsing/             # Prefilter + row parser (Stage 1.6, 1.7)
│   ├── item_extraction/     # Unit regex, brand lexicon, LLM is_food gate (Stage 2)
│   ├── normalization/       # Abbreviation map + fuzzy + LLM fallback (Stage 3)
│   ├── expiry/               # Shelf-life lookup + confidence scoring (Stage 4)
│   ├── ner/archive/          # Retired DistilBERT NER — historical record only
│   └── pipeline.py          # Orchestrator (rewiring in progress)
├── db/seeds/                # Database seed data
├── migrations/               # Alembic DB migrations
├── PRD.md                   # Product requirements
├── ML_Pipeline.md           # Full pipeline architecture
├── OCR_Training.md          # PaddleOCR setup + TrOCR fine-tuning guide + full training history
├── NER_Training.md          # Retired DistilBERT NER — historical record
├── Architecture.md          # System architecture
├── API_Spec.md              # REST API specification
└── DB_Schema.md             # Database schema
```

---

## Accuracy Targets

| Stage | Metric | Target | Current |
|-------|--------|--------|---------|
| OCR | Real-receipt accuracy | Beat fine-tuned TrOCR baseline | ✅ PaddleOCR (pretrained) — correct on item/price/qty lines across all test receipts; CER/WER not applicable (pretrained, no held-out labeled set) |
| Row Parser | Field extraction accuracy | Correctly extract qty/price/discount/total | ✅ 100% match on 4-receipt sample; not yet measured at scale |
| Item Field Extraction | is_food / unit / brand accuracy | Not yet formally targeted | Not yet measured — no labeled sample built |
| Normalization | Match rate | ≥ 80% | Built, not yet tested against real Stage 1.5–2 output |
| Expiry | MAE (days) | ≤ 1.5 | Built, not yet tested against real pipeline output |
| End-to-end | Item accuracy | ≥ 85% | Not yet measured — blocked on full pipeline wiring (`pipeline.py`) + Stage 3/4 real-data validation |

---

## Inference Latency Budget (CPU, per receipt)

**Not yet re-measured end-to-end since the Stage 2 restructure.**

| Stage | Target |
|-------|--------|
| Image preprocessing | < 200ms |
| OCR (PaddleOCR) | ~5-6s |
| Row reconstruction + prefilter + row parser | Not yet measured — expected negligible (pure Python, no model) |
| Item Field Extraction | Not yet measured — LLM call (is_food) likely dominates |
| Normalization | < 300ms (target, unvalidated) |
| Expiry prediction | < 100ms (target, unvalidated) |
| **Total** | **Needs full re-measurement — not yet run end-to-end on real data** |
