# SmartStock — AI-Powered Inventory & Waste Reduction

Households waste ~30% of purchased food annually. SmartStock eliminates that by turning a grocery receipt photo into a live, expiry-aware inventory — with zero manual data entry.

---

## What it does

1. You photograph your grocery receipt
2. SmartStock extracts every item via a custom-trained OCR + NER pipeline (no third-party Vision API)
3. Each item gets a predicted expiry date based on category and storage context
4. A virtual fridge dashboard tracks everything, color-coded by urgency
5. 48 hours before something expires, you get a push notification with a recipe that uses it

---

## Why the ML is built in-house

Most receipt apps wrap Google Cloud Vision or AWS Textract as a black-box API. SmartStock's pipeline is evaluated and owned end-to-end, even where the best-performing model turned out to be pretrained rather than fine-tuned:

- OCR: a fine-tuned TrOCR model was built in-house first (see OCR_Training.md); pretrained PaddleOCR was evaluated against it and won on both accuracy and CPU latency, so it's used as-is — a deliberate, measured choice, not a fallback to an unowned API
- Custom NER lets us extract food entities, quantities, and units in one pass without post-processing hacks
- Full control over the pipeline means measurable accuracy targets and swappable components, not a black-box API response

---

## ML Pipeline

```
Receipt Image
     │
     ▼
┌─────────────────────────────────┐
│  Stage 1: OCR                   │
│  PaddleOCR (pretrained PP-OCRv6)│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 2: NER                   │
│  DistilBERT (fine-tuned)        │
│  Current: F1 0.907 ✅           │
│  Target:  F1 ≥ 0.88             │
└────────────────┬────────────────┘
                 │
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
  { name, qty, unit, expiry, confidence }
```

---

## Current ML Status

### Stage 1 — OCR ✅ Complete

| Model | PaddleOCR (PP-OCRv6, pretrained — no fine-tuning) |
|-------|------------------------------------------|
| Config | Small det/rec models; doc-orientation, unwarping, textline-orientation disabled |
| CPU latency | ~5-6s/receipt |
| Note | Fine-tuned TrOCR (CER 0.0631) was tried first — see OCR_Training.md — but pretrained PaddleOCR beat it on both accuracy and speed |


### Stage 2 — NER ✅ Complete

| Model | distilbert-base-uncased, fine-tuned |
|-------|--------------------------------------|
| Entities | FOOD_ITEM, QUANTITY, UNIT, BRAND, PRICE, OTHER |
| F1 score | **0.907** |

### Stage 3 — Normalization

Three-pass: direct abbreviation lookup (~800 entries, handles ~65% of cases) → fuzzy match via rapidfuzz → LLM fallback for unresolved tokens.

### Stage 4 — Expiry Prediction

Rule-based lookup against a shelf-life reference database with 180+ items. Confidence score exposed per item — items below 0.60 are flagged for user review in the confirmation modal.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML training | PyTorch, HuggingFace Transformers, PEFT (LoRA), Kaggle T4 — NER stage only; OCR (PaddleOCR) is pretrained, no training infra used |
| Backend | FastAPI (Python), PostgreSQL, SQLAlchemy 2.0, Alembic migrations |
| ML service | FastAPI (Python) — PaddleOCR (own runtime) for OCR, ONNX for NER |
| Frontend | React (web-first, mobile-responsive) |
| Recipe suggestions | Spoonacular API |
| Notifications | Push via scheduled daily job |

---

## Repo Structure

```
Smart-Stock/
├── app/                    # Frontend (React)
├── ml_service/             # FastAPI ML inference service
│   ├── ocr/                # PaddleOCR loader + preprocessor
│   ├── ner/                # DistilBERT NER + entity parser
│   ├── normalization/      # Abbreviation map + fuzzy + LLM fallback
│   └── expiry/             # Shelf-life lookup + confidence scoring
├── db/seeds/               # Database seed data
├── migrations/             # Alembic DB migrations
├── PRD.md                  # Product requirements
├── ML_Pipeline.md          # Full pipeline architecture
├── OCR_Training.md         # PaddleOCR setup + TrOCR fine-tuning guide + full training history
├── NER_Training.md         # DistilBERT NER training guide
├── Architecture.md         # System architecture
├── API_Spec.md             # REST API specification
└── DB_Schema.md            # Database schema
```

---

## Accuracy Targets

| Stage | Metric | Target | Current |
|-------|--------|--------|---------|
| OCR | Real-receipt accuracy | Beat fine-tuned TrOCR baseline | ✅ PaddleOCR (pretrained) — correct on item/price/qty lines across all test receipts; CER/WER not applicable (pretrained, no held-out labeled set) |
| NER | F1 | ≥ 0.88 | **0.907** ✅ (DistilBERT — alternatives being evaluated, not yet locked in) |
| Normalization | Match rate | ≥ 80% | Built, not yet tested against real OCR→NER output |
| Expiry | MAE (days) | ≤ 1.5 | Built, not yet tested against real pipeline output |
| End-to-end | Item accuracy | ≥ 85% | Not yet measured — blocked on NER lock-in + Stage 3/4 real-data validation |

---

## Inference Latency Budget (CPU, per receipt)

| Stage | Target |
|-------|--------|
| Image preprocessing | < 200ms |
| OCR (PaddleOCR) | < ~5-6s |
| NER (DistilBERT ONNX) | < 500ms |
| Normalization | < 300ms |
| Expiry prediction | < 100ms |
| **Total** | **< 10s** |