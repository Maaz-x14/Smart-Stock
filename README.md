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

Most receipt apps wrap Google Cloud Vision or AWS Textract. SmartStock trains its own models on receipt-domain data. This matters because:

- Off-the-shelf OCR is not tuned for receipt fonts, thermal print artifacts, or abbreviations like `ORG STRWBRY 1LB`
- Custom NER lets us extract food entities, quantities, and units in one pass without post-processing hacks
- Full control over the pipeline means measurable accuracy targets, not black-box API responses

---

## ML Pipeline

```
Receipt Image
     │
     ▼
┌─────────────────────────────────┐
│  Stage 1: OCR                   │
│  TrOCR (fine-tuned)             │
│  Current: CER 0.0631            │
│  Target:  CER ≤ 0.05            │
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

### Stage 1 — OCR (Active)

| Model | microsoft/trocr-base-printed, fine-tuned |
|-------|------------------------------------------|
| Adapter | LoRA on decoder (r=16), encoder blocks 10–11 unfrozen |
| Training data | CORD + SROIE + WildReceipt (50,657 filtered line crops) |
| Infrastructure | Kaggle T4 GPU, 30hr/week |
| Best val CER | **0.0631** (target: ≤ 0.05) |
| Training approach | Line-crop strategy: TrOCR fed individual receipt lines, not full images. CER dropped from 0.757 → 0.063 after switching. |

Key decisions made during training:
- Filtered 26% of WildReceipt training data — tiny crops (< 50px wide) caused the model to hallucinate entire receipt footers. Removing them dropped CER from 0.0687 → 0.0631.
- Partial encoder unfreeze (ViT blocks 10+11) enables receipt-domain visual adaptation without full fine-tuning cost
- Generation config bug in the base model (`max_length=20`) silently truncated all outputs at test time — fixed by explicit override at save

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
| ML training | PyTorch, HuggingFace Transformers, PEFT (LoRA), Kaggle T4 |
| Backend | Node.js, PostgreSQL, Alembic migrations |
| ML service | FastAPI (Python), ONNX inference |
| Frontend | React (web-first, mobile-responsive) |
| Recipe suggestions | Spoonacular API |
| Notifications | Push via scheduled daily job |

---

## Repo Structure

```
Smart-Stock/
├── app/                    # Frontend (React)
├── ml_service/             # FastAPI ML inference service
│   ├── ocr/                # TrOCR loader + preprocessor
│   ├── ner/                # DistilBERT NER + entity parser
│   ├── normalization/      # Abbreviation map + fuzzy + LLM fallback
│   └── expiry/             # Shelf-life lookup + confidence scoring
├── db/seeds/               # Database seed data
├── migrations/             # Alembic DB migrations
├── PRD.md                  # Product requirements
├── ML_Pipeline.md          # Full pipeline architecture
├── OCR_Training.md         # TrOCR fine-tuning guide + full training history
├── NER_Training.md         # DistilBERT NER training guide
├── Architecture.md         # System architecture
├── API_Spec.md             # REST API specification
└── DB_Schema.md            # Database schema
```

---

## Accuracy Targets

| Stage | Metric | Target | Current |
|-------|--------|--------|---------|
| OCR | CER | ≤ 0.05 | 0.0631 |
| OCR | WER | ≤ 0.10 | 0.231 |
| NER | F1 | ≥ 0.88 | **0.907** ✅ |
| Normalization | Match rate | ≥ 80% | In progress |
| Expiry | MAE (days) | ≤ 1.5 | In progress |
| End-to-end | Item accuracy | ≥ 85% | In progress |

---

## Inference Latency Budget (CPU, per receipt)

| Stage | Target |
|-------|--------|
| Image preprocessing | < 200ms |
| OCR (TrOCR ONNX) | < 1500ms |
| NER (DistilBERT ONNX) | < 500ms |
| Normalization | < 300ms |
| Expiry prediction | < 100ms |
| **Total** | **< 3s** |