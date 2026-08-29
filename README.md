# SmartStock — AI-Powered Inventory & Waste Reduction

> **Turn a grocery receipt into an expiry-aware digital inventory.**

SmartStock is an AI-powered inventory and food-waste reduction system that turns grocery receipt images into structured, expiry-aware inventory. Instead of manually logging every purchase, users upload a receipt and SmartStock handles OCR, layout reconstruction, item extraction, food classification, normalization, and shelf-life prediction.

The project follows an **evidence-driven ML philosophy**: components are benchmarked against real receipts, and previous approaches are replaced when testing shows a better solution.

---

## The Problem

Households waste food partly because inventory is difficult to maintain:

- **Memory gap:** users forget what is already in the fridge or pantry and buy duplicates.
- **Expiration oversight:** food gets forgotten until it expires.

Manual inventory apps depend on users maintaining a log. Smart fridges solve automation with expensive hardware.

**SmartStock uses the receipt as the inventory record.**

---

## What It Does

```text
Receipt Photo
     ↓
OCR + Layout Reconstruction
     ↓
Structured Item Extraction
     ↓
Food / Non-Food Gate
     ↓
Food Name Normalization
     ↓
Shelf-Life Prediction
     ↓
Expiry-Aware Inventory
     ↓
Alerts + Recipes + Waste Tracking
```

Example:

```text
"ORG STRWBRY 1LB"
        ↓
item_name="ORG STRWBRY", quantity=1, unit="lb"
        ↓
is_food=true
        ↓
"Strawberries"
        ↓
Produce + Fridge
        ↓
predicted expiry + confidence
```

Non-food items are **not silently discarded**. They remain visible as detected-but-excluded items and skip food-specific normalization and expiry processing.

---

## ML Pipeline

<img src="assets/svgs/smart_stock_pipeline.svg" alt="Smart-Stock ML pipeline" width="620" />

### Stage 1 — OCR

**PaddleOCR / PP-OCRv6**, pretrained, extracts receipt text and bounding boxes.

A fine-tuned TrOCR model was built first. Real-receipt testing showed that PaddleOCR was both more accurate on the tested item/quantity/price lines and dramatically faster: approximately **5–6s/receipt on CPU** versus roughly **480s** for the earlier TrOCR path.

### Stage 1.5 — Row Reconstruction

OCR produces text boxes rather than logical receipt rows. SmartStock estimates skew, deskews y-positions, clusters boxes into rows, and orders fields left-to-right.

Validated on four real Pakistani retail receipts.

### Stage 1.6 — Prefilter

Rule-based filtering removes receipt metadata such as GST numbers, invoice information, payment rows, totals, dates, and other non-item content.

### Stage 1.7 — Header-Driven Row Parser

Real Pakistani receipts use different layouts:

```text
Quantity | Price | Discount | Total
Qty | Item | Rate | Amount
Quantity Price | Discount | Total
M.R.P | Price | Qty/Wt | Tax(%) | Disc. Amount
```

Rather than assuming a fixed format, SmartStock detects the receipt's own headers and maps fields using x-position.

It extracts `item_name`, `quantity`, `price`, `discount`, and `total`, with handling for merged headers, OCR numeric corruption, fallback parsing, and split rows.

**Validation:** 100% field match on the initial four real receipt samples.

### Stage 2 — Item Field Extraction

The original DistilBERT NER stage was retired after row parsing made its original task obsolete.

| Field | Method |
|---|---|
| Unit | Regex + OCR surface variants |
| Brand | Curated lexicon + RapidFuzz |
| `is_food` | LLM binary classification gate |

The brand matcher intentionally does not guess unmatched food nouns as brands. A missing brand is valid.

### Stage 3 — Food Name Normalization

Receipt printers abbreviate names:

```text
ORG STRWBRY   → Strawberries
CHKN BRST BNLS → Boneless Chicken Breast
DAHI          → Yogurt
MURG QEEMA    → Minced Chicken
```

Three-pass normalization:

1. Abbreviation lookup
2. Fuzzy matching
3. LLM fallback

**Status:** implemented; real Stage 1.5–2 re-validation pending.

### Stage 4 — Expiry Prediction

Expiry prediction is intentionally not a trained regression model. The system uses a shelf-life reference database with confidence-aware lookup:

```text
Exact item + storage context
        ↓
Category fallback
        ↓
Hard default
```

**Status:** implemented; real Stage 1.5–2 re-validation pending.

---

## Why We Retired NER

SmartStock originally used fine-tuned DistilBERT NER to tag `FOOD`, `QTY`, `UNIT`, and `PRICE`. It achieved **F1 = 0.907** on its evaluation setup.

It was still retired.

Real receipt testing showed that quantity, price, discount, and total are fundamentally **layout fields**. Once the receipt's column structure is detected, asking a token classifier to infer them is unnecessary.

The architecture became:

```text
OCR boxes
   ↓
row reconstruction
   ↓
header-driven structural parsing
   ↓
small deterministic / semantic components
```

The NER experiment remains documented as historical work.

---

## Current Status

| Component | Status |
|---|---|
| OCR | ✅ Complete — PaddleOCR selected after real-receipt comparison |
| Row reconstruction | ✅ Complete — validated on 4 receipts |
| Prefilter | ✅ Complete |
| Row parser | ✅ Complete — 100% match on initial 4-receipt sample |
| Unit extraction | ✅ Complete |
| Brand matcher | ✅ Complete |
| `is_food` classifier | ✅ Complete — real API smoke-tested |
| Stage 2 extractor | ⏳ Components complete; orchestration pending |
| Pipeline orchestrator | ⏳ Pending |
| Normalization | ✅ Built; real-data validation pending |
| Expiry prediction | ✅ Built; real-data validation pending |
| End-to-end validation | ⏳ Pending |
| End-to-end latency | ⏳ Needs re-measurement |
| `is_food` evaluation set | ⏳ Needs labeled data |

The repository deliberately distinguishes **implemented**, **validated**, and **measured** work.

---

## Architecture

```text
React Frontend
     │
     ▼
FastAPI Backend
     │
     ▼
OCR → Row Reconstruction → Prefilter → Row Parser
                         → Item Field Extraction
                         → Normalization → Expiry
     │
     ▼
PostgreSQL
Inventory · Shelf Life · Alerts · Waste Log · Users
```

Recipe suggestions use Spoonacular; expiry alerts are driven by scheduled jobs.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite, React Query, Zustand |
| Backend | FastAPI, Python, Uvicorn |
| OCR | PaddleOCR / PP-OCRv6 |
| Parsing | Python, regex, RapidFuzz |
| Item extraction | Regex, brand lexicon, LLM API |
| Normalization | Lookup tables, RapidFuzz, LLM fallback |
| Expiry | PostgreSQL shelf-life reference + confidence scoring |
| Database | PostgreSQL, SQLAlchemy 2.0, Alembic |
| Authentication | JWT |
| Recipes | Spoonacular API |
| Historical ML | PyTorch, HuggingFace Transformers, PEFT/LoRA, Kaggle T4 |

---

## Repository Structure

```text
Smart-Stock/
├── app/                    # FastAPI application + DB models
├── ml_service/
│   ├── ocr/               # OCR + row reconstruction
│   ├── parsing/            # Prefilter + row parser
│   ├── item_extraction/   # Unit, brand, is_food
│   ├── normalization/     # Food-name normalization
│   ├── expiry/             # Shelf-life + expiry prediction
│   ├── ner/                # Retired NER record
│   └── pipeline.py         # End-to-end orchestrator
├── db/seeds/               # Reference data
├── migrations/             # Alembic migrations
├── assets/svgs/            # Pipeline diagrams
├── PRD.md
├── Architecture.md
├── ML_Pipeline.md
├── Item_Extraction.md
├── OCR_Training.md
├── NER_Training.md
├── Normalization_Training.md
├── Expiry_Training.md
├── API_Spec.md
└── DB_Schema.md
```

---

## Engineering Philosophy

> **Don't keep a model because it is impressive. Keep it because the evidence says it belongs there.**

That principle has already produced several architecture changes:

- fine-tuned TrOCR → pretrained PaddleOCR
- DistilBERT NER → structural row parsing
- trained item-field extraction → regex + lexicon + LLM gate
- heuristic brand candidates → conservative lexicon-only matching
- ML expiry regression → reference-data lookup + confidence scoring

The goal is not to attach an AI model to an inventory app. The goal is to build a receipt-understanding system that survives messy real-world data.

---

## Documentation

- [`Architecture.md`](Architecture.md) — system design
- [`ML_Pipeline.md`](ML_Pipeline.md) — complete pipeline
- [`Item_Extraction.md`](Item_Extraction.md) — Stage 2 design
- [`OCR_Training.md`](OCR_Training.md) — OCR evaluation + TrOCR history
- [`Normalization_Training.md`](Normalization_Training.md) — normalization
- [`Expiry_Training.md`](Expiry_Training.md) — expiry prediction
- [`API_Spec.md`](API_Spec.md) — REST API
- [`DB_Schema.md`](DB_Schema.md) — PostgreSQL schema
- [`PRD.md`](PRD.md) — product requirements
- [`NER_Training.md`](NER_Training.md) — retired NER experiment

---

## Roadmap

- [ ] Wire Stage 2 extractor
- [ ] Wire end-to-end pipeline orchestrator
- [ ] Build labeled `is_food` evaluation set
- [ ] Re-validate Normalization + Expiry on real receipt output
- [ ] Re-measure end-to-end latency
- [ ] Expand receipt-format coverage
- [ ] Run full end-to-end accuracy evaluation

---

## License

See the repository for licensing information.
