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

**The pipeline is built, wired end-to-end (`pipeline.py`), and validated against real Pakistani retail receipts** — see Current Status below.

### Stage 1 — OCR

**PaddleOCR / PP-OCRv6**, pretrained, extracts receipt text and bounding boxes.

A fine-tuned TrOCR model was built first. Real-receipt testing showed that PaddleOCR was both more accurate on the tested item/quantity/price lines and dramatically faster: approximately **5.4s/receipt** (measured, see ML_Pipeline.md §9) versus roughly **480s** for the earlier TrOCR path.

### Stage 1.5 — Row Reconstruction

OCR produces text boxes rather than logical receipt rows. SmartStock estimates skew, deskews y-positions, clusters boxes into rows, and orders fields left-to-right.

Validated on four real Pakistani retail receipts. Negligible latency (~2ms measured).

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

**Validation:** 100% field match on the initial four real receipt samples. Negligible latency (~3ms measured).

### Stage 2 — Item Field Extraction

The original DistilBERT NER stage was retired after row parsing made its original task obsolete.

| Field | Method |
|---|---|
| Unit | Regex + OCR surface variants |
| Brand | Curated lexicon + RapidFuzz |
| `is_food` | LLM binary classification gate (Groq, batched in chunks of 5) |

The brand matcher intentionally does not guess unmatched food nouns as brands. A missing brand is valid.

**Measured latency: ~5.6s/receipt** (Groq-dependent, the primary rate-limit risk stage — see ML_Pipeline.md §9).

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
3. LLM fallback (Groq, with an explicit UNKNOWN abstention path — see Normalization.md §5.4)

**Status:** validated against real pipeline output — 97.2% canonical match rate on synthetic eval, 34/36 items resolved end-to-end on real receipts. Measured latency: ~922ms/receipt mean (cache-dependent).

### Stage 4 — Expiry Prediction

Expiry prediction is intentionally not a trained regression model. The system uses a shelf-life reference database with confidence-aware lookup:

```text
Exact item + storage context
        ↓
Category fallback
        ↓
Hard default
```

**Status:** validated against real pipeline output. 100% within ±2 days on a synthetic 16-case eval. Negligible latency (~22ms measured).

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
| `is_food` classifier | ✅ Complete — batched, validated on real receipts, reasoning-loop bug fixed |
| Stage 2 extractor | ✅ Complete — orchestrator wired |
| Pipeline orchestrator | ✅ Complete — `pipeline.py`, validated end-to-end on real receipts |
| Normalization | ✅ Validated on real pipeline output — LLM fallback abstention fix landed |
| Expiry prediction | ✅ Validated on real pipeline output |
| End-to-end validation | ✅ Done — 2 of 4 real receipts run through the full pipeline; remaining 2 not yet tried |
| End-to-end latency | ✅ Measured — ~11.9s mean / ~11.5s median across 2 receipts (over the 10s budget on one, near it on the other); OCR + Item Field Extraction are the bottleneck stages |
| `is_food` evaluation set | ⏳ Still needs labeled data — real pipeline runs show correct classification but this isn't a substitute for a formal measurement |

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
| Item extraction | Regex, brand lexicon, LLM API (Groq) |
| Normalization | Lookup tables, RapidFuzz, LLM fallback (Groq) |
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
│   ├── ocr/                # OCR + row reconstruction
│   ├── parsing/             # Prefilter + row parser
│   ├── item_extraction/    # Unit, brand, is_food
│   ├── normalization/      # Food-name normalization
│   ├── expiry/              # Shelf-life + expiry prediction
│   ├── ner/                 # Retired NER record
│   └── pipeline.py          # End-to-end orchestrator (built, validated)
├── db/seeds/                # Reference data
├── migrations/              # Alembic migrations
├── assets/svgs/             # Pipeline diagrams
├── benchmark_latency.py     # Per-stage + end-to-end latency benchmark (#33)
├── check_cache.py           # normalization_cache inspect/clear CLI
├── test_pipeline_local.py   # Local end-to-end pipeline test harness
├── PRD.md
├── Architecture.md
├── ML_Pipeline.md
├── Item_Extraction.md
├── Normalization.md
├── Expiry.md
├── OCR_Training.md
├── NER_Training.md
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
- [`ML_Pipeline.md`](ML_Pipeline.md) — complete pipeline, including measured latency
- [`Item_Extraction.md`](Item_Extraction.md) — Stage 2 design
- [`Normalization.md`](Normalization.md) — Stage 3 design
- [`Expiry.md`](Expiry.md) — Stage 4 design
- [`OCR_Training.md`](OCR_Training.md) — OCR evaluation + TrOCR history
- [`API_Spec.md`](API_Spec.md) — REST API
- [`DB_Schema.md`](DB_Schema.md) — PostgreSQL schema
- [`PRD.md`](PRD.md) — product requirements
- [`NER_Training.md`](NER_Training.md) — retired NER experiment

---

## Roadmap

- [ ] Build labeled `is_food` evaluation set (#34)
- [ ] Validate remaining 2 real receipts (1.jpg, 4.jpg) end-to-end
- [ ] Optimize latency — OCR and Item Field Extraction are the two bottleneck stages, end-to-end currently exceeds the 10s budget on larger receipts
- [ ] Expand receipt-format coverage (multi-line/Tax% header format, issue #23)
- [ ] Run full end-to-end accuracy evaluation against ground truth
- [ ] Fix Stage 2 fail-safe gap for empty-content classification responses (#52, open)

---

## License

See the repository for licensing information.
