# HANDOFF.md — Context Document for Next Chat Session
## Smart-Stock: AI-Powered Inventory & Waste Reduction System

---

## What This Document Is

This is a context handoff document. The previous chat session is getting long and token-heavy. This document brings the next session up to speed on:
- What Smart-Stock is
- What has been built and completed
- What is currently in progress
- What needs to be done next
- What files exist (do NOT regenerate these)

The following files will be provided as context alongside this document:
- `Model_Training.md` — full TrOCR fine-tuning guide (Stage 1)
- `ML_Pipeline.md` — full ML pipeline architecture

---

## Project Summary

Smart-Stock is a portfolio/CV project targeting Big Tech (Google, Meta) roles. It is an AI-powered inventory management system that reads grocery receipts, predicts expiry dates, and eliminates food waste.

**Core differentiator:** The OCR and NER pipeline is trained in-house — not a third-party API wrapper.

**Full stack:** React + TypeScript (frontend) → FastAPI (backend) → PostgreSQL → ML Pipeline

**Six documentation files exist:**
- `PRD.md` — Product Requirements Document
- `Architecture.md` — System design and data flow
- `API_Spec.md` — All endpoints with request/response schemas
- `DB_Schema.md` — Full PostgreSQL schema with SQL DDL
- `ML_Pipeline.md` — End-to-end ML pipeline (4 stages)
- `Model_Training.md` — TrOCR + DistilBERT training guide

Do NOT regenerate any of these. They are complete.

---

## The ML Pipeline — 4 Stages

```
Receipt Image
      |
      v
Stage 1: OCR (TrOCR)          ← currently training
      | raw text lines
      v
Stage 2: NER (DistilBERT)     ← next task
      | entities: FOOD_ITEM, QUANTITY, UNIT, PRICE
      v
Stage 3: Normalization
      | canonical item names
      v
Stage 4: Expiry Prediction
      | predicted_expiry_date + confidence
      v
Structured inventory items
```

---

## Stage 1 Status — OCR (TrOCR) — Nearly Complete

### What was done
- Fine-tuning `microsoft/trocr-base-printed` on CORD + SROIE datasets
- Full pipeline built on Kaggle with T4 GPU
- Training currently running — 10 epochs, ~12 hours total

### Datasets used
- `naver-clova-ix/cord-v2` — Indonesian restaurant receipts, bounding boxes via `valid_line[].words[].quad` grouped by `group_id`
- `sizhkhy/SROIE` — English scanned receipts, bounding boxes per word via `bboxes` field
- Both datasets converted to **line-level crops** — individual text line images paired with their text
- SROIE train duplicated 2× for more English receipt signal
- Final dataset: ~31,000 train examples saved as Kaggle dataset `smart_stock_dataset_v2`

### Key architectural decision
TrOCR was pretrained on single text line images. Feeding it full receipt images caused CER ~0.83. Switching to line-level crops dropped CER to **0.30 at epoch 1** — a massive improvement.

### Production inference plan
In production, a full receipt image is first passed through an **OpenCV line detector** (morphological ops) to split it into line crops. Each crop is then passed to TrOCR individually. This keeps training and production consistent.

```python
# Production flow
line_crops = detect_line_crops(full_receipt_image)  # OpenCV
for crop in line_crops:
    text = trocr_model.generate(crop)  # one line at a time
```

### Current training results (epoch 1 of 10)
| Epoch | Train Loss | Val Loss | CER | WER |
|---|---|---|---|---|
| 1 | 5.64 | 1.14 | 0.300 | 0.434 |

Previous run (full images, no line crops) best CER was 0.757 at epoch 10. Line crops already beat that at epoch 1.

### Training config
- `learning_rate=2e-5`, `warmup_ratio=0.06`, `max_grad_norm=1.0`
- `num_train_epochs=10`, `batch_size=8` (effective 16 with DataParallel)
- `lr_scheduler_type="cosine"`, `fp16=True`

### After training completes
1. Save model as Kaggle dataset (`trocr-smart-stock-best`)
2. Run Optuna hyperparameter search (10 trials, 20% of data, Bayesian optimization)
3. Retrain with best params if significant improvement found

---

## Stage 2 — NER (DistilBERT) — Starting Next

### Goal
Take raw text output from TrOCR (e.g. `"ORG STRWBRY 1LB 2.99"`) and extract structured food entities:

```
Input:  "ORG STRWBRY 1LB 2.99"
Output: {FOOD_ITEM: "STRWBRY", QUANTITY: "1", UNIT: "LB", PRICE: "2.99"}
```

### Model
`distilbert-base-uncased` fine-tuned for token classification (NER).

### Entity labels (BIO tagging)
| Label | Meaning | Example |
|---|---|---|
| `B-FOOD` | Beginning of food item | `ORG` |
| `I-FOOD` | Inside food item | `STRWBRY` |
| `B-QTY` | Quantity | `1` |
| `B-UNIT` | Unit of measure | `LB` |
| `B-PRICE` | Price | `2.99` |
| `O` | Outside / irrelevant | store name, date, tax |

### Dataset approach — Option A (CORD remapping)
CORD's `valid_line` has words with categories like `menu.nm`, `menu.cnt`, `menu.price`. These map directly to BIO tags:
- `menu.nm` words → `B-FOOD` / `I-FOOD`
- `menu.cnt` words → `B-QTY`
- `menu.price` words → `B-PRICE`
- Everything else → `O`

SROIE has `ner_tags` but they are COMPANY/DATE/ADDRESS/TOTAL — not food entities. SROIE is **not used** for NER training.

### What needs to be built
1. NER dataset builder — extract word-level annotations from CORD `valid_line`, convert to BIO tags
2. Tokenization with label alignment (WordPiece splits words into subwords — labels must be aligned)
3. DistilBERT fine-tuning with `seqeval` F1 evaluation
4. ONNX export for production inference

### Target metrics
- Entity-level F1 ≥ 0.88
- Precision ≥ 0.90
- Recall ≥ 0.86

---

## Key Lessons Learned (for awareness, not to repeat)

- **CORD schema:** `gt_parse.menu` has text only. Bounding boxes are in `valid_line[].words[].quad` grouped by `group_id`.
- **SROIE NER tags** are COMPANY/DATE/ADDRESS/TOTAL — useless for food NER.
- **Kaggle disk:** ~20GB working space. 31k preprocessed float32 tensors = ~52GB — won't fit. Use on-the-fly `TorchDataset` preprocessing instead of `.map()`.
- **Kaggle persistence:** Quick Save = code only. Save & Run All = outputs committed. Always use Save & Run All for training runs.
- **Transformers 4.46+:** `evaluation_strategy` → `eval_strategy`, no `tokenizer=` in Trainer, generation params on `model.generation_config` not `model.config`.

---

## What To Do In This Session

1. Build `NER_Training.md` — equivalent of `Model_Training.md` but for DistilBERT NER
2. Write the CORD → BIO tag dataset builder
3. Write tokenization + label alignment code
4. Write fine-tuning config and training loop
5. Write seqeval evaluation
6. Write ONNX export

Do NOT touch `Model_Training.md` unless explicitly asked. That file is handled separately.
