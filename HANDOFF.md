# HANDOFF.md — Context Document for Next Chat Session
## Smart-Stock: AI-Powered Inventory & Waste Reduction System

---

## Project Summary

Smart-Stock is a portfolio/CV project targeting Big Tech (Google, Meta) roles. It is an AI-powered inventory management system that reads grocery receipts, predicts expiry dates, and eliminates food waste.

**Core differentiator:** OCR and NER pipeline trained in-house — not a third-party API wrapper.

**Full stack:** React + TypeScript (frontend) → FastAPI (backend) → PostgreSQL → ML Pipeline

### Documentation files (DO NOT regenerate)
- `PRD.md` — Product Requirements Document
- `Architecture.md` — System design and data flow
- `API_Spec.md` — All endpoints with request/response schemas
- `DB_Schema.md` — Full PostgreSQL schema with SQL DDL
- `ML_Pipeline.md` — End-to-end ML pipeline (4 stages)
- `OCR_Training.md` — TrOCR training guide (Stage 1)
- `NER_Training.md` — DistilBERT NER training guide (Stage 2)

---

## ML Pipeline — 4 Stages

```
Receipt Image
      |
      v
Stage 1: OCR (TrOCR)          ✅ COMPLETE
      | raw text lines
      v
Stage 2: NER (DistilBERT)     ✅ COMPLETE
      | entities: FOOD, QTY, UNIT, PRICE
      v
Stage 3: Normalization         ← NEXT TASK
      | canonical item names
      v
Stage 4: Expiry Prediction
      | predicted_expiry_date + confidence
      v
Structured inventory items
```

---

## Stage 1 — OCR (TrOCR) ✅ COMPLETE

### What was built
- Fine-tuned `microsoft/trocr-base-printed` on CORD v2 + SROIE datasets
- Trained on Kaggle T4 GPU, 10 epochs

### Key architectural decision
TrOCR is pretrained on single-line images. Feeding it full receipts caused CER ~0.757. Switching to **line-level crops** dropped CER to **0.140 at epoch 9** — the single most important fix.

**Production inference:** full receipt → OpenCV line detector → per-line crops → TrOCR (one line at a time)

### Datasets
- `naver-clova-ix/cord-v2` — Indonesian restaurant receipts. Bounding boxes via `valid_line[].words[].quad`, grouped by `group_id`. Line crops extracted from `menu.*` categories only.
- `sizhkhy/SROIE` — English scanned receipts. Words grouped into lines by Y-coordinate proximity (≤15px = same line). SROIE train duplicated 2× for more English signal.
- Final dataset: ~31,000 line crops, saved as Kaggle dataset `smart-stock-dataset-v2`

### Final results (Run 2, epoch 9)
| Metric | Target | Achieved |
|---|---|---|
| CER | ≤ 5% | **14%** (still improving, not final) |
| WER | ≤ 10% | **25%** |

> Training resumed from `checkpoint-15536` after session timeout. Best checkpoint saved as Kaggle dataset `trocr-smart-stock-best`.

### Critical gotchas (do not repeat)
- CORD `ground_truth` is a raw JSON string — must `json.loads()` it
- Bounding boxes are in `valid_line[].words[].quad`, NOT in `gt_parse.menu`
- Generation params go on `model.generation_config`, not `model.config`
- When resuming from checkpoint saved with `fp16=True`, must set `fp16=True` in new session's `TrainingArguments` or scaler will be `None`
- `Missing keys: ['decoder.output_projection.weight']` on checkpoint resume — harmless, ignore
- Use `eval_strategy` not `evaluation_strategy` (transformers 4.46+)

---

## Stage 2 — NER (DistilBERT) ✅ COMPLETE

### What was built
- Fine-tuned `distilbert-base-uncased` for token classification on receipt NER
- BIO tagging scheme: `B-FOOD`, `I-FOOD`, `B-QTY`, `B-UNIT`, `B-PRICE`, `O`
- Trained on Kaggle T4 GPU, 15 epochs

### Label schema
```python
LABEL2ID = {"O": 0, "B-FOOD": 1, "I-FOOD": 2, "B-QTY": 3, "B-UNIT": 4, "B-PRICE": 5}
```
No `I-QTY`, `I-UNIT`, `I-PRICE` — those are always single tokens. If a consecutive same-entity appears, emit `B-` again.

### Datasets (three-source strategy)
| Dataset | HF Path | Size | Entities | Weight |
|---|---|---|---|---|
| CORD v2 | `naver-clova-ix/cord-v2` | ~11k lines | FOOD, QTY, PRICE | 1.0× |
| TASTEset | `dmargutierrez/TASTESet` | 700 recipes | FOOD, QTY, UNIT | 0.5× |
| Synthetic | Generated in-notebook | 5,000 lines | FOOD, QTY, UNIT, PRICE | 0.8× |

**CORD → BIO mapping:**
- `menu.nm` / `menu.sub_nm` → FOOD
- `menu.cnt` → QTY
- `menu.unitprice` / `menu.price` → PRICE
- everything else → O

**Note:** SROIE's NER tags are COMPANY/DATE/ADDRESS/TOTAL — completely useless for food NER. Not used.

### Final results (Run 2)
| Metric | Target | Achieved |
|---|---|---|
| Entity-level F1 | ≥ 0.88 | **0.907** ✅ |
| Precision | ≥ 0.90 | **0.930** ✅ |
| Recall | ≥ 0.86 | **0.885** ✅ |

> Best model saved as Kaggle dataset `distilbert-ner-smart-stock-best`.

### Critical gotchas (do not repeat)
- `LABEL2ID` must be defined globally before any loader function runs
- WordPiece alignment: only first subword of each word gets the real label; continuation subwords get `-100` (ignored by loss)
- Pass `is_split_into_words=True` to tokenizer — input is a pre-tokenized word list
- seqeval reports entity-level F1, not token accuracy — always use seqeval, not sklearn accuracy
- Optuna `save_strategy="no"` during HP search trials — otherwise checkpoints from every trial fill disk (`OSError: [Errno 28] No space left on device`)
- Add `save_total_limit=2` in final training to cap disk usage

---

## Stage 3 — Normalization 🔲 NEXT TASK

### Goal
Convert raw, abbreviated NER output tokens into canonical food names that match the `shelf_life_reference` database table.

```
NER output: {food: "ORG STRWBRY", quantity: "1", unit: "LB"}
     ↓
Normalized: {canonical_name: "Strawberries", quantity: 1.0, unit: "lb", category: "Produce"}
```

### Method: Three-Pass Approach (already designed in ML_Pipeline.md)

**Pass 1 — Direct Lookup (~65% of cases)**
Curated abbreviation dictionary (`~800 entries`):
```python
ABBREVIATION_MAP = {
    "STRWBRY": "Strawberries",
    "MLKWHL": "Whole Milk",
    "CHKN BRST": "Chicken Breast",
    ...
}
```

**Pass 2 — Fuzzy Matching**
`rapidfuzz` against all `canonical_name` values in `shelf_life_reference`:
```python
match, score, _ = process.extractOne(raw_token, canonical_names_list, scorer=fuzz.token_sort_ratio)
if score >= 80:
    return match
```

**Pass 3 — LLM Fallback (≤20% of cases)**
For unresolved tokens (score < 80), call LLM:
```
Prompt: "This is a line item from a US grocery store receipt: '{raw_token}'.
What common food item does this refer to? Reply with just the canonical food name, nothing else."
```
Results cached in `normalization_cache` DB table.

**Unit normalization:**
```python
UNIT_MAP = {"LB": "lb", "LBS": "lb", "OZ": "oz", "GAL": "gal", "GL": "gal",
            "CT": "count", "PK": "pack", "EA": "each"}
```

**Category assignment:** Lookup `category` from `shelf_life_reference` by `canonical_name`. Fallback to keyword classifier if not found.

### What needs to be built for Stage 3

1. `Normalization_Training.md` — document the full normalization pipeline, just like NER_Training.md and OCR_Training.md
2. Build the abbreviation map (`ABBREVIATION_MAP`) — ~800 or more entries for common Global/US/Pakistani grocery abbreviations
3. Build the `shelf_life_reference` table — canonical food names + shelf life in days by storage context (Fridge/Freezer/Pantry)
4. Implement the three-pass normalizer as a Python module (`ml_service/normalization/`)
5. Implement unit normalization
6. Implement category keyword classifier (fallback)
7. Wire LLM fallback (Ollama `llama3.2:1b` local or Claude API or Groq API to use llama), 
8. Set up `normalization_cache` table to avoid redundant LLM calls
9. Evaluate: Canonical Match Rate ≥ 80%, LLM Fallback Rate ≤ 20%

### Target metrics
| Metric | Target |
|---|---|
| Canonical Match Rate (Pass 1 + 2) | ≥ 80% |
| LLM Fallback Rate (Pass 3) | ≤ 20% |

---

## Kaggle Workflow Reminders

- **Quick Save** = code only (no outputs persisted)
- **Save & Run All** = outputs committed — always use this for training runs
- Kaggle disk: ~20GB working space — watch for `OSError: [Errno 28] No space left on device`
- Save important outputs as Kaggle datasets from the Output tab after a run completes
- Kaggle input path format: `/kaggle/input/datasets/{username}/{dataset-slug}/{folder-name}`
