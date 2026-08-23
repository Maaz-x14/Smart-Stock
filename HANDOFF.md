# HANDOFF.md — Context Document for Next Chat Session

## Smart-Stock: AI-Powered Inventory & Waste Reduction System

---

## Project Summary

Smart-Stock is a portfolio/CV project targeting Big Tech roles. AI-powered inventory management system that reads grocery receipts, predicts expiry dates, and eliminates food waste.

**Core differentiator:** OCR and NER pipeline trained in-house — not a third-party API wrapper.

**Full stack:** React + TypeScript → FastAPI → PostgreSQL → ML Pipeline

### Documentation files (DO NOT regenerate)

- `PRD.md` — Product Requirements Document
- `Architecture.md` — System design and data flow
- `API_Spec.md` — All endpoints with request/response schemas
- `DB_Schema.md` — Full PostgreSQL schema with SQL DDL
- `ML_Pipeline.md` — End-to-end ML pipeline (4 stages)
- `OCR_Training.md` — TrOCR training guide (Stage 1) — always attach this
- `NER_Training.md` — DistilBERT NER training guide (Stage 2)

---

## ML Pipeline — 4 Stages

```
Receipt Image
      |
      v
Stage 1: OCR (TrOCR)          ⏳ IN PROGRESS — CER 0.0631, target ≤ 0.05
      | raw text lines
      v
Stage 2: NER (DistilBERT)     ✅ COMPLETE — F1 0.907
      | entities: FOOD, QTY, UNIT, PRICE
      v
Stage 3: Normalization         🔲 NOT STARTED
      | canonical item names
      v
Stage 4: Expiry Prediction     🔲 NOT STARTED
      | predicted_expiry_date + confidence
      v
Structured inventory items
```

---

## Stage 1 — OCR (TrOCR) ⏳ IN PROGRESS

### Current state

- **Best val CER: 0.0631** (checkpoint from v4 run 1, stored as `trocr-smart-stock-best`)
- **Target: CER ≤ 0.05**
- Optuna re-run completed this session — new LR found but no CER improvement (0.0652 best vs 0.0631 previous best)
- Model is at a genuine plateau. The remaining 25.2% of val samples with CER > 0 are dominated by WildReceipt label noise and multi-line crop mismatches — not fixable by LR tuning

### Architecture

- Base: `microsoft/trocr-base-printed`
- Adapter: LoRA on decoder (q_proj, v_proj, r=16, lora_alpha=32)
- Encoder: blocks 10+11 unfrozen, rest frozen
- Trainable: 1,523,712 LoRA + 14,171,136 encoder = ~15.7M of 335M total

### Dataset — v4 (current, active)

| Split | Samples | Notes                            |
| ----- | ------- | -------------------------------- |
| Train | 50,657  | CORD + SROIE (2×) + WildReceipt |
| Val   | 1,800   | CORD + WildReceipt               |
| Test  | 8,720   | CORD + SROIE + WildReceipt       |

Filtered from v3 (68,463) by: width ≥ 50px, height ≥ 10px, label ≥ 2 chars, non-food blocklist (34 tokens). Filtering alone dropped CER from 0.0687 → 0.0631.

### Training history

| Run                   | Dataset | Val CER          | Notes                            |
| --------------------- | ------- | ---------------- | -------------------------------- |
| v3 runs 1–3          | v3      | 0.0687           | Frozen encoder, checkpoint-51348 |
| v3 encoder unfreeze   | v3      | 0.0706           | No improvement                   |
| v4 run 1 (clean data) | v4      | **0.0631** | Best — checkpoint-31665         |
| v4 run 2              | v4      | 0.0637           | Plateau confirmed                |
| v4 run 3 (Optuna LR)  | v4      | 0.0652           | New LR 8.84e-5, no improvement   |

### Val error distribution (current)

| CER Bucket     | Count | %     |
| -------------- | ----- | ----- |
| 0.00 (perfect) | 1,346 | 74.8% |
| 0.01–0.05     | 61    | 3.4%  |
| 0.06–0.10     | 161   | 8.9%  |
| 0.11–0.20     | 151   | 8.4%  |
| 0.21–0.50     | 63    | 3.5%  |
| > 0.50         | 18    | 1.0%  |

Per-source: CORD 0.027, SROIE 0.042, WildReceipt 0.073

### Remaining approaches (in priority order)

| Technique              | Status         | Notes                                                                                                                         |
| ---------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| LoRA rank r=16 → r=32 | **Next** | Only remaining architectural lever. After 3 failed LR attempts, this is the last thing worth trying before declaring ceiling. |
| PaddleOCR              | Dropped        | Full pipeline rewrite cost not justified yet                                                                                  |

### Kaggle dataset inputs (current)

| Slug                        | Contents                | Path                                                                                                         |
| --------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------ |
| `smart-stock-dataset-v4`  | Training data           | `/kaggle/input/datasets/maazahmad69/smart-stock-dataset-v4/smart_stock_dataset_v4`                         |
| `trocr-smart-stock-model` | Best model (0.0631)     | `/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock-best/trocr-smart-stock-best` |
| `smart-stock-model-data`  | checkpoint-31665 (temp) | `/kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock/checkpoint-31665`             |
| `wild-receipt`            | Raw WildReceipt         | `/kaggle/input/datasets/maazahmad69/wild-receipt/wildreceipt`                                              |

**Rule:** dataset outputs → `smart-stock-dataset-v4` only. Model outputs → `trocr-smart-stock-model` only. Never mix.

### Key bugs fixed (do not repeat)

- `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` must be absolute first line before any import
- `dataset.filter()` writes cache to `/kaggle/input/` (read-only) — use `dataset.select(indices)` instead
- `generation_config.max_length=20` in saved model silently truncates all outputs — override to 256 in Cell 15 before saving
- `model.generate()` after `merge_and_unload()` — model is stale, use `merged_model.generate()`
- `label_smoothing_factor` incompatible with transformers 5.0.0 + TrOCR — remove from training args
- Draft mode runs on CPU (180hr estimates) — always Save & Run All
- Trainer epoch overflow on resume — fresh optimizer when config changes materially

### Test CER is always > 1.0 — this is a known issue

Cell 16 (test eval) loads the OLD model from `MODEL_INPUT` (trocr-smart-stock-best on Kaggle), not the newly trained model in `/kaggle/working/trocr-smart-stock-best`. The newly trained model isn't uploaded to Kaggle until after the session. This is expected — val CER is the real metric during training.

---

## Stage 2 — NER (DistilBERT) ✅ COMPLETE

| Metric    | Target  | Achieved           |
| --------- | ------- | ------------------ |
| F1        | ≥ 0.88 | **0.907** ✅ |
| Precision | ≥ 0.90 | **0.930** ✅ |
| Recall    | ≥ 0.86 | **0.885** ✅ |

Model saved as Kaggle dataset `distilbert-ner-smart-stock-best`.

Label schema: `B-FOOD`, `I-FOOD`, `B-QTY`, `B-UNIT`, `B-PRICE`, `O`

### Critical gotchas

- `LABEL2ID` must be defined globally before any loader function
- WordPiece alignment: only first subword gets real label, continuations get `-100`
- Pass `is_split_into_words=True` to tokenizer
- Always use seqeval for entity-level F1, not sklearn accuracy
- `save_strategy="no"` during Optuna trials — otherwise disk fills up

---

## Stage 3 — Normalization 🔲 NOT STARTED

### Goal

```
NER output: {food: "ORG STRWBRY", qty: "1", unit: "LB"}
     ↓
Normalized: {canonical_name: "Strawberries", quantity: 1.0, unit: "lb", category: "Produce"}
```

### Designed approach (from ML_Pipeline.md)

Three passes:

1. Direct lookup — abbreviation dict (~800 entries, ~65% of cases)
2. Fuzzy match — rapidfuzz against `shelf_life_reference` canonical names (threshold 80)
3. LLM fallback — for unresolved tokens (target ≤ 20% of cases), results cached in DB

### What needs to be built

1. `ABBREVIATION_MAP` — ~800 entries (US + Pakistani grocery abbreviations)
2. `shelf_life_reference` table — canonical names + shelf life by storage context
3. `ml_service/normalization/` Python module (three-pass normalizer)
4. Unit normalization (`LB` → `lb`, `GAL` → `gal`, etc.)
5. Category keyword classifier (fallback)
6. LLM fallback wiring (Groq API with llama — cheapest option)
7. `normalization_cache` DB table
8. `Normalization_Training.md`

### Target metrics

| Metric                          | Target |
| ------------------------------- | ------ |
| Canonical Match Rate (Pass 1+2) | ≥ 80% |
| LLM Fallback Rate (Pass 3)      | ≤ 20% |

---

## Stage 4 — Expiry Prediction 🔲 NOT STARTED

Rule-based lookup against `shelf_life_reference` DB + confidence scoring. Designed in `ML_Pipeline.md`. Not started pending Stage 3 completion.

---

## Kaggle Workflow Rules

- **Quick Save** = code only (no outputs) — never use for training
- **Save & Run All** = outputs committed — always use this
- Draft mode = CPU only = useless for training
- After each run: download outputs, upload as new version of `trocr-smart-stock-model`
- Keep only the best checkpoint in `trocr-smart-stock-model` — delete rest after upload
- `smart-stock-model-data` is a temp dataset holding checkpoint-31665 — clean this up once checkpoint is properly uploaded to `trocr-smart-stock-model`

---

## GitHub

Repo: `github.com/Maaz-x14/Smart-Stock`
Active branch: `ml/ocr-cer-improvement`
Open issue: `[ML] OCR Stage 1 — Improve TrOCR CER from 0.0631 to ≤ 0.05`

**⚠️ kaggle.json is publicly exposed in the repo — revoke API token at kaggle.com → Account → API → Expire, then remove the file and add to .gitignore**

I'm building SmartStock, an AI-powered receipt intelligence system. Attaching HANDOFF.md for full context — read it before anything else.

Current situation: OCR Stage 1 (TrOCR) is at val CER 0.0631, target ≤ 0.05. Three training runs on v4 clean data haven't moved it further. Optuna re-run just completed — new LR (8.84e-5) found but run produced 0.0652, worse than current best. The model is plateaued.

What I need help with next: Try LoRA rank increase r=16 → r=32 — the last remaining architectural lever before moving on. Then if that doesn't work, declare OCR ceiling and move to Stage 3 (Normalization).

My preferences: never assume missing info, push back on bad ideas, distinguish facts from speculation, minimum explanation needed, ask one question at a time, propose design before writing code.
