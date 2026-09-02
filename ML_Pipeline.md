# ML_Pipeline.md — Machine Learning Pipeline

## Smart-Stock: OCR → Row Reconstruction → Prefilter → Row Parser → Item Field Extraction → Normalization → Expiry Prediction

**Version:** 2.0 — restructured after discovering real Pakistani retail receipts use no fixed header format; DistilBERT NER removed in favor of header-driven parsing + lightweight per-field extraction.

---

## 0. Why This Changed (read before the rest of this doc)

The original pipeline (v1.0) assumed a single-line-per-item OCR format (`"STRWBRY 1 LB 2.99"`) and used a fine-tuned DistilBERT NER model to tag FOOD/QTY/UNIT/PRICE tokens within that line.

Testing against real PaddleOCR output on real Pakistani retail receipts (issue #14 onward) found this assumption wrong on two counts:

1. **OCR output is box-per-field, not line-per-item.** Item name, quantity, price, discount, and total arrive as separate unordered text boxes that must be reconstructed into logical rows first (deskew + y-position clustering).
2. **Receipt header formats vary per store, with no standard.** `Quantity | Price | Discount | Total`, `Qty | Item | Rate | Amount`, `M.R.P | Price | Qty/Wt | Tax(%) | Disc. Amount`, and more — all seen across a handful of real receipts. A model trained on one synthetic format cannot generalize to this variety; a **header-driven parser that reads each receipt's own column layout** does.

Once row parsing extracts quantity/price/discount/total structurally (by matching header position, not by asking a model to infer them), DistilBERT NER's job shrank to almost nothing — the only real judgment left is "is this item text actually food" (a semantic question, better suited to an LLM gate than a small trained classifier with no labeled data) plus unit/brand extraction from the item name string (both regex/lexicon-solvable, no model needed).

**Net result: DistilBERT NER is retired.** See NER_Training.md for the preserved historical record — real training work, just solving a problem that no longer exists in this pipeline.

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML PIPELINE                                 │
│                                                                 │
│  Receipt Image                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 1: OCR                                            │   │
│  │  Model: PaddleOCR (pretrained PP-OCRv6)                  │   │
│  │  Input:  Receipt image (JPEG/PNG/PDF)                    │   │
│  │  Output: Text + bounding boxes per detected text box     │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 1.5: Row Reconstruction                           │   │
│  │  Method: Deskew correction + y-position clustering       │   │
│  │  Input:  Text boxes (unordered)                          │   │
│  │  Output: Logical rows (left-to-right ordered)            │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 1.6: Prefilter                                    │   │
│  │  Method: Rule-based keyword/regex drop                   │   │
│  │  Input:  Logical rows                                    │   │
│  │  Output: Rows with metadata (GST#, Invoice#, etc) removed│   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 1.7: Row Parser                                   │   │
│  │  Method: Header detection (fuzzy) + x-position column map│   │
│  │  Input:  Filtered rows                                   │   │
│  │  Output: {item_name, quantity, price, discount, total}   │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: Item Field Extraction                          │   │
│  │  Method: Regex (unit) + fuzzy lexicon (brand)            │   │
│  │          + LLM gate (is_food)                            │   │
│  │  Input:  item_name string                                │   │
│  │  Output: {is_food, brand, unit}                          │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                     is_food=false → surfaced to user, stops here│
│                     is_food=true  ▼                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 3: Normalization                                  │   │
│  │  Method: Lookup → Fuzzy match → LLM canonical naming     │   │
│  │  Input:  item_name (is_food=true only)                   │   │
│  │  Output: Canonical item records                          │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 4: Expiry Prediction                              │   │
│  │  Method: Rule-based baseline + shelf_life_reference DB   │   │
│  │  Input:  Canonical item + storage context                │   │
│  │  Output: predicted_expiry_date, confidence score         │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  Structured Output: List[InventoryItem]                         │
└─────────────────────────────────────────────────────────────────┘
```

<img src="assets/svgs/smart_stock_pipeline.svg" alt="Pipeline" width="500" />

---

## 2. Stage 1: OCR — Text Extraction

Unchanged from v1.0.

### Model

**PaddleOCR** (PP-OCRv6, pretrained — no fine-tuning) — small det/rec models, doc-orientation/unwarping/textline-orientation disabled for speed.

PaddleOCR's CNN-based detector+recognizer (DBNet + CRNN/SVTR) beat a 2-month fine-tuned TrOCR on both accuracy and latency (~5-6s/receipt vs ~480s/receipt) without any fine-tuning. See OCR_Training.md for the full comparison.

### Output

```python
result = ocr.predict(image_path)
res = result[0]
texts  = res["rec_texts"]   # recognized text per box
scores = res["rec_scores"]  # confidence per box
boxes  = res["rec_boxes"]   # [x1, y1, x2, y2] per box — now extracted (was previously unused)
polys  = res["rec_polys"]   # 4-point polygon per box — used for skew estimation
```

**Note:** `rec_boxes`/`rec_polys` were always present in PaddleOCR's result dict but not extracted in v1.0 — Stage 1.5 needed them, so extraction was added. No re-run of detection required.

Module: `ml_service/ocr/model.py`

---

## 3. Stage 1.5: Row Reconstruction

**New in v2.0.** Solves: one text box ≠ one logical receipt row. Item name, qty, price, discount, and total arrive as separate unordered boxes.

### Method

1. Compute y-center and height per box from `rec_boxes`.
2. Estimate the receipt's skew angle from `rec_polys` (median angle of each box's top edge, via `arctan2`) — real photographed receipts are rarely perfectly flat.
3. Correct each box's y-center for skew-induced drift: `y_corrected = y_center - slope * x_center`.
4. Sort boxes by corrected y, cluster into rows where consecutive y-differences fall under a tolerance (proportional to box height).
5. Within each row, sort left-to-right by x.

```python
def cluster_rows_deskewed(texts, scores, boxes, polys, y_tol_ratio=0.3):
    # estimate skew angle, correct y-positions, cluster into rows
    ...
```

Module: `ml_service/ocr/row_reconstruction.py`

**Validated** against 4 real receipts — correctly reconstructs rows including cases with meaningful skew (verified: a row split by drift, e.g. "Total Items/Quantity" separated from its value "3/67.00", merges correctly after deskew correction).

---

## 4. Stage 1.6: Prefilter

**Relocated from `ml_service/ner/` to `ml_service/parsing/`** — was never NER-specific logic, just ran before NER in v1.0.

### Method

Rule-based keyword/regex drop. Runs on reconstructed rows (joined into one line for matching), before Row Parser sees them.

```python
import re

DROP_KEYWORDS = {
    "GST", "INVOICE", "TRANSACTION", "POS", "CUSTOMER", "CNIC", "PAYMENTS",
    "TOTAL", "DISCOUNT", "ROUNDING", "TAX BREAKUP", "MRP", "NON MRP",
    "CHANGE DUE", "CASH", "THANK YOU", "VISIT AGAIN", "COME AGAIN",
    "CASHIER", "OPERATOR", "SAVE RECEIPT", "SUBTOTAL", "VAT",
}

PRICE_ONLY_RE   = re.compile(r"^Rs?\.?\s*[\d,]+\.?\d*$", re.IGNORECASE)
PHONE_RE        = re.compile(r"^\+?\d[\d\-\s]{7,}$")
DATE_RE         = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
TIME_RE         = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?$", re.IGNORECASE)
SEPARATOR_RE    = re.compile(r"^[-=*_]{3,}$")
PERCENT_ONLY_RE = re.compile(r"^\d{1,2}(\.\d+)?\s*%$")
BARCODE_RE      = re.compile(r"^\d{8,}$")

def should_drop_row(row: list[dict]) -> bool:
    return should_drop_line(" ".join(item["text"] for item in row))
```

**Important interaction with Row Parser:** the header row itself (e.g. `"Discount | Total"`) is excluded from prefilter checks — Row Parser detects and consumes the header row before prefilter runs on the remaining rows, otherwise the header would be wrongly dropped (its own keywords overlap with `DROP_KEYWORDS`).

Module: `ml_service/parsing/prefilter.py`

**Known gap:** doesn't catch every leaked non-item row (e.g. "Ex1. Amt", bank names like "HBL") — low priority, doesn't corrupt item data, just leaves a few harmless stray entries.

---

## 5. Stage 1.7: Row Parser

**New in v2.0.** Solves: no two Pakistani retail receipts share a header format. Real headers seen: `Quantity | Price | Discount | Total`, `Qty | Item | Rate | Amount`, `Quantity Price | Discount | Total` (merged), `M.R.P | Price | Qty/Wt | Tax(%) | Disc. Amount`. Position-only heuristics don't generalize across this variety.

### Method

1. **Header detection:** scan rows (up to the "Sales Items" marker) for fuzzy matches against column synonym lists (`QUANTITY`, `PRICE`, `TOTAL`, `DISCOUNT`, `ITEM`). Strict `ratio` match first; falls back to `partial_ratio` only when strict match fails, to catch merged header tokens (`"Quantity Price"`) without over-matching plain single-word headers (fixes a real bug where `"Discount"` alone was fuzzy-matching `"Total"` too).
2. **Column mapping:** record each header token's x-position → build a per-receipt column schema.
3. **Field extraction:** for each data row, map each token to its nearest header column by x-distance; numeric tokens get cleaned (strips OCR noise like stray carets, fixes multi-dot decimals) before parsing.
4. **Fallback (no header detected):** leading non-numeric token = item name; rightmost numeric token = total (receipts consistently place total last); remaining numbers split by magnitude/integer heuristics (low_confidence_parse flagged).
5. **Split-row merge:** item name and its numeric fields sometimes print on separate physical receipt lines, sometimes on the same line — a merge pass combines a name-only row with an immediately following numbers-only row into one item.

```python
def parse_receipt_rows(rows) -> list[dict]:
    # returns [{item_name, quantity, price, discount, total, low_confidence_parse}, ...]
```

Module: `ml_service/parsing/row_parser.py`

**Validated** against 4 real receipts — output matches ground truth exactly on all fields after fixing: header-scan window (was too small, missed headers past row 10), merged-column double-write bug, and OCR noise in numeric tokens (`"11^9.00"` → 119.00, `"Rs7.948.80"` → 7948.80).

**Known unsupported format (issue #23):** multi-line headers with a `Tax (%)` column and 6 fields per item, seen on one receipt (item code + name fused, sub-values merged in one OCR token like `"0.00(0) 3.00"`). Deferred — needs its own parsing path, not a patch to the current header logic.

**Note:** `quantity` from this stage is the field the rest of the pipeline uses (`DB_Schema.md inventory_items.quantity`). `price`, `discount`, `total` are extracted but **not persisted** — confirmed unused per `DB_Schema.md` (no price/discount/total columns on `inventory_items`).

---

## 6. Stage 2: Item Field Extraction

**Replaces the retired DistilBERT NER stage.** Operates on the `item_name` string that Row Parser produced. Output contract: `{is_food: bool, brand: str | None, unit: str | None}`.

### Why not a trained model

- **Unit** is a deterministic regex-extractable pattern embedded in item_name (`"Pakola MIk Uht 250M1"` → unit=ml, `"Ponam Sugar 1kg"` → unit=kg), including known OCR corruption patterns (`ml`→`M1`, `l`→`1`). No training data needed.
- **Brand** is a leading-token fuzzy match against a curated lexicon, reusing the same `rapidfuzz` infrastructure as Stage 3 Pass 2. No labeled training data exists for this; a lexicon is directly inspectable and extendable as new brands appear, unlike a trained classifier.
- **is_food** is the one genuinely semantic judgment — but there's no labeled dataset for this specific binary task (the retired DistilBERT was trained for a different task shape: full-line BIO tagging, not isolated item-name classification), and building one would be expensive with thin coverage across unseen store types. An LLM gate needs no training data and generalizes to novel item text (pharmacy items, cosmetics, random SKUs) far better than a small classifier trained on a handful of labeled examples would.

### Unit Extraction

```python
UNIT_PATTERNS = {
    r"(\d+\.?\d*)\s*(ml|m1)\b": "ml",   # M1 = common OCR misread of ml
    r"(\d+\.?\d*)\s*(kg)\b":    "kg",
    r"(\d+\.?\d*)\s*(g|gm)\b":  "g",
    r"(\d+\.?\d*)\s*(lb|lbs)\b":"lb",
    ...
}
```

Module: `ml_service/item_extraction/unit_extractor.py`

### Brand Extraction

Fuzzy match leading token(s) of item_name against a curated brand lexicon (same `rapidfuzz` approach as Stage 3 Pass 2). Not every item has a brand (e.g. "Beef Mince") — no match is a valid, expected outcome, not a failure.

Module: `ml_service/item_extraction/brand_matcher.py`

### is_food Gate

LLM call, binary classification. Two entry points:

```
Single-item (extract_item_fields()):
  Prompt: "Is '{item_name}' a food or grocery item? Reply yes or no."
  -> ONE Groq call per item.

Batched (extract_item_fields_batch()), production path since Issue #46:
  -> ONE Groq call per chunk of up to 5 items, numbered-list-in /
     JSON-array-out, strict index correspondence, fail-safe to
     UNKNOWN for the whole chunk on any malformed/wrong-length response.
```

**Why batching, not concurrency:** an earlier design considered concurrent per-item calls (bounded by a semaphore) to fit the pipeline's 10-second upload budget (see §9's latency discussion, and Item_Extraction.md's original per-item latency measurement of ~658ms/item). Real testing during `pipeline.py`'s build (#25) found concurrency insufficient — it bounds parallel *requests*, not *tokens/minute*, and Groq's free-tier TPM limit (8000/min) was exhausted regardless of concurrency setting once total token volume across ~15-20 items was considered. Batching (chunk size 5) fixes this directly by cutting Groq calls ~5x per receipt. Full incident/fix record: Item_Extraction.md §4.1, HANDOFF.md.

Module: `ml_service/item_extraction/food_classifier.py`

**Distinct from Stage 3 Pass 3's LLM call** — Stage 2 asks "is this food at all", Stage 3 Pass 3 asks "what's the canonical name for this (already-confirmed) food item". Different questions; both needed. `is_food = false` skips Stage 3 and Stage 4 entirely — no shelf-life to predict for a non-food item — and the item is surfaced to the user in the confirmation modal flagged as excluded, rather than silently dropped.

Module: `ml_service/item_extraction/extractor.py` orchestrates all three and returns the combined `{is_food, brand, unit}` result. `extract_item_fields_batch()` is the production entry point `pipeline.py` calls; unit/brand extraction still runs per-item (local, not the bottleneck), only is_food classification is batched.

**Open consideration, not yet decided:** Stage 2's is_food call and Stage 3 Pass 3's canonical-naming call could be merged into one LLM round-trip for is_food=true items. Not implemented — keeping them separate is simpler to reason about and debug first.

**Known gap, low priority:** Groq's `gpt-oss-20b` can enter a reasoning-loop failure on severely corrupted OCR input, consuming its full token budget without emitting content, even with `reasoning_effort="low"` applied. Observed on 2/36 items in real pipeline testing, both tied to a hand blocking part of the receipt during photo capture. Pipeline fails safe correctly (item surfaced, not dropped). Tracked as GitHub Issue #48, not actively being worked. Full detail: Item_Extraction.md §4.2.

---

## 7. Stage 3: Normalization

Mechanism unchanged from v1.0 — still a three-pass approach. **Now only runs for items where Stage 2 returned `is_food = true`.**

### Goal

Convert raw, abbreviated, retailer-specific food tokens into canonical food names that match the `shelf_life_reference` table.

`"ORG STRWBRY 1LB"` → `{canonical_name: "Strawberries", quantity: 1.0, unit: "lb", category: "Produce"}`

### Method: Three-Pass Approach

**Pass 1 — Direct Lookup**
Curated abbreviation dictionary (~580 entries after dedup), handles the large majority of real-world cases (97.2% canonical match rate + LLM fallback combined in the latest eval — see Normalization.md).

**Pass 2 — Fuzzy Matching**
`rapidfuzz` fuzzy-match against `shelf_life_reference.canonical_name` values, threshold ≥ 80.

**Pass 3 — LLM Cleaning (Fallback)**
For tokens Pass 1 and 2 cannot resolve, sends to LLM for canonical name resolution. Cached in `normalization_cache`. Runs **per-item, not batched** — a deliberate scope decision (Issue #46) distinct from Stage 2's batching, since Pass 3's call volume is naturally low (cache/fuzzy-match miss only) and batching here would trade accuracy risk for throughput this stage doesn't need.

**Validated against real pipeline output** (this session, via `test_pipeline_local.py`) — 2 real receipts, 34/36 items resolved correctly end-to-end. See Normalization.md §9.1 for detail.

### Unit / Category

Unchanged from v1.0 — see original `UNIT_MAP` and `CATEGORY_KEYWORDS` in `ml_service/normalization/unit_normalizer.py` and `category_classifier.py`.

---

## 8. Stage 4: Expiry Prediction

Unchanged from v1.0. Rule-based lookup from `shelf_life_reference` + confidence scoring. See `ml_service/expiry/predictor.py` and Expiry.md.

**Validated against real pipeline output** as part of this session's `test_pipeline_local.py` runs, alongside Stage 3 — see Expiry.md / Normalization.md §9.1.

---

## 9. Pipeline Execution

### Inference Code Structure

```
ml_service/
├── pipeline.py               # Orchestrates all stages (#25 - built and wired this session,
│                              #   validated against 2 real receipts via test_pipeline_local.py)
├── ocr/
│   ├── model.py              # PaddleOCR loader + inference
│   └── row_reconstruction.py # Deskew + y-clustering (Stage 1.5)
├── parsing/
│   ├── prefilter.py          # Rule-based metadata-row drop (Stage 1.6)
│   └── row_parser.py         # Header detection + field extraction (Stage 1.7)
├── item_extraction/
│   ├── unit_extractor.py     # Regex unit parsing
│   ├── brand_matcher.py      # Fuzzy lexicon match
│   ├── food_classifier.py    # LLM is_food gate - single-item AND batched (chunk size 5, Issue #46)
│   └── extractor.py          # Orchestrates the above (Stage 2) - extract_item_fields_batch()
│                              #   is the production entry point pipeline.py calls
├── normalization/
│   ├── abbreviation_map.py
│   ├── fuzzy_matcher.py
│   └── llm_fallback.py       # Stage 3 Pass 3 — distinct from Stage 2's LLM call, stays per-item
├── expiry/
│   └── predictor.py          # Shelf-life lookup + confidence
├── ner/
│   └── archive/
│       └── ner-training.ipynb  # Historical record only — not imported anywhere
└── models/
    └── trocr-smart-stock/    # Retired TrOCR weights (superseded by PaddleOCR, see OCR_Training.md) — cleanup decision separate from this restructure
```

### Latency Budget (per receipt, CPU inference)

**Note:** Stage 2's LLM-call latency dominates and is now understood in detail (see Item_Extraction.md's per-item measurement, ~658ms/item single-call, and the batching fix in §6 above that cuts total Groq call count ~5x per receipt). **Full end-to-end wall-clock latency for the built pipeline has still not been formally measured** — only informal evidence exists from `test_pipeline_local.py` log timestamps (~8-12s of visible Groq-call time across 2 test receipts), which excludes OCR/Row Reconstruction/Row Parser time and DB round-trips. A real timer around `process_receipt()` is the next concrete step (see HANDOFF.md).

| Stage                  | Target Latency                                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| Image preprocessing    | N/A — handled internally by PaddleOCR                                                    |
| OCR (PaddleOCR)        | ~5-6s (measured)                                                                          |
| Row reconstruction     | Not yet measured — expected negligible (pure Python, no model)                           |
| Prefilter + Row Parser | Not yet measured — expected negligible                                                   |
| Item Field Extraction  | ~658ms per single-item LLM call (measured); batched in chunks of 5 in production, reducing total call count ~5x per receipt but per-call latency itself is unchanged |
| Normalization          | < 300ms (target, unvalidated against real timing data — correctness validated, not speed) |
| Expiry prediction      | < 100ms (target, unvalidated against real timing data — correctness validated, not speed) |
| **Total**        | **Needs full wall-clock re-measurement — pipeline is built and correctness-validated on 2 real receipts, but no formal end-to-end timer has been run yet** |

---

## 10. Evaluation Metrics

### OCR

Unchanged from v1.0 — see original doc content. CER/WER not measured (pretrained, no labeled eval set); qualitative validation only.

### Row Parser (new)

| Metric                    | Definition                                                              | Status                                                         |
| ------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------|
| Field extraction accuracy | % of quantity/price/discount/total correctly extracted vs. ground truth | Validated on 4 receipts, 100% match — not yet tested at scale |
| Header detection rate     | % of receipts where a header row is correctly found                     | Not formally measured beyond the 4-receipt sample              |

### Item Field Extraction (new, replaces NER metrics)

| Metric               | Definition                                    | Status                                      |
| -------------------- | ---------------------------------------------- | ------------------------------------------- |
| is_food accuracy     | LLM gate correctness vs. labeled sample       | Not yet measured — no labeled sample built (#34). Real pipeline runs (2 receipts, 36 items) show correct classification on all resolvable items, but this is not a substitute for a formal labeled-sample measurement. |
| Unit extraction rate | % of items with a unit successfully extracted | Not yet measured                            |
| Brand match rate     | % of items with a brand successfully matched  | Not yet measured                            |

**Retired:** NER Entity-level F1/Precision/Recall metrics — no model, nothing to score. Historical DistilBERT results (F1 0.907) preserved in NER_Training.md for the record.

### Normalization

| Metric               | Definition                     | Target | Status |
| -------------------- | ------------------------------- | ------ | ------ |
| Canonical Match Rate | % items resolved by Pass 1 + 2 | ≥ 80% | 97.2% on synthetic eval (Normalization.md §9). Real pipeline runs: 34/36 end-to-end. |
| LLM Fallback Rate    | % items needing Pass 3         | ≤ 20% | 11.1% on synthetic eval. |

### Expiry Prediction

| Metric                   | Definition                            | Target      | Status |
| ------------------------ | -------------------------------------- | ----------- | ------ |
| MAE (days)               | Mean absolute error vs. actual expiry | ≤ 1.5 days | 100% within ±2 days on synthetic 16-case eval (Expiry.md) |
| High-confidence accuracy | Accuracy when confidence ≥ 0.85      | ≥ 92%      | Not yet separately broken out by confidence tier |

### End-to-End

| Metric              | Definition                                           | Target | Status |
| ------------------- | ------------------------------------------------------ | ------ | ------ |
| Item-level Accuracy | % items correctly extracted + named on test receipts | ≥ 85% | 34/36 = 94.4% across 2 real receipts this session (informal, not yet a formal measurement against ground truth) |
| Processing Time     | Wall clock, full pipeline, CPU                       | < 10s  | **Not yet measured** — see Latency Budget section above and HANDOFF.md next steps |

**Pipeline is now built and running end-to-end on real data** (`pipeline.py`, #25, validated against 2 of 4 available real receipts) — a major change from the prior "pipeline.py currently empty" status. Formal wall-clock latency measurement and broader real-receipt coverage (2 more receipts available, not yet tried) remain the next concrete steps.
