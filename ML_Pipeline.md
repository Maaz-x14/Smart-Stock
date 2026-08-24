# ML_Pipeline.md — Machine Learning Pipeline

## Smart-Stock: OCR → NER → Normalization → Expiry Prediction

**Version:** 1.0

---

## 1. Pipeline Overview

The Smart-Stock ML pipeline transforms a raw receipt image into structured, expiry-annotated inventory items. It consists of four sequential stages:

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
│  │  Output: Raw text string                                 │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: Named Entity Recognition (NER)                 │   │
│  │  Model: DistilBERT (fine-tuned on annotated receipt NER) │   │
│  │  Input:  Raw text tokens                                 │   │
│  │  Output: Tagged entities: FOOD_ITEM, QUANTITY, UNIT,     │   │
│  │          BRAND, PRICE, OTHER                             │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 3: Normalization                                  │   │
│  │  Method: Fuzzy match + lookup table + LLM cleaning       │   │
│  │  Input:  Raw NER entities                                │   │
│  │  Output: Canonical item records with qty/unit parsed     │   │
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

---

## 2. Stage 1: OCR — Text Extraction

### Model

**PaddleOCR** (PP-OCRv6, pretrained — no fine-tuning) — small det/rec models, doc-orientation/unwarping/textline-orientation disabled for speed. Ships its own integrated text detector, so no separate line-segmentation stage is needed.

PaddleOCR's CNN-based detector+recognizer (DBNet + CRNN/SVTR) proved a better structural fit for CPU-served receipt OCR than TrOCR's transformer decoder — beat a 2-month fine-tuned TrOCR on both accuracy and latency (~5-6s/receipt vs ~480s/receipt) without any fine-tuning. See OCR_Training.md Decision Log for the full comparison.

### Input Preprocessing

Handled internally by PaddleOCR's pipeline (detection runs on the full image directly — no separate deskew/binarize/resize/line-crop stage required, unlike the original TrOCR approach).

### Output

```python
result = ocr.predict(image_path)
res = result[0]
texts  = res["rec_texts"]   # list of recognized text lines
scores = res["rec_scores"]  # confidence per line
```

### OCR Post-processing

- Price line filtering: remove lines matching pattern `r'^\d+\.\d{2}$'` (price-only lines)
- Header/footer stripping: remove store name, date, total, tax lines using keyword heuristics (GST, Invoice, Transaction, POS, Customer, CNIC, Payments, Total, Discount, Rounding, Tax Breakup — planned pre-filter before Stage 2 NER)

## 3. Stage 2: NER — Entity Extraction

### Model

**DistilBERT** (`distilbert-base-uncased`), fine-tuned for token classification on an annotated receipt NER corpus. F1 0.907 (test), exceeds target.

**Locked in.** Benchmarked against BERT-base, RoBERTa-base, ModernBERT-base (same `ner_splits`, 15 epochs) — DistilBERT won on both F1 (0.9156) and CPU latency (28.75ms). See NER_Training.md §23 for full comparison.

### Pre-filter (rule-based, before NER)

Runs on OCR output lines before they reach the NER model. Drops lines that are 100% certainly not items — pure noise, no food-semantic judgment involved (that judgment is NER + Stage 3's job, not this filter's). Module: `ml_service/ner/prefilter.py`.

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

def should_drop_line(line: str) -> bool:
    stripped = line.strip()

    if len(stripped) < 3:
        return True
    if any(kw in stripped.upper() for kw in DROP_KEYWORDS):
        return True
    if "@" in stripped or "www." in stripped.lower() or "http" in stripped.lower() or ".com" in stripped.lower():
        return True
    for pattern in (PRICE_ONLY_RE, PHONE_RE, DATE_RE, TIME_RE, SEPARATOR_RE, PERCENT_ONLY_RE, BARCODE_RE):
        if pattern.match(stripped):
            return True
    return False
```

**Scope:** drops standalone price/total/date/contact/barcode lines only — never strips a price token off an item line (`"STRWBRY 1 LB 2.99"` stays whole; NER still tags `B-PRICE` on it). Not yet validated against real PaddleOCR receipt output — risk noted in NER_Training.md-adjacent testing: substring keyword match on "CASH"/"TOTAL" could false-positive on a product name that happens to contain those words.

### Entity Labels (BIO tagging scheme)

| Label       | Meaning                | Example                    |
| ----------- | ---------------------- | -------------------------- |
| `B-FOOD`  | Beginning of food item | `ORG` in `ORG STRWBRY` |
| `I-FOOD`  | Inside food item       | `STRWBRY`                |
| `B-QTY`   | Quantity               | `1`                      |
| `B-UNIT`  | Unit of measure        | `LB`, `GAL`, `CT`    |
| `B-BRAND` | Brand name             | `ORGANIC VALLEY`         |
| `B-PRICE` | Price token            | `2.99`                   |
| `O`       | Outside / irrelevant   | Store name, date, etc.     |

### Input

Tokenized receipt text lines, after OCR post-processing **and** the rule-based pre-filter (see below — drops pure-noise lines before NER ever sees them). Lines are processed independently. Max sequence length: 128 tokens.

### Output

Token-level entity tags. Post-processing groups consecutive `B-FOOD` / `I-FOOD` tags into entity spans:

```
Input:  ["ORG", "STRWBRY", "1", "LB", "2.99"]
Output: [B-FOOD, I-FOOD, B-QTY, B-UNIT, B-PRICE]

Grouped: {
  food_tokens: ["ORG", "STRWBRY"],
  quantity: "1",
  unit: "LB",
  price: "2.99"
}
```

---

## 4. Stage 3: Normalization

### Goal

Convert raw, abbreviated, retailer-specific food tokens into canonical food names that match the `shelf_life_reference` table.

`"ORG STRWBRY 1LB"` → `{canonical_name: "Strawberries", quantity: 1.0, unit: "lb", category: "Produce"}`

### Method: Three-Pass Approach

**Pass 1 — Direct Lookup**
Check a curated abbreviation dictionary (manually built from common retail receipt shortcodes):

```python
ABBREVIATION_MAP = {
    "STRWBRY": "Strawberries",
    "MLKWHL": "Whole Milk",
    "CHKN BRST": "Chicken Breast",
    "GRK YGRT": "Greek Yogurt",
    ...
}
```

~800 entries covering the most common grocery items. This handles ~65% of real-world cases.

**Pass 2 — Fuzzy Matching**
If Pass 1 misses, use `rapidfuzz` library to fuzzy-match against all `canonical_name` values in `shelf_life_reference`:

```python
from rapidfuzz import process, fuzz

match, score, _ = process.extractOne(
    raw_token,
    canonical_names_list,
    scorer=fuzz.token_sort_ratio
)
if score >= 80:
    return match
```

Handles misspellings, partial matches, and ordering variations.

**Pass 3 — LLM Cleaning (Fallback)**
For tokens that pass 1 and 2 cannot resolve (score < 80), send to a lightweight LLM (Ollama `llama3.2:1b` running locally, or Claude API call):

```
Prompt: "This is a line item from a US grocery store receipt: '{raw_token}'.
What common food item does this refer to? Reply with just the canonical food name, nothing else."
```

Results are cached in a `normalization_cache` table to avoid redundant LLM calls.

### Unit Normalization

```python
UNIT_MAP = {
    "LB": "lb", "LBS": "lb",
    "OZ": "oz",
    "GAL": "gal", "GL": "gal",
    "CT": "count", "PK": "pack",
    "EA": "each",
}
```

### Category Assignment

After canonical name is resolved, look up `category` from `shelf_life_reference` by `canonical_name`. If not found, use a simple keyword classifier:

```python
CATEGORY_KEYWORDS = {
    "Produce": ["berries", "apple", "lettuce", "tomato", ...],
    "Dairy":   ["milk", "cheese", "yogurt", "butter", ...],
    "Meat":    ["chicken", "beef", "pork", "salmon", ...],
    "Pantry":  ["bread", "pasta", "rice", "sauce", ...],
    "Frozen":  ["frozen", "ice cream", ...],
}
```

---

## 5. Stage 4: Expiry Prediction

### Method

Hybrid: rule-based lookup from `shelf_life_reference` + confidence scoring based on match quality.

### Algorithm

```python
def predict_expiry(
    canonical_name: str,
    storage_context: str,
    purchase_date: date,
    normalization_confidence: float
) -> tuple[date, float]:

    # 1. Look up shelf life reference
    ref = db.query(ShelfLifeReference).filter_by(
        canonical_name=canonical_name,
        storage_context=storage_context
    ).first()

    if ref:
        shelf_life_days = ref.shelf_life_days_avg
        base_confidence = 0.95
    else:
        # 2. Category-level fallback
        ref = db.query(ShelfLifeReference).filter_by(
            category=item_category,
            storage_context=storage_context
        ).order_by(ShelfLifeReference.shelf_life_days_avg).first()
        shelf_life_days = ref.shelf_life_days_avg if ref else 7
        base_confidence = 0.70

    # 3. Adjust confidence by normalization quality
    final_confidence = base_confidence * normalization_confidence

    predicted_expiry = purchase_date + timedelta(days=shelf_life_days)
    return predicted_expiry, round(final_confidence, 3)
```

### Confidence Score Interpretation

| Range        | Meaning                                           |
| ------------ | ------------------------------------------------- |
| 0.90 – 1.00 | High confidence: exact match in reference table   |
| 0.70 – 0.89 | Medium: category-level fallback or fuzzy match    |
| 0.50 – 0.69 | Low: LLM normalization used, or no category match |
| < 0.50       | Very low: flag for user review                    |

Items with confidence < 0.60 are surfaced in the confirmation modal with a warning indicator, prompting the user to verify the predicted expiry date manually.

---

## 6. Pipeline Execution

### Inference Code Structure

```
ml_service/
├── pipeline.py          # Orchestrates all 4 stages
├── ocr/
│   ├── model.py         # PaddleOCR loader + inference
│   └── preprocessor.py  # Optional receipt cleanup/helpers
├── ner/
│   ├── prefilter.py     # Rule-based noise-line drop, before NER
│   ├── model.py         # DistilBERT loader + inference
│   └── entity_parser.py # BIO tag grouping
├── normalization/
│   ├── abbreviation_map.py
│   ├── fuzzy_matcher.py
│   └── llm_fallback.py
├── expiry/
│   └── predictor.py     # Shelf-life lookup + confidence
└── models/              # Serialized model weights
    ├── paddleocr        # PaddleOCR's own runtime — not ONNX
    └── distilbert_ner.onnx
```

### Latency Budget (per receipt, CPU inference)

**Note:** Total budget below is stale — OCR alone (~5-6s) already exceeds the original <3000ms total target set when OCR was assumed to be TrOCR+ONNX at <1500ms. Revisit this budget once Stage 2 NER model is locked in and Stage 3/4 are validated against real pipeline output.

| Stage                 | Target Latency     |
| --------------------- | ------------------ |
| Image preprocessing   | N/A — handled internally by PaddleOCR |
| OCR (PaddleOCR)        | ~5-6s (measured, pretrained, not yet latency-optimized) |
| NER (DistilBERT ONNX) | < 500ms            |
| Normalization         | < 300ms            |
| Expiry prediction     | < 100ms            |
| **Total**       | **~6-7s (target needs revision, see note above)** |

---

## 7. Evaluation Metrics

### OCR

CER/WER targets below were set for a fine-tuned model (TrOCR) with a held-out labeled test set. PaddleOCR is used pretrained — no fine-tuning eval loop produces these numbers today. Real-receipt accuracy was validated qualitatively instead (correct item/price/qty extraction across test receipts). A formal CER/WER benchmark against annotated real receipts is still open — see OCR_Training.md.

| Metric                     | Definition                  | Target | Status |
| -------------------------- | --------------------------- | ------ | ------ |
| Character Error Rate (CER) | Edit distance / total chars | ≤ 5%  | Not measured for PaddleOCR — no labeled eval set yet |
| Word Error Rate (WER)      | Word-level edit distance    | ≤ 10% | Not measured for PaddleOCR — no labeled eval set yet |
| Line Detection Rate        | % of receipt lines captured | ≥ 95% | Not formally measured — qualitatively all lines captured on test receipts |

### NER

| Metric          | Definition                 | Target  |
| --------------- | -------------------------- | ------- |
| Entity-level F1 | F1 over FOOD_ITEM entities | ≥ 0.88 |
| Precision       | TP / (TP + FP)             | ≥ 0.90 |
| Recall          | TP / (TP + FN)             | ≥ 0.86 |

### Normalization

| Metric               | Definition                     | Target |
| -------------------- | ------------------------------ | ------ |
| Canonical Match Rate | % items resolved by Pass 1 + 2 | ≥ 80% |
| LLM Fallback Rate    | % items needing Pass 3         | ≤ 20% |

### Expiry Prediction

| Metric                   | Definition                            | Target      |
| ------------------------ | ------------------------------------- | ----------- |
| MAE (days)               | Mean absolute error vs. actual expiry | ≤ 1.5 days |
| High-confidence accuracy | Accuracy when confidence ≥ 0.85      | ≥ 92%      |

### End-to-End

| Metric              | Definition                                           | Target |
| ------------------- | ---------------------------------------------------- | ------ |
| Item-level Accuracy | % items correctly extracted + named on test receipts | ≥ 85% |
| Processing Time     | Wall clock, full pipeline, CPU                       | < 10s  |
