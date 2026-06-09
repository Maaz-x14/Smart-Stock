# Expiry_Training.md — Stage 4 Expiry Prediction Guide
## Smart-Stock: Stage 4 — Expiry Date Prediction

**Version:** 1.0 (Initial build — rule-based lookup + confidence scoring)
**Environment:** Local Python / FastAPI container (no GPU required)
**Method:** shelf_life_reference DB lookup → category fallback → confidence scoring

---

## ⚠️ Critical Design Reality Check

Before any code: Stage 4 is **not a training pipeline**. It is a production module that runs at inference time inside `ml_service/`. There is no model to fine-tune, no GPU, no dataset, no Trainer.

| Stage | What it does | Where it lives |
|---|---|---|
| Stage 1 (OCR) | Fine-tune TrOCR on receipt images | Kaggle notebook |
| Stage 2 (NER) | Fine-tune DistilBERT for token classification | Kaggle notebook |
| Stage 3 (Normalization) | Rule-based + fuzzy + LLM text normalization | `ml_service/normalization/` |
| **Stage 4 (Expiry)** | **Shelf-life lookup + confidence scoring → predicted expiry date** | **`ml_service/expiry/`** |

Stage 4 is pure logic — it reads the `shelf_life_reference` table seeded in Stage 3 and computes a predicted expiry date from `purchase_date + shelf_life_days_avg`. The confidence score propagates forward from Stage 3's normalization quality into the final expiry prediction.

---

## 1. Overview

Stage 4 takes a `NormalizedItem` from Stage 3 and produces a predicted expiry date with a confidence score. That output is what gets written to the `inventory_items` table and displayed on the frontend dashboard.

```
Input  (from Stage 3):   NormalizedItem(canonical_name="Strawberries", category="Produce",
                                        quantity=1.0, unit="lb", confidence=0.95)
                         + storage_context="Fridge"
                         + purchase_date=date(2025, 6, 1)

Output (to inventory):   ExpiryPrediction(predicted_expiry=date(2025, 6, 6),
                                          shelf_life_days=5,
                                          confidence=0.9025,
                                          source="exact_match")
```

### Why hybrid rule-based, not ML regression

A trained regression model for shelf life would require a large dataset of (food item, storage condition, actual expiry date) pairs — data that does not exist in any public dataset at the granularity needed. The `shelf_life_reference` table, built from food safety guidelines and seeded in Stage 3, encodes this knowledge directly.

The confidence scoring system gives the frontend everything it needs to surface low-confidence predictions for user review — the same outcome a regression model's prediction interval would provide, without requiring training data that doesn't exist.

### Prediction algorithm — two-level lookup

```
NormalizedItem + storage_context + purchase_date
      |
      v
┌─────────────────────────────────────────────────┐
│  Level 1: Exact Match                           │
│  Query shelf_life_reference WHERE               │
│    canonical_name = item.canonical_name         │
│    AND storage_context = storage_context        │
│  → shelf_life_days_avg                          │
│  → base_confidence = 0.95                       │
└──────────────────┬──────────────────────────────┘
                   │ no row found
                   v
┌─────────────────────────────────────────────────┐
│  Level 2: Category Fallback                     │
│  Query shelf_life_reference WHERE               │
│    category = item.category                     │
│    AND storage_context = storage_context        │
│  → median shelf_life_days_avg across category   │
│  → base_confidence = 0.70                       │
└──────────────────┬──────────────────────────────┘
                   │ no rows found
                   v
┌─────────────────────────────────────────────────┐
│  Level 3: Hard Default                          │
│  shelf_life_days = 7 (1 week)                  │
│  base_confidence = 0.40                         │
└──────────────────┬──────────────────────────────┘
                   │
                   v
   final_confidence = base_confidence × normalization_confidence
   predicted_expiry = purchase_date + timedelta(days=shelf_life_days)
```

---

## 2. Module Structure

```
ml_service/
├── pipeline.py                 # Orchestrates all 4 stages
├── normalization/
│   └── normalizer.py           # Produces NormalizedItem → input to Stage 4
├── expiry/
│   ├── __init__.py
│   └── predictor.py            # Full Stage 4 logic — predict_expiry() entry point
└── models/
    ├── trocr.onnx
    └── distilbert_ner.onnx
```

Stage 4 is a single file. It has one public function: `predict_expiry()`. Nothing outside `expiry/` imports from submodules — `pipeline.py` calls `predict_expiry()` directly.

---

## 3. Storage Context

Storage context is the user-selected (or category-defaulted) storage condition for an item. It is the second dimension of the `shelf_life_reference` lookup — the same canonical name has different shelf lives depending on where it is stored.

| Storage Context | Meaning | Example items |
|---|---|---|
| `Fridge` | Refrigerated (0–5°C) | Dairy, raw meat, opened sauces, most produce |
| `Freezer` | Frozen (−18°C) | Raw meat (bulk), frozen veg, ice cream |
| `Pantry` | Room temperature, dry | Rice, pasta, canned goods, onions, bananas |

**Default assignment by category** — when the user doesn't specify a storage context, the predictor applies a sensible default:

```python
CATEGORY_DEFAULT_STORAGE = {
    "Produce":   "Fridge",
    "Dairy":     "Fridge",
    "Meat":      "Fridge",
    "Pantry":    "Pantry",
    "Frozen":    "Freezer",
    "Beverages": "Pantry",
    "Bakery":    "Pantry",
    "Other":     "Fridge",   # conservative default
}
```

The frontend confirmation modal lets users override this per item. The override is stored in `inventory_items.storage_context`.

---

## 4. Confidence Scoring

Confidence is a single float in [0, 1] that represents how reliable the predicted expiry date is. It is the product of two independent signals:

```
final_confidence = base_confidence × normalization_confidence
```

**`base_confidence`** — set by which level of the expiry lookup resolved the item:

| Level | Source | base_confidence |
|---|---|---|
| Level 1 | Exact match in `shelf_life_reference` | 0.95 |
| Level 2 | Category-level median fallback | 0.70 |
| Level 3 | Hard default (7 days) | 0.40 |

**`normalization_confidence`** — carried forward from Stage 3's `NormalizedItem.confidence`:

| Stage 3 Pass | normalization_confidence |
|---|---|
| Pass 1 (abbreviation map) | 1.00 |
| Pass 2 (fuzzy match, score ≥ 80) | 0.80 – 0.99 |
| Pass 3 (LLM fallback) | 0.70 |

**Combined confidence interpretation:**

| Range | Meaning | Frontend behaviour |
|---|---|---|
| 0.90 – 1.00 | High — exact shelf-life data + reliable normalization | Show predicted date, no warning |
| 0.70 – 0.89 | Medium — some uncertainty in name or shelf life | Show predicted date, subtle indicator |
| 0.50 – 0.69 | Low — LLM normalization or category fallback | Show predicted date + yellow warning |
| < 0.50 | Very low — hard default shelf life and/or LLM name | Flag for manual review in confirmation modal |

Items with `final_confidence < 0.60` are surfaced in the frontend confirmation modal with a warning and a prompt for the user to verify or override the predicted expiry date.

---

## 5. Implementation

```python
# ml_service/expiry/predictor.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median

from sqlalchemy.orm import Session

from app.models import ShelfLifeReference
from ml_service.normalization.normalizer import NormalizedItem


# Storage context default per category — used when user has not specified
CATEGORY_DEFAULT_STORAGE: dict[str, str] = {
    "Produce":   "Fridge",
    "Dairy":     "Fridge",
    "Meat":      "Fridge",
    "Pantry":    "Pantry",
    "Frozen":    "Freezer",
    "Beverages": "Pantry",
    "Bakery":    "Pantry",
    "Other":     "Fridge",
}

# Base confidence by lookup level
BASE_CONFIDENCE_EXACT    = 0.95
BASE_CONFIDENCE_CATEGORY = 0.70
BASE_CONFIDENCE_DEFAULT  = 0.40

# Hard-default shelf life when no reference data exists at all
DEFAULT_SHELF_LIFE_DAYS = 7

# Confidence threshold below which item is flagged for user review
REVIEW_THRESHOLD = 0.60


@dataclass
class ExpiryPrediction:
    predicted_expiry:   date
    shelf_life_days:    int
    confidence:         float
    source:             str    # "exact_match" | "category_fallback" | "hard_default"
    storage_context:    str
    flag_for_review:    bool   # True if confidence < REVIEW_THRESHOLD


def _resolve_storage_context(item: NormalizedItem, storage_context: str | None) -> str:
    """
    Return the storage context to use for this item.
    Uses user-provided value if given; falls back to category default.
    """
    if storage_context:
        return storage_context
    return CATEGORY_DEFAULT_STORAGE.get(item.category, "Fridge")


def _level1_exact(
    canonical_name: str,
    storage_context: str,
    db: Session,
) -> int | None:
    """
    Query shelf_life_reference for an exact (canonical_name, storage_context) match.
    Returns shelf_life_days_avg or None if not found.
    """
    ref = (
        db.query(ShelfLifeReference)
        .filter_by(canonical_name=canonical_name, storage_context=storage_context)
        .first()
    )
    return ref.shelf_life_days_avg if ref else None


def _level2_category(
    category: str,
    storage_context: str,
    db: Session,
) -> int | None:
    """
    Query shelf_life_reference for all items in this category + storage_context.
    Returns the median shelf_life_days_avg across the category, or None if no rows.
    """
    rows = (
        db.query(ShelfLifeReference.shelf_life_days_avg)
        .filter_by(category=category, storage_context=storage_context)
        .all()
    )
    if not rows:
        return None
    values = [r[0] for r in rows]
    return int(median(values))


def predict_expiry(
    item:             NormalizedItem,
    purchase_date:    date,
    db:               Session,
    storage_context:  str | None = None,
) -> ExpiryPrediction:
    """
    Predict expiry date for a normalized inventory item.

    Args:
        item:             NormalizedItem output from Stage 3 normalize_entity()
        purchase_date:    Date the item was purchased (from receipt or user input)
        db:               Active SQLAlchemy session
        storage_context:  "Fridge", "Freezer", or "Pantry". If None, defaults
                          by category via CATEGORY_DEFAULT_STORAGE.

    Returns:
        ExpiryPrediction with predicted_expiry, confidence, source, and review flag.

    Examples:
        item = NormalizedItem("Strawberries", 1.0, "lb", "Produce", 1, 1.0)
        result = predict_expiry(item, date(2025, 6, 1), db)
        # → ExpiryPrediction(predicted_expiry=date(2025, 6, 6), shelf_life_days=5,
        #                    confidence=0.9025, source="exact_match", ...)

        item = NormalizedItem("Exotic Mango Variety", 1.0, None, "Produce", 3, 0.70)
        result = predict_expiry(item, date(2025, 6, 1), db, storage_context="Fridge")
        # → ExpiryPrediction(predicted_expiry=date(2025, 6, 7), shelf_life_days=6,
        #                    confidence=0.49, source="category_fallback",
        #                    flag_for_review=True)
    """
    resolved_storage = _resolve_storage_context(item, storage_context)

    # ── Level 1: Exact match ──────────────────────────────────────────────────
    shelf_life_days = _level1_exact(item.canonical_name, resolved_storage, db)
    if shelf_life_days is not None:
        base_confidence = BASE_CONFIDENCE_EXACT
        source = "exact_match"

    # ── Level 2: Category fallback ────────────────────────────────────────────
    else:
        shelf_life_days = _level2_category(item.category, resolved_storage, db)
        if shelf_life_days is not None:
            base_confidence = BASE_CONFIDENCE_CATEGORY
            source = "category_fallback"

        # ── Level 3: Hard default ─────────────────────────────────────────────
        else:
            shelf_life_days = DEFAULT_SHELF_LIFE_DAYS
            base_confidence = BASE_CONFIDENCE_DEFAULT
            source = "hard_default"

    # ── Confidence scoring ────────────────────────────────────────────────────
    final_confidence = round(base_confidence * item.confidence, 4)
    predicted_expiry = purchase_date + timedelta(days=shelf_life_days)
    flag_for_review  = final_confidence < REVIEW_THRESHOLD

    return ExpiryPrediction(
        predicted_expiry=predicted_expiry,
        shelf_life_days=shelf_life_days,
        confidence=final_confidence,
        source=source,
        storage_context=resolved_storage,
        flag_for_review=flag_for_review,
    )
```

---

## 6. Pipeline Integration

Stage 4 is the final step in `pipeline.py`. After `normalize_entity()` returns a `NormalizedItem`, `predict_expiry()` is called immediately with the purchase date and optional storage context.

```python
# ml_service/pipeline.py  (Stage 4 integration excerpt)

from datetime import date
from sqlalchemy.orm import Session

from ml_service.normalization.normalizer import normalize_entity, NormalizedItem
from ml_service.expiry.predictor import predict_expiry, ExpiryPrediction


def run_pipeline(
    receipt_image,           # PIL Image or file path
    purchase_date: date,
    db: Session,
    storage_overrides: dict[str, str] | None = None,
) -> list[dict]:
    """
    Full 4-stage pipeline: receipt image → structured inventory items with expiry dates.

    Args:
        receipt_image:     Receipt image (PIL Image)
        purchase_date:     Date of purchase (from receipt header or user input)
        db:                Active SQLAlchemy session
        storage_overrides: Optional dict mapping canonical_name → storage_context
                           for items the user has pre-specified storage for.

    Returns:
        List of inventory item dicts ready for DB insert into inventory_items.
    """
    # Stage 1: OCR — receipt image → raw text lines
    # (TrOCR inference — see ml_service/ocr/model.py)
    raw_lines = run_ocr(receipt_image)

    # Stage 2: NER — raw text lines → entity dicts
    # (DistilBERT inference — see ml_service/ner/model.py)
    entities = run_ner(raw_lines)

    inventory_items = []

    for entity in entities:
        # Stage 3: Normalization — entity → NormalizedItem
        normalized: NormalizedItem | None = normalize_entity(
            food_tokens=entity["food_tokens"],
            raw_quantity=entity.get("quantity"),
            raw_unit=entity.get("unit"),
            db=db,
        )

        if normalized is None:
            # Unresolvable token — skip; will appear as "unrecognized item" in UI
            continue

        # Stage 4: Expiry prediction — NormalizedItem → ExpiryPrediction
        storage_ctx = None
        if storage_overrides:
            storage_ctx = storage_overrides.get(normalized.canonical_name)

        expiry: ExpiryPrediction = predict_expiry(
            item=normalized,
            purchase_date=purchase_date,
            db=db,
            storage_context=storage_ctx,
        )

        inventory_items.append({
            "canonical_name":    normalized.canonical_name,
            "quantity":          normalized.quantity,
            "unit":              normalized.unit,
            "category":          normalized.category,
            "storage_context":   expiry.storage_context,
            "purchase_date":     purchase_date.isoformat(),
            "predicted_expiry":  expiry.predicted_expiry.isoformat(),
            "shelf_life_days":   expiry.shelf_life_days,
            "confidence":        expiry.confidence,
            "expiry_source":     expiry.source,
            "flag_for_review":   expiry.flag_for_review,
            "normalization_pass": normalized.normalization_pass,
        })

    return inventory_items
```

---

## 7. inventory_items Table

Stage 4's output is written to the `inventory_items` table. This is the table the frontend dashboard reads.

### 7.1 Schema

```sql
CREATE TABLE inventory_items (
    id               SERIAL PRIMARY KEY,
    user_id          INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    canonical_name   VARCHAR(100) NOT NULL,
    quantity         NUMERIC(8,2) NOT NULL,
    unit             VARCHAR(20),
    category         VARCHAR(50)  NOT NULL,
    storage_context  VARCHAR(20)  NOT NULL,
    purchase_date    DATE         NOT NULL,
    predicted_expiry DATE         NOT NULL,
    shelf_life_days  INTEGER      NOT NULL,
    confidence       NUMERIC(5,4) NOT NULL,
    expiry_source    VARCHAR(30)  NOT NULL,   -- exact_match | category_fallback | hard_default
    flag_for_review  BOOLEAN      NOT NULL DEFAULT FALSE,
    normalization_pass INTEGER    NOT NULL,   -- 1, 2, or 3 — which Stage 3 pass resolved the name
    status           VARCHAR(20)  NOT NULL DEFAULT 'ACTIVE',  -- ACTIVE | CONSUMED | WASTED
    created_at       TIMESTAMP    NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_inv_user_expiry   ON inventory_items (user_id, predicted_expiry);
CREATE INDEX idx_inv_user_status   ON inventory_items (user_id, status);
CREATE INDEX idx_inv_flag_review   ON inventory_items (flag_for_review) WHERE flag_for_review = TRUE;
```

### 7.2 SQLAlchemy model

Add this to `app/models.py`:

```python
from sqlalchemy import Column, Integer, String, Date, Boolean, Numeric, ForeignKey, DateTime, func

class InventoryItem(Base):
    __tablename__ = "inventory_items"

    id                 = Column(Integer,     primary_key=True)
    user_id            = Column(Integer,     ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    canonical_name     = Column(String(100), nullable=False)
    quantity           = Column(Numeric(8,2),nullable=False)
    unit               = Column(String(20))
    category           = Column(String(50),  nullable=False)
    storage_context    = Column(String(20),  nullable=False)
    purchase_date      = Column(Date,        nullable=False)
    predicted_expiry   = Column(Date,        nullable=False)
    shelf_life_days    = Column(Integer,     nullable=False)
    confidence         = Column(Numeric(5,4),nullable=False)
    expiry_source      = Column(String(30),  nullable=False)
    flag_for_review    = Column(Boolean,     nullable=False, default=False)
    normalization_pass = Column(Integer,     nullable=False)
    status             = Column(String(20),  nullable=False, default="ACTIVE")
    created_at         = Column(DateTime,    nullable=False, server_default=func.now())
    updated_at         = Column(DateTime,    nullable=False, server_default=func.now(), onupdate=func.now())
```

### 7.3 Alembic migration

After adding `InventoryItem` to `app/models.py`, generate and apply the migration:

```bash
alembic revision --autogenerate -m "add inventory_items table"
alembic upgrade head
```

---

## 8. Evaluation

No Kaggle, no GPU. Run locally.

### 8.1 Test harness

```python
# ml_service/expiry/evaluate.py
# Usage: python -m ml_service.expiry.evaluate

from datetime import date
from dataclasses import dataclass
from app.db import SessionLocal
from ml_service.normalization.normalizer import NormalizedItem
from ml_service.expiry.predictor import predict_expiry, REVIEW_THRESHOLD

# Ground truth: (NormalizedItem, storage_context, expected_shelf_life_days_approx)
# "approx" = expected days ± 2 counts as correct (shelf life is a range, not a point)
TEST_CASES = [
    # Exact match expected — all seeded in shelf_life_reference
    (NormalizedItem("Strawberries",    1.0, "lb",  "Produce",   1, 1.00), "Fridge",   5),
    (NormalizedItem("Chicken Breast",  1.0, "lb",  "Meat",      1, 1.00), "Fridge",   2),
    (NormalizedItem("Whole Milk",      1.0, "gal", "Dairy",     1, 1.00), "Fridge",   7),
    (NormalizedItem("Basmati Rice",    1.0, "kg",  "Pantry",    1, 1.00), "Pantry", 730),
    (NormalizedItem("Butter",          1.0, "lb",  "Dairy",     1, 1.00), "Freezer", 90),
    (NormalizedItem("Dahi",            1.0, None,  "Dairy",     1, 1.00), "Fridge",   5),
    (NormalizedItem("Mutton",          1.0, "kg",  "Meat",      1, 1.00), "Fridge",   3),
    (NormalizedItem("Paneer",          1.0, None,  "Dairy",     1, 1.00), "Fridge",   5),
    # Category fallback expected — not seeded by name, fallback by category
    (NormalizedItem("Exotic Berry Mix",1.0, None,  "Produce",   2, 0.80), "Fridge",   5),
    (NormalizedItem("Artisan Cheese",  1.0, None,  "Dairy",     2, 0.80), "Fridge",  14),
    # LLM-resolved items — lower confidence, may hit category fallback
    (NormalizedItem("Eggs",            1.0, None,  "Dairy",     3, 0.70), "Fridge",  35),
    (NormalizedItem("Coriander",       1.0, None,  "Produce",   3, 0.70), "Fridge",   7),
    # Storage context defaulting — no storage_context passed, uses category default
    (NormalizedItem("Onions",          2.0, "lb",  "Produce",   1, 1.00), None,      30),
    (NormalizedItem("Frozen Peas",     1.0, "bag", "Frozen",    1, 1.00), None,     270),
]

TOLERANCE_DAYS = 2  # predicted shelf_life_days within ±2 of expected = correct


@dataclass
class EvalResult:
    total:            int
    exact_match_hits: int
    category_hits:    int
    default_hits:     int
    correct:          int
    accuracy:         float
    avg_confidence:   float
    flagged_count:    int


def run_evaluation() -> EvalResult:
    db = SessionLocal()

    exact_hits = cat_hits = default_hits = correct = 0
    confidences = []
    flagged = 0

    purchase_date = date(2025, 6, 1)

    for item, storage_ctx, expected_days in TEST_CASES:
        result = predict_expiry(item, purchase_date, db, storage_context=storage_ctx)

        within_tolerance = abs(result.shelf_life_days - expected_days) <= TOLERANCE_DAYS
        if within_tolerance:
            correct += 1

        if result.source == "exact_match":
            exact_hits += 1
        elif result.source == "category_fallback":
            cat_hits += 1
        else:
            default_hits += 1

        if result.flag_for_review:
            flagged += 1

        confidences.append(result.confidence)

        status = "✅" if within_tolerance else "❌"
        print(
            f"  {status} [{result.source[:8]:8s}] "
            f"{item.canonical_name:25s} | "
            f"storage={result.storage_context:7s} | "
            f"predicted={result.shelf_life_days:4d}d | "
            f"expected≈{expected_days:4d}d | "
            f"conf={result.confidence:.3f}"
            + (" ⚠ review" if result.flag_for_review else "")
        )

    db.close()

    total = len(TEST_CASES)
    return EvalResult(
        total=total,
        exact_match_hits=exact_hits,
        category_hits=cat_hits,
        default_hits=default_hits,
        correct=correct,
        accuracy=round(correct / total, 3),
        avg_confidence=round(sum(confidences) / len(confidences), 3),
        flagged_count=flagged,
    )


if __name__ == "__main__":
    result = run_evaluation()

    print(f"\n{'─' * 55}")
    print(f"Total cases:          {result.total}")
    print(f"Exact match:          {result.exact_match_hits}")
    print(f"Category fallback:    {result.category_hits}")
    print(f"Hard default:         {result.default_hits}")
    print(f"Correct (±{TOLERANCE_DAYS}d):        {result.correct}")
    print(f"Accuracy:             {result.accuracy:.1%}  (target ≥ 90%)")
    print(f"Avg confidence:       {result.avg_confidence:.3f}")
    print(f"Flagged for review:   {result.flagged_count}")
```

Run it:

```bash
python -m ml_service.expiry.evaluate
```

Expected output (all shelf_life_reference rows seeded):

```
  ✅ [exact_ma] Strawberries              | storage=Fridge  | predicted=   5d | expected≈   5d | conf=0.950
  ✅ [exact_ma] Chicken Breast            | storage=Fridge  | predicted=   2d | expected≈   2d | conf=0.950
  ✅ [exact_ma] Whole Milk                | storage=Fridge  | predicted=   7d | expected≈   7d | conf=0.950
  ...
  ✅ [category] Exotic Berry Mix          | storage=Fridge  | predicted=   5d | expected≈   5d | conf=0.560 ⚠ review
  ...
────────────────────────────────────────────────────
Total cases:          14
Exact match:          11
Category fallback:     2
Hard default:          0
Correct (±2d):        13
Accuracy:             92.9%  (target ≥ 90%)
Avg confidence:       0.842
Flagged for review:    2
```

### 8.2 Target metrics

| Metric | Definition | Target |
|---|---|---|
| Shelf-life accuracy (±2 days) | % predictions within 2 days of reference avg | ≥ 90% |
| Exact match rate | % items resolved by Level 1 (exact DB lookup) | ≥ 75% |
| Category fallback rate | % items falling through to Level 2 | ≤ 20% |
| Hard default rate | % items hitting Level 3 (no reference data) | ≤ 5% |
| MAE (days) | Mean absolute error vs shelf_life_days_avg in reference | ≤ 1.5 days |

### 8.3 If targets aren't met

| Symptom | Action |
|---|---|
| Hard default rate > 5% | Items that hit Level 3 are not in `shelf_life_reference`. Print their `canonical_name` values and add them to `SHELF_LIFE_DATA` in `shelf_life_seed.py`, then re-run the seed. |
| Exact match rate < 75% | The canonical name from Stage 3 doesn't exactly match the `canonical_name` in the DB. Print `item.canonical_name` and compare against `shelf_life_reference`. Fix either the seed data spelling or the `ABBREVIATION_MAP` entry in Stage 3. |
| MAE > 1.5 days | Category fallback median is off for a specific category. Add more representative items to that category in `SHELF_LIFE_DATA` to bring the median closer to reality. |
| Confidence too low across the board | Stage 3 is hitting Pass 3 (LLM) too often. Expand `ABBREVIATION_MAP` in Stage 3 to raise normalization_confidence. |

---

## 9. Setup & Requirements

No new packages needed — Stage 4 uses only SQLAlchemy (already installed in Stage 3).

### 9.1 Add InventoryItem to app/models.py

Add the `InventoryItem` model from Section 7.2 to `app/models.py`. This is the only file that needs to change.

### 9.2 Generate and apply the migration

```bash
alembic revision --autogenerate -m "add inventory_items table"
alembic upgrade head
```

### 9.3 Verify

```bash
python -c "
from app.db import SessionLocal
from app.models import InventoryItem
db = SessionLocal()
print('inventory_items table exists, row count:', db.query(InventoryItem).count())
db.close()
"
```

### 9.4 Run evaluation

```bash
python -m ml_service.expiry.evaluate
```

### 9.5 Runtime estimate

| Operation | Latency |
|---|---|
| Level 1 exact query | < 2ms |
| Level 2 category query + median | < 5ms |
| Level 3 hard default (no DB call) | < 1ms |
| Full receipt (10–15 items) | < 50ms |

Stage 4 is the fastest stage in the pipeline — all computation is a DB read and simple arithmetic.

---

## Appendix: Troubleshooting

| Issue | Fix |
|---|---|
| `Level 1 always misses` for a seeded item | `canonical_name` case mismatch. The DB row may be `"strawberries"` but the item is `"Strawberries"`. The seed script and `ABBREVIATION_MAP` must agree exactly. Run `db.query(ShelfLifeReference.canonical_name).all()` to inspect actual values. |
| `Level 2 returns wrong shelf life` | Category median is being skewed by outliers (e.g. pantry items with 3650-day life). Inspect the category rows: `db.query(ShelfLifeReference).filter_by(category="Pantry", storage_context="Pantry").all()`. Consider using a weighted median or filtering to only the 25th–75th percentile. |
| `predicted_expiry` in the past | `purchase_date` passed as `None` or wrong year. Always validate `purchase_date` before calling `predict_expiry()`. Add a check: `assert purchase_date <= date.today()`. |
| `confidence` always 0.40 | Stage 3 normalization failed and fell through to hard default (`item.confidence = 0.0` shouldn't happen, but `normalization_pass = 0` would be the signal). Check that `normalize_entity()` never returns a `NormalizedItem` with `confidence = 0`. |
| `storage_context` wrong for an item | `CATEGORY_DEFAULT_STORAGE` assigned the wrong default. Override it in `storage_overrides` from `pipeline.py`, or let the user correct it in the frontend confirmation modal. |
| `InventoryItem` not in DB after migration | `alembic upgrade head` ran but `InventoryItem` wasn't imported in `migrations/env.py` via `Base.metadata`. Confirm `from app.models import Base` is present in `migrations/env.py` and that `InventoryItem` is defined in `app/models.py` before running autogenerate. |
| `alembic revision --autogenerate` detects no changes | Model was added to `app/models.py` after the migration env was set up but `Base` isn't picking it up. Make sure `InventoryItem` is in the same `Base = declarative_base()` as the other models, not a separate Base instance. |
