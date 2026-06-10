# Expiry_Training.md — Stage 4 Expiry Prediction Guide
## Smart-Stock: Stage 4 — Expiry Date Prediction

**Version:** 1.0 (Initial build — rule-based lookup + confidence scoring)
**Environment:** Local Python / FastAPI container (no GPU required)
**Method:** shelf_life_reference DB lookup → category fallback → confidence scoring

---

## ⚠️ Critical Design Reality Check

Before any code: Stage 4 is **not a training pipeline**. It is a production module that runs at inference time inside `ml_service/`. There is no GPU, no dataset, no Trainer, no checkpoint.

| Stage | What it does | Where it lives |
|---|---|---|
| Stage 1 (OCR) | Fine-tune TrOCR on receipt images | Kaggle notebook |
| Stage 2 (NER) | Fine-tune DistilBERT for token classification | Kaggle notebook |
| Stage 3 (Normalization) | Rule-based + fuzzy + LLM text normalization | `ml_service/normalization/` ✅ |
| **Stage 4 (Expiry)** | **Shelf-life lookup + confidence scoring → predicted expiry date** | **`ml_service/expiry/`** |

Stages 1 and 2 produce model artifacts saved to disk and reloaded at inference time. Stages 3 and 4 are pure logic — they run directly as Python inside the API server with no model weights. Stage 4 reads the `shelf_life_reference` table seeded in Stage 3 and computes a predicted expiry date from `purchase_date + shelf_life_days_avg`. The confidence score propagates forward from Stage 3's normalization quality.

---

## 1. Overview

Stage 4 takes a `NormalizedItem` from Stage 3 and produces a predicted expiry date with a confidence score. That result is written to `inventory_items` and drives the frontend dashboard colour-coding, alerts, and recipe suggestions.

```
Input  (from Stage 3):   NormalizedItem(canonical_name="Strawberries", category="Produce",
                                        quantity=1.0, unit="lb", normalization_pass=1, confidence=1.0)
                         + storage_context="Fridge"
                         + purchase_date=date(2025, 6, 1)

Output (to inventory):   ExpiryPrediction(predicted_expiry=date(2025, 6, 6),
                                          shelf_life_days=5,
                                          confidence=0.9025,
                                          source="exact_match",
                                          flag_for_review=False)
```

### Why rule-based, not an ML regression model

A trained regression model for shelf life would require a large labelled dataset of (food item, storage condition, actual expiry date) pairs — data that does not exist in any public dataset at the granularity needed. The `shelf_life_reference` table, built from food safety guidelines and seeded in Stage 3, encodes this knowledge directly.

The confidence scoring system gives the frontend everything it needs to surface low-confidence predictions for user review — the same outcome a regression model's prediction interval would provide, without requiring training data that does not exist.

### Prediction algorithm — two-level lookup

```
NormalizedItem + storage_context + purchase_date
      |
      v
┌─────────────────────────────────────────────────┐
│  Level 1: Exact Match                           │
│  Query shelf_life_reference WHERE               │
│    canonical_name = item.canonical_name AND     │
│    storage_context = storage_context            │
│  → shelf_life_days_avg                          │
│  → base_confidence = 0.95                       │
└──────────────────┬──────────────────────────────┘
                   │ no row found
                   v
┌─────────────────────────────────────────────────┐
│  Level 2: Category Fallback                     │
│  Query shelf_life_reference WHERE               │
│    category = item.category AND                 │
│    storage_context = storage_context            │
│  → median of shelf_life_days_avg for category   │
│  → base_confidence = 0.70                       │
└──────────────────┬──────────────────────────────┘
                   │ no rows found
                   v
┌─────────────────────────────────────────────────┐
│  Level 3: Hard Default                          │
│  shelf_life_days = 7 (one week)                 │
│  base_confidence = 0.40                         │
└──────────────────┬──────────────────────────────┘
                   │
                   v
   final_confidence = base_confidence × normalization_confidence
   predicted_expiry = purchase_date + timedelta(days=shelf_life_days)
```

---

## 2. Module Structure

All files listed below need to exist. `predictor.py` contains the core logic. `evaluate.py` is the test harness. `__init__.py` makes the folder a package.

```
ml_service/
└── expiry/
    ├── __init__.py      ← empty — makes expiry/ a package so evaluate.py can be run as a module
    ├── predictor.py     ← Full Stage 4 logic — predict_expiry() public entry point
    └── evaluate.py      ← Test harness — run with: python -m ml_service.expiry.evaluate
```

The `app/` package (`app/models.py`, `app/db.py`) was created in Stage 3 and is already present. No new files are needed there for Stage 4 — only a new model class (`InventoryItem`) is added to the existing `app/models.py`.

---

## 3. Storage Context

Storage context is the user-selected (or category-defaulted) storage condition for an item. It is the second dimension of every `shelf_life_reference` lookup — the same canonical name has a completely different shelf life depending on where it is stored.

| Storage Context | Meaning | Typical items |
|---|---|---|
| `Fridge` | Refrigerated (0–5°C) | Dairy, raw meat, opened sauces, most produce |
| `Freezer` | Frozen (−18°C) | Raw meat (bulk), frozen veg, ice cream |
| `Pantry` | Room temperature, dry | Rice, pasta, canned goods, onions, bananas |

**Default assignment by category** — used when the user has not specified a storage context:

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

The frontend confirmation modal lets users override this per item. The override is stored in `inventory_items.storage_context` and passed into `predict_expiry()` on subsequent calls.

---

## 4. Confidence Scoring

Confidence is a single float in [0, 1] representing how reliable the predicted expiry date is. It is the product of two independent signals:

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

**Combined confidence interpretation — what the frontend does with it:**

| Range | Meaning | Frontend behaviour |
|---|---|---|
| 0.90 – 1.00 | High — exact shelf-life data + reliable normalization | Show predicted date, no warning |
| 0.70 – 0.89 | Medium — some uncertainty in name or shelf life | Show predicted date, subtle indicator |
| 0.50 – 0.69 | Low — LLM normalization or category fallback used | Show predicted date + yellow warning |
| < 0.50 | Very low — hard default shelf life and/or LLM name | Flag for manual review in confirmation modal |

Items with `final_confidence < 0.60` have `flag_for_review = True` in the output and are surfaced in the frontend confirmation modal with a prompt for the user to verify or correct the predicted expiry date.

---

## 5. predictor.py

Full implementation. This is the only file with logic — everything else in `expiry/` is either empty (`__init__.py`) or the test harness (`evaluate.py`).

```python
# ml_service/expiry/predictor.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median

from sqlalchemy.orm import Session

from app.models import ShelfLifeReference
from ml_service.normalization.normalizer import NormalizedItem


# ── Constants ────────────────────────────────────────────────────────────────

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

BASE_CONFIDENCE_EXACT    = 0.95
BASE_CONFIDENCE_CATEGORY = 0.70
BASE_CONFIDENCE_DEFAULT  = 0.40

DEFAULT_SHELF_LIFE_DAYS  = 7   # hard fallback when shelf_life_reference has no data at all
REVIEW_THRESHOLD         = 0.60  # below this, flag_for_review = True


# ── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class ExpiryPrediction:
    predicted_expiry:  date
    shelf_life_days:   int
    confidence:        float
    source:            str   # "exact_match" | "category_fallback" | "hard_default"
    storage_context:   str
    flag_for_review:   bool


# ── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_storage_context(item: NormalizedItem, storage_context: str | None) -> str:
    """Use user-provided storage context if given; otherwise default by category."""
    if storage_context:
        return storage_context
    return CATEGORY_DEFAULT_STORAGE.get(item.category, "Fridge")


def _level1_exact(canonical_name: str, storage_context: str, db: Session) -> int | None:
    """
    Exact lookup: canonical_name + storage_context → shelf_life_days_avg.
    Returns None if no matching row exists.
    """
    ref = (
        db.query(ShelfLifeReference)
        .filter_by(canonical_name=canonical_name, storage_context=storage_context)
        .first()
    )
    return ref.shelf_life_days_avg if ref else None


def _level2_category(category: str, storage_context: str, db: Session) -> int | None:
    """
    Category fallback: all items in this category + storage_context → median shelf_life_days_avg.
    Returns None if no rows found for this category/storage combination.
    """
    rows = (
        db.query(ShelfLifeReference.shelf_life_days_avg)
        .filter_by(category=category, storage_context=storage_context)
        .all()
    )
    if not rows:
        return None
    return int(median(r[0] for r in rows))


# ── Public API ───────────────────────────────────────────────────────────────

def predict_expiry(
    item:            NormalizedItem,
    purchase_date:   date,
    db:              Session,
    storage_context: str | None = None,
) -> ExpiryPrediction:
    """
    Predict expiry date for a normalized inventory item.

    Args:
        item:            NormalizedItem produced by Stage 3 normalize_entity()
        purchase_date:   Date the item was purchased (from receipt or user input)
        db:              Active SQLAlchemy session
        storage_context: "Fridge", "Freezer", or "Pantry".
                         If None, defaults by item.category via CATEGORY_DEFAULT_STORAGE.

    Returns:
        ExpiryPrediction with predicted_expiry, shelf_life_days, confidence,
        source, storage_context, and flag_for_review.

    Examples:
        # Exact match: Strawberries + Fridge → 5 days → confidence 0.95
        item = NormalizedItem("Strawberries", 1.0, "lb", "Produce", 1, 1.0)
        result = predict_expiry(item, date(2025, 6, 1), db, storage_context="Fridge")
        # ExpiryPrediction(predicted_expiry=date(2025, 6, 6), shelf_life_days=5,
        #                  confidence=0.9025, source="exact_match", flag_for_review=False)

        # Category fallback: unknown item in Produce + Fridge → median produce days
        item = NormalizedItem("Exotic Mango Variety", 1.0, None, "Produce", 3, 0.70)
        result = predict_expiry(item, date(2025, 6, 1), db)
        # ExpiryPrediction(confidence=0.49, source="category_fallback", flag_for_review=True)
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

    # ── Final confidence + expiry ─────────────────────────────────────────────
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

## 6. evaluate.py

Full test harness. Mirrors the structure of `ml_service/normalization/evaluate.py` exactly.

```python
# ml_service/expiry/evaluate.py
# Usage: python -m ml_service.expiry.evaluate

from dataclasses import dataclass
from datetime import date

from app.db import SessionLocal
from ml_service.normalization.normalizer import NormalizedItem
from ml_service.expiry.predictor import predict_expiry, REVIEW_THRESHOLD

# ── Ground truth test cases ──────────────────────────────────────────────────
# Format: (NormalizedItem, storage_context_or_None, expected_shelf_life_days)
# All canonical_names must exist in shelf_life_reference (seeded in Stage 3).
# storage_context=None tests the category-default logic.
# "expected" is the shelf_life_days_avg from the seed data — exact match expected for Level 1 cases.

TEST_CASES = [
    # ── Level 1 exact match expected ─────────────────────────────────────────
    # (canonical_name matches shelf_life_reference exactly)
    (NormalizedItem("Strawberries",   1.0, "lb",  "Produce",   1, 1.00), "Fridge",   5),
    (NormalizedItem("Chicken Breast", 1.0, "lb",  "Meat",      1, 1.00), "Fridge",   2),
    (NormalizedItem("Whole Milk",     1.0, "gal", "Dairy",     1, 1.00), "Fridge",   7),
    (NormalizedItem("Basmati Rice",   1.0, "kg",  "Pantry",    1, 1.00), "Pantry", 730),
    (NormalizedItem("Butter",         1.0, "lb",  "Dairy",     1, 1.00), "Freezer", 90),
    (NormalizedItem("Eggs",           1.0, None,  "Dairy",     1, 1.00), "Fridge",  35),
    (NormalizedItem("Dahi",           1.0, None,  "Dairy",     1, 1.00), "Fridge",   5),
    (NormalizedItem("Mutton",         1.0, "kg",  "Meat",      1, 1.00), "Fridge",   3),
    (NormalizedItem("Paneer",         1.0, None,  "Dairy",     1, 1.00), "Fridge",   5),
    (NormalizedItem("Onions",         2.0, "lb",  "Produce",   1, 1.00), "Pantry",  30),
    (NormalizedItem("Frozen Peas",    1.0, "bag", "Frozen",    1, 1.00), "Freezer",270),
    (NormalizedItem("Coriander",      1.0, None,  "Produce",   1, 1.00), "Fridge",   7),

    # ── Level 1 with storage context defaulting ───────────────────────────────
    # storage_context=None → resolved from CATEGORY_DEFAULT_STORAGE
    (NormalizedItem("Spinach",        1.0, None,  "Produce",   1, 1.00), None,       5),
    (NormalizedItem("Greek Yogurt",   1.0, None,  "Dairy",     1, 1.00), None,      14),
    (NormalizedItem("Pasta",          1.0, None,  "Pantry",    1, 1.00), None,     730),
    (NormalizedItem("Ice Cream",      1.0, None,  "Frozen",    1, 1.00), None,      60),

    # ── Level 1 with Pass 2 normalization confidence (fuzzy match) ────────────
    (NormalizedItem("Strawberries",   1.0, "lb",  "Produce",   2, 0.91), "Fridge",   5),
    (NormalizedItem("Chicken Breast", 1.0, "lb",  "Meat",      2, 0.85), "Fridge",   2),

    # ── Level 1 with Pass 3 normalization confidence (LLM fallback) ──────────
    # confidence drops to 0.95 × 0.70 = 0.665 — above review threshold
    (NormalizedItem("Whole Milk",     1.0, "gal", "Dairy",     3, 0.70), "Fridge",   7),

    # ── Level 2 category fallback expected ───────────────────────────────────
    # canonical_name not in shelf_life_reference — falls through to category median
    (NormalizedItem("Dragon Fruit",   1.0, None,  "Produce",   2, 0.80), "Fridge",   5),
    (NormalizedItem("Artisan Cheese", 1.0, None,  "Dairy",     2, 0.80), "Fridge",  14),
]

TOLERANCE_DAYS = 2  # within ±2 days of expected = correct


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

    print(f"\n{'─' * 58}")
    print(f"Total cases:          {result.total}")
    print(f"Exact match (L1):     {result.exact_match_hits}")
    print(f"Category fallback (L2):{result.category_hits}")
    print(f"Hard default (L3):    {result.default_hits}")
    print(f"Correct (±{TOLERANCE_DAYS}d):        {result.correct}")
    print(f"Accuracy:             {result.accuracy:.1%}   (target ≥ 90%)")
    print(f"Avg confidence:       {result.avg_confidence:.3f}")
    print(f"Flagged for review:   {result.flagged_count}")
```

---

## 7. app/models.py — Add InventoryItem

`app/models.py` was created in Stage 3. It currently contains `ShelfLifeReference` and `NormalizationCache`. Add `InventoryItem` to the same file — do not create a new file.

```python
# Append to the bottom of app/models.py
# (ShelfLifeReference and NormalizationCache are already above this)

from sqlalchemy import Column, Integer, String, Date, Boolean, Numeric, ForeignKey, DateTime, func

class InventoryItem(Base):
    __tablename__ = "inventory_items"

    id                 = Column(Integer,      primary_key=True)
    user_id            = Column(Integer,      ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    canonical_name     = Column(String(100),  nullable=False)
    quantity           = Column(Numeric(8,2), nullable=False)
    unit               = Column(String(20))
    category           = Column(String(50),   nullable=False)
    storage_context    = Column(String(20),   nullable=False)
    purchase_date      = Column(Date,         nullable=False)
    predicted_expiry   = Column(Date,         nullable=False)
    shelf_life_days    = Column(Integer,      nullable=False)
    confidence         = Column(Numeric(5,4), nullable=False)
    expiry_source      = Column(String(30),   nullable=False)  # exact_match | category_fallback | hard_default
    flag_for_review    = Column(Boolean,      nullable=False,  default=False)
    normalization_pass = Column(Integer,      nullable=False)  # 1, 2, or 3 from Stage 3
    status             = Column(String(20),   nullable=False,  default="ACTIVE")  # ACTIVE | CONSUMED | WASTED
    created_at         = Column(DateTime,     nullable=False,  server_default=func.now())
    updated_at         = Column(DateTime,     nullable=False,  server_default=func.now(), onupdate=func.now())
```

> The SQL DDL equivalent of this table is documented in `DB_Schema.md` — no need to write raw SQL here.

---

## 8. Evaluation

### 8.1 Run the evaluation

```bash
# From Smart-Stock/ root with .venv active
python -m ml_service.expiry.evaluate
```

No extra setup needed — `app/`, `app/models.py`, `app/db.py`, and `shelf_life_reference` are all already in place from Stage 3.

### 8.2 Expected output

```
  ✅ [exact_ma] Strawberries              | storage=Fridge   | predicted=   5d | expected≈   5d | conf=0.9025
  ✅ [exact_ma] Chicken Breast            | storage=Fridge   | predicted=   2d | expected≈   2d | conf=0.9025
  ✅ [exact_ma] Whole Milk                | storage=Fridge   | predicted=   7d | expected≈   7d | conf=0.9025
  ✅ [exact_ma] Basmati Rice              | storage=Pantry   | predicted= 730d | expected≈ 730d | conf=0.9025
  ✅ [exact_ma] Butter                    | storage=Freezer  | predicted=  90d | expected≈  90d | conf=0.9025
  ...
  ✅ [category] Dragon Fruit              | storage=Fridge   | predicted=   5d | expected≈   5d | conf=0.5600 ⚠ review
  ✅ [category] Artisan Cheese            | storage=Fridge   | predicted=  14d | expected≈  14d | conf=0.5600 ⚠ review
  ──────────────────────────────────────────────────────────
  Total cases:           21
  Exact match (L1):      19
  Category fallback (L2): 2
  Hard default (L3):      0
  Correct (±2d):         21
  Accuracy:              100.0%  (target ≥ 90%)
  Avg confidence:        0.873
  Flagged for review:    2
```

### 8.3 Target metrics

| Metric | Definition | Target |
|---|---|---|
| Accuracy (±2 days) | % predictions within 2 days of seed reference avg | ≥ 90% |
| Exact match rate | % items resolved by Level 1 | ≥ 75% |
| Category fallback rate | % items falling to Level 2 | ≤ 20% |
| Hard default rate | % items hitting Level 3 | ≤ 5% |
| MAE (days) | Mean absolute error vs. `shelf_life_days_avg` in reference | ≤ 1.5 days |

### 8.4 If targets aren't met

| Symptom | Action |
|---|---|
| Hard default rate > 5% | Items hitting Level 3 are missing from `shelf_life_reference`. Print their `canonical_name`, add them to `SHELF_LIFE_DATA` in `shelf_life_seed.py`, re-run the seed. |
| Exact match rate < 75% | The canonical name from Stage 3 doesn't match the DB spelling exactly. Run `db.query(ShelfLifeReference.canonical_name).all()` and compare. Fix the seed data or the `ABBREVIATION_MAP` entry in Stage 3 so spellings agree. |
| MAE > 1.5 days | Category fallback median is off for a specific category. Inspect: `db.query(ShelfLifeReference).filter_by(category="X", storage_context="Y").all()`. Add more representative items to bring the median closer to reality. |
| Confidence too low across the board | Stage 3 is resolving too many items via Pass 3 (LLM). Expand `ABBREVIATION_MAP` in Stage 3 to push more items to Pass 1, raising `normalization_confidence` from 0.70 to 1.00. |

---

## 9. Setup & Requirements

### 9.1 No new packages

Stage 4 uses only SQLAlchemy — already installed in Stage 3. Nothing to `pip install`.

### 9.2 Files to create

| File | Action |
|---|---|
| `ml_service/expiry/__init__.py` | Create empty — makes `expiry/` a package |
| `ml_service/expiry/predictor.py` | Already exists from project setup — paste content from Section 5 |
| `ml_service/expiry/evaluate.py` | Create new — paste content from Section 6 |
| `app/models.py` | Already exists — append `InventoryItem` class from Section 7 |

```bash
touch ml_service/expiry/__init__.py
```

### 9.3 Alembic migration for inventory_items

After appending `InventoryItem` to `app/models.py`:

```bash
alembic revision --autogenerate -m "add inventory_items table"
alembic upgrade head
```

Verify:

```bash
python -c "
from app.db import SessionLocal
from app.models import InventoryItem
db = SessionLocal()
print('inventory_items table OK, row count:', db.query(InventoryItem).count())
db.close()
"
```

### 9.4 Project layout after Stage 4

```
Smart-Stock/
├── app/
│   ├── __init__.py
│   ├── models.py          ← ShelfLifeReference, NormalizationCache, InventoryItem
│   └── db.py
├── db/
│   └── seeds/
│       └── shelf_life_seed.py
├── migrations/
│   └── versions/
│       ├── c04767df306c_create_shelf_life_reference_and_.py   ← Stage 3 migration
│       └── xxxx_add_inventory_items_table.py                  ← Stage 4 migration
├── alembic.ini
├── ml_service/
│   ├── pipeline.py
│   ├── normalization/        ← Stage 3 ✅
│   │   ├── __init__.py
│   │   ├── normalizer.py
│   │   ├── preprocessor.py
│   │   ├── abbreviation_map.py
│   │   ├── fuzzy_matcher.py
│   │   ├── llm_fallback.py
│   │   ├── unit_normalizer.py
│   │   ├── category_classifier.py
│   │   └── evaluate.py
│   └── expiry/               ← Stage 4 ✅
│       ├── __init__.py
│       ├── predictor.py
│       └── evaluate.py
└── .env
```

### 9.5 Runtime estimate

| Operation | Latency |
|---|---|
| Level 1 exact query | < 2ms |
| Level 2 category query + median | < 5ms |
| Level 3 hard default (no DB call) | < 1ms |
| Full receipt (10–15 items, all Level 1) | < 30ms |

Stage 4 is the fastest stage in the pipeline — all computation is a DB read and simple arithmetic.

---

## Appendix: Troubleshooting

| Issue | Fix |
|---|---|
| `Level 1 always misses` for a seeded item | Case mismatch between Stage 3 output and the seed data. The DB row may be `"strawberries"` but the item is `"Strawberries"`. Run `db.query(ShelfLifeReference.canonical_name).all()` to inspect actual values. The `ABBREVIATION_MAP` values and seed data `canonical_name` values must be identical strings. |
| `Level 2 returns wrong shelf life` | Category median is skewed by outliers (e.g. pantry items with 3650-day life mixed with 5-day items). Inspect: `db.query(ShelfLifeReference).filter_by(category="Pantry", storage_context="Pantry").all()`. Add more representative items to the seed data to pull the median toward the realistic centre. |
| `predicted_expiry` is in the past | `purchase_date` is wrong — probably `None` or the wrong year. Add a guard: `assert purchase_date <= date.today()` before calling `predict_expiry()`. |
| `ModuleNotFoundError: No module named 'ml_service'` when running evaluate | Run from the project root (`Smart-Stock/`), not from inside `ml_service/`. The command must be `python -m ml_service.expiry.evaluate` from `Smart-Stock/`. |
| `ModuleNotFoundError: No module named 'app'` | Same as above — must run from project root. Or activate `.venv` first. |
| `InventoryItem` not appearing after `alembic upgrade head` | `InventoryItem` was added to `app/models.py` after the `migrations/env.py` was set up. Confirm `from app.models import Base` is in `env.py` and `InventoryItem` is declared under the same `Base = declarative_base()` instance as the other models. Then re-run `alembic revision --autogenerate`. |
| `alembic revision --autogenerate` detects no changes | Alembic is comparing to the already-applied migration state. If `InventoryItem` was in the DB before you added the class, Alembic sees no diff. Drop the table manually and re-run, or add a manual migration. |
| `storage_context` is wrong for a specific item | `CATEGORY_DEFAULT_STORAGE` assigned the wrong default. Pass the correct value explicitly as `storage_context` argument. The frontend confirmation modal also lets the user override this. |
| `flag_for_review` is True for a high-confidence item | Check that `item.confidence` from Stage 3 is not unexpectedly low. If Stage 3 resolved via Pass 3 (LLM), `normalization_confidence = 0.70` — multiply by `base_confidence 0.95` → `0.665`, which is above the 0.60 threshold but will become below if base drops to 0.70 (category fallback). This is correct behaviour. |
