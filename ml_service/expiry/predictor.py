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
    "Produce":   "Fridge",    # fresh fruits/vegetables
    "Dairy":     "Fridge",    # milk, yogurt, cheese
    "Meat":      "Fridge",    # raw meat/fish
    "Pantry":    "Pantry",    # dry staples
    "Frozen":    "Freezer",   # frozen goods
    "Beverages": "Pantry",    # unopened juice, soda, tea, coffee
    "Bakery":    "Pantry",    # bread, baked goods
    "Snacks":    "Pantry",    # chips, crackers, packaged snacks
    "Condiments & Sauces": "Fridge",   # opened bottles/jars (ketchup, mayo, salsa)
    "Prepared Meals":      "Fridge",   # cooked dishes, ready-to-eat
    "Breakfast Foods":     "Pantry",   # cereals, pancake mix, oats
    "Confectionery":       "Pantry",   # chocolates, candies, sweets
    "Baby Food":           "Pantry",   # unopened formula, jars, pouches
    "Other":               "Fridge",   # safe fallback
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