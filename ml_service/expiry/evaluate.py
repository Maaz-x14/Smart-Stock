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