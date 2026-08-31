"""
Stage 4 evaluation harness.
Usage: python -m ml_service.expiry.evaluate

REWRITTEN (Maaz, this session):
  - Test cases moved to data/eval_test_cases.json (was inline TEST_CASES
    of hand-constructed NormalizedItem objects).
  - CRITICAL FIX: previously this harness hand-built NormalizedItem(...)
    directly, bypassing Stage 3's normalize_entity() entirely. It looked
    like an integration test but only ever exercised Stage 4 in
    isolation against fabricated inputs - it would pass even if Stage 3
    was completely broken. Now calls the real normalize_entity() first
    (using the rewritten interface, item_name/quantity/unit/db) and
    feeds ITS output into predict_expiry() - this is now a genuine
    Stage 3 + Stage 4 integration test.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from datetime import date

from app.db import SessionLocal
from ml_service.normalization.normalizer import normalize_entity
from ml_service.expiry.predictor import predict_expiry, REVIEW_THRESHOLD

_DATA_PATH = Path(__file__).parent / "data" / "expiry_eval_test_cases.json"

TOLERANCE_DAYS = 2  # within +-2 days of expected = correct


def _load_test_cases() -> list[tuple[str, str | None, int]]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(c["item_name"], c["storage_context"], c["expected_days"]) for c in data["cases"]]


TEST_CASES = _load_test_cases()


@dataclass
class EvalResult:
    total:              int
    normalization_fails: int  # NEW: cases where Stage 3 itself failed to resolve
    exact_match_hits:   int
    category_hits:      int
    default_hits:       int
    correct:            int
    accuracy:           float
    avg_confidence:     float
    flagged_count:      int


def run_evaluation() -> EvalResult:
    db = SessionLocal()

    exact_hits = cat_hits = default_hits = correct = normalization_fails = 0
    confidences = []
    flagged = 0

    purchase_date = date(2025, 6, 1)

    for item_name, storage_ctx, expected_days in TEST_CASES:
        # Real Stage 3 call - not a fabricated NormalizedItem.
        item = normalize_entity(item_name, 1.0, None, db)

        if item is None:
            normalization_fails += 1
            print(f"  FAIL (Stage 3 could not resolve): {item_name}")
            continue

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

        status = "PASS" if within_tolerance else "FAIL"
        print(
            f"  {status} [{result.source[:8]:8s}] "
            f"{item.canonical_name:25s} | "
            f"storage={result.storage_context:7s} | "
            f"predicted={result.shelf_life_days:4d}d | "
            f"expected~{expected_days:4d}d | "
            f"conf={result.confidence:.3f}"
            + (" [REVIEW]" if result.flag_for_review else "")
        )

    db.close()

    total = len(TEST_CASES)
    scored = total - normalization_fails
    return EvalResult(
        total=total,
        normalization_fails=normalization_fails,
        exact_match_hits=exact_hits,
        category_hits=cat_hits,
        default_hits=default_hits,
        correct=correct,
        accuracy=round(correct / scored, 3) if scored else 0.0,
        avg_confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        flagged_count=flagged,
    )


if __name__ == "__main__":
    result = run_evaluation()

    print(f"\n{'-' * 58}")
    print(f"Total cases:            {result.total}")
    print(f"Stage 3 resolution fails:{result.normalization_fails}")
    print(f"Exact match (L1):       {result.exact_match_hits}")
    print(f"Category fallback (L2): {result.category_hits}")
    print(f"Hard default (L3):      {result.default_hits}")
    print(f"Correct (+-{TOLERANCE_DAYS}d):          {result.correct}")
    print(f"Accuracy:               {result.accuracy:.1%}   (target >= 90%, of resolved cases)")
    print(f"Avg confidence:         {result.avg_confidence:.3f}")
    print(f"Flagged for review:     {result.flagged_count}")
