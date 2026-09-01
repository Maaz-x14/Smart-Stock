"""
Stage 3 evaluation harness.
Usage: python -m ml_service.normalization.evaluate

REWRITTEN:
  - Test cases moved to data/eval_test_cases.json (was inline TEST_CASES).
  - Calls now use the rewritten normalize_entity(item_name, quantity,
    unit, db) signature (see normalizer.py) instead of the old
    (food_tokens: list[str], raw_quantity, raw_unit, db) NER-era call -
    this script was previously calling a signature that doesn't exist
    anymore and would have failed outright.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ml_service.normalization.normalizer import normalize_entity

_DATA_PATH = Path(__file__).parent / "data" / "eval_test_cases.json"


def _load_test_cases() -> list[tuple[str, str]]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(c["item_name"], c["expected"]) for c in data["cases"]]


TEST_CASES = _load_test_cases()


@dataclass
class EvalResult:
    total:        int
    pass1_hits:   int
    pass2_hits:   int
    pass3_hits:   int
    failures:     int
    correct:      int
    accuracy:     float
    llm_fallback_rate: float
    canonical_match_rate: float


def run_evaluation(db_url: str) -> EvalResult:
    engine  = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    db      = Session()

    pass1 = pass2 = pass3 = failures = correct = 0

    for item_name, expected in TEST_CASES:
        # quantity=1.0, unit=None - these test cases exercise name
        # resolution only; quantity/unit pass through unchanged from
        # Stage 2, nothing to validate at this layer.
        item = normalize_entity(item_name, 1.0, None, db)

        if item is None:
            failures += 1
            print(f"  FAIL (unresolved): {item_name}")
            continue

        match = item.canonical_name.lower() == expected.lower()
        if match:
            correct += 1

        if item.normalization_pass == 1:
            pass1 += 1
        elif item.normalization_pass == 2:
            pass2 += 1
        elif item.normalization_pass == 3:
            pass3 += 1

        status = "PASS" if match else "FAIL"
        print(f"  {status} Pass {item.normalization_pass} | {item_name:30s} -> {item.canonical_name:25s} (expected: {expected}) | category={item.category}")

    total = len(TEST_CASES)
    return EvalResult(
        total=total,
        pass1_hits=pass1,
        pass2_hits=pass2,
        pass3_hits=pass3,
        failures=failures,
        correct=correct,
        accuracy=round(correct / total, 3),
        llm_fallback_rate=round(pass3 / total, 3),
        canonical_match_rate=round((pass1 + pass2) / total, 3),
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    import os

    result = run_evaluation(os.environ["DATABASE_URL"])

    print(f"\n{'-'*50}")
    print(f"Total:               {result.total}")
    print(f"Pass 1 (map):        {result.pass1_hits}")
    print(f"Pass 2 (fuzzy):      {result.pass2_hits}")
    print(f"Pass 3 (LLM):        {result.pass3_hits}")
    print(f"Failures:            {result.failures}")
    print(f"Correct:             {result.correct}")
    print(f"Accuracy:            {result.accuracy:.1%}")
    print(f"Canonical match rate:{result.canonical_match_rate:.1%}  (target >= 80%)")
    print(f"LLM fallback rate:   {result.llm_fallback_rate:.1%}  (target <= 20%)")
