# ml_service/normalization/evaluate.py
# Usage: python -m ml_service.normalization.evaluate

from dataclasses import dataclass
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ml_service.normalization.normalizer import normalize_entity

# Ground truth: (food_tokens, expected_canonical_name)
# Drawn from a held-out set of real receipt lines not used to build ABBREVIATION_MAP
TEST_CASES = [
    # Pass 1 expected
    (["ORG", "STRWBRY"],          "Strawberries"),
    (["CHKN", "BRST", "BNLS"],    "Chicken Breast"),
    (["GRK", "YGRT", "PLN"],      "Greek Yogurt"),
    (["MLKWHL"],                   "Whole Milk"),
    (["DAHI"],                     "Dahi"),
    (["BASMATI"],                  "Basmati Rice"),
    (["MURG", "QEEMA"],            "Minced Chicken"),
    (["PALAK"],                    "Spinach"),
    (["LAHSUN"],                   "Garlic"),
    (["MILKPAK"],                  "UHT Milk"),
    # Pass 2 expected (slight misspellings / unseen variants)
    (["STRABERRY"],                "Strawberries"),
    (["CHICKIN", "BREAST"],        "Chicken Breast"),
    (["YOGHURT"],                  "Plain Yogurt"),
    (["WHOLE", "MILK"],            "Whole Milk"),
    (["CORIANDER", "LEAVES"],      "Coriander"),
    # Pass 3 expected (novel tokens)
    (["ANDAY"],                    "Eggs"),          # Urdu for eggs
    (["MACCHI"],                   "Fish"),        # Urdu for fish (approximate)
    (["SAFAID", "MIRCH"],          "White Pepper"),  # Urdu — may not be in map
]


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

    for food_tokens, expected in TEST_CASES:
        item = normalize_entity(food_tokens, None, None, db)

        if item is None:
            failures += 1
            print(f"  ❌ FAIL (unresolved): {' '.join(food_tokens)}")
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

        status = "✅" if match else "❌"
        print(f"  {status} Pass {item.normalization_pass} | {' '.join(food_tokens):30s} → {item.canonical_name:25s} (expected: {expected})")

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
    import os
    result = run_evaluation(os.environ["DATABASE_URL"])

    print(f"\n{'─'*50}")
    print(f"Total:               {result.total}")
    print(f"Pass 1 (map):        {result.pass1_hits}")
    print(f"Pass 2 (fuzzy):      {result.pass2_hits}")
    print(f"Pass 3 (LLM):        {result.pass3_hits}")
    print(f"Failures:            {result.failures}")
    print(f"Correct:             {result.correct}")
    print(f"Accuracy:            {result.accuracy:.1%}")
    print(f"Canonical match rate:{result.canonical_match_rate:.1%}  (target ≥ 80%)")
    print(f"LLM fallback rate:   {result.llm_fallback_rate:.1%}  (target ≤ 20%)")