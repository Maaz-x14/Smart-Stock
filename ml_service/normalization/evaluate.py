# ml_service/normalization/evaluate.py
# Usage: python -m ml_service.normalization.evaluate

from dataclasses import dataclass
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ml_service.normalization.normalizer import normalize_entity

# Ground truth: (food_tokens, expected_canonical_name)
# Drawn from a held-out set of real receipt lines not used to build ABBREVIATION_MAP
TEST_CASES = [
    # Pass 1 expected (direct abbreviations / known variants)
    (["AMROOD"],                 "Guavas"),          # Urdu
    (["ANAR"],                   "Pomegranates"),    # Urdu
    (["CHAUNSA"],                "Mangoes"),         # Pakistani mango variety
    (["SEEKH", "KBAB"],          "Seekh Kebab"),     # Common kebab spelling
    (["MURG", "QEEMA"],          "Minced Chicken"),  # Already tested, keep for consistency
    (["PALAK"],                  "Spinach"),         # Urdu
    (["SHALGAM"],                "Turnips"),         # Urdu
    (["MOOLI"],                  "Radishes"),        # Urdu
    (["LAUKI"],                  "Bottle Gourd"),    # Urdu
    (["KARELA"],                 "Bitter Gourd"),    # Urdu

    # Pass 2 expected (slight misspellings / unseen but close)
    (["STRWBERRY"],              "Strawberries"),    # OCR slip
    (["CHKN", "THIGH"],          "Chicken Thighs"),  # Slight variant
    (["YOUGURT"],                "Yogurt"),          # Common misspelling
    (["WHOLE", "MLK"],           "Whole Milk"),      # Token split
    (["CORIANDER", "LEAF"],      "Coriander"),       # Variation
    (["BASMATI", "CHAWAL"],      "Basmati Rice"),    # Urdu + English hybrid
    (["PANEER"],                 "Paneer"),          # Direct but not in Pass 1 list
    (["MANGO", "LASSI"],         "Lassi"),           # Drink variant

    # Pass 3 expected (novel/local tokens, Pakistani style)
    (["ANDA"],                   "Eggs"),            # Urdu singular
    (["ANDA", "DOZEN"],          "Eggs"),            # Pack style
    (["MACCHI"],                 "Fish"),            # Urdu for fish
    (["SAFAID", "MIRCH"],        "White Pepper"),    # Urdu phrase
    (["DUDH"],                   "Milk"),            # Urdu word
    (["CHAWAL"],                 "Rice"),            # Urdu
    (["ATTA"],                   "Atta Flour"),      # Urdu/Hindi
    (["MAIDA"],                  "Maida"),           # South Asian flour
    (["SUJI"],                   "Semolina"),        # Urdu
    (["DALDA"],                  "Cooking Oil"),     # Pakistani brand
    (["DESI", "GHEE"],           "Ghee"),            # Local term
    (["CHAI"],                   "Tea Bags"),        # Local beverage
    (["TAPAL"],                  "Tea Bags"),        # Pakistani brand
    (["BROOKE", "BOND"],         "Tea Bags"),        # Brand
    (["NAN"],                    "Naan"),            # Common spelling
    (["CHAPATI"],                "Roti"),            # Synonym
    (["PARATHA"],                "Paratha"),         # Synonym
    (["BIRYANI"],                "Biryani"),  # Cooked dish
    (["HALWA"],                  "Halva"),   # Sweet dish
    (["PAKORA"],                 "Fritter"),  # Fried snack
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