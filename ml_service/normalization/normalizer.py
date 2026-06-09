# ml_service/normalization/normalizer.py

from dataclasses import dataclass
from sqlalchemy.orm import Session

from .preprocessor       import preprocess_token
from .abbreviation_map   import pass1_lookup
from .fuzzy_matcher      import pass2_fuzzy
from .llm_fallback       import pass3_llm
from .unit_normalizer    import parse_quantity_unit, ParsedQuantity
from .category_classifier import assign_category


@dataclass
class NormalizedItem:
    canonical_name:    str
    quantity:          float
    unit:              str | None
    category:          str
    normalization_pass: int    # 1, 2, or 3 — which pass resolved the name
    confidence:        float   # 1.0 for Pass 1; score/100 for Pass 2; 0.70 for Pass 3


def normalize_entity(
    food_tokens:   list[str],
    raw_quantity:  str | None,
    raw_unit:      str | None,
    db:            Session,
) -> NormalizedItem | None:
    """
    Normalize a single NER entity into a canonical inventory item.

    Args:
        food_tokens:  word list from NER, e.g. ["ORG", "STRWBRY"]
        raw_quantity: quantity string from NER, e.g. "1"
        raw_unit:     unit string from NER, e.g. "LB"
        db:           active SQLAlchemy session

    Returns:
        NormalizedItem with canonical_name, quantity, unit, category, confidence.
        Returns None if all three passes fail to produce a result.
    """
    # Join tokens into a single string for normalization
    raw_token = " ".join(food_tokens)

    # Clean the token before any pass
    cleaned = preprocess_token(raw_token)

    if not cleaned:
        return None

    canonical_name = None
    confidence     = 0.0
    norm_pass      = 0

    # ── Pass 1: Abbreviation Map ──────────────────────────────────────────────
    result = pass1_lookup(cleaned)
    if result:
        canonical_name = result
        confidence     = 1.0
        norm_pass      = 1

    # ── Pass 2: Fuzzy Match ───────────────────────────────────────────────────
    if canonical_name is None:
        result, score = pass2_fuzzy(cleaned, db)
        if result:
            canonical_name = result
            confidence     = score
            norm_pass      = 2

    # ── Pass 3: LLM Fallback ──────────────────────────────────────────────────
    if canonical_name is None:
        result, score = pass3_llm(raw_token, db)   # pass raw (uncleaned) to LLM — more context
        if result:
            canonical_name = result
            confidence     = score
            norm_pass      = 3

    if canonical_name is None:
        return None

    # ── Unit & Quantity ───────────────────────────────────────────────────────
    parsed: ParsedQuantity = parse_quantity_unit(raw_quantity, raw_unit)

    # ── Category Assignment ───────────────────────────────────────────────────
    category = assign_category(canonical_name, db)

    return NormalizedItem(
        canonical_name=canonical_name,
        quantity=parsed.quantity,
        unit=parsed.unit,
        category=category,
        normalization_pass=norm_pass,
        confidence=confidence,
    )