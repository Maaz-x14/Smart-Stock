"""
Stage 3 orchestrator: normalize_entity().

INTERFACE REWRITTEN (Maaz, this session) - the old normalize_entity()
took (food_tokens: list[str], raw_quantity: str | None, raw_unit: str |
None) - a direct NER-tagger contract from the retired DistilBERT
architecture. It could never receive what the current pipeline
actually produces.

Current pipeline shape, end to end:
  Stage 1.7 Row Parser  -> {item_name, quantity, price, discount, total}
  Stage 2 extractor.py  -> ItemFields{unit, brand, is_food} (#29)
                            + the item_name string itself

normalize_entity() now takes:
  - item_name: str          the ORIGINAL raw text from Row Parser
                             (used for Pass 1/2/3 name resolution AND
                             for the frozen-signal category check,
                             which needs the raw "FRZ BROC"-style text,
                             not the already-stripped canonical name)
  - quantity: float         already a clean float from Row Parser -
                             nothing left for Stage 3 to parse
  - unit: str | None        already extracted by Stage 2's
                             unit_extractor.py - Stage 3 only
                             canonicalizes it (normalize_unit), does
                             not re-parse fused tokens
  - db: Session

is_food gating is NOT this function's job - extractor.py / the
pipeline orchestrator (#25) decides whether to call normalize_entity()
at all based on Stage 2's is_food result. This function assumes it is
only ever called for items already confirmed food.
"""

from dataclasses import dataclass
from sqlalchemy.orm import Session

from .preprocessor        import preprocess_token
from .abbreviation_map    import pass1_lookup
from .fuzzy_matcher       import pass2_fuzzy
from .llm_fallback        import pass3_llm
from .unit_normalizer     import normalize_unit
from .category_classifier import assign_category


@dataclass
class NormalizedItem:
    canonical_name:      str
    quantity:            float
    unit:                str | None
    category:            str
    normalization_pass:  int    # 1, 2, or 3 - which pass resolved the name
    confidence:          float  # 1.0 for Pass 1; score/100 for Pass 2; 0.70 for Pass 3


def normalize_entity(
    item_name: str,
    quantity:  float,
    unit:      str | None,
    db:        Session,
) -> NormalizedItem | None:
    """
    Normalize a single Stage 2 item into a canonical inventory item.

    Args:
        item_name: raw item text from Stage 1.7 Row Parser, e.g.
                   "ORG STRWBRY" (unit/price already stripped out by
                   Row Parser structurally - this is NOT the full
                   original receipt line, just the item-name column)
        quantity:  already-parsed quantity from Row Parser
        unit:      already-extracted unit from Stage 2's
                   unit_extractor.py (may be None)
        db:        active SQLAlchemy session

    Returns:
        NormalizedItem, or None if all three passes fail to resolve
        a canonical name.
    """
    cleaned = preprocess_token(item_name)

    if not cleaned:
        return None

    canonical_name = None
    confidence     = 0.0
    norm_pass      = 0

    # -- Pass 1: Abbreviation Map --
    result = pass1_lookup(cleaned)
    if result:
        canonical_name = result
        confidence     = 1.0
        norm_pass      = 1

    # -- Pass 2: Fuzzy Match --
    if canonical_name is None:
        result, score = pass2_fuzzy(cleaned, db)
        if result:
            canonical_name = result
            confidence     = score
            norm_pass      = 2

    # -- Pass 3: LLM Fallback (raw item_name, not preprocessed - more context) --
    if canonical_name is None:
        result, score = pass3_llm(item_name, db)
        if result:
            canonical_name = result
            confidence     = score
            norm_pass      = 3

    if canonical_name is None:
        return None

    # -- Unit normalization (canonicalize only, Stage 2 already extracted it) --
    normalized_unit = normalize_unit(unit)

    # -- Category assignment (raw_token passed for frozen-signal priority check) --
    category = assign_category(canonical_name, db, raw_token=item_name)

    return NormalizedItem(
        canonical_name=canonical_name,
        quantity=quantity,
        unit=normalized_unit,
        category=category,
        normalization_pass=norm_pass,
        confidence=confidence,
    )
