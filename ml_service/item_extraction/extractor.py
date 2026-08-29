"""
Stage 2 component: orchestrator.

Wires unit_extractor.py -> brand_matcher.py -> food_classifier.py into
the Stage 2 contract: { unit, brand, is_food }.

Design (locked, see Item_Extraction.md / chat record):
  - food_classifier runs on remaining_text (post unit+brand strip), not
    the original item_name - chosen as the default because a bare
    product name is a cleaner classification signal than a full string
    still containing brand/unit noise. If real data shows this causes
    confusion (e.g. loses useful context), revert to item_name - not
    settled permanently, just the starting default.
  - EXCEPTION: if unit+brand extraction consumes the ENTIRE item_name
    (remaining_text == ""), fall back to the original item_name for
    classification. A fully-consumed string (e.g. "NESTLE 1L" -> unit
    stripped, brand stripped -> "") would otherwise feed an empty
    string to the classifier, which food_classifier.py's own empty-
    input guard turns into UNKNOWN by design - incorrectly, since the
    item was likely food (a brand+unit-only receipt line almost always
    is). This fallback prevents that false UNKNOWN.
  - is_food is bool | None on ItemFields - None means unknown (API
    failure or malformed LLM response), not "confidently not food".
    Matches food_classifier.py's real output shape. Downstream pipeline
    gating must treat None the same as False (skip Normalization/
    Expiry, surface to user) - see API_Spec.md §2 - but that equivalence
    is a pipeline-gating decision, not asserted by this dataclass.
"""

from dataclasses import dataclass
from time import perf_counter

from .unit_extractor import extract_unit
from .brand_matcher import extract_brand
from .food_classifier import classify_is_food


@dataclass
class ItemFields:
    unit: str | None
    brand: str | None
    is_food: bool | None  # None = unknown (see module docstring)


@dataclass
class ItemFieldsTiming:
    """Per-stage latency in ms - for #33-style latency measurement.
    Not part of the production contract, opt-in via extract_item_fields(debug=True)."""
    unit_ms: float
    brand_ms: float
    food_ms: float
    total_ms: float


def extract_item_fields(item_name: str, debug: bool = False) -> ItemFields | tuple[ItemFields, ItemFieldsTiming]:
    """
    Orchestrates Stage 2: unit extraction -> brand extraction ->
    is_food classification, in that order (each stage consumes the
    previous stage's remaining_text).

    debug=True: also prints the fallback trace (what text fed the food
    classifier and why) and the raw classify_is_food() result, and
    returns (ItemFields, ItemFieldsTiming) instead of just ItemFields.
    For diagnosing/measuring, not the production call shape - callers
    should not pass debug=True in the real pipeline.
    """
    t0 = perf_counter()
    unit_result = extract_unit(item_name)
    t1 = perf_counter()

    brand_result = extract_brand(unit_result.remaining_text)
    t2 = perf_counter()

    # Fallback: if unit+brand extraction consumed the whole string,
    # classify on the original item_name instead of an empty string.
    stripped_remaining = brand_result.remaining_text.strip()
    used_fallback = not stripped_remaining
    classification_input = stripped_remaining or item_name

    if debug:
        print(f"    [debug] remaining_text after brand strip: {brand_result.remaining_text!r}")
        print(f"    [debug] used_fallback={used_fallback}  classification_input={classification_input!r}")

    food_result = classify_is_food(classification_input)
    t3 = perf_counter()

    if debug:
        print(f"    [debug] food_result: outcome={food_result.outcome.value} "
              f"confidence={food_result.confidence} raw_response={food_result.raw_response!r} "
              f"reasoning={food_result.reasoning!r}")

    fields = ItemFields(
        unit=unit_result.unit,
        brand=brand_result.brand,
        is_food=food_result.is_food,
    )

    if not debug:
        return fields

    timing = ItemFieldsTiming(
        unit_ms=(t1 - t0) * 1000,
        brand_ms=(t2 - t1) * 1000,
        food_ms=(t3 - t2) * 1000,
        total_ms=(t3 - t0) * 1000,
    )
    return fields, timing


if __name__ == "__main__":
    # Smoke test - requires GROQ_API_KEY in environment (food_classifier
    # dependency). Includes a case exercising the empty-remaining_text
    # fallback: "NESTLE 1L" should have unit+brand fully strip it down
    # to "", triggering fallback to the original string for classification.
    #
    # debug=True prints per-item fallback trace + raw food_classifier
    # result, and reports per-stage latency (unit/brand/food/total) -
    # useful ad-hoc data point for #33, not a replacement for a real
    # end-to-end pipeline latency measurement.
    samples = [
        "ORG STRWBRY 1LB",
        "Supravit-M Tablet 10's",
        "NESTLE 1L",          # unit+brand likely consume the whole string
        "CHICKEN BREAST 500G",
        "Dettol Antiseptic 500ML",
    ]
    for s in samples:
        r, timing = extract_item_fields(s, debug=True)
        print(f"{s!r:30} -> unit={r.unit!r:8} brand={r.brand!r:10} is_food={r.is_food!r}")
        print(f"    [timing] unit={timing.unit_ms:.1f}ms brand={timing.brand_ms:.1f}ms "
              f"food={timing.food_ms:.1f}ms total={timing.total_ms:.1f}ms")
        print()
