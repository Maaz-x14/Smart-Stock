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
  - Batch entry point (Issue #46): extract_item_fields_batch() runs
    unit/brand extraction per item (local, no API - not the bottleneck),
    then classifies is_food for ALL items in the batch via
    classify_is_food_chunked() in chunks of 5, instead of one Groq call
    per item. Order is preserved item-for-item throughout. This is the
    call pipeline.py's process_receipt() uses for Stage 2 - the old
    per-item extract_item_fields() stays for standalone/debug use and
    is what extract_item_fields_batch() calls internally per item for
    the unit/brand portion.
"""

from dataclasses import dataclass
from time import perf_counter

from .unit_extractor import extract_unit
from .brand_matcher import extract_brand
from .food_classifier import classify_is_food, classify_is_food_chunked


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

    NOTE (Issue #46): this is the single-item path, ONE Groq call per
    call to this function. pipeline.py's process_receipt() now uses
    extract_item_fields_batch() instead, which batches the is_food
    call across the whole receipt. This function remains for
    standalone/debug/test use and is the per-item building block
    extract_item_fields_batch() uses for unit/brand extraction.
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


def extract_item_fields_batch(item_names: list[str]) -> list[ItemFields]:
    """
    Batch Stage 2 entry point (Issue #46). Returns ItemFields in the
    SAME ORDER as item_names, one per input.

    unit_extractor and brand_matcher run per-item (local, regex/lexicon
    based, not the bottleneck - no reason to batch them). Only the
    is_food classification step is batched: all items' classification
    inputs (post unit+brand strip, same fallback-to-item_name rule as
    extract_item_fields()) are collected and sent through
    classify_is_food_chunked() in chunks of 5, cutting Groq calls from
    N to ceil(N/5) per receipt.

    A batch-level classification failure only affects the items in
    that chunk (fail safe to UNKNOWN) - see classify_is_food_batch()
    docstring. unit/brand fields are unaffected either way, since
    they're computed independently before batching.
    """
    if not item_names:
        return []

    unit_results = [extract_unit(name) for name in item_names]
    brand_results = [extract_brand(u.remaining_text) for u in unit_results]

    classification_inputs = []
    for original_name, brand_result in zip(item_names, brand_results):
        stripped_remaining = brand_result.remaining_text.strip()
        classification_inputs.append(stripped_remaining or original_name)

    food_results = classify_is_food_chunked(classification_inputs, batch_size=5)

    return [
        ItemFields(
            unit=unit_result.unit,
            brand=brand_result.brand,
            is_food=food_result.is_food,
        )
        for unit_result, brand_result, food_result in zip(unit_results, brand_results, food_results)
    ]


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

    print("--- batch smoke test ---")
    batch_fields = extract_item_fields_batch(samples)
    for s, r in zip(samples, batch_fields):
        print(f"{s!r:30} -> unit={r.unit!r:8} brand={r.brand!r:10} is_food={r.is_food!r}")
