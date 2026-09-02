"""
Stage 2 component: is_food classification.

LLM-based binary gate: is this receipt item food/beverage for human
consumption? Replaces the old #18 scope (NOT_FOOD gate inside Stage 3's
LLM fallback) - this decision now belongs here, before Normalization
and Expiry ever run.

Design (locked, see Item_Extraction.md / chat record):
  - Model: Groq `openai/gpt-oss-20b`. Selected after testing 4 candidates
    (Groq, Gemini, 2x OpenRouter free) against a 15-item hand-labeled
    sample - only candidate with 15/15 valid JSON + 15/15 correct, no
    rate-limit or reliability issues. See Item_Extraction.md for full
    comparison table.
  - Prompt: strict JSON-only output, binary + confidence score, 3
    few-shot examples anchoring format. The prompt instructs no
    reasoning in the output - Groq's gpt-oss models still produce a
    reasoning trace, but on a SEPARATE `reasoning` field, not inline
    in `content`. This is expected behavior for this model family, not
    a prompt failure - content stays clean JSON regardless, and the
    reasoning trace is captured (not parsed) for debugging/audit.
  - Provider abstraction: FOOD_CLASSIFIER_PROVIDER / FOOD_CLASSIFIER_MODEL
    env vars select the backend. Only "groq" is wired currently - the
    other 3 tested candidates are NOT validated fallbacks (see doc),
    do not promote them without re-testing.
  - No retry: any API failure, timeout, or malformed response ->
    immediate is_food=None (unknown). Explicit choice, not a shortcut -
    keeps failure handling simple and fast; retries were considered and
    rejected for this stage.
  - No explicit request timeout override - uses SDK default.
  - unknown is surfaced to the user identically to is_food=False (per
    API_Spec.md §2's existing "detected but excluded" flow) - it is
    NOT silently treated as food, and NOT silently dropped. Reuses
    infra that already exists rather than inventing a new state.
  - Confidence threshold: NOT YET SET. classify_is_food() returns the
    raw (is_food, confidence) pair from the model; thresholding into
    unknown based on low confidence is deferred until real confidence
    distributions exist (Phase 4, undecided) - see extractor.py (#29)
    for where that decision will need to be applied once made.
  - reasoning_effort="low" (Issue #46): Groq's gpt-oss models spend part
    of max_tokens on an internal reasoning trace before emitting
    content. On some prompts this consumed the ENTIRE budget, producing
    empty content -> false UNKNOWN, independent of rate limiting.
    reasoning_effort is a real, documented Groq API param that caps how
    much of the budget goes to reasoning vs the final answer. Applied
    to both the single-item and batch call paths.
  - Batch classification (Issue #46): classify_is_food() alone does
    ONE Groq call per item. At 15-20 items/receipt this blew through
    Groq free-tier TPM (8000/min) regardless of concurrency settings -
    concurrency bounds parallel requests, not tokens/min. 
    classify_is_food_batch() / classify_is_food_chunked() batch N items
    into one call (default chunk size 5, see Issue #46 decision - large
    enough to meaningfully cut call count, small enough to keep
    hallucination/misordering risk low and index-tracking trivial).
    Fail-safe: any malformed/wrong-length/wrong-order batch response ->
    every item in that batch resolves to UNKNOWN. No partial trust
    within a batch - see classify_is_food_batch() docstring.
"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a binary food classifier for retail receipt items. Given an item name, output ONLY a JSON object in this exact shape:

{"is_food": true|false, "confidence": 0.0-1.0}

Rules:
- No explanation, no reasoning, no text before or after the JSON.
- "is_food" = true only for edible food/beverage items for human consumption.
- "is_food" = false for non-food items (medicine, cleaning products, toiletries, electronics, etc.).
- If genuinely ambiguous, still output your best guess with a lower confidence score - do not refuse, do not add caveats.

Examples:
Item: "ORG STRWBRY" -> {"is_food": true, "confidence": 0.95}
Item: "Supravit-M Tablet 10's" -> {"is_food": false, "confidence": 0.9}
Item: "XYZFOODS RICE" -> {"is_food": true, "confidence": 0.6}"""


# Batch system prompt (Issue #46). XML-tagged sections (role/instructions/
# examples/output_format) per Anthropic prompt-engineering guidance -
# keeps a strict, rigid contract for a multi-item response the model
# must not deviate from (exact array length, exact index mapping, no
# cross-item "theme" bias, no partial refusal).
BATCH_SYSTEM_PROMPT = """<role>
You are a binary food classifier for retail receipt items. You classify a LIST of items in one pass.
</role>

<instructions>
You will receive a numbered list of items, one per line, formatted as "N. item_name".
For EACH item, decide if it is food/beverage for human consumption.
Output ONLY a JSON array, with exactly one object per input item, in the SAME ORDER as the input.
Each object has this exact shape: {"index": N, "is_food": true|false, "confidence": 0.0-1.0}

Rules:
- The array length MUST exactly equal the number of input items. Do not skip, merge, or add items.
- "index" must match the input item's number exactly (1-based, matching the input list).
- "is_food" = true only for edible food/beverage items for human consumption.
- "is_food" = false for non-food items (medicine, cleaning products, toiletries, electronics, household goods, etc.).
- If an item is genuinely ambiguous, still output your best guess with a lower confidence score. Do not refuse, do not add caveats, do not skip the item.
- Treat every item independently. Earlier items in the list must NOT influence your judgment on later items - do not assume a "theme" for the batch (e.g. do not assume all items are groceries just because most are).
- No explanation, no reasoning, no markdown, no text before or after the JSON array.
</instructions>

<examples>
<example>
Input:
1. ORG STRWBRY
2. Supravit-M Tablet 10's
3. XYZFOODS RICE
4. Dettol Antiseptic
5. NESTLE 1L

Output:
[{"index": 1, "is_food": true, "confidence": 0.95}, {"index": 2, "is_food": false, "confidence": 0.9}, {"index": 3, "is_food": true, "confidence": 0.6}, {"index": 4, "is_food": false, "confidence": 0.95}, {"index": 5, "is_food": true, "confidence": 0.7}]
</example>

<example>
Input:
1. Colgate Total 100ml
2. FRESH CHKN BREAST
3. AA Battery 4pk

Output:
[{"index": 1, "is_food": false, "confidence": 0.95}, {"index": 2, "is_food": true, "confidence": 0.9}, {"index": 3, "is_food": false, "confidence": 0.95}]
</example>
</examples>

<output_format>
A single JSON array only. No prose, no code fences, no trailing commentary.
</output_format>"""


class ClassificationOutcome(Enum):
    FOOD = "food"
    NOT_FOOD = "not_food"
    UNKNOWN = "unknown"  # API failure OR malformed response - not a confidence tier (yet)


@dataclass
class FoodClassificationResult:
    outcome: ClassificationOutcome
    confidence: float | None  # None when outcome is UNKNOWN via failure
    raw_response: str | None = None  # kept for debugging/logging, not returned to API layer
    reasoning: str | None = None  # provider's reasoning trace, if exposed (e.g. Groq gpt-oss).
                                    # Not parsed/relied on for the decision - content is the
                                    # source of truth - but kept for debugging/audit.

    @property
    def is_food(self) -> bool | None:
        """Convenience accessor matching the Stage 2 contract shape
        ({is_food: bool | None}) - None means unknown, treated the same
        as False by downstream gating (see module docstring)."""
        if self.outcome == ClassificationOutcome.FOOD:
            return True
        if self.outcome == ClassificationOutcome.NOT_FOOD:
            return False
        return None


def _parse_response(raw: str) -> tuple[bool, float] | None:
    """Strict parse of expected {"is_food": bool, "confidence": float}
    shape. Returns None on ANY deviation - no partial trust, no
    guessing at malformed output. Mirrors the parser proven in the
    Phase 1 test harness (test_food_classifier_candidates.py)."""
    if raw is None:
        return None
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return None
        if "is_food" not in data or "confidence" not in data:
            return None
        if not isinstance(data["is_food"], bool):
            return None
        if not isinstance(data["confidence"], (int, float)):
            return None
        confidence = float(data["confidence"])
        if not (0.0 <= confidence <= 1.0):
            return None
        return data["is_food"], confidence
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _parse_batch_response(raw: str, expected_count: int) -> list[tuple[bool, float]] | None:
    """Strict parse of the batch JSON array. Returns None on ANY deviation
    from the expected shape/length/order - no partial trust. On None,
    caller must mark EVERY item in the batch as UNKNOWN (fail-safe,
    per Issue #46 scope decision - we cannot safely guess which index
    maps to which item if the array is malformed or the wrong length)."""
    if raw is None:
        return None
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()
        data = json.loads(cleaned)
        if not isinstance(data, list):
            return None
        if len(data) != expected_count:
            return None

        results: list[tuple[bool, float] | None] = [None] * expected_count
        for entry in data:
            if not isinstance(entry, dict):
                return None
            if "index" not in entry or "is_food" not in entry or "confidence" not in entry:
                return None
            idx = entry["index"]
            if not isinstance(idx, int) or not (1 <= idx <= expected_count):
                return None
            if not isinstance(entry["is_food"], bool):
                return None
            if not isinstance(entry["confidence"], (int, float)):
                return None
            confidence = float(entry["confidence"])
            if not (0.0 <= confidence <= 1.0):
                return None
            pos = idx - 1
            if results[pos] is not None:
                # duplicate index in response - malformed, fail safe
                return None
            results[pos] = (entry["is_food"], confidence)

        if any(r is None for r in results):
            # some index never showed up in the response - malformed, fail safe
            return None
        return results  # type: ignore[return-value]
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _call_groq(item_name: str, model: str) -> tuple[str, str | None]:
    """Single call, no retry (see module docstring for why). Any
    exception propagates to the caller, which converts it to UNKNOWN.
    Returns (content, reasoning) - Groq's gpt-oss models return
    reasoning on a separate field from content, not inline in the
    JSON. We don't parse it, but it's useful for debugging/logging
    (see module docstring)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Item: "{item_name}"'},
        ],
        temperature=0,
        # max_tokens covers reasoning + content combined for this model
        # (see Item_Extraction.md - gpt-oss-20b reasons on a separate
        # field, but it still consumes the token budget). 100 was too
        # low - a real truncation was observed on an ambiguous item
        # ("NESTLE 1L") where longer reasoning ate the budget before
        # content finished, producing malformed JSON -> false UNKNOWN.
        # Raised with headroom; not derived from a formal worst-case
        # token count, just observed failure + margin.
        max_tokens=300,
        # reasoning_effort caps how much of max_tokens goes to the
        # internal reasoning trace vs final content (Issue #46 fix #2) -
        # a documented Groq API param for gpt-oss models, not a guess.
        # Without this, reasoning was observed to consume the ENTIRE
        # budget on some prompts, leaving empty content -> false UNKNOWN,
        # independent of max_tokens size or rate limiting.
        reasoning_effort="low",
        # No explicit timeout override - SDK default, per design decision.
    )
    message = resp.choices[0].message
    reasoning = getattr(message, "reasoning", None)
    return message.content, reasoning


def _call_groq_batch(item_names: list[str], model: str) -> tuple[str, str | None]:
    """Batch call, no retry (same no-retry design as single-item path).
    max_tokens scales with batch size since output is one JSON object
    per item; reasoning_effort=low keeps the reasoning trace from
    eating the content budget (Issue #46, fix #2)."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )
    numbered = "\n".join(f"{i+1}. {name}" for i, name in enumerate(item_names))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": f"Input:\n{numbered}"},
        ],
        temperature=0,
        # ~80 tokens/item output budget (JSON object is short) + floor of
        # 300 for small batches. Not derived from a formal worst-case
        # count - observed single-item shape + margin, same empirical
        # approach as the single-item path's max_tokens=300.
        max_tokens=max(300, 80 * len(item_names)),
        reasoning_effort="low",
    )
    message = resp.choices[0].message
    reasoning = getattr(message, "reasoning", None)
    return message.content, reasoning


# Provider registry - each entry returns (content, reasoning). reasoning
# may be None for providers that don't expose it. Only "groq" is
# currently backed by real test results (see Item_Extraction.md). Do
# not add Gemini/OpenRouter here without re-running the Phase 1-style
# comparison first.
_PROVIDERS = {
    "groq": _call_groq,
}


def classify_is_food(item_name: str) -> FoodClassificationResult:
    """
    Classify whether a receipt item is food. Never raises - any failure
    (API error, timeout, malformed response) resolves to UNKNOWN, which
    callers must treat identically to NOT_FOOD for pipeline gating
    (surfaced to user, skips Normalization/Expiry) per API_Spec.md §2.

    Provider/model selected via FOOD_CLASSIFIER_PROVIDER /
    FOOD_CLASSIFIER_MODEL env vars. Defaults to the validated choice:
    Groq openai/gpt-oss-20b.
    """
    if not item_name or not item_name.strip():
        return FoodClassificationResult(outcome=ClassificationOutcome.UNKNOWN, confidence=None)

    provider_name = os.environ.get("FOOD_CLASSIFIER_PROVIDER", "groq")
    model = os.environ.get("FOOD_CLASSIFIER_MODEL", "openai/gpt-oss-20b")

    call_fn = _PROVIDERS.get(provider_name)
    if call_fn is None:
        # Unconfigured/unknown provider - fail safe to UNKNOWN, don't guess.
        return FoodClassificationResult(outcome=ClassificationOutcome.UNKNOWN, confidence=None)

    try:
        raw, reasoning = call_fn(item_name, model)
    except Exception:
        # No retry (design decision) - any failure -> UNKNOWN immediately.
        return FoodClassificationResult(outcome=ClassificationOutcome.UNKNOWN, confidence=None, raw_response=None)

    parsed = _parse_response(raw)
    if parsed is None:
        # Malformed output - same UNKNOWN path as an API failure.
        return FoodClassificationResult(outcome=ClassificationOutcome.UNKNOWN, confidence=None, raw_response=raw, reasoning=reasoning)

    is_food, confidence = parsed
    outcome = ClassificationOutcome.FOOD if is_food else ClassificationOutcome.NOT_FOOD
    return FoodClassificationResult(outcome=outcome, confidence=confidence, raw_response=raw, reasoning=reasoning)


def classify_is_food_batch(item_names: list[str]) -> list[FoodClassificationResult]:
    """
    Classify a batch of receipt items in ONE Groq call (Issue #46).
    Returns results in the SAME ORDER as item_names, one result per input.

    Fail-safe: if the batch call fails, times out, or the response is
    malformed/wrong-length/wrong-order in any way, EVERY item in the
    batch resolves to UNKNOWN. We do not attempt partial recovery -
    per Issue #46 scope decision, mixing trusted and guessed results
    within one batch is worse than failing the whole batch safe.

    Empty strings in item_names resolve to UNKNOWN individually and are
    still sent to the model as "(blank)" placeholders to preserve index
    alignment (never filtered out - array length must match input length).

    Only backed by the "groq" provider currently (same restriction as
    the single-item path) - see module docstring.
    """
    if not item_names:
        return []

    provider_name = os.environ.get("FOOD_CLASSIFIER_PROVIDER", "groq")
    model = os.environ.get("FOOD_CLASSIFIER_MODEL", "openai/gpt-oss-20b")

    if provider_name != "groq":
        # Batch path only implemented for groq currently - fail safe.
        return [FoodClassificationResult(outcome=ClassificationOutcome.UNKNOWN, confidence=None) for _ in item_names]

    safe_names = [n.strip() if n and n.strip() else "(blank)" for n in item_names]

    try:
        raw, reasoning = _call_groq_batch(safe_names, model)
    except Exception:
        return [FoodClassificationResult(outcome=ClassificationOutcome.UNKNOWN, confidence=None) for _ in item_names]

    parsed = _parse_batch_response(raw, len(item_names))
    if parsed is None:
        return [
            FoodClassificationResult(outcome=ClassificationOutcome.UNKNOWN, confidence=None, raw_response=raw, reasoning=reasoning)
            for _ in item_names
        ]

    results = []
    for original_name, (is_food, confidence) in zip(item_names, parsed):
        if not original_name or not original_name.strip():
            results.append(FoodClassificationResult(outcome=ClassificationOutcome.UNKNOWN, confidence=None))
            continue
        outcome = ClassificationOutcome.FOOD if is_food else ClassificationOutcome.NOT_FOOD
        results.append(FoodClassificationResult(outcome=outcome, confidence=confidence, raw_response=raw, reasoning=reasoning))
    return results


def classify_is_food_chunked(item_names: list[str], batch_size: int = 5) -> list[FoodClassificationResult]:
    """
    Chunk item_names into batches of `batch_size` (default 5, per Issue
    #46 decision - cuts call count ~5x while keeping index-tracking
    trivial and hallucination/misordering risk low) and call
    classify_is_food_batch per chunk. This is the entry point
    extractor.py's extract_item_fields_batch() calls for Stage 2
    is_food classification.

    A failure in one chunk (-> all UNKNOWN for that chunk) does NOT
    affect other chunks - chunks are independent Groq calls.
    """
    if not item_names:
        return []
    results: list[FoodClassificationResult] = []
    for start in range(0, len(item_names), batch_size):
        chunk = item_names[start:start + batch_size]
        results.extend(classify_is_food_batch(chunk))
    return results


if __name__ == "__main__":
    # Smoke test - requires GROQ_API_KEY in environment.
    samples = ["ORG STRWBRY", "Supravit-M Tablet 10's", "CHICKEN BREAST", "Dettol Antiseptic"]
    for s in samples:
        r = classify_is_food(s)
        print(f"{s!r:30} -> outcome={r.outcome.value:10} is_food={r.is_food!r:6} confidence={r.confidence}")
        print(f"    reasoning: {r.reasoning}")

    print("\n--- batch smoke test ---")
    batch_results = classify_is_food_chunked(samples, batch_size=5)
    for s, r in zip(samples, batch_results):
        print(f"{s!r:30} -> outcome={r.outcome.value:10} is_food={r.is_food!r:6} confidence={r.confidence}")
