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
        # No explicit timeout override - SDK default, per design decision.
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


if __name__ == "__main__":
    # Smoke test - requires GROQ_API_KEY in environment.
    samples = ["ORG STRWBRY", "Supravit-M Tablet 10's", "CHICKEN BREAST", "Dettol Antiseptic"]
    for s in samples:
        r = classify_is_food(s)
        print(f"{s!r:30} -> outcome={r.outcome.value:10} is_food={r.is_food!r:6} confidence={r.confidence}")
        print(f"    reasoning: {r.reasoning}")
