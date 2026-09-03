"""
Stage 3 Pass 3: LLM fallback via Groq.

FIXED (Maaz, this session): GROQ_MODEL was "llama-3.1-8b-instant",
confirmed DEPRECATED by Groq on 2026-06-17 (discovered during #28
research - see Item_Extraction.md). This was dead on arrival. Updated
to openai/gpt-oss-20b, the same model validated for Stage 2's
food_classifier.py (#28) - reuses an already-tested, working choice
rather than picking a new untested one for Stage 3.

FIXED (Issue #46): added reasoning_effort="low". gpt-oss-20b spends
part of max_tokens on an internal reasoning trace before emitting
content; on at least one real query during testing this consumed the
ENTIRE 300-token budget, producing empty content (raw_content='')
that failed the len(canonical_name) < 2 check downstream - not a rate
limit, a genuine token-budget bug, same root cause as food_classifier.py's
version of this bug. reasoning_effort is a real, documented Groq API
param, not a guess.

Stays per-item (not batched) per Issue #46 scope decision - Pass 3
only fires on a cache + fuzzy-match miss, so call volume per receipt
is naturally low, and batching would trade accuracy risk for
throughput this stage doesn't need.

PROMPT REWRITE (Maaz, this session - #51): every wrong canonical
found in real-receipt testing (Tapal Tea Bags -> "Dried Lychee",
Everyday Instnt -> "Instant Noodles", Everyday Tea Whtnr -> "White
Tea") came from this pass, using the old one-line prompt that had no
abstention path ("Reply with just the canonical food name, nothing
else."). That forces a confident-sounding answer even when the model
has no real basis for one - the tokens weren't misread (OCR
confidence would be high on all three), they're just abbreviated/
brand-prefixed strings the model pattern-matched to the nearest
plausible-sounding food word instead of admitting it couldn't
resolve them.

New prompt is a single strict one-shot instruction (no follow-up
turns available at this call site) that: (1) explicitly allows and
requires "UNKNOWN" when the token can't be confidently resolved to a
specific known food, (2) gives worked examples of the exact failure
pattern seen in testing (brand name / ambiguous abbreviation ->
UNKNOWN, not a guessed food), (3) still resolves clearly-decodable
abbreviations normally so this doesn't regress good Pass 3 hits.

_clean_llm_response's UNKNOWN handling: "UNKNOWN" (any case) is
treated as a miss, same as empty/malformed content - returns
(None, 0.0), item falls through with no canonical name, same as any
other Pass 3 failure today. No new downstream state to handle.

NOT YET DONE (deferred per this session's plan): OCR-confidence-aware
spelling correction. Evidence gathered this session shows it wouldn't
address either real failure found (both tokens were almost certainly
read correctly by OCR - the failure is ambiguity/hallucination, not
misread characters) - see #51 discussion. Scoped as a possible
secondary signal, not implemented here.

DEBUGGING (Issue #46 follow-up, still open): empty-content failures
from the reasoning trace eating the token budget are a Stage 2
(food_classifier.py) problem in the latest testing, not confirmed
recurring here in Stage 3 - the debug logging below is left in place
unchanged in case it recurs.
"""

import os
import re
import httpx
from sqlalchemy.orm import Session
from app.models import NormalizationCache

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "openai/gpt-oss-20b"
LLM_CONFIDENCE = 0.70

PROMPT_TEMPLATE = """Receipt line item: '{raw_token}'

Resolve this to a specific canonical food name, ONLY if the words present clearly identify one specific food or drink.

Two brand situations are different - do not treat them the same:
- If the brand name itself IS the product (e.g. "Milo", "Nutella", "Sprite" - the brand name is what you'd ask for by name), resolve it to that product directly, even with abbreviations or unit codes attached (e.g. "Milo Drnk 180M1" -> Milo).
- If the brand is a general company/store name attached to an abbreviated, unclear product description (e.g. "Everyday Instnt", "Kimtiaz" with no product word at all), and you cannot tell which SPECIFIC product it refers to, output UNKNOWN rather than guessing.

Do not output UNKNOWN just because a brand name is present or the text is abbreviated - only output UNKNOWN when, after accounting for the brand, there is no clear specific product identifiable.

Output only the food name or the word UNKNOWN. No explanation.

Examples:
'CHKN BRST BNLS' -> Chicken Breast
'TAPAL TEA BAGS DNEDR ELCHI' -> Tea Bags
'Milo Drnk 180M1' -> Milo
'Everyday Instnt' -> UNKNOWN
'Kimtiaz' -> UNKNOWN

Now resolve: '{raw_token}'"""

def _cache_lookup(raw_token: str, db: Session) -> str | None:
    """Return cached canonical name for raw_token, or None if not cached."""
    entry = db.query(NormalizationCache).filter_by(raw_token=raw_token.upper()).first()
    if entry:
        entry.hit_count += 1
        db.commit()
        return entry.canonical_name
    return None


def _cache_store(raw_token: str, canonical_name: str, db: Session) -> None:
    """Store a new LLM result in the cache."""
    entry = NormalizationCache(
        raw_token=raw_token.upper(),
        canonical_name=canonical_name,
        source="llm",
    )
    db.add(entry)
    db.commit()


def _clean_llm_response(response_text: str) -> str:
    """
    Strip any stray punctuation, quotes, or explanation the LLM added despite instructions.
    Returns the first line, title-cased.
    """
    text = response_text.strip()
    text = text.split("\n")[0].strip()
    text = re.sub(r'^["\']|["\']$', '', text).strip()
    text = text.rstrip(".,;:")
    return text.title()


def pass3_llm(raw_token: str, db: Session) -> tuple[str | None, float]:
    """
    LLM fallback via Groq API.
    Checks cache first. On miss, calls Groq and caches the result.

    Returns (canonical_name, confidence) or (None, 0.0) on failure,
    or on an explicit model UNKNOWN (model declined to guess).
    """
    cached = _cache_lookup(raw_token, db)
    if cached:
        return cached, LLM_CONFIDENCE

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set")

    prompt = PROMPT_TEMPLATE.format(raw_token=raw_token)

    try:
        response = httpx.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.0,
                # Caps reasoning-trace token spend so it doesn't crowd out
                # actual content (Issue #46, fix #2) - confirmed root cause
                # of at least one raw_content='' failure in testing.
                "reasoning_effort": "low",
            },
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        print(f"[llm_fallback] Groq API error for token '{raw_token}': {e}")
        return None, 0.0

    response_json = response.json()
    message = response_json["choices"][0]["message"]
    content = message["content"]
    # Groq's gpt-oss models may expose a separate reasoning trace field,
    # not inline in content (same shape as food_classifier.py's provider
    # call). None if this response doesn't expose it.
    reasoning = message.get("reasoning")
    canonical_name = _clean_llm_response(content)

    if not canonical_name or len(canonical_name) < 2:
        # Debug logging (Issue #46 follow-up): capture reasoning length/
        # content, finish_reason, and usage to determine whether the
        # reasoning trace is still consuming the token budget despite
        # reasoning_effort="low".
        reasoning_len = len(reasoning) if reasoning else 0
        finish_reason = response_json["choices"][0].get("finish_reason")
        usage = response_json.get("usage", {})
        print(
            f"[llm_fallback debug] raw_token={raw_token!r} raw_content={content!r} "
            f"cleaned={canonical_name!r} reasoning_len={reasoning_len} "
            f"finish_reason={finish_reason!r} usage={usage!r}"
        )
        if reasoning:
            print(f"[llm_fallback debug] reasoning_text={reasoning!r}")
        return None, 0.0

    if canonical_name.strip().upper() == "UNKNOWN":
        # Model explicitly declined to guess - correct, safe outcome per
        # #51's new prompt. Not cached (nothing useful to cache), not
        # logged as an error - this is the fix working as intended.
        return None, 0.0

    _cache_store(raw_token, canonical_name, db)

    return canonical_name, LLM_CONFIDENCE
