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
"""

import os
import re
import httpx
from sqlalchemy.orm import Session
from app.models import NormalizationCache

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "openai/gpt-oss-20b"
LLM_CONFIDENCE = 0.70


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

    Returns (canonical_name, confidence) or (None, 0.0) on failure.
    """
    cached = _cache_lookup(raw_token, db)
    if cached:
        return cached, LLM_CONFIDENCE

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set")

    prompt = (
        f"This is a line item from a grocery store receipt: '{raw_token}'. "
        f"What common food item does this refer to? Reply with just the canonical food name, nothing else."
    )

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

    content = response.json()["choices"][0]["message"]["content"]
    canonical_name = _clean_llm_response(content)

    if not canonical_name or len(canonical_name) < 2:
        print(f"[llm_fallback debug] raw_token={raw_token!r} raw_content={content!r} cleaned={canonical_name!r}")
        return None, 0.0

    _cache_store(raw_token, canonical_name, db)

    return canonical_name, LLM_CONFIDENCE
