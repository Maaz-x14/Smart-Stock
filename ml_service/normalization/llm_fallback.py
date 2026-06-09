# ml_service/normalization/llm_fallback.py

import os
import re
import httpx
from sqlalchemy.orm import Session
from app.models import NormalizationCache

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"
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
    # Take only the first line in case the LLM added an explanation
    text = text.split("\n")[0].strip()
    # Remove surrounding quotes
    text = re.sub(r'^["\']|["\']$', '', text).strip()
    # Remove trailing punctuation
    text = text.rstrip(".,;:")
    return text.title()


def pass3_llm(raw_token: str, db: Session) -> tuple[str | None, float]:
    """
    LLM fallback via Groq API.
    Checks cache first. On miss, calls Groq and caches the result.

    Returns (canonical_name, confidence) or (None, 0.0) on failure.
    """
    # 1. Cache lookup
    cached = _cache_lookup(raw_token, db)
    if cached:
        return cached, LLM_CONFIDENCE

    # 2. Groq API call
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
                "max_tokens": 20,    # canonical name is never > 5 words
                "temperature": 0.0,  # deterministic — we want the same answer every time
            },
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        # Network or API error — fail gracefully, don't crash the pipeline
        print(f"[llm_fallback] Groq API error for token '{raw_token}': {e}")
        return None, 0.0

    content = response.json()["choices"][0]["message"]["content"]
    canonical_name = _clean_llm_response(content)

    if not canonical_name or len(canonical_name) < 2:
        return None, 0.0

    # 3. Store in cache
    _cache_store(raw_token, canonical_name, db)

    return canonical_name, LLM_CONFIDENCE