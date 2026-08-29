"""
Stage 2 component: brand extraction.

Hybrid resolution: curated lexicon (external data file) + conservative
token-level detection kept OUT of the production path (see decision
below). Deterministic, confidence-gated - explicitly NOT a semantic
brand classifier.

Design:
  1. Tokenize item_name (remaining_text from unit_extractor).
  2. Normalize punctuation/case for matching ("&" <-> "and", strip
     apostrophes/hyphens variance) so "Head & Shoulders" / "HEAD AND
     SHOULDERS" / "HEAD SHOULDERS" resolve identically.
  3. LEXICON MATCH FIRST, on the full (un-stripped) token set.
     Match -> high-confidence brand, matched token(s) stripped from
     remaining_text.
  4. Only if NO lexicon match: strip descriptor stopwords (this is
     now dead-end cleanup, not a gate before matching) and return
     brand=None.
  5. Lexicon growth is MANUAL, OFFLINE: edit data/brand_lexicon.json
     after human review of real receipt data. This module never
     auto-grows itself.

BUG FIXED THIS SESSION: original order was
"strip descriptors -> lexicon match", which meant a real brand like
"Fresh St" (Al-Fatah) or "Fresh Choice" would have "Fresh" stripped
as a descriptor before the lexicon ever saw it, corrupting/losing
brand matches whose name overlaps a descriptor word. Order flipped to
lexicon-match-first; stripping only happens on the no-match path.

LEXICON SOURCE: data/brand_lexicon.json (Maaz's curated list from
Imtiaz/Al-Fatah store browsing + web search, Aug 2026). NOT a complete
Pakistan brand database - a growing seed list. See file's _meta field.
Unverified against real Smart-Stock receipt OCR output.

DESCRIPTOR STOPWORD LIST IS ALSO UNVERIFIED - same caveat. Food nouns
(chicken, milk, rice, etc.) are deliberately EXCLUDED from this list -
those are the product name, not a descriptor, and belong to Stage 3
Normalization's concern, not brand extraction's. Including them here
risked both false-stripping legitimate names and colliding with brand
substrings (e.g. "milk" inside "Milkpak").
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

_LEXICON_PATH = Path(__file__).parent / "data" / "brand_lexicon.json"
_FUZZY_MATCH_THRESHOLD = 88  # conservative; tune against real data

# --- Descriptor stopwords (UNVERIFIED, see module docstring) -------------
# Only used on the NO-MATCH path now, for remaining_text cleanup - never
# as a pre-filter before lexicon matching. Food nouns deliberately
# excluded (see docstring).
_DESCRIPTOR_STOPWORDS = {
    # freshness / origin
    "fresh", "freshly", "organic", "local", "imported",
    "frozen", "chilled", "homemade",
    # quality / marketing
    "premium", "classic", "original", "special", "select",
    "choice", "best", "value", "economy", "deluxe",
    "natural", "pure", "real", "authentic",
    # size / quantity descriptors
    "large", "small", "medium", "jumbo", "mini",
    "family", "regular", "extra",
    # dietary / formulation
    "diet", "light", "lite", "low", "fat", "free",
    "salted", "unsalted",
    # packaging
    "pack", "packet", "box", "bag", "pouch", "bottle",
    "tin", "can", "jar", "sachet",
    # product qualifiers
    "mix", "mixed", "assorted", "combo", "variety",
    "flavoured", "flavored", "plain", "hot", "mild",
    "spicy", "sweet",
}

_TOKEN_PATTERN = re.compile(r"[A-Za-z&']+")


def _load_lexicon() -> dict[str, str]:
    """Load brand_lexicon.json -> flat {surface_form: canonical_brand}."""
    with open(_LEXICON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    flat: dict[str, str] = {}
    for canonical, surfaces in data["brands"].items():
        for surface in surfaces:
            flat[_normalize(surface)] = canonical
    return flat


def _normalize(text: str) -> str:
    """Normalize punctuation/case so equivalent brand strings match.
    '&' <-> 'and', apostrophes/hyphens stripped, lowercased, collapsed
    whitespace. Applied identically to lexicon surface forms and input
    tokens so both sides compare on equal footing."""
    t = text.lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[''\-]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_SURFACE_TO_CANONICAL = _load_lexicon()
_ALL_SURFACE_FORMS = list(_SURFACE_TO_CANONICAL.keys())


@dataclass
class BrandExtractionResult:
    brand: str | None
    remaining_text: str


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def _lexicon_match(tokens: list[str]) -> tuple[str | None, str | None]:
    """Fuzzy match token/bigram/trigram windows against the lexicon,
    on the FULL token set (no pre-stripping). Returns
    (canonical_brand, matched_original_substring) or (None, None)."""
    if not tokens:
        return None, None

    windows = list(tokens)
    for n in (2, 3):
        windows += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    best_match = None
    best_score = 0
    best_window = None
    for window in windows:
        result = process.extractOne(
            _normalize(window), _ALL_SURFACE_FORMS, scorer=fuzz.ratio
        )
        if result and result[1] >= _FUZZY_MATCH_THRESHOLD and result[1] > best_score:
            best_match, best_score, best_window = result[0], result[1], window

    if best_match:
        return _SURFACE_TO_CANONICAL[best_match], best_window
    return None, None


def extract_brand(remaining_text: str) -> BrandExtractionResult:
    """
    Extract brand from remaining_text (already stripped of unit+qty by
    unit_extractor). Lexicon-match-first, never guesses: no match ->
    brand=None. See module docstring for the match-order bug fix and
    why descriptor stripping only applies on the no-match path.
    """
    if not remaining_text:
        return BrandExtractionResult(brand=None, remaining_text=remaining_text or "")

    tokens = _tokenize(remaining_text)

    canonical, matched_window = _lexicon_match(tokens)
    if canonical:
        remaining = re.sub(
            re.escape(matched_window), "", remaining_text, flags=re.IGNORECASE
        ).strip()
        remaining = re.sub(r"\s{2,}", " ", remaining)
        return BrandExtractionResult(brand=canonical, remaining_text=remaining)

    # No match: descriptor stripping is cleanup only, not a re-attempt gate.
    cleaned_tokens = [t for t in tokens if t.lower() not in _DESCRIPTOR_STOPWORDS]
    return BrandExtractionResult(brand=None, remaining_text=" ".join(cleaned_tokens))


if __name__ == "__main__":
    samples = [
        "NESTLE MILK",
        "ORG STRWBRY",
        "OLPER'S MILK",
        "BEEF MINCE",
        "XYZFOODS RICE",
        "SHAN MASALA",
        "FRESH ST BROCCOLI",       # the bug case: Fresh St is a real brand
        "FRESH CHOICE JUICE",      # another descriptor/brand collision
        "HEAD AND SHOULDERS SHAMPOO",
        "HEAD & SHOULDERS SHAMPOO",
        "HEAD SHOULDERS SHAMPOO",
        "K&N'S CHICKEN NUGGETS",
    ]
    for s in samples:
        r = extract_brand(s)
        print(f"{s!r:32} -> brand={r.brand!r:20} remaining={r.remaining_text!r}")
