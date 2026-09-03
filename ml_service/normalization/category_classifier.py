"""
Stage 3 category assignment.

Data moved to data/category_keywords.json (was inline dicts with real
collision bugs - see that file's _meta).

FIX: assign_category() now takes an optional `raw_token` (the original,
unstripped receipt text, e.g. "FRZ BROC") in addition to
`canonical_name`. If raw_token contains a frozen-signal keyword
("frozen"/"frzn"/"frz"), category is Frozen immediately - checked
BEFORE any other keyword list, using information (that the raw token
said "frozen") that the canonical name alone ("Broccoli") can never
recover. This is the correct fix, not a keyword-list edit: no amount
of keyword-list reshuffling can let "Broccoli" alone disambiguate
fresh vs frozen - the signal has to come from upstream, before
normalization stripped it.

If raw_token is not provided (backward-compat / DB-only callers),
falls through to canonical_name matching only, same as before.

FIX (Issue #49): Pass 2's keyword classifier used raw substring
matching (`kw in name_lower`), which caused false hits when a
keyword was a substring of an unrelated word - e.g. Produce's
"berry" is a substring of "strawberry", so "Pakola Strawberry Milk"
matched Produce before Dairy's "milk" keyword was ever checked
(Produce is earlier in dict iteration order). Fixed with a
word-boundary regex match instead of substring containment. Note:
the Pass 0 frozen-signal check above still uses substring matching
(`signal in raw_lower`) - same class of bug, out of scope for #49,
flagged for a follow-up if a frozen-signal keyword ever collides
with an unrelated word.
"""

import json
import re
from pathlib import Path

from sqlalchemy.orm import Session
from app.models import ShelfLifeReference

_DATA_PATH = Path(__file__).parent / "data" / "category_keywords.json"


def _load_data() -> tuple[list[str], dict[str, list[str]]]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["frozen_signal_keywords"], data["categories"]


FROZEN_SIGNAL_KEYWORDS, CATEGORY_KEYWORDS = _load_data()


def assign_category(canonical_name: str, db: Session, raw_token: str | None = None) -> str:
    """
    Assign category to a canonical food name.

    Priority:
    0. Frozen-signal check on raw_token (if provided) - catches
       "frozen X" before X's own keyword would route it elsewhere.
    1. Exact match in shelf_life_reference (most reliable)
    2. Keyword classifier fallback (approximate, word-boundary match)
    3. "Other" if no match

    Returns category string.
    """
    # Pass 0: frozen-signal priority check (see module docstring)
    if raw_token:
        raw_lower = raw_token.lower()
        if any(signal in raw_lower for signal in FROZEN_SIGNAL_KEYWORDS):
            return "Frozen"

    # Pass 1: DB lookup
    ref = db.query(ShelfLifeReference).filter_by(canonical_name=canonical_name).first()
    if ref:
        return ref.category

    # Pass 2: keyword classifier (word-boundary match, Issue #49 -
    # substring matching caused false hits, e.g. "berry" inside
    # "strawberry" routing a Dairy item to Produce)
    name_lower = canonical_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(kw)}\b", name_lower) for kw in keywords):
            return category

    return "Other"
