"""
Stage 3 Pass 1: direct abbreviation lookup.

Data moved to data/abbreviation_map.json (was a ~500-entry inline dict -
see that file's _meta for provenance/dedup notes). This module only
loads and looks up.

TOKEN-SCAN FIX (Maaz, this session - #51 follow-up): the original
pass1_lookup only tried the *whole* cleaned string as one key (exact,
then space-collapsed). This misses real receipt tokens that carry
extra noise words beyond what preprocess_token's prefix/suffix
stripping catches - e.g. "TAPAL TEA BAGS DNEDR ELCHI 50'S" never
collapses down to "TAPAL" or "TEA" even though both are valid map
keys, because preprocess_token only trims known prefix/suffix
modifiers, not arbitrary interior noise. Every one of these fell
through to Pass 3, which then hallucinated a wrong-but-confident
canonical name (root cause of #51).

Added a bounded token-scan: split the cleaned string into words, try
decreasing-length contiguous word windows (bigram before unigram, so
"TEA BAGS" is preferred over a lone "TEA" hit if both exist) against
the map, return the first hit found by scanning left to right. This
stays a deterministic dictionary lookup - no fuzzy/LLM risk - and
only fires after the existing whole-string exact/collapsed checks
miss, so it can't change any currently-correct Pass 1 result.
"""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "abbreviation_map.json"

# Max contiguous words to try as a single map key during token-scan.
# 2 covers real multi-word keys observed in the map (e.g. "CHKN BRST")
# without the combinatorics of scanning longer windows.
_MAX_SCAN_WINDOW = 2


def _load_map() -> dict[str, str]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["map"]


ABBREVIATION_MAP: dict[str, str] = _load_map()


def _token_scan(cleaned_token: str) -> str | None:
    """
    Scan cleaned_token's words left to right, trying each contiguous
    window (longest first, up to _MAX_SCAN_WINDOW) against the map.
    Returns the first hit, or None if nothing in the string matches
    any known key.
    """
    words = cleaned_token.upper().split()

    for start in range(len(words)):
        for window in range(min(_MAX_SCAN_WINDOW, len(words) - start), 0, -1):
            candidate = " ".join(words[start:start + window])
            result = ABBREVIATION_MAP.get(candidate)
            if result:
                return result

    return None


def pass1_lookup(cleaned_token: str) -> str | None:
    """
    Direct dictionary lookup. Returns canonical name or None on miss.
    Tries exact match on the whole string first, then a joined-words
    variant of the whole string, then a token-scan for a known key
    embedded inside a longer noisy string.
    """
    upper = cleaned_token.upper()

    result = ABBREVIATION_MAP.get(upper)
    if result:
        return result

    # Try collapsing spaces: "CHKN BRST" might appear as "CHKNBRST"
    collapsed = upper.replace(" ", "")
    result = ABBREVIATION_MAP.get(collapsed)
    if result:
        return result

    # Try finding a known key as a substring token/bigram within a
    # longer noisy string (e.g. "TAPAL TEA BAGS DNEDR ELCHI 50'S").
    return _token_scan(upper)
