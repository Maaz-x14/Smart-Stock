"""
Stage 3 Pass 1: direct abbreviation lookup.

Data moved to data/abbreviation_map.json (was a ~500-entry inline dict -
see that file's _meta for provenance/dedup notes). This module only
loads and looks up.
"""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "abbreviation_map.json"


def _load_map() -> dict[str, str]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["map"]


ABBREVIATION_MAP: dict[str, str] = _load_map()


def pass1_lookup(cleaned_token: str) -> str | None:
    """
    Direct dictionary lookup. Returns canonical name or None on miss.
    Tries exact match first, then a joined-words variant.
    """
    result = ABBREVIATION_MAP.get(cleaned_token.upper())
    if result:
        return result

    # Try collapsing spaces: "CHKN BRST" might appear as "CHKNBRST"
    collapsed = cleaned_token.upper().replace(" ", "")
    return ABBREVIATION_MAP.get(collapsed)
