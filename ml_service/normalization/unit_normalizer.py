"""
Stage 3 unit normalization.

INTERFACE REWRITTEN (Maaz, this session) - the old parse_quantity_unit()
was NER-era: it expected raw_qty/raw_unit strings from a NER tagger and
did its own fused-token regex parsing ("1LB" -> 1.0, "lb"). That
architecture is retired. Stage 2 (ml_service/item_extraction/) now owns
fused-token unit extraction entirely (unit_extractor.py, #26) - by the
time Stage 3 sees an item, the unit is already a clean string (or
None), and quantity is a float already parsed by Stage 1.7's Row
Parser. Stage 3's remaining job is just CANONICALIZING that unit string
(e.g. "LBS" -> "lb", "Litre" -> "l") against a shared vocabulary -
not re-extracting it from raw text.

New interface: normalize_unit(unit: str | None) -> str | None.
Quantity is NOT touched here anymore - Row Parser already produced a
clean float; there is nothing left for Stage 3 to parse. Callers that
need quantity pass it through unchanged from Stage 1.7's output.

Data (canonicalization map) moved to data/unit_map.json - see that
file's _meta for why this consolidates what used to be three
independently-drifted UNIT_MAP copies (this file, and one embedded in
Normalization_Training.md, which had already gone out of sync with
each other).
"""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "unit_map.json"


def _load_map() -> dict[str, str]:
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["map"]


UNIT_MAP: dict[str, str] = _load_map()


def normalize_unit(unit: str | None) -> str | None:
    """
    Canonicalize a unit string already extracted by Stage 2's
    unit_extractor.py. Returns None if unit is None or unrecognized -
    never guesses.

    Examples:
      "LB"    -> "lb"
      "Litre" -> "l"
      None    -> None
      "XYZ"   -> None   (unrecognized, not a guess)
    """
    if not unit:
        return None
    return UNIT_MAP.get(unit.upper().strip())
