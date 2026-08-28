"""
Stage 2 component: unit extraction.

Extracts a unit token (and its preceding quantity, if fused) from a raw
item_name string produced by the Row Parser (Stage 1.7). Regex + lookup
only — no model, no training data.

Design:
  1. Normalize case for matching (original string is not mutated except
     for the matched span being stripped).
  2. Match a NUMBER+UNIT fused pattern (e.g. "1LB", "500ML") since real
     receipt samples show no space between quantity and unit.
  3. Vocabulary includes known units + OCR-corruption variants.
  4. No match -> unit=None. Never force a guess.

OCR corruption table is SPECULATIVE — built from generic visual-confusion
pairs (l/1/I, O/0, S/5, B/8, Z/2, G/6), NOT validated against real
Smart-Stock receipt OCR output. Treat as a first draft; replace entries
once real corrupted-unit examples are observed in PaddleOCR output.
"""

import re
from dataclasses import dataclass

# --- Canonical unit vocabulary -------------------------------------------
# category -> canonical unit -> set of accepted surface forms (incl. corruption)
_CANONICAL_UNITS = {
    "kg": {"kg", "kgs", "kg.", "k9", "kq", "kqs", "kq.", "k9", "k9s", "k9."},          # k9/kq: g->9/q visual noise (speculative)
    "g":  {"g", "gm", "gms", "grams", "gram", "9m", "9ms", "9rams", "9ram"},   # "9" for lone g (speculative, risky - see note)
    "lb": {"lb", "lbs", "1b", "1bs", "ib", "ibs"},                  # l->1/i (speculative)
    "oz": {"oz", "0z", "o2", "02"},                               # o->0
    "l":  {"l", "ltr", "ltrs", "liter", "litre"},# l->1 (HIGH collision risk, see note)
    "ml": {"ml", "m1", "mI"},                         # l->1/I (documented in HANDOFF)
    "gal": {"gal", "gallon", "9al", "9allon"},                  # g->9
    "pcs": {"pcs", "pc", "pieces", "piece", "pc's", "pcs."},
    "dozen": {"dozen", "dz", "doz"},
    "pack": {"pack", "pkt", "packet", "pk"},
    "container": {"container", "ctr"},
}

# Reverse lookup: surface form -> canonical unit
_SURFACE_TO_CANONICAL = {
    form.lower(): canon
    for canon, forms in _CANONICAL_UNITS.items()
    for form in forms
}

# Sort surface forms longest-first so e.g. "ltrs" matches before "l"
_SURFACE_FORMS_SORTED = sorted(_SURFACE_TO_CANONICAL.keys(), key=len, reverse=True)

# NOTE on risky single-character entries ("9" for g, "1" for l):
# These are high false-positive risk because they collide with legitimate
# quantity digits. They are ONLY matched when directly fused to a number
# per the regex below (e.g. "500" + "9" -> reject, no leading digit before
# the unit char itself in that position) - see _FUSED_PATTERN comment.
# Recommend disabling "9"->g and "1"->l entries entirely until real
# corrupted examples justify them. Left in as documented, flagged, and
# commented for a deliberate decision rather than silently omitted.

_UNIT_ALTERNATION = "|".join(re.escape(f) for f in _SURFACE_FORMS_SORTED)

# Matches: <number><optional space><unit token>, unit token is fused or
# separated by at most one space. Number captured for reference (not
# currently returned separately - row parser already owns quantity).
_FUSED_PATTERN = re.compile(
    rf"(?P<qty>\d+(?:\.\d+)?)\s?(?P<unit>{_UNIT_ALTERNATION})\b",
    re.IGNORECASE,
)


@dataclass
class UnitExtractionResult:
    unit: str | None          # canonical unit, or None if no confident match
    matched_span: str | None  # original substring matched (for stripping/audit)
    remaining_text: str       # item_name with matched span removed


def extract_unit(item_name: str) -> UnitExtractionResult:
    """
    Extract a unit from an item_name string.

    Returns UnitExtractionResult with unit=None if no confident match -
    this function never guesses.
    """
    if not item_name:
        return UnitExtractionResult(unit=None, matched_span=None, remaining_text=item_name or "")

    match = _FUSED_PATTERN.search(item_name)
    if not match:
        return UnitExtractionResult(unit=None, matched_span=None, remaining_text=item_name)

    surface_unit = match.group("unit").lower()
    canonical = _SURFACE_TO_CANONICAL.get(surface_unit)

    # Guard against the risky single-char entries unless truly fused to a digit
    # with no separating space (reduces false positives on isolated "1" or "9").
    if surface_unit in {"1", "9"} and " " in match.group(0):
        return UnitExtractionResult(unit=None, matched_span=None, remaining_text=item_name)

    matched_span = match.group(0)
    remaining = (item_name[: match.start()] + item_name[match.end():]).strip()
    remaining = re.sub(r"\s{2,}", " ", remaining)

    return UnitExtractionResult(
        unit=canonical,
        matched_span=matched_span,
        remaining_text=remaining,
    )


if __name__ == "__main__":
    # Smoke tests using real examples from API_Spec.md / HANDOFF.md context
    samples = [
        "ORG STRWBRY 1LB",
        "WHOLE MILK 1GAL",
        "Supravit-M Tablet 10's",
        "NESTLE MILK 1L",
        "CHICKEN 500G",
        "RICE 5KG",
    ]
    for s in samples:
        r = extract_unit(s)
        print(f"{s!r:35} -> unit={r.unit!r:12} remaining={r.remaining_text!r}")
