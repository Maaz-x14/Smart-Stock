"""
Stage 3 preprocessing: strip quality modifiers before Pass 1/2/3 lookup.

REWRITTEN (Maaz, this session): the old preprocess_token() re-implemented
unit-stripping regex logic that duplicates Stage 2's unit_extractor.py
(#26, already closed) - two independently-maintained unit vocabularies
that could silently disagree. Also stripped trailing prices
("$2.99") - not relevant anymore since Stage 1.7's Row Parser already
separates price from item_name structurally; by the time Stage 3 sees
text, price was never in it to begin with.

What Stage 3 preprocessing still legitimately owns: quality/dietary
prefix modifiers (ORG, FF, GF, LOWFAT, etc.) that are useful signal for
confirming "this is food" but aren't part of the canonical food name
and would break Pass 1/2 lookup if left in. This is NOT unit
extraction and NOT brand extraction - those are Stage 2's job,
already done by the time this module runs.

Input is now expected to be Stage 2's `remaining_text` (post
unit+brand strip) or the classification_input extractor.py used for
food_classifier - i.e. already unit-clean and often brand-clean.
STRIP_SUFFIXES (unit tokens) intentionally REMOVED from this module -
if a unit string somehow survives into this input, that's a Stage 2
bug to fix there, not something to silently re-strip here.
"""

import re

STRIP_PREFIXES = {
    "ORG", "ORGC", "ORGANIC",
    "FF",                          # Fat-free
    "LF", "LOWFAT", "LO-FAT",     # Low-fat
    "RF", "REDFAT",                # Reduced-fat
    "WHL", "WHOLE",
    "FRZN", "FRZ", "FZ",          # Frozen (keep if it disambiguates, e.g. "FZ CORN")
    "GF",                          # Gluten-free
    "NS", "NSA",                   # No-salt-added
    "NF",                          # Non-fat
    "LS",                          # Low-sodium
    "RAW",
    "FRESH",
    "SMKD",                        # Smoked
    "SLCD", "SLC",                 # Sliced
    "DICED",
    "BNLS", "BNLESS",              # Boneless
    "SKNLS",                       # Skinless
    "LN", "LEAN",
    "XL", "LG", "SM", "MED",      # Size modifiers
    "PKG", "PCK",                  # Package (sometimes prefixed)
}


def preprocess_token(raw: str) -> str:
    """
    Uppercase, strip quality prefix/suffix modifiers. Returns cleaned
    token ready for Pass 1 lookup.

    Does NOT strip units or prices - Stage 2 already produced clean,
    unit-free, price-free text by the time this runs.

    Examples:
      "ORG STRWBRY"   -> "STRWBRY"
      "CHKN BRST BNLS"-> "CHKN BRST"
      "DAHI"          -> "DAHI"
    """
    token = raw.upper().strip()

    words = token.split()
    while words and words[0] in STRIP_PREFIXES:
        words.pop(0)
    while words and words[-1] in STRIP_PREFIXES:
        words.pop()

    return " ".join(words)
