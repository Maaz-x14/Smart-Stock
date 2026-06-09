# ml_service/normalization/preprocessor.py

# Strip noise before attempting any match

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

STRIP_SUFFIXES = {
    # Units that get fused to the token
    "LB", "LBS", "OZ", "GL", "GAL", "CT", "PK", "PCS",
    "KG", "G", "GR", "GRM",       # Metric
    "ML", "L", "LTR",             # Metric liquid
    "BX", "BG", "BAG", "BTL", "CAN", "JAR", "TUB",
    "DOZ", "DZ",
    "PT", "QT",
}

def preprocess_token(raw: str) -> str:
    """
    Uppercase, strip trailing price/quantity patterns, remove quality
    prefix/suffix modifiers. Returns cleaned token ready for Pass 1 lookup.

    Examples:
      "ORG STRWBRY 1LB"  -> "STRWBRY"
      "CHKN BRST BNLS"   -> "CHKN BRST"
      "MLK FL CRM 1GL"   -> "MLK FL CRM"
      "DAHI 1KG"         -> "DAHI"
    """
    import re

    token = raw.upper().strip()

    # Remove trailing price: "2.99", "$2.99"
    token = re.sub(r'\$?\d+\.\d{2}$', '', token).strip()

    # Remove standalone numeric quantity+unit suffix: "1LB", "2KG", "500G"
    token = re.sub(r'\s*\d+\.?\d*\s*(LB|LBS|OZ|GL|GAL|KG|G|GR|GRM|ML|L|LTR|CT|PK|PCS|BX|BAG)\b', '', token).strip()

    # Remove leading quality prefixes
    words = token.split()
    while words and words[0] in STRIP_PREFIXES:
        words.pop(0)

    # Remove trailing quality/size suffixes
    while words and words[-1] in STRIP_PREFIXES | STRIP_SUFFIXES:
        words.pop()

    return " ".join(words)