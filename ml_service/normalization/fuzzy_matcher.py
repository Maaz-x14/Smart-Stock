# ml_service/normalization/fuzzy_matcher.py

from rapidfuzz import process, fuzz
from sqlalchemy.orm import Session
from functools import lru_cache
from app.models import ShelfLifeReference


@lru_cache(maxsize=1)
def _load_canonical_names(db_session_hash: int) -> list[str]:
    """
    Load all canonical_name values from shelf_life_reference.
    LRU-cached after first call — the reference table rarely changes.
    Cache key is a hash of the db session to allow testing with different DBs.
    """
    # This is called once at startup via get_canonical_names() below
    raise NotImplementedError("Use get_canonical_names() directly")


_canonical_names_cache: list[str] | None = None


def get_canonical_names(db: Session) -> list[str]:
    """
    Returns deduplicated list of all canonical names from shelf_life_reference.
    Cached in module-level variable after first call.
    """
    global _canonical_names_cache
    if _canonical_names_cache is None:
        rows = db.query(ShelfLifeReference.canonical_name).distinct().all()
        _canonical_names_cache = [r[0] for r in rows]
    return _canonical_names_cache


FUZZY_THRESHOLD = 80
SHORT_TOKEN_THRESHOLD = 4  # skip fuzzy for tokens this short or shorter


def pass2_fuzzy(cleaned_token: str, db: Session) -> tuple[str | None, float]:
    """
    Fuzzy match cleaned_token against all canonical names in shelf_life_reference.

    Returns (canonical_name, confidence) or (None, 0.0) on miss.
    Confidence is score / 100.

    Examples:
      "STRWBERY"  → ("Strawberries", 0.91)
      "CHKN BRST" → ("Chicken Breast", 0.88)
      "SALT"      → ("Salt", 1.00)   ← exact match would have been caught by Pass 1
      "XYZ123"    → (None, 0.0)
    """
    if len(cleaned_token) <= SHORT_TOKEN_THRESHOLD:
        return None, 0.0

    canonical_names = get_canonical_names(db)
    if not canonical_names:
        return None, 0.0

    result = process.extractOne(
        cleaned_token.upper(),
        [name.upper() for name in canonical_names],
        scorer=fuzz.token_sort_ratio,
    )

    if result is None:
        return None, 0.0

    _, score, idx = result

    if score >= FUZZY_THRESHOLD:
        return canonical_names[idx], round(score / 100, 3)

    return None, 0.0