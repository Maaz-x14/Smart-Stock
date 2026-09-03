"""
Check/clear stale NormalizationCache entries.

Why this exists: llm_fallback.py's pass3_llm() checks NormalizationCache
BEFORE calling Groq at all. If a raw_token was cached under an old prompt
version, changing the prompt has zero effect on that token until its
cache row is cleared - the LLM never gets re-invoked. This explains why
'Everyday Instnt' -> 'Instant Noodles' persisted identically across three
different Pass 3 prompt rewrites.

Usage:
    python check_cache.py check "EVERYDAY INSTNT"       # inspect one entry
    python check_cache.py check-all                      # list all entries
    python check_cache.py clear "EVERYDAY INSTNT"        # delete one entry
    python check_cache.py clear-all                      # delete ALL entries (re-evaluate everything under new prompt)

Note: raw_token is stored .upper() (see llm_fallback.py's _cache_lookup/
_cache_store), so lookups here also upper-case the input for a match.
"""

import sys
from app.db import SessionLocal
from app.models import NormalizationCache


def check(token: str):
    db = SessionLocal()
    entry = db.query(NormalizationCache).filter_by(raw_token=token.upper()).first()
    if entry:
        print(f"FOUND: raw_token={entry.raw_token!r} canonical_name={entry.canonical_name!r} "
              f"source={entry.source!r} hit_count={entry.hit_count} created_at={entry.created_at}")
    else:
        print(f"No cache entry for {token.upper()!r}")
    db.close()


def check_all():
    db = SessionLocal()
    entries = db.query(NormalizationCache).all()
    if not entries:
        print("Cache is empty.")
    for e in entries:
        print(f"raw_token={e.raw_token!r:40} canonical_name={e.canonical_name!r:25} "
              f"source={e.source!r:6} hit_count={e.hit_count} created_at={e.created_at}")
    db.close()


def clear(token: str):
    db = SessionLocal()
    entry = db.query(NormalizationCache).filter_by(raw_token=token.upper()).first()
    if entry:
        db.delete(entry)
        db.commit()
        print(f"Deleted cache entry for {token.upper()!r}")
    else:
        print(f"No cache entry for {token.upper()!r} - nothing to delete")
    db.close()


def clear_all():
    db = SessionLocal()
    count = db.query(NormalizationCache).delete()
    db.commit()
    print(f"Deleted {count} cache entries. Every token will re-hit Pass 3 (Groq) on next resolve.")
    db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "check" and len(sys.argv) == 3:
        check(sys.argv[2])
    elif cmd == "check-all":
        check_all()
    elif cmd == "clear" and len(sys.argv) == 3:
        clear(sys.argv[2])
    elif cmd == "clear-all":
        clear_all()
    else:
        print(__doc__)
        sys.exit(1)
