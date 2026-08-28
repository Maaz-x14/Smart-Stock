# Item name
#    ↓
# Normalize/tokenize
#    ↓
# Known brand lexicon ──→ high-confidence brand
#    │
#    ↓ no match
# Remove qty/unit/product descriptors
#    ↓
# Candidate tokens
#    ↓
# Conservative brand heuristics
#    ↓
# High confidence → brand
# Low confidence  → None
#    ↓
# Human-reviewed candidates
#    ↓
# Lexicon growth