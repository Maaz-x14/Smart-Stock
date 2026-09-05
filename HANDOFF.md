# HANDOFF.md — Session Summary

**Session focus:** #33 (latency re-measurement) end-to-end, doc backfill for #49/#50/#51/#32, README.md rewrite.

---

## What got done this session

### #33 — Latency re-measurement (closed)
- Built `benchmark_latency.py` — per-stage + end-to-end timing, mirrors `pipeline.py`'s stage sequence without modifying it.
- Iterated on the script twice: added a rate-limit watcher (Groq free-tier 429s were inflating retry-backoff into apparent latency — those runs are now detected and excluded, not averaged in) and dropped default reps from 5 to 1 after hitting exactly that problem on a real run.
- Real numbers (4 runs, 2 receipts — 2.jpg, 3.jpg — 1 rep each): end-to-end mean 11.9s / median 11.5s. OCR (~5.4s) and Item Field Extraction (~5.6s) are the two bottleneck stages, ~90% of total latency combined. Normalization/expiry/row-parsing stages negligible.
- Results written into `ML_Pipeline.md` §9 (Latency Budget) and §10 (Processing Time row), replacing the old "not yet measured" placeholders.
- **Caveats on record:** n=2 per receipt (no real p95), normalization cache was warm between the two runs per receipt (not cold-cache numbers), 2.jpg is over the 10s budget, 3.jpg is near it.

### Doc backfill for previously-closed issues
- **#49** (dup row + category bug): dup row confirmed genuine (not a bug, no doc change needed). Category classifier fix (word-boundary matching instead of substring) documented in `Normalization.md` §5.6.
- **#50** (Kimtiaz reasoning-loop): fix documented in `Item_Extraction.md` §4.2 — new batch is_food prompt explicitly forbids inventing food identities from partial letter overlap. Verified via commit diff + the actual shipped `BATCH_SYSTEM_PROMPT`, not just the session's earlier claim.
- **#51** (Tapal Tea Bags → Dried Lychee): fix documented in `Normalization.md` §5.4 — Pass 3 prompt now distinguishes brand-is-product vs. brand-is-generic-company-name, with an UNKNOWN abstention path. Also documented the cache gotcha (`check_cache.py clear-all` needed to test a prompt change against a previously-seen token) since it cost real debugging time this session.
- **#32**: already reflected in `ML_Pipeline.md` from the previous session; confirmed no further doc changes needed.
- All four verified against actual GitHub issue comments/commits before writing anything — two of the four (#50, #51) had no resolution text in their issue body, only in commit messages, so those were pulled directly rather than assumed.

### README.md rewrite
Was badly stale — marked the pipeline orchestrator, Stage 2 extractor, and end-to-end validation as "pending" when all three were done in a prior session. Rewritten: Current Status table now reflects built/validated/measured accurately, real latency numbers added inline per stage, doc list fixed (`Normalization.md`/`Expiry.md`, not the old `_Training.md` names), roadmap replaced with actual open items instead of stale checkboxes.

---

## Open issues, current state

| # | Title | Priority |
|---|---|---|
| #52 | Stage 2 fail-safe not firing on empty-content responses (is_food=True instead of UNKNOWN) | Should do next — real correctness bug, root cause already suspected (`_parse_batch_response` likely only validates array-shape, not per-item content), not yet confirmed by reading `extractor.py` |
| #30 | DB migration: add brand column to inventory_items | Blocks persisting real Stage 2 output |
| #31 | Update Pydantic schemas for Stage 2 extraction fields | Same blocker as #30 |
| #34 | Build labeled is_food evaluation dataset | Needed before claiming real is_food accuracy anywhere |
| #23 | New receipt format: multi-line header + Tax(%) (5.jpg) | Deferred — needs its own parsing path |
| #48 | Stage 3 LLM reasoning loop on severely corrupted OCR | Low priority, fails safe correctly, low frequency |
| #16 | Confidence-score garbage-line filter | Needs real distribution analysis first |

## Next session plan (proposed order)
1. **#52** — read `extractor.py`'s `_parse_batch_response`, confirm/fix the fail-safe gap. Small, well-scoped, already-suspected root cause.
2. **#30 + #31** — DB migration + schema updates to actually persist Stage 2 output (brand, unit, is_food). Nothing downstream of the ML pipeline currently stores this.
3. **#34** — build the labeled is_food eval set once persistence exists to pull real examples from.
4. Validate 1.jpg/4.jpg through the full pipeline (cheap, closes #32's known gap).
5. #23/#48/#16 remain deferred, low priority.

## Known state to remember
- Groq free-tier TPM: back-to-back Stage 2/3 calls across reps or receipts risk 429s. Space out real pipeline runs.
- `normalization_cache` is checked before every Pass 3 LLM call — clear it (`check_cache.py clear-all`) before testing any Pass 3 prompt change, or results will look unchanged when they aren't.
- `benchmark_latency.py` is in repo root, not `ml_service/` — it's a dev tool, not pipeline code.
