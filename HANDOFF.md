# HANDOFF.md — Context Document for Next Chat Session

## Smart-Stock: AI-Powered Inventory & Waste Reduction System

---

## Writing style for this project (important)

- No over-explanation. Short, direct answers.
- Propose approach before code. Ask one clarifying question at a time, don't assume.
- Surgical edits to existing docs/code — never full rewrites unless explicitly asked.
- Push back on bad ideas. Distinguish facts from speculation.
- Individual per-file commits.
- Ruthless mentor mode: stress-test everything, don't sugarcoat.
- Present files at end of response as attachments, not pasted inline — so specific lines can be edited without re-pasting whole files. **Always the FULL file, never a diff/snippet** — the person applies it directly. (Diffs were tried this session for doc updates and failed to apply cleanly — back to full files only.)

---

## Project Summary

Smart-Stock: portfolio/CV project. Reads grocery receipts, predicts expiry dates, reduces food waste. Full stack: React + TypeScript -> FastAPI -> PostgreSQL -> ML Pipeline (7 stages).

**Docs, current status:**

| Doc | Status |
|---|---|
| PRD.md, Architecture.md, API_Spec.md, DB_Schema.md, OCR_Training.md, NER_Training.md | Current as of previous handoff, unchanged this session |
| ML_Pipeline.md | **Updated this session (#33).** §9 Latency Budget filled in with real measured numbers (was "not yet measured" placeholder). §10 Processing Time row updated with the same. |
| Item_Extraction.md | **Updated this session.** §4.2 now documents #50's fix (batch is_food prompt rewrite) instead of listing it as an open gap. |
| Normalization.md | **Updated this session.** §5.4 documents #51's fix (Pass 3 abstention path) + the cache-clear testing gotcha. §5.6 documents #49's category-classifier fix. §9.1's stale `Tapal Tea Bags -> 'Dried Lychee'` example corrected to reflect the fix. |
| Expiry.md | Current as of previous handoff, unchanged this session |
| README.md | **Rewritten this session.** Was badly stale — pipeline orchestrator, Stage 2 extractor, and end-to-end validation were all marked "pending" despite being done in a prior session. Now reflects built/validated/measured status accurately, doc list fixed (`Normalization.md`/`Expiry.md`, not old `_Training.md` names), roadmap replaced with real open items. |

**Naming note:** `Normalization_Training.md`/`Expiry_Training.md` renamed to `Normalization.md`/`Expiry.md` (prior session) - neither stage trains a model (3-pass lookup + rule-based tiers), the `_Training` suffix was a leftover from NER-era naming. `OCR_Training.md` keeps its name - TrOCR fine-tuning genuinely happened there.

---

## Pipeline (unchanged structure)

```
Receipt Image
  -> Stage 1: OCR (PaddleOCR, pretrained) - text + bounding boxes
  -> Stage 1.5: Row Reconstruction - deskew + y-position clustering
  -> Stage 1.6: Prefilter - drops metadata rows
  -> Stage 1.7: Row Parser - header-driven column mapping -> {item_name, quantity, price, discount, total}
  -> Stage 2: Item Field Extraction - regex unit + fuzzy brand lexicon + LLM is_food gate (BATCHED, chunk size 5) -> {unit, brand, is_food}
       is_food=false/unknown -> surfaced to user flagged "excluded", stops here
       is_food=true  |
  -> Stage 3: Normalization - 3-pass lookup (map -> fuzzy -> LLM) -> NormalizedItem
  -> Stage 4: Expiry Prediction - 3-tier rule-based lookup -> ExpiryPrediction
  -> Structured Inventory Item
```

---

## THIS SESSION'S WORK (#33 latency re-measurement + doc backfill for last session's fixes)

No new bugs found this session — pure measurement + documentation work, closing out debt from the prior session (#49/#50/#51 were fixed then, but never written up in the docs).

### #33 — Latency re-measurement, CLOSED
Built `benchmark_latency.py`: calls the same stage functions `process_receipt()` calls, individually timed, without modifying `pipeline.py`. Iterated twice on the design:
- v1 (`--reps 5`, no rate-limit handling): run on real Groq free tier, hit cascading 429s — retry-backoff got counted as pipeline latency, badly inflating Item Field Extraction/Normalization numbers.
- v2 (current): `--reps` default dropped to 1, added a logging-handler-based rate-limit watcher that flags any run hitting a 429 mid-stage so it's excluded from stats instead of silently averaged in.

**Real numbers** (4 runs — 2.jpg x2, 3.jpg x2, 1 rep each, spaced out manually to avoid TPM limits): end-to-end mean 11.9s / median 11.5s. Per-stage: OCR ~5.4s, Item Field Extraction ~5.6s (these two are ~90% of total), Normalization ~0.9s mean (cache-dependent, cache was warm between the two runs per receipt — not a cold-cache number), row stages + expiry negligible (<25ms). 2.jpg (14.3s) is over PRD.md §6's 10s budget; 3.jpg (9.5s) is near/under it. n=2 per receipt — no real p95 reported.

Written into `ML_Pipeline.md` §9 and §10. `benchmark_latency.py` committed to repo root (dev tool, not pipeline code).

### Doc backfill — #49, #50, #51, #32
Prior session fixed these but never updated the docs. This session:
- Pulled actual GitHub issue comments/commits for #49/#50/#51 before writing anything — #50 and #51 had no resolution text in the issue body itself, only in commit messages (`17cdb8c` for #51's Pass 3 prompt, the shipped `BATCH_SYSTEM_PROMPT` for #50), so those were read directly rather than assumed from memory.
- #49: dup-row finding not worth documenting (was a non-bug). Category classifier fix -> `Normalization.md` §5.6.
- #50: batch is_food prompt fix -> `Item_Extraction.md` §4.2. Kept the writeup generic (describes the reasoning-loop *mechanism*, not the specific example strings) per explicit request — a reader without prior context should still understand the failure mode.
- #51: Pass 3 abstention-path fix + the cache-clear testing gotcha -> `Normalization.md` §5.4. Stale `Tapal Tea Bags -> 'Dried Lychee'` example in §9.1 corrected.
- #32: already reflected from the prior session, confirmed no further changes needed.

### README.md rewrite
Full rewrite, not surgical — justified because the drift was structural (whole status table wrong), not a few stale lines. Current Status table now correct, real latency numbers inline, roadmap replaced with actual open items (#34, #52, 1.jpg/4.jpg, latency optimization, #23).

---

## Prompt iteration notes

Not applicable this session — no prompts were touched. See prior handoff (preserved in git history / `Item_Extraction.md` §4.2, `Normalization.md` §5.4) for the #50/#51 prompt-design reasoning if it needs to be revisited.

---

## Known follow-up, NOT fixed this session (very low priority)

**Pass 3 resolves multi-product brand names too eagerly.** Example: `Pakola MIk Uht 250M1` -> `'Pakola'` (pass 3, confidence 0.28, hard_default fallback). Still deferred — needs a labeled eval set to tune safely (#34), not another blind prompt edit. Unchanged from prior handoff, not touched this session.

---

## Stage-by-stage status (Stages 1-4 substance unchanged this session)

### Stages 1, 1.5, 1.6, 1.7 - unchanged this session. All DONE.

### Stage 2 - Item Field Extraction: unchanged this session (code). #50's fix documented this session — see above.

### Stage 3 - Normalization: unchanged this session (code). #49/#51's fixes documented this session — see above.

### Stage 4 - Expiry Prediction: unchanged this session.

---

## Real findings, carried forward as open items

1. **Latency: now formally measured (#33, closed this session).** See "THIS SESSION'S WORK" above for numbers. 2.jpg is consistently over the 10s budget; 3.jpg is near/under it. OCR and Item Field Extraction are the two stages worth optimizing if this gets picked up — everything else is negligible.
2. **Groq free-tier TPM is fragile under back-to-back calls** — confirmed hard this session (the v1 benchmark script's `--reps 5` triggered cascading 429s). Real pipeline runs (not just benchmarking) should still be spaced out; not yet a problem in normal single-receipt-at-a-time usage, but worth remembering before any batch-testing script.
3. **Pakola multi-product-brand gap** — see "Known follow-up" above, unchanged.
4. Carried from prior sessions, still open/unconfirmed: Pass 3 `ANDA DOZEN -> Andouille` gap, Pass 3 free-text output testability.

---

## GitHub issue status (as of this handoff)

**Closed this session:** #33 (latency re-measurement).

**No code fixed this session** — #49/#50/#51 were already fixed/closed in the prior session; this session only wrote up their docs (see above).

**Still open, unchanged from previous handoffs:**
- #16 - confidence-score garbage-line filter, unblocked, not picked up yet
- #23 - 5.jpg multi-line header + Tax(%) format, deferred
- #34 - labeled `is_food` sample, not started — needed for real is_food accuracy measurement and the Pakola follow-up
- #48 - Stage 3 reasoning-loop gap on severely corrupted OCR — still flagged as "likely improved by #50's prompt rewrite, not formally confirmed." Not rechecked this session either.
- #52 - Stage 2 fail-safe not firing on empty-content classification (`is_food=True` instead of `UNKNOWN`). Confirmed filed and open (was at risk of being lost last handoff). **Not yet investigated** — still needs a read of `extractor.py`'s `_parse_batch_response()` before any fix is proposed.

**Open, from the earlier-filed set, status:**
- #30 - DB migration: add `brand` column - still not run
- #31 - Update Pydantic schemas (`brand`, `is_food` in `/receipts/upload` response) - still not done
- #32 - Real end-to-end validation - 2 of 4 real receipts tested (2.jpg, 3.jpg). 1.jpg, 4.jpg still not run. Closed as complete per PRD's stated acceptance criteria; this gap is a documented known limitation, not a reason it was reopened.

---

## Immediate next steps (in order)

1. **#52** — read `extractor.py`'s `_parse_batch_response()`, confirm or rule out the suspected cause (array-shape validated but not per-item content), fix if confirmed. Small, well-scoped, already flagged twice now as the right next move.
2. **#30 + #31** — DB migration + Pydantic schema updates so Stage 2's real output (brand, unit, is_food) can actually be persisted. Nothing downstream of the ML pipeline stores this yet.
3. **Build the labeled eval set (#34)** — blocks both real is_food accuracy measurement and the Pakola brand-resolution follow-up. Do this once #30/#31 give a real place to pull persisted examples from, rather than hand-collecting from logs again.
4. **1.jpg/4.jpg** — cheap, closes out #32's one remaining known gap.
5. #23/#48/#16 remain deferred, low priority, unchanged.

---

## Key decisions made this session (context for future reference, not to be re-litigated without new evidence)

- **Diffs abandoned for doc updates, back to full files.** Tried unified-diff patches for `Item_Extraction.md`/`Normalization.md` to save tokens; `git apply` failed on malformed hunk headers (hand-written line-count math, error-prone). Full files are the reliable path — see "Writing style" above, updated to reflect this.
- **Full-file rewrite justified for README.md specifically**, as an exception to the surgical-edit default — the drift was structural (an entire status table wrong), not a few stale lines a patch could fix cleanly.
- **#50's writeup deliberately kept example-free in the failure-mechanism description** (per explicit instruction) — describes the reasoning-loop pattern generically so it's understandable without prior session context, rather than naming the specific items that triggered it.
- **Latency benchmark excludes rate-limited runs from stats rather than averaging them in** — retry-backoff time is not pipeline latency, and blending it in would understate how much of "slowness" is actually Groq TPM contention vs. real compute cost.

Carried from prior sessions, still valid: Stage 2's `food_classifier.py` model is Groq `openai/gpt-oss-20b`; Stage 3/4 interface is `normalize_entity(item_name, quantity, unit, db)`; frozen-signal category check runs on raw unstripped `item_name`; doc renames (`Normalization.md`/`Expiry.md`); test-fixture errors documented as such, not silently patched.
