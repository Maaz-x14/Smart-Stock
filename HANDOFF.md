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
- Present files at end of response as attachments, not pasted inline — so specific lines can be edited without re-pasting whole files. **Always the FULL file, never a diff/snippet** — the person applies it directly.

---

## Project Summary

Smart-Stock: portfolio/CV project. Reads grocery receipts, predicts expiry dates, reduces food waste. Full stack: React + TypeScript -> FastAPI -> PostgreSQL -> ML Pipeline (7 stages).

**Docs, current status:**

| Doc | Status |
|---|---|
| PRD.md, Architecture.md, API_Spec.md, DB_Schema.md, OCR_Training.md, NER_Training.md, README.md | Current as of previous handoff, unchanged this session |
| Item_Extraction.md | Updated this session - batch classification (Issue #46) documented (§4.1, §4.2), extractor.py batch entry point documented |
| Normalization.md | Updated this session - `reasoning_effort` fix documented in Pass 3 (§5.4), real pipeline validation results added (§9.1), reasoning-loop known gap added (§10) |
| Expiry.md | Current as of previous handoff, unchanged this session |
| ML_Pipeline.md | **Needs update, not yet done** - Stage 2 section (§6) still describes the old concurrency-based design; pipeline.py's structure diagram (§9) is stale (pipeline.py is no longer empty). See "Docs still needing updates" below. |

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

## Stage-by-stage status (updated)

### Stages 1, 1.5, 1.6, 1.7 - unchanged this session, see previous handoff / respective docs. All DONE.

### Stage 2 - Item Field Extraction: COMPLETE, batching added this session (Issue #46)

All 4 modules built and smoke-tested against real APIs (Groq), from prior sessions:
- `unit_extractor.py` (#26), `brand_matcher.py` (#27), `food_classifier.py` (#28), `extractor.py` (#29) - see Item_Extraction.md for full detail, unchanged in substance this session.

**This session's work (Issue #46):** `pipeline.py` (#25) was built and wired end-to-end, then real testing (`test_pipeline_local.py`) surfaced Groq TPM 429s that the originally-planned concurrency approach (`STAGE2_CONCURRENCY` semaphore) did not fix - concurrency bounds parallel requests, not tokens/minute. Root-caused and fixed:

1. `food_classifier.py`: added `classify_is_food_batch()` / `classify_is_food_chunked()` (chunk size 5) - batches is_food classification instead of one Groq call per item. XML-structured batch prompt with 2 multishot examples, fail-safe-to-UNKNOWN-per-chunk on any malformed/wrong-length response.
2. `extractor.py`: added `extract_item_fields_batch()` - the new production entry point. Unit/brand stay per-item (local, not the bottleneck); only is_food is batched.
3. `pipeline.py`: removed `STAGE2_CONCURRENCY`/`Semaphore`/`gather` fan-out entirely, replaced with a single `extract_item_fields_batch()` call.
4. `reasoning_effort="low"` added to both `food_classifier.py` calls (single-item and batch) and `llm_fallback.py` (Stage 3 Pass 3) - fixes a separate real bug where Groq's `gpt-oss-20b` reasoning trace was observed consuming the entire token budget on some prompts, producing empty content.

**Validated:** 2 real receipts via `test_pipeline_local.py`, 36 items total. Zero Groq 429s in either run. See Item_Extraction.md §4.1 for full validation detail.

**Known remaining gap (not this session's target, tracked separately):** `reasoning_effort="low"` reduces but does not fully eliminate the reasoning-loop failure mode - 2/36 items (both with severely corrupted OCR input, hand blocking part of the receipt) still hit `finish_reason='length'` with the reasoning trace looping and never emitting content. Pipeline fails safe correctly (item surfaced, not dropped). Tracked as GitHub Issue #48, low priority, not fixed this session. Full root-cause detail: Item_Extraction.md §4.2, Normalization.md §5.4.

Full detail, bug records, worked examples: Item_Extraction.md.

### Stage 3 - Normalization: COMPLETE, `reasoning_effort` fix applied this session (Issue #46)

Interface rewrite and core bug fixes are from the prior session (see below, unchanged in substance). **This session added:** `reasoning_effort="low"` to `llm_fallback.py`'s Groq call (same fix/same root cause as Stage 2, see above) plus debug logging (`reasoning` field length, `finish_reason`, `usage`) used to root-cause the empty-content failures found in real pipeline testing.

Real pipeline validation this session (distinct from the synthetic 36-case eval below): 2 real receipts, 36 items total. 34/36 resolved correctly via Pass 1/2/3; the 2 failures were the reasoning-loop cases above (Issue #48), not a Stage 3 logic bug - confirmed by a second, cleaner-OCR receipt resolving 16/16 with zero failures.

**Prior session's work (unchanged, still valid):** `normalize_entity()` interface rewrite from the old broken NER-era signature to `normalize_entity(item_name: str, quantity: float, unit: str | None, db) -> NormalizedItem | None`; 6 real bugs found and fixed (deprecated Groq model, category-keyword collisions via frozen-signal priority check, duplicate dict keys, dead code, duplicated unit-stripping logic, unit_normalizer interface simplification). Synthetic eval: 36 test cases, 35/36 exact-match correct (97.2% corrected accuracy). One unresolved finding from that session: `ANDA DOZEN -> Andouille`, not root-caused.

Full detail: Normalization.md.

### Stage 4 - Expiry Prediction: COMPLETE, unchanged this session

No changes this session. Prior session: dependency chain fixed (unblocked by Stage 3's interface fix), `evaluate.py` fixed to call the real `normalize_entity()` instead of hand-building `NormalizedItem` objects (was passing 100% even with Stage 3 broken). Real eval: 16/16 correct within +/-2 days (100%).

Full detail: Expiry.md.

---

## Real findings this session, carried forward as open items

1. **Latency not yet formally re-measured end-to-end.** Only informal evidence from `test_pipeline_local.py` log timestamps: receipt 1 (20 items) showed ~12s of visible Groq-call time, receipt 2 (16 items) ~8s - but this excludes OCR/Row Reconstruction/Row Parser time (Stage 1/1.5/1.7 ran before the first logged Groq timestamp) and DB round-trip time in Stage 3/4. **Needs a real wall-clock timer around `process_receipt()`** to get an actual number against PRD.md §6's 10-second budget - not done yet, was flagged as the next step, not picked up this session.
2. **Stage 3 Pass 3 reasoning-loop gap (Issue #48, low priority):** see above. Not being actively worked, revisit only if frequency increases in real usage.
3. **Category classifier inconsistency observed in real pipeline testing (not yet fixed):** `'Pakola Flvor Mlk Strwbry 125M1'` resolved `category=Produce` while its choco sibling correctly resolved `category=Dairy` - likely a keyword-priority interaction, not investigated further. Flagged in Normalization.md §5.6, no issue filed yet.
4. **Duplicate row observed in real pipeline testing (not investigated):** `'Everyday Instnt'` appeared twice in one receipt's parsed items, causing 2 redundant Stage 2/3/4 calls for the same string. Unclear whether this is a genuine duplicate line on the physical receipt or a Stage 1.7 Row Parser bug - not root-caused, not filed as an issue yet. Worth checking before the next latency measurement, since it inflates call count for reasons unrelated to Issue #46.
5. Carried from prior session, still open: Pass 3 (`llm_fallback.py`) `ANDA DOZEN -> Andouille` gap (unrelated to this session's reasoning-loop finding - a different, semantic-resolution failure), Pass 3 free-text output testability (exact-match assertions don't fit LLM output well).

---

## Docs still needing updates (not done yet, flagged for next session)

- **ML_Pipeline.md**: §6 (Stage 2 section) still describes the old concurrency-only design consideration (`asyncio.Semaphore`/`STAGE2_CONCURRENCY`) as the plan - needs updating to reflect that batching (chunk size 5) was the approach actually shipped, and why (concurrency alone didn't solve the real TPM problem). §9's inference code structure diagram still shows `pipeline.py` as "currently empty — rewiring in progress" - it's built now (#25 closed). §9's latency budget table is stale (still says "not yet re-measured," which remains true, but the surrounding context describing an unbuilt pipeline is outdated).
- **README.md**: not reviewed this session - unknown whether it references Stage 2's architecture/concurrency approach. Check before next session if it's user-facing enough to matter.

---

## GitHub issue status (as of this handoff)

**Closed this session:** #46 (Stage 2 batch food classification, fixes Groq TPM 429s).

**Opened this session:** #48 (Stage 3 LLM reasoning loop on severely corrupted OCR input - low priority, fails safe, tracked not fixed).

**Closed prior sessions:** #26, #27, #28, #29.

**Open, unchanged from previous handoffs:**
- #16 - confidence-score garbage-line filter, unblocked, not picked up yet
- #23 - 5.jpg multi-line header + Tax(%) format, deferred
- #19 - superseded by #32 (user decision, prior session)
- #48 - Stage 3 reasoning-loop gap, low priority (new this session)

**Open, from the 10 filed two sessions ago, status:**
1. `pipeline.py` orchestrator (#25) - **DONE this session.** Built, wired end-to-end, tested against 2 real receipts. Batching (not the originally-planned concurrency-only approach) was needed to make it actually work under Groq's free-tier TPM limit - see Issue #46.
2. `unit_extractor.py` - done (#26)
3. `brand_matcher.py` - done (#27)
4. `food_classifier.py` - done (#28), extended this session (#46)
5. `extractor.py` - done (#29), extended this session (#46)
6. DB migration: add `brand` column - still not run
7. Update Pydantic schemas (`brand`, `is_food` in `/receipts/upload` response) - still not done
8. Real end-to-end validation (#32, replaces #19) - **partially done this session** via `test_pipeline_local.py` against 2 real receipts (36 items), but not formally closed/tracked as complete - worth confirming whether #32's original scope is now satisfied or needs more receipts
9. Re-measure pipeline latency end-to-end (#33) - **still not done.** See "Real findings" #1 above - next concrete step.
10. Build labeled `is_food` sample (#34) - not started, still blocking real Stage 2 accuracy measurement

---

## Immediate next steps (in order)

1. **Update ML_Pipeline.md** (§6 Stage 2 section, §9 pipeline structure + latency budget) - flagged above, not done this session, should happen before further drift.
2. **#33 - Real end-to-end latency measurement.** Add a wall-clock timer around `process_receipt()` in `test_pipeline_local.py` or `pipeline.py` itself, rerun against both existing test receipts, get a real number against PRD.md §6's 10-second budget. The informal ~8-12s figures from this session's logs are Groq-call-time-only, not the real total.
3. **Investigate the `'Everyday Instnt'` duplicate-row finding** before or alongside the latency measurement - it's inflating call count in the current test data for reasons unrelated to Issue #46's fix, would distort a fresh latency number if not understood first.
4. Once latency is measured: #6/#7 (DB/API wiring), then decide if #32 needs more real-receipt coverage or can be closed.
5. Real-receipt validation coverage: 2 of 4 available real receipts (2.jpg, 3.jpg) have now been run through the full pipeline this session. 1.jpg and 4.jpg have not yet been tried - worth running before considering #32 fully closed.

---

## Key decisions made this session (context for future reference, not to be re-litigated without new evidence)

- **Stage 2 latency/rate-limit fix: batching (chunk size 5), not concurrency.** The prior session's plan (concurrency via `asyncio.Semaphore`) was tried first as part of building #25, found insufficient in real testing (concurrency bounds parallel requests, not tokens/minute - the actual constraint), and replaced with batching. This reverses the prior session's stated preference for concurrency-first ("doesn't touch a working tested contract... batching deferred, not rejected outright") - the deferral ended once real testing showed concurrency alone didn't solve the actual problem.
- **Batch chunk size: 5**, decided over one-call-per-receipt - cuts call count ~5x (the main goal) while keeping index-tracking trivial and hallucination/misordering risk low on the batch response array. Not empirically tested at other chunk sizes.
- **Stage 3 Pass 3 stays per-item, not batched** - deliberate scope decision made alongside Stage 2's batching work. Pass 3's call volume is naturally low (cache/fuzzy-match miss only, ~11% fallback rate), and batching would trade accuracy risk for throughput this stage doesn't need.
- **`reasoning_effort="low"` added to all 3 Groq call sites** (Stage 2 single-item, Stage 2 batch, Stage 3 Pass 3) - real, documented Groq API parameter, fixes empty-content responses caused by the reasoning trace consuming the full token budget. **Does not fully eliminate the failure mode** on severely corrupted/ambiguous input - accepted as a known, low-priority, fail-safe-covered gap (Issue #48) rather than chased further, given low observed frequency (2/36 items) and confirmed correlation with physically-obstructed OCR input specifically.
- **XML-structured prompt for batch classification** (`<role>`, `<instructions>`, `<examples>`, `<output_format>`) - applied Anthropic's documented prompt-engineering guidance for prompts mixing instructions/examples/context, per explicit request this session.

Carried from prior sessions, still valid:
- Stage 2's `food_classifier.py` model: Groq `openai/gpt-oss-20b`, chosen over Gemini and 2 OpenRouter free candidates after real testing (see Item_Extraction.md).
- Stage 3/4 interface: `normalize_entity(item_name, quantity, unit, db)` is the permanent contract.
- Frozen-signal category fix: checked on raw, unstripped `item_name`, before any keyword-based category matching.
- Docs renamed: `Normalization_Training.md` -> `Normalization.md`, `Expiry_Training.md` -> `Expiry.md`. `_Training` suffix reserved for docs describing genuine model fine-tuning.
- Test-fixture errors documented as such, not silently patched to make an eval look better.
