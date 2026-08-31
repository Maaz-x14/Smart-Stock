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
- Present files at end of response as attachments, not pasted inline — so specific lines can be edited without re-pasting whole files.

---

## Project Summary

Smart-Stock: portfolio/CV project. Reads grocery receipts, predicts expiry dates, reduces food waste. Full stack: React + TypeScript -> FastAPI -> PostgreSQL -> ML Pipeline (7 stages).

**Docs, current status:**

| Doc | Status |
|---|---|
| PRD.md, Architecture.md, API_Spec.md, DB_Schema.md, ML_Pipeline.md, OCR_Training.md, NER_Training.md, README.md | Current as of previous handoff, unchanged this session |
| Item_Extraction.md | Current - Stage 2, all 4 modules built, closed via #26-#29 |
| Normalization.md (renamed from Normalization_Training.md) | Rewritten this session - full depth, real bugs documented, real eval results |
| Expiry.md (renamed from Expiry_Training.md) | Rewritten this session - full depth, real bugs documented, real eval results |

**Naming note:** `Normalization_Training.md`/`Expiry_Training.md` renamed to `Normalization.md`/`Expiry.md` this session - neither stage trains a model (3-pass lookup + rule-based tiers), the `_Training` suffix was a leftover from NER-era naming. `OCR_Training.md` keeps its name - TrOCR fine-tuning genuinely happened there.

---

## Pipeline (unchanged structure, now further along)

```
Receipt Image
  -> Stage 1: OCR (PaddleOCR, pretrained) - text + bounding boxes
  -> Stage 1.5: Row Reconstruction - deskew + y-position clustering
  -> Stage 1.6: Prefilter - drops metadata rows
  -> Stage 1.7: Row Parser - header-driven column mapping -> {item_name, quantity, price, discount, total}
  -> Stage 2: Item Field Extraction - regex unit + fuzzy brand lexicon + LLM is_food gate -> {unit, brand, is_food}
       is_food=false/unknown -> surfaced to user flagged "excluded", stops here
       is_food=true  |
  -> Stage 3: Normalization - 3-pass lookup (map -> fuzzy -> LLM) -> NormalizedItem
  -> Stage 4: Expiry Prediction - 3-tier rule-based lookup -> ExpiryPrediction
  -> Structured Inventory Item
```

---

## Stage-by-stage status (updated)

### Stages 1, 1.5, 1.6, 1.7 - unchanged this session, see previous handoff / respective docs. All DONE.

### Stage 2 - Item Field Extraction: COMPLETE (closed #26, #27, #28, #29 this and prior sessions)

All 4 modules built and smoke-tested against real APIs (Groq):
- `unit_extractor.py` (#26) - regex + lookup, OCR-corruption table flagged speculative/unvalidated
- `brand_matcher.py` (#27) - lexicon-match-first order (bug fix: was strip-then-match, destroyed brands overlapping descriptor words like "Fresh St"), ~200-brand seed lexicon in `data/brand_lexicon.json`
- `food_classifier.py` (#28) - Groq `openai/gpt-oss-20b`, selected after testing 4 candidates (Groq/Gemini/2x OpenRouter). Real bug found+fixed: `max_tokens=100` was truncating long reasoning traces on ambiguous items, causing false UNKNOWN - raised to 300.
- `extractor.py` (#29) - orchestrates the three above. `remaining_text`-with-fallback-to-`item_name` design for the food classifier input. Measured latency: ~658ms/item avg (food classification dominant) - flagged as a real problem for #25, sequential calls across a 15-20 item receipt would blow PRD.md Section 6's 10-second budget.

Full detail, bug records, and worked examples: Item_Extraction.md.

### Stage 3 - Normalization: COMPLETE, interface rewritten, real bugs fixed, evaluated (this session)

**Was structurally broken** - `normalize_entity()` had the old NER-era signature `(food_tokens: list[str], raw_quantity, raw_unit)`, which could never receive what the real pipeline (Stage 1.7 -> Stage 2) actually produces. Rewritten to `normalize_entity(item_name: str, quantity: float, unit: str | None, db) -> NormalizedItem | None`.

Real bugs found and fixed this session:
1. `GROQ_MODEL = "llama-3.1-8b-instant"` in `llm_fallback.py` - confirmed deprecated by Groq (found during #28 research). Fixed to `openai/gpt-oss-20b`.
2. `CATEGORY_KEYWORDS` had genuine collisions (`"milk"` in both Dairy/Beverages; `"broccoli"`/`"mango"`/`"strawberries"` in both Produce/Frozen) - dict iteration order silently picked a winner. Fixed with a frozen-signal priority check: `assign_category()` now checks the raw (unstripped) `item_name` for `frozen`/`frzn`/`frz` before any keyword matching - using information the canonical name alone can't recover.
3. `abbreviation_map.py` had duplicate dict keys (harmless but sloppy) - deduplicated.
4. `fuzzy_matcher.py` had dead broken code (`_load_canonical_names`, always raised NotImplementedError) - deleted.
5. `preprocessor.py` duplicated Stage 2's unit-stripping - removed, now only strips quality/dietary modifiers.
6. `unit_normalizer.py` interface simplified to canonicalization-only (`normalize_unit(unit) -> unit`), since Stage 2 already extracts units.

All data extracted to `data/*.json` files: `abbreviation_map.json` (~580 entries), `category_keywords.json`, `unit_map.json`, `eval_test_cases.json`.

**Real eval run (first time, live DB + real Groq):** 36 test cases, 32/36 exact-match correct (88.9%), but 3 of the 4 failures are test-fixture errors (LLM gave reasonable-but-differently-phrased answers: "Strawberry" vs "Strawberries", "Yogurt" vs "Plain Yogurt", correct Frozen category but compound-name mismatch on "Frozen Broccoli"). Corrected accuracy: 35/36 = 97.2%. One real unresolved finding: `ANDA DOZEN -> Andouille` (Pass 3 LLM reliability gap, not root-caused).

Full detail: Normalization.md.

### Stage 4 - Expiry Prediction: COMPLETE, core logic was already sound, dependency chain fixed, evaluated (this session)

`predict_expiry()`'s tiered lookup logic (exact -> category median -> hard default) did not need a rewrite. Two real problems, both fixed:
1. Depended on `NormalizedItem` from the broken Stage 3 interface - unblocked once Stage 3 was fixed.
2. `evaluate.py` was not actually an integration test - it hand-built `NormalizedItem` objects directly, bypassing `normalize_entity()` entirely. Would have passed 100% even with Stage 3 completely broken. Fixed: now calls the real `normalize_entity()` first.

`CATEGORY_DEFAULT_STORAGE` extracted to `data/category_default_storage.json`.

**Real eval run (first genuine Stage 3+4 integration test):** 16 cases, 16/16 correct within +/-2 days (100%), avg confidence 0.934, 0 flagged for review. Confirms the tiered fallback and confidence-propagation math (`final_confidence = tier_base x item.confidence`) work correctly.

Full detail: Expiry.md.

---

## Real findings this session, carried forward as open items

1. **Stage 2 latency problem (#25 blocker):** ~658ms/item avg for food classification, sequential. A 15-20 item receipt would take 10+ seconds for Stage 2 alone, breaking the PRD's 10-second upload budget. Decision made: start with Option A (concurrent/async calls per receipt), not batching. Concurrency doesn't touch the already-tested single-item `food_classifier.py` contract; batching would require a prompt/parser redesign, untested. Groq's `openai/gpt-oss-20b` free tier: RPM 30, RPD 1K, TPM 8K, TPD 200K - concurrency alone will need real load-testing against these limits before calling it production-ready; batching may become necessary later, not deciding that now.
2. **Pass 3 (`llm_fallback.py`) reliability gap:** `ANDA DOZEN -> Andouille`, unexplained, not root-caused. Watch for a pattern.
3. **Pass 3 testability:** free-text LLM output can't be reliably tested with exact-match string assertions - a structural limitation, not a bug. Future eval work should consider fuzzy/semantic matching for Pass 3 cases specifically.

---

## GitHub issue status (as of this handoff)

**Closed this session (or prior, confirmed closed):** #26, #27, #28, #29.

**Open, unchanged from previous handoff:**
- #16 - confidence-score garbage-line filter, unblocked, not picked up yet
- #23 - 5.jpg multi-line header + Tax(%) format, deferred
- #19 - user is deleting this directly (superseded by #32, per user decision last session)

**Open, from the 10 filed last session, status:**
1. `pipeline.py` orchestrator (#25) - NEXT UP. Needs Stage 2's concurrency design (see above) built in, not bolted on after.
2. `unit_extractor.py` - done (#26)
3. `brand_matcher.py` - done (#27)
4. `food_classifier.py` - done (#28)
5. `extractor.py` - done (#29)
6. DB migration: add `brand` column - still not run
7. Update Pydantic schemas (`brand`, `is_food` in `/receipts/upload` response) - still not done
8. Refile end-to-end validation (#32, replaces #19) - open, not started; now realistically unblocked since Stage 2/3/4 all individually work
9. Re-measure pipeline latency end-to-end (#33) - partially informed by Stage 2's ~658ms/item finding, but no full pipeline run yet
10. Build labeled `is_food` sample (#34) - not started, still blocking real Stage 2 accuracy measurement

---

## Immediate next steps (in order)

1. **#25 - `pipeline.py` orchestrator.** Wire Stage 1 -> 1.5 -> 1.6 -> 1.7 -> 2 -> 3 -> 4, with `is_food` gating (Stage 3/4 only run when Stage 2's `is_food` is True; False/None/unknown both route to "surfaced, excluded"). Must include concurrent Stage 2 food-classification calls per receipt (Option A, per above) - this is a design requirement for #25, not an afterthought.
2. Once #25 exists and runs: #6/#7 (DB/API wiring), then #8/#9/#10 (validation/measurement now that there's a real pipeline to measure).
3. Real-receipt validation (1-4.jpg) for all of Stage 2/3/4 - flagged as a known gap in every doc this session, still not done. Real test data, not hand-picked strings.

---

## Key decisions made this session (context for future reference, not to be re-litigated without new evidence)

- Stage 2's `food_classifier.py` model: Groq `openai/gpt-oss-20b`, chosen over Gemini and 2 OpenRouter free candidates after real testing (see Item_Extraction.md). Not locked in blind.
- Stage 2 latency fix: concurrency (Option A) before batching (Option B) - additive, doesn't touch a working tested contract. Batching deferred, not rejected outright.
- Stage 3/4 interface rewrite: `normalize_entity(item_name, quantity, unit, db)` replaces the old NER-era signature. This is now the permanent contract Stage 3 expects from the pipeline orchestrator.
- Frozen-signal category fix: checked on raw, unstripped `item_name`, before any keyword-based category matching - not a keyword-list edit, a structural fix, because the signal is lost once "frozen" gets stripped as a quality modifier.
- Docs renamed: `Normalization_Training.md` -> `Normalization.md`, `Expiry_Training.md` -> `Expiry.md`. Naming convention going forward: `_Training` suffix only for docs describing genuine model fine-tuning (OCR_Training.md, NER_Training.md - historical).
- Test-fixture errors are documented as such, not silently patched to make an eval look better - both the raw and corrected accuracy numbers are reported in Normalization.md.
