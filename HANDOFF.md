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
| Item_Extraction.md | Unchanged this session - still reflects Issue #46 batch classification work. **Needs update**: Stage 2 prompt rewrite (this session, #50) not yet documented. |
| Normalization.md | Unchanged this session. **Needs update**: §9.1's real-pipeline validation example still cites `Tapal Tea Bags -> 'Dried Lychee'` as a correct resolved example — this is now WRONG, it was one of the bugs fixed this session (#51). §5.4/§10 reasoning-loop content needs a note that Pass 3's prompt was rewritten this session (UNKNOWN escape hatch + brand-is-product distinction). |
| Expiry.md | Current as of previous handoff, unchanged this session |
| ML_Pipeline.md | Still needs update from before (Stage 2 concurrency->batching, §9 structure diagram) - not done this session either, still stale. |

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

## THIS SESSION'S WORK (real-receipt testing surfaced 4 distinct bugs, all fixed/closed)

Continuing from prior session's real-receipt pipeline testing (2.jpg, 3.jpg via `test_pipeline_local.py`). Found and worked through:

### #49 — CLOSED (not a bug)
Two sub-issues, both investigated and resolved:
- **Duplicate row** ('Everyday Instnt' x2 on 3.jpg): row-dump debug (raw_token + y-coord) confirmed 76px y-separation, each with its own independent price row (qty 24, Rs936 each). Genuine duplicate line printed on the physical receipt, not an OCR/row-reconstruction bug. No dedup logic added.
- **Category classifier inconsistency** ('Pakola Strwbry Milk' -> category=Produce instead of Dairy): root-caused to substring matching in `category_classifier.py`'s `assign_category()` — Produce's keyword "berry" is a substring of "strawberry", matched before Dairy's "milk" was checked (dict insertion order + `kw in name_lower` logic). Fixed with word-boundary matching (`\bkw\b`). Verified fixed in pipeline output.

### #51 — Pass 3 hallucination on real receipt tokens, FIXED, ready to close on PR merge
Root cause: `llm_fallback.py`'s original one-line Pass 3 prompt ("Reply with just the canonical food name, nothing else") had no abstention path, forcing a confident-sounding wrong guess on abbreviated/ambiguous tokens. Real examples: `Tapal Tea Bags Dnedr Elchi` -> `'Dried Lychee'`, `Everyday Instnt` -> `'Instant Noodles'`, `Everyday Tea Whtnr` -> `'White Tea'`.

Two fixes, both required (neither alone was sufficient):
1. **`abbreviation_map.py`**: Pass 1 only tried the whole cleaned string as one key (exact/collapsed). Added token-scan (bigram-then-unigram, left to right) so a known key embedded in a longer noisy string (e.g. "TAPAL TEA BAGS DNEDR ELCHI 50'S" containing "TAPAL" and "TEA BAGS") resolves at Pass 1, never reaching Pass 3 at all. This is the primary fix — most of #51's examples now resolve here.
2. **`llm_fallback.py`**: rewrote the Pass 3 prompt through several iterations (see "prompt iteration notes" below) to add an explicit `UNKNOWN` escape hatch, and a brand-resolution rule distinguishing single-product brands (e.g. "Milo" — resolve directly) from ambiguous/multi-product or non-food brand names (e.g. "Kimtiaz", a mall name — UNKNOWN).

**Cache gotcha hit and fixed during this work:** `NormalizationCache` (keyed on `raw_token.upper()`) caches Pass 3 results and is checked *before* Groq is called. Three different prompt rewrites in a row produced byte-identical wrong output for "Everyday Instnt" because the cache was serving a stale pre-fix result the whole time — not a prompt problem. Added `check_cache.py` (pushed to main directly, not part of the #51/#50 PR) to inspect/clear cache entries. **Lesson for next session: clear relevant cache entries before concluding a Pass 3 prompt change "didn't work."**

### #50 — Stage 2 reasoning-loop / fail-safe, FIXED, ready to close on PR merge
Root cause: `food_classifier.py`'s batch/single prompts had a "do not refuse, still output your best guess" instruction that pushed the model to keep reasoning toward a forced answer instead of bailing early on unrecognizable garbage — directly causing `finish_reason='length'` reasoning-loops on items like `'Kimtiaz'` (a mall name) and `'Peek Frns Cocnt Crnch Farm Hose F/P'` (severely garbled OCR), both burning the full token budget and returning empty content.

Rewrote both prompts (batch `BATCH_SYSTEM_PROMPT` and single-item `SYSTEM_PROMPT`) through 2 iterations:
- v1: added an instruction to output `is_food=false` immediately on garbled/unrecognizable text. **Caused a real regression** — over-applied to noisy-but-legible items (`Bush Essence Mango 28M1`, `Milo Drnk 180M1` both got dropped as non-food, previously correct).
- v2 (current): sharpened the rule to explicitly distinguish "OCR noise/abbreviation attached to a real food/brand word" (still food — the noise elsewhere isn't disqualifying) from "no food/brand word present anywhere" (not food). Added a 3rd few-shot example set built from this session's real failure cases.

**Confirmed via 2 full pipeline reruns:** Kimtiaz and Peek Frns both correctly `is_food=False` now, no more reasoning-loop, no regression on Bush Essence Mango / Milo / any other previously-correct item.

### New issue this session (see #52 below): Pass 3 brand-resolution over-eager on multi-product brands
Not fixed this session, deliberately deferred (see "known follow-up" below).

---

## Prompt iteration notes (why this took several passes — context if the pattern needs to be understood again)

Both `food_classifier.py` and `llm_fallback.py` prompts went through multiple rewrites this session. Pattern observed each time: fixing one failure mode (forced-guess-on-garbage) risked introducing the opposite failure mode (over-cautious UNKNOWN/false on genuinely resolvable items). The fix that stuck, both times, was **not** more/stricter prose instructions — it was **adding a clearer boundary rule** ("does a real food/brand word exist in the text, yes or no") **plus concrete few-shot examples drawn from actual observed failures**, rather than abstract instruction-stacking. Any future prompt tuning on these two files should test against the full set of real examples now baked into both prompts' few-shot sections (Kimtiaz, Peek Frns, Tapal, Bush Essence Mango, Milo, Everyday Instnt) before considering a change safe.

---

## Known follow-up, NOT fixed this session (very low priority)

**Pass 3 resolves multi-product brand names too eagerly.** Example: `Pakola MIk Uht 250M1` -> `'Pakola'` (pass 3, confidence 0.28, hard_default fallback). The brand-is-product rule added for Milo (Milo makes effectively one product, so "Milo" alone is a valid canonical name) over-applies to brands like Pakola that sell multiple distinct products (milk, sodas, juices) — "Pakola" alone isn't a specific food. Severity is low: lands at 0.28 confidence, same low-trust bucket as UNKNOWN, not a wrong-but-confident hallucination like the original #51 bugs. Deliberately not fixed this session to avoid a 5th prompt iteration without a proper eval set. **Next step if picked up: needs a labeled eval set first** (see "Immediate next steps" below) — do not attempt another blind prompt edit on this.

---

## Stage-by-stage status (Stages 1-4 substance unchanged this session except as noted above)

### Stages 1, 1.5, 1.6, 1.7 - unchanged this session. All DONE.

### Stage 2 - Item Field Extraction: batching (Issue #46) from prior session, prompt rewrite (#50) this session
See "THIS SESSION'S WORK" above for #50. Batching/chunking design (chunk size 5, `extract_item_fields_batch()`) unchanged from prior session.

### Stage 3 - Normalization: Pass 1 token-scan + Pass 3 prompt rewrite (#51) this session
See "THIS SESSION'S WORK" above. `normalize_entity()` interface and Pass 1/2 core logic otherwise unchanged from prior sessions.

### Stage 4 - Expiry Prediction: unchanged this session.

---

## Real findings, carried forward as open items

1. **Latency: partially measured this session, not yet meeting budget on all receipts.** `process_receipt()` wall-clock via `test_pipeline_local.py`: 2.jpg ranged 12-16s across reruns (varies with cache state / Groq call count), 3.jpg ranged 7-14s. PRD.md §6's budget is 10s. 2.jpg is consistently over budget; 3.jpg is borderline, sometimes under. **No sub-stage breakdown yet** — can't tell how much is OCR vs Stage 2 batch calls vs Stage 3 Pass 3 calls vs DB round-trips. This is Issue #33, still open, still needs sub-stage timers before any latency optimization is proposed.
2. **Groq 429 rate-limiting observed during this session's testing** (3.jpg run), auto-retried successfully by the client on a 1-5s backoff. Not fatal, but resurfacing under load — same rate-limit class as the original #46 batching fix. Not investigated further this session, flag if it starts failing (not just retrying) in future runs.
3. **Pakola multi-product-brand gap** — see "Known follow-up" above.
4. Carried from prior sessions, still open/unconfirmed: Pass 3 `ANDA DOZEN -> Andouille` gap (unrelated semantic-resolution failure, not investigated this session), Pass 3 free-text output testability.

---

## GitHub issue status (as of this handoff)

**Closed this session:** #49 (not a bug — duplicate row confirmed genuine, category classifier bug fixed).

**Fixed this session, closing on PR merge:** #51 (Pass 3 hallucination), #50 (Stage 2 reasoning-loop/fail-safe).

**Opened this session:**
- #52 — Stage 2 fail-safe not firing on empty-content classification (`is_food=True` instead of `UNKNOWN` on `finish_reason='length'` malformed responses). Suspected cause: `extractor.py`'s `_parse_batch_response()` likely only validates array-shape, not per-item empty content. **Not yet investigated — needs a read of the batch-parsing code before any fix is proposed.** This is a different, more serious failure mode than #50 (unsafe-direction fail-safe gap vs. reasoning-loop inefficiency) — was originally going to be filed but the tool call was declined; **still needs to be filed at the start of next session if not already done.**

**Still open, unchanged from previous handoffs:**
- #16 - confidence-score garbage-line filter, unblocked, not picked up yet
- #23 - 5.jpg multi-line header + Tax(%) format, deferred
- #33 - latency re-measurement, partially done this session (see "Real findings" #1), sub-stage timers still needed
- #34 - labeled `is_food` sample, not started — now also needed for the Pakola follow-up, not just Stage 2 accuracy measurement
- #48 - Stage 3 reasoning-loop gap on severely corrupted OCR — **likely improved/resolved as a side effect of #50's prompt rewrite, not confirmed.** Worth a quick recheck next session rather than assuming still-open.

**Open, from the earlier-filed set, status:**
6. DB migration: add `brand` column - still not run
7. Update Pydantic schemas (`brand`, `is_food` in `/receipts/upload` response) - still not done
8. Real end-to-end validation (#32) - 2 of 4 real receipts tested (2.jpg, 3.jpg) across this and prior session. 1.jpg, 4.jpg still not run.

---

## Immediate next steps (in order)

1. **File issue #52 properly** (see above — attempted this session, tool call declined by user, needs to actually be filed) before starting new work, so it isn't lost.
2. **Build the labeled eval set** (blocks both #34 and the Pakola follow-up). This was flagged multiple times this session as the right next move instead of further blind prompt iteration — a small set of real items with known-correct answers (Kimtiaz->not food, Milo->Milo, Pakola Milk->?, Tapal->Tea Bags, Bush Essence Mango->Mangoes, etc.) run as a batch eval, not one-off pipeline reruns eyeballed by hand.
3. **#33 — sub-stage latency timers.** Wall-clock total is now known (partially) but not broken down. Needed before proposing any latency fix, especially since 2.jpg is consistently over the 10s budget.
4. **Recheck #48** — may already be resolved by #50's prompt rewrite (same reasoning-loop failure mode, same root symptom). Quick confirm, don't assume.
5. **Docs**: Normalization.md §9.1 has a now-incorrect example (Tapal -> Dried Lychee cited as correct) that needs fixing before anyone reads it and gets confused. ML_Pipeline.md still stale from before, unresolved multiple sessions running.
6. Once latency is measured: #6/#7 (DB/API wiring), then 1.jpg/4.jpg real-receipt validation, then decide if #32 can close.

---

## Key decisions made this session (context for future reference, not to be re-litigated without new evidence)

- **Prompt fixes over data fixes for #51/#50.** Both root causes were prompt-design issues (forced guessing, no abstention path), not missing abbreviation-map coverage or missing training data. Explicitly decided NOT to extend `abbreviation_map.json` — the gap was Pass 1's matching *strategy* (whole-string only), not missing entries; the map already had `TAPAL`/`TEA` as valid keys before the token-scan fix.
- **Word-boundary/anchor-word matching over more few-shot examples, when the two conflict.** Each prompt rewrite that just added more prose or more examples without sharpening the underlying rule tended to regress something else. The stable fixes were the ones that named a clear, checkable distinction (word-boundary in Pass 1, brand-is-product vs. brand-is-company in Pass 3, anchor-word-present vs. absent in Stage 2).
- **Deferred the Pakola-style multi-product-brand gap rather than iterating a 5th time blind.** Explicit call: needs a labeled eval set to tune safely, not another single-example prompt patch.
- **`check_cache.py` added as a standing utility**, not a one-off script — pushed directly to main. Any future Pass 3 prompt change should include a cache-clear step in the test process, not just a code change.

Carried from prior sessions, still valid: Stage 2's `food_classifier.py` model is Groq `openai/gpt-oss-20b`; Stage 3/4 interface is `normalize_entity(item_name, quantity, unit, db)`; frozen-signal category check runs on raw unstripped `item_name`; doc renames (`Normalization.md`/`Expiry.md`); test-fixture errors documented as such, not silently patched.
