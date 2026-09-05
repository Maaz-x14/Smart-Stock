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
- Present files at end of response as attachments, not pasted inline — so specific lines can be edited without re-pasting whole files. **Always the FULL file, never a diff/snippet** — the person applies it directly. (Diffs were tried once for doc updates and failed to apply cleanly — full files only, permanently.)

---

## Project Summary

Smart-Stock: portfolio/CV project. Reads grocery receipts, predicts expiry dates, reduces food waste. Full stack: React + TypeScript -> FastAPI -> PostgreSQL -> ML Pipeline (7 stages).

**Docs, current status:**

| Doc | Status |
|---|---|
| PRD.md, Architecture.md, API_Spec.md, DB_Schema.md, OCR_Training.md, NER_Training.md | Current as of previous handoff, unchanged this session |
| ML_Pipeline.md | Updated prior session (#33 latency numbers). Unchanged this session. |
| Item_Extraction.md | Updated prior session (#50 fix writeup). Unchanged this session. |
| Normalization.md | Updated prior session (#49/#51 fix writeups). Unchanged this session. |
| Expiry.md | Current as of previous handoff, unchanged this session |
| README.md | Rewritten prior session. Unchanged this session. |

**Naming note:** `Normalization_Training.md`/`Expiry_Training.md` renamed to `Normalization.md`/`Expiry.md` (earlier session) - neither stage trains a model (3-pass lookup + rule-based tiers), the `_Training` suffix was a leftover from NER-era naming. `OCR_Training.md` keeps its name - TrOCR fine-tuning genuinely happened there.

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

## THIS SESSION'S WORK (#52 investigation, CLOSED — no bug found in current code)

### #52 — Stage 2 fail-safe not firing on empty-content classification, CLOSED (verified correct, not a code fix)
Read `_parse_batch_response()` and `classify_is_food_batch()` in `food_classifier.py` end to end against the original report (`finish_reason='length'` + empty content -> `is_food=True` instead of `UNKNOWN`).

**Could not reproduce in current code.** Traced every failure path by hand:
- `raw is None` -> `_parse_batch_response` returns `None` immediately.
- `raw == ""` -> `json.loads("")` raises `JSONDecodeError` -> caught -> `None`.
- Truncated mid-array, wrong length, missing/duplicate index, non-bool `is_food` (int `1`/`0` correctly rejected — `isinstance(1, bool)` is `False` in Python), out-of-range confidence, non-list top level — all correctly return `None`.
- `classify_is_food_batch()`: `parsed is None` -> every item in the chunk resolves to `UNKNOWN`. No partial trust, no path that could produce `True` from malformed input.

Wrote `test_issue_52_failsafe.py` — 15 regression tests (11 unit tests directly on `_parse_batch_response`, 4 end-to-end on `classify_is_food_batch()` with the Groq call mocked, no live API/rate-limit risk). All 15 pass against current code.

**Working conclusion (labeled as such, not verified):** the original report predates this project's #50/#51 prompt+parsing hardening work. Most likely the parsing code was already this strict when #52 was filed, and whatever produced the false `is_food=True` was a different, already-fixed path (or a version-shift in this file not directly attributable to #52 specifically) — not confirmed, no historical diff was reviewed to prove this. Not chasing further: current code is verified correct and now has permanent regression coverage against this exact failure shape, which is what actually matters going forward.

Closed via issue comment + `state: closed`, `state_reason: completed`. `test_issue_52_failsafe.py` committed to repo root.

---

## Prompt iteration notes

Not applicable this session — no prompts were touched. See `Item_Extraction.md` §4.2 / `Normalization.md` §5.4 for the #50/#51 prompt-design reasoning if it needs to be revisited.

---

## Known follow-up, NOT fixed this session (very low priority)

**Pass 3 resolves multi-product brand names too eagerly.** Example: `Pakola MIk Uht 250M1` -> `'Pakola'` (pass 3, confidence 0.28, hard_default fallback). Still deferred — needs a labeled eval set to tune safely (#34), not another blind prompt edit. Unchanged, not touched this session.

---

## Stage-by-stage status (Stages 1-4 substance unchanged this session)

### Stages 1, 1.5, 1.6, 1.7 - unchanged this session. All DONE.

### Stage 2 - Item Field Extraction: no code change this session. #52 investigated and closed (no bug found) — see above.

### Stage 3 - Normalization: unchanged this session.

### Stage 4 - Expiry Prediction: unchanged this session.

---

## Real findings, carried forward as open items

1. **Latency: formally measured (#33, closed prior session).** 2.jpg consistently over the 10s budget; 3.jpg near/under it. OCR and Item Field Extraction are the two stages worth optimizing if picked up.
2. **Groq free-tier TPM is fragile under back-to-back calls** — confirmed prior session. Real pipeline runs should be spaced out.
3. **Pakola multi-product-brand gap** — see "Known follow-up" above, unchanged.
4. Carried from prior sessions, still open/unconfirmed: Pass 3 `ANDA DOZEN -> Andouille` gap, Pass 3 free-text output testability.
5. **No persisted DB/API path for Stage 2 output yet** (#30/#31, still open) — is_food/brand/unit are computed correctly but nothing downstream stores them. This is the next real blocker, not a nice-to-have.

---

## GitHub issue status (as of this handoff)

**Closed this session:** #52 (fail-safe investigated, verified already correct, closed with regression tests — not a code fix).

**Still open, unchanged from previous handoffs:**
- #16 - confidence-score garbage-line filter, unblocked, not picked up yet
- #23 - 5.jpg multi-line header + Tax(%) format, deferred
- #34 - labeled `is_food` sample, not started — needed for real is_food accuracy measurement and the Pakola follow-up
- #48 - Stage 3 reasoning-loop gap on severely corrupted OCR — still flagged as "likely improved by #50's prompt rewrite, not formally confirmed." Not rechecked this session.

**Open, from the earlier-filed set, status:**
- #30 - DB migration: add `brand` column - still not run. **Next up.**
- #31 - Update Pydantic schemas (`brand`, `is_food` in `/receipts/upload` response) - still not done. **Next up.**
- #32 - Real end-to-end validation - 2 of 4 real receipts tested (2.jpg, 3.jpg). 1.jpg, 4.jpg still not run. Closed as complete per PRD's stated acceptance criteria; this gap is a documented known limitation, not a reason it was reopened.

---

## Immediate next steps (in order)

1. **#30 + #31** — DB migration (`brand` column on `inventory_items`) + Pydantic schema updates (`brand`, `is_food` in the upload response) so Stage 2's real output can actually be persisted and returned via the API. Nothing downstream of the ML pipeline stores this yet — this is now the main blocker, everything else (eval sets, further tuning) is easier once real data exists to pull from.
2. **Build the labeled eval set (#34)** — blocks real is_food accuracy measurement and the Pakola brand-resolution follow-up. Do this once #30/#31 give a real persisted-data source, rather than hand-collecting from logs again.
3. **1.jpg/4.jpg** — cheap, closes out #32's one remaining known gap.
4. #23/#48/#16 remain deferred, low priority, unchanged.

---

## Key decisions made this session (context for future reference, not to be re-litigated without new evidence)

- **#52 closed without a code change.** Verified via manual trace + 15 regression tests that current `_parse_batch_response`/`classify_is_food_batch` already fail safe correctly on every documented malformed/empty/truncated shape. Root cause of the *original* report was not identified (no historical diff reviewed) — explicitly documented as unconfirmed rather than invented, per this project's "distinguish facts from speculation" rule. The regression suite is the durable outcome, not a fix.
- **Mocking `_call_groq_batch` for tests, not a real API call.** Keeps the regression suite fast (0.13s for 15 tests), free, and independent of Groq rate limits — appropriate here since the thing under test is pure parsing/control-flow logic, not model behavior.

Carried from prior sessions, still valid: Stage 2's `food_classifier.py` model is Groq `openai/gpt-oss-20b`; Stage 3/4 interface is `normalize_entity(item_name, quantity, unit, db)`; frozen-signal category check runs on raw unstripped `item_name`; doc renames (`Normalization.md`/`Expiry.md`); test-fixture errors documented as such, not silently patched.
