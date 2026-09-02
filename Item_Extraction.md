# Item_Extraction.md — Stage 2: Item Field Extraction

**Version:** 1.1  
**Status:** ✅ Complete. All four Stage 2 modules implemented and smoke-tested: `unit_extractor.py`, `brand_matcher.py`, `food_classifier.py`, `extractor.py`. Batch classification (Issue #46) validated against 2 real receipts via `test_pipeline_local.py` — 36 items total, correct classification, zero Groq 429s. Still not validated against the real #34 labeled dataset.

---

## 0. What this stage replaced

Stage 2 was originally "NER" — a fine-tuned DistilBERT model tagging FOOD/QTY/UNIT/PRICE within a receipt line (F1 0.907, see NER_Training.md). Once the header-driven Row Parser (Stage 1.7) started extracting qty/price/discount/total structurally, NER's remaining job shrank to just what's left inside `item_name`: unit, brand, and whether the item is food at all. That doesn't need a trained model — regex, a fuzzy lexicon, and an LLM gate cover it with no labeled data and no served model. Full reasoning: ML_Pipeline.md §0, HANDOFF.md.

---

## 1. Position in the pipeline

```
Stage 1.7 Row Parser output
  { item_name, quantity, price, discount, total }
        │
        ▼
┌───────────────────────────────────────────┐
│  Stage 2: Item Field Extraction           │
│                                           │
│  item_name                                │
│     │                                     │
│     ▼                                     │
│  unit_extractor.py  ──▶ unit, remaining   │
│     │                                     │
│     ▼                                     │
│  brand_matcher.py   ──▶ brand, remaining  │
│     │                                     │
│     ▼                                     │
│  food_classifier.py ──▶ is_food           │
│     │                                     │
│  extractor.py (orchestrates the above)    │
│     ▼                                     │
└────────────────────┬─────────────┬────────┘
        is_food=false│             │is_food=true
                     ▼             ▼
              surfaced to    Stage 3 Normalization
              user, flagged  Stage 4 Expiry
              "excluded",
              stops here
```

**Contract in:** `item_name: str` (raw text from Row Parser, e.g. `"ORG STRWBRY 1LB"`)
**Contract out:** `{ unit: str | None, brand: str | None, is_food: bool }`

`is_food=false` items never reach Normalization/Expiry — surfaced to the user flagged "detected but excluded," per API_Spec.md §2 / PRD.md §5.1.

---

## 2. Module: `unit_extractor.py`

**Status:** ✅ Built, smoke-tested. Closed via #26.

**Approach:** Regex + lookup. No model, no training data.

**Input:** `item_name: str`
**Output:** `UnitExtractionResult { unit: str | None, matched_span: str | None, remaining_text: str }`

### Design

1. Canonical unit vocabulary (kg, g, lb, oz, l, ml, gal, pcs, dozen, pack, container) with known surface-form variants.
2. Fused number+unit regex match (`\d+(\.\d+)?\s?(unit)\b`) — real receipt samples show quantity and unit fused with no space (`1LB`, `500ML`).
3. No confident match → `unit = None`. Never guesses.
4. Matched span is stripped from `item_name`, producing `remaining_text` for `brand_matcher.py` to consume next.

### OCR corruption handling — flagged as speculative

The unit vocabulary includes surface-form variants for likely OCR misreads (`l`→`i`, `g`→`q`, `o`→`0`), built from generic visual-confusion character pairs — **not derived from real Smart-Stock receipt OCR output.** Two initially-considered entries (`"9"→g`, `"1"→l`) were explicitly **rejected** — too high a collision risk against legitimate quantity digits in an inventory app. Documented and gated out in code rather than silently omitted.

**Open item:** validate the corruption table against real PaddleOCR output from the 4 real receipt samples (1-4.jpg). Currently untested against real data.

### Example

| Input | unit | remaining_text |
|---|---|---|
| `ORG STRWBRY 1LB` | `lb` | `ORG STRWBRY` |
| `WHOLE MILK 1GAL` | `gal` | `WHOLE MILK` |
| `Supravit-M Tablet 10's` | `None` | `Supravit-M Tablet 10's` |
| `CHICKEN 500G` | `g` | `CHICKEN` |

---

## 3. Module: `brand_matcher.py`

**Status:** ✅ Built, smoke-tested, iterated after a real design bug was caught. Closed via #27.

**Approach:** Hybrid — curated lexicon (external data file) + fuzzy matching. Explicitly **not** a semantic brand classifier; deterministic and confidence-gated.

**Input:** `remaining_text: str` (output of `unit_extractor.py`)
**Output:** `BrandExtractionResult { brand: str | None, remaining_text: str }`

### Design

```
tokenize(remaining_text)
        │
        ▼
normalize (lowercase, "&"↔"and", strip punctuation variance)
        │
        ▼
build 1/2/3-gram windows over tokens
        │
        ▼
fuzzy match windows against brand_lexicon.json  (threshold: 88, untuned)
        │
   match found? ──── yes ──▶ brand = canonical name
        │                    strip matched window from remaining_text
        no
        │
        ▼
strip descriptor stopwords (cleanup only — NOT a pre-filter)
        │
        ▼
brand = None
```

### Bug fixed this session: match order

**Original (wrong) order:** strip descriptors → lexicon match.
**Problem:** real brands overlap descriptor words — `Fresh St` (Al-Fatah) and `Fresh Choice` (Imtiaz) both contain "Fresh," which would be stripped as a descriptor before the lexicon ever saw the token, silently destroying a real brand match.
**Fix:** lexicon match now runs first, against the full unstripped token set. Descriptor stripping only happens as cleanup on the no-match path, never as a gate.

### Punctuation normalization

`"Head & Shoulders"`, `"HEAD AND SHOULDERS"`, and `"HEAD SHOULDERS"` all resolve to the same canonical brand — `&` is normalized to `and`, and apostrophes/hyphens are stripped before comparison, applied identically to both lexicon surface forms and input tokens.

### Data file: `data/brand_lexicon.json`

Lexicon was deliberately split out of the Python module into its own JSON file rather than buried inline — see file's `_meta` field for provenance. ~200 brands, sourced from:
- Maaz's manual curation from browsing Imtiaz/Al-Fatah store listings
- Web search for major Pakistani/international FMCG brands (Aug 2026)

**Not a complete Pakistan brand database — a growing seed list.** Unverified against real receipt OCR output. Lexicon growth is a **manual, offline** process — edit the JSON file after human review of real data. `brand_matcher.py` does not auto-grow itself; letting OCR noise write directly into the production lexicon was explicitly rejected as a design (see §3.1 below).

### Descriptor stopwords

Used only for `remaining_text` cleanup on the no-match path — never as a pre-match filter (see bug above). Covers freshness/origin, quality/marketing, size, dietary, and packaging terms.

**Deliberately excludes food nouns** (`chicken`, `milk`, `rice`, etc.) — those are the product name, not a descriptor, and belong to Stage 3 Normalization's concern. Including them here risked both false-stripping legitimate product names and colliding with brand substrings (e.g. "milk" inside "Milkpak").

Unverified against real data — same caveat as the lexicon.

### 3.1 Decision: candidate/heuristic detection removed from production path

An earlier version of this module included a "conservative candidate" heuristic: any leftover token that was alphabetic, ≥3 characters, and not a stopword was flagged as a low-confidence brand candidate for manual review.

**Removed after smoke testing showed it can't distinguish an unmatched brand from the food noun itself** — e.g. `"ORG STRWBRY"` (no brand present) flagged `"STRWBRY"` as a candidate, which is simply the food name. Without a food-name lexicon to exclude against, every leftover token looks like a brand candidate.

`extract_brand()` is now lexicon-only: no match → `brand = None`, full stop. If candidate discovery is wanted later, it belongs in a **separate, explicitly offline/low-confidence function** — never mixed into the production extraction path.

### Example

| Input (remaining_text) | brand | remaining_text out |
|---|---|---|
| `NESTLE MILK` | `Nestle` | `MILK` |
| `ORG STRWBRY` | `None` | `ORG STRWBRY` |
| `FRESH ST BROCCOLI` | `Fresh St` | `BROCCOLI` |
| `HEAD AND SHOULDERS SHAMPOO` | `Head & Shoulders` | `SHAMPOO` |
| `BEEF MINCE` | `None` | `BEEF MINCE` |

### Known gaps

- Lexicon and stopword list unverified against real receipts.
- Fuzzy threshold (88) untuned — no real data to tune against yet.
- Matching cost is O(tokens² × lexicon size) per item (1/2/3-gram windows × ~200+ surface forms) — untested for latency at pipeline scale; flag for #33 (latency re-measurement) once wired in.

<img src="assets/svgs/brand_matcher_flow.svg" alt="Pipeline" width="500" />

---

## 4. Module: `food_classifier.py`

**Status:** ✅ Built, smoke-tested against the real Groq API (not mocked). Closed via #28.

**Approach:** LLM gate, binary classification (`is_food: bool`). No labeled data exists for this exact task shape — the old DistilBERT NER was trained for full-line entity tagging, which doesn't transfer to a single yes/no classification.

**Input:** `remaining_text` (post unit + brand extraction — closer to the bare product name, e.g. `"BROCCOLI"`, `"Supravit-M Tablet"`)

**Output:** `is_food: bool`

**Distinct from Stage 3 Pass 3's LLM call** — this asks "is this food at all," Stage 3 Pass 3 asks "what's the canonical name for this already-confirmed food item." Different questions, deliberately not merged (flagged in `HANDOFF.md` as a possible future optimization, not decided).

**Not yet decided:**

- Confidence threshold — deferred until real confidence-score distributions exist (see Phase 4). `classify_is_food()` returns the raw confidence value; thresholding into `unknown` based on low confidence is not yet applied anywhere.

### Fallback / unknown-state design (Phase 5 — implemented)

Any API failure, timeout, or malformed response → immediate `UNKNOWN` outcome. **No retry** — deliberate choice, not a shortcut. `UNKNOWN` is surfaced to the user identically to `is_food=False` (reuses the existing "detected but excluded" flow from API_Spec.md §2) rather than inventing a new UI state or silently dropping/accepting the item.

Verified against real failure conditions (bad/missing API key during testing) — correctly returned `UNKNOWN`, not a crash or a silent guess.

No explicit request timeout override — uses SDK default.

### Reasoning field (found during smoke testing, not originally planned)

Groq's `openai/gpt-oss-20b` returns a reasoning trace on a **separate `reasoning` field**, not inline in `content` — `content` stays clean, parseable JSON regardless. This is expected model behavior, not a prompt failure. The reasoning trace is now captured on `FoodClassificationResult.reasoning` for debugging/audit — not parsed or relied on for the actual decision, `content` remains the source of truth.

### Smoke test result

4/4 correct against real Groq API calls (not the original 15-item Phase 1 sample — a smaller manual re-check post-implementation):

| Item | outcome | confidence |
|---|---|---|
| `ORG STRWBRY` | food | 0.95 |
| `Supravit-M Tablet 10's` | not_food | 0.9 |
| `CHICKEN BREAST` | food | 0.95 |
| `Dettol Antiseptic` | not_food | 0.99 |

**Known gap:** `raw_response` only stores `.content`; full completion metadata (token usage, etc.) is not logged. Not currently needed, flagged as a future logging opportunity if debugging requires it.

**Related, not blocking implementation:** #34 — build a labeled `is_food` sample to actually measure this gate's accuracy at scale. No labeled data exists yet; the 15+4 items tested so far are hand-picked sanity checks, not a real accuracy measurement.

### Batch classification (Issue #46 — added this session)

**Problem:** `classify_is_food()` alone makes one Groq call per item. At 15-20 items/receipt this exhausted Groq's free-tier TPM (8000/min) — confirmed via `x-ratelimit-remaining-tokens` dropping toward zero mid-receipt and a wall of 429s in `test_pipeline_local.py`. An earlier fix attempt (`STAGE2_CONCURRENCY`, bounding parallel requests via `asyncio.Semaphore`) did not solve this — concurrency bounds parallel requests, not tokens/minute, so the same total token volume still blew the TPM budget regardless of concurrency setting.

**Fix:** `classify_is_food_batch(item_names: list[str])` sends N items in one Groq call; `classify_is_food_chunked(item_names, batch_size=5)` splits a full receipt into chunks of 5 and calls `classify_is_food_batch` per chunk. Chunk size 5 chosen over one-call-per-receipt: cuts call count ~5x (the main goal) while keeping index-tracking trivial and hallucination/misordering risk low on the array response — a larger single batch (e.g. all 20 items) was judged higher-risk for the model losing track of index alignment, though this wasn't empirically tested at larger sizes.

**Batch prompt design:** XML-structured (`<role>`, `<instructions>`, `<examples>`, `<output_format>`), per Anthropic prompt-engineering guidance for prompts mixing instructions/examples/context. Two multishot examples included. Key rules beyond the single-item prompt: strict input/output length match, 1-based index correspondence, and an explicit instruction not to let earlier items bias judgment on later ones ("no theme bias").

**Fail-safe parsing:** `_parse_batch_response()` requires exact array length match and valid, unique 1-based indices covering every position — any deviation (wrong length, duplicate index, malformed entry) fails the **entire batch** to `UNKNOWN`, not just the malformed entries. No partial trust within a batch, matching the existing single-item philosophy of "no partial trust, no guessing at malformed output."

**`reasoning_effort="low"` also added** to both single-item and batch Groq calls in this same change — a separate, real bug found during Issue #46 testing: gpt-oss-20b's reasoning trace was observed consuming the *entire* token budget on certain prompts, producing empty content that failed downstream parsing regardless of rate limiting. `reasoning_effort` is a real, documented Groq API parameter. **Caveat found in follow-up testing (see below): this reduces but does not eliminate the failure mode** — see Section 4.2.

**Validation:** 2 real receipts via `test_pipeline_local.py`, 36 items total (20 + 16). Zero Groq 429s in either run — Groq's `x-ratelimit-remaining-tokens` header stayed healthy throughout (never dropped below ~3300 of 8000). All non-food/junk OCR lines correctly excluded; all resolvable food items correctly classified `is_food=True`.

### 4.2 Known gap: reasoning-loop failure on severely corrupted OCR input

`reasoning_effort="low"` does not fully prevent Groq's `gpt-oss-20b` from entering a token-exhausting reasoning loop on certain inputs. Observed on 2/36 tested items, both with severely corrupted OCR (a hand blocking part of the physical receipt during photo capture): `'Kimtiaz'` and `'Peek Frns Cocnt Crnch Farm Hose F/P'`.

Confirmed via response metadata (`finish_reason='length'`, `completion_tokens=300` = the max_tokens ceiling, `reasoning_tokens=298` of that 300): the model's reasoning trace enters a literal repetition loop ("Could be X but maybe Y?? Could be X but maybe Y??" repeated near-verbatim 5-6 times) and never reaches content before hitting the token limit. This is a genuine model behavior on ambiguous input, not a config/prompt bug — `reasoning_effort` caps *effort level*, not a hard reasoning-token ceiling, and Groq did not bound this specific failure mode.

This affects Stage 2's `food_classifier.py` and Stage 3's `llm_fallback.py` identically (same model, same failure class). In the observed cases it surfaced via Stage 3 (the item passed Stage 2's `is_food=True` gate but then failed to normalize in `llm_fallback.py`). Both stages fail safe correctly on this failure — item surfaced to the user with null downstream fields, never dropped or silently wrong.

**On a second test receipt with cleaner OCR (no physical obstruction), 16/16 items resolved with zero failures of this kind** — supports the working conclusion that this is specifically triggered by genuinely unresolvable/severely corrupted input, not a systemic issue with the fix. Tracked as a separate low-priority follow-up (GitHub Issue #48) rather than reopening #46 — not chasing further right now given low observed frequency and correct fail-safe behavior.

## Model selection (Phase 1 — completed)

Original 7-candidate shortlist was cut down before testing: several (Qwen3 14B/8B, Gemma 3 12B, Nemotron Nano 9B V2, GLM-4.5-Air) turned out to be stale/unconfirmed on Groq's own docs, and Llama 3.1 8B Instant was found **deprecated by Groq on 2026-06-17** (replaced by GPT-OSS 20B). Broader OpenRouter free-tier browsing also surfaced mostly large agentic/reasoning models, embeddings, and TTS — wrong shape for a small binary classifier. Final tested set: 4 candidates across Groq, Gemini, and OpenRouter.

**Test:** 15-item hand-labeled sample (not the real #34 dataset — a stand-in for pre-selection sanity checking), locked prompt (see Phase 2), strict JSON-only output required.

| Candidate | Valid JSON | Correct | Avg latency | Result |
|---|---|---|---|---|
| **Groq: `openai/gpt-oss-20b`** | 15/15 | 15/15 | 1037ms | ✅ **Selected** |
| Gemini: `gemini-3.5-flash-lite` | 14/15 | 14/15 | 2808ms | ❌ Rate-limited (15 RPM free tier — too tight for a receipt's worth of items in one upload) |
| OpenRouter: `google/gemma-4-26b-a4b-it:free` | 0/15 | 0/15 | N/A | ❌ Upstream 429, shared free-pool rate limit |
| OpenRouter: `liquid/lfm-2.5-2.6b:free` | 0/15 | 0/15 | N/A | ❌ Unreliable — truncated/malformed JSON and `None` responses independent of rate limits |

**Decision: Groq `openai/gpt-oss-20b`.** Clean sweep — no errors, no rate-limit friction, fastest and only 100%-correct candidate on the sanity sample. The other three failed on infrastructure reliability, not classification ability.

Provider abstraction (`FOOD_CLASSIFIER_PROVIDER`/`FOOD_CLASSIFIER_MODEL` config) is kept regardless, so a fallback provider can be swapped in later — but no fallback has been validated; the other 3 candidates are not production-ready fallbacks based on this test.

### Phase 2 — Prompt Design (locked, matches implementation)

System Prompt:

You are a binary food classifier for retail receipt items. Given an item name, output ONLY a JSON object in this exact shape:

{"is_food": true|false, "confidence": 0.0-1.0}

Rules:

* No explanation, no reasoning, no text before or after the JSON.
* "is_food" = true only for edible food/beverage items for human consumption.
* "is_food" = false for non-food items (medicine, cleaning products, toiletries, electronics, etc.).
* If genuinely ambiguous, still output your best guess with a lower confidence score - do not refuse, do not add caveats.

Examples:
Item: "ORG STRWBRY" -> {"is_food": true, "confidence": 0.95}
Item: "Supravit-M Tablet 10's" -> {"is_food": false, "confidence": 0.9}
Item: "XYZFOODS RICE" -> {"is_food": true, "confidence": 0.6}

User:

Item: "{item_name}"

---

### Phase 3 — Provider Abstraction

`food_classifier.py` calls a swappable LLM interface.

Config picks provider + model (`FOOD_CLASSIFIER_PROVIDER`, `FOOD_CLASSIFIER_MODEL`) — no hardcoded Groq/OpenRouter calls inside the classification logic itself.

---

### Phase 4 — Threshold

Deferred until real confidence-score distributions exist.

No number picked yet.

---

### Phase 5 — Fallback / Unknown State, Caching

As agreed: API-down + low-confidence unify into one unknown path, reusing the existing surfaced/flagged item pattern.

Caching is latency/cost only.

---

### Phase 6 — Build + Smoke Test

Implementation pending once the prompt and provider abstraction are finalized.

---

## 5. Module: `extractor.py`

**Status:** ✅ Built, smoke-tested. Closed via #29. Batch entry point added Issue #46.

**Role:** Orchestrates the three components above into the Stage 2 contract.

**Two entry points now exist:**
- `extract_item_fields(item_name)` — single-item, ONE Groq call. Kept for standalone/debug/test use; also the per-item building block the batch path uses for unit/brand extraction.
- `extract_item_fields_batch(item_names: list[str])` — the production entry point `pipeline.py` calls. Unit/brand extraction still runs per-item (local, regex/lexicon-based, not the bottleneck — no reason to batch). Only the is_food classification step is batched via `classify_is_food_chunked(..., batch_size=5)`. Order preserved item-for-item throughout.

```python
def extract_item_fields(item_name: str) -> ItemFields:
    unit_result = extract_unit(item_name)
    brand_result = extract_brand(unit_result.remaining_text)

    classification_input = brand_result.remaining_text.strip() or item_name
    food_result = classify_is_food(classification_input)

    return ItemFields(
        unit=unit_result.unit,
        brand=brand_result.brand,
        is_food=food_result.is_food,
    )
```

### Design decision: food_classifier input

Runs on `remaining_text` (bare product name, post unit+brand strip) by default — chosen as a cleaner classification signal than the full string with brand/unit noise still in it. Not settled permanently: if real data shows this loses useful context, revert to `item_name`.

**Fallback:** if unit+brand extraction consumes the entire `item_name` (`remaining_text` becomes `""`), classification falls back to the original `item_name`. Example: `"NESTLE 1L"` → unit strips `"1L"`, brand strips `"Nestle"` → `remaining_text = ""` → falls back to classifying `"NESTLE 1L"` directly, rather than feeding an empty string (which `food_classifier.py`'s own empty-input guard turns into `UNKNOWN`).

`ItemFields.is_food` is `bool | None` — `None` means unknown (API failure or malformed response), matching `food_classifier.py`'s real output shape. Downstream pipeline gating (skip Normalization/Expiry, surface to user) must treat `None` the same as `False`, per API_Spec.md §2 — that equivalence is enforced at the pipeline-gating layer, not by this dataclass.

### Real bug found via this module's smoke testing

Testing the empty-`remaining_text` fallback (`"NESTLE 1L"`) surfaced a genuine bug in `food_classifier.py`, not this module: the fallback correctly fed `"NESTLE 1L"` to the classifier, but the model's reasoning trace ran long on this ambiguous item, and `max_tokens=100` was shared between reasoning and content — the response got cut off mid-JSON (`'{"is_food": true'`), which the strict parser correctly rejected, producing a false `UNKNOWN` for an item that was actually food.

**Fixed in `food_classifier.py`:** `max_tokens` raised 100 → 300 (headroom, not a formally derived worst case). Confirmed fixed — `"NESTLE 1L"` now resolves `is_food=True` with the full reasoning trace intact.

### Latency (ad-hoc measurement via `debug=True` timing, not a formal benchmark)

| Item | unit | brand | food classification | total |
|---|---|---|---|---|
| `Supravit-M Tablet 10's` | 0.0ms | 0.8ms | 689.5ms | 690.3ms |
| `NESTLE 1L` | 0.1ms | 0.4ms | 713.4ms | 713.8ms |
| `CHICKEN BREAST 500G` | 0.0ms | 0.2ms | 614.1ms | 614.4ms |
| `Dettol Antiseptic 500ML` | 0.0ms | 0.5ms | 614.2ms | 614.7ms |
| **Average (excl. first call)** | ~0.03ms | ~0.5ms | **~657.8ms** | **~658.3ms** |

First call (`ORG STRWBRY 1LB`) excluded — ~6800ms, presumed cold-start/connection warm-up (client init, DNS/TLS), not representative of steady-state cost. Unconfirmed — not re-tested to isolate the cause.

**Implication for #25 (pipeline orchestrator):** unit/brand extraction is negligible (<1ms combined); food classification dominates entirely at ~650-700ms/item per single-item call. A receipt with 15-20 items classified **sequentially** would take ~10-14+ seconds for Stage 2 alone, breaking PRD.md §6's 10-second upload budget.

**Resolved via Issue #46:** `extract_item_fields_batch()` (see Section 5 above) batches is_food classification in chunks of 5, cutting Groq calls from N to ceil(N/5) per receipt — this was the actual fix applied in `pipeline.py`, not per-item concurrency (an earlier concurrency-only approach was tried first and found insufficient — see Section 4.1). Per-item latency numbers above still describe the underlying single-call cost; end-to-end Stage 2 latency after batching has not yet been formally re-benchmarked (see HANDOFF.md open items).

---

## 6. Accuracy status

No component in this stage has been measured against real data yet — matches the broader pipeline status in README.md/PRD.md.

| Component | Target | Current |
|---|---|---|
| Unit extraction | Not yet formally targeted | Smoke-tested on doc examples only |
| Brand extraction | Not yet formally targeted | Smoke-tested on doc examples only |
| is_food classification | Not yet formally targeted | Built, smoke-tested (19 hand-picked items across two test rounds, 100% correct) — not yet measured against a real labeled dataset (#34) |
| End-to-end Stage 2 | Not yet formally targeted | Wired (#29). Smoke-tested, 5/5 correct after max_tokens fix. ~658ms/item avg (food classification dominant, unit/brand negligible) — not yet re-tested against real pipeline throughput at receipt scale |

---

## 7. Open items (carried from HANDOFF.md, not re-litigated here)

- `extractor.py`'s remaining_text-vs-item_name choice for the food gate is a default, not final — revisit if real data shows confusion.
- Real-receipt validation for `unit_extractor.py` and `brand_matcher.py` — not yet run.
- Latency of brand fuzzy-matching at scale — not yet measured.
- `food_classifier.py` confidence threshold — deferred until real confidence distributions exist.
- `food_classifier.py` real-accuracy measurement — blocked on #34 (labeled dataset), no data yet.
- **Sequential food-classification latency (~650-700ms/item) will break PRD.md §6's 10-second upload budget on multi-item receipts** — needs concurrent/batched classification, to be addressed in #25 (pipeline orchestrator), not in `extractor.py`.
- First-call latency outlier (~6800ms on cold start) observed but not confirmed/isolated — presumed connection warm-up.
