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

---

## Project Summary

Smart-Stock: portfolio/CV project. Reads grocery receipts, predicts expiry dates, reduces food waste. Full stack: React + TypeScript → FastAPI → PostgreSQL → ML Pipeline (now 7 stages, was 4 — see below).

**Docs (update in place, don't regenerate):** PRD.md, Architecture.md, API_Spec.md, DB_Schema.md, ML_Pipeline.md, OCR_Training.md, NER_Training.md, Normalization_Training.md, Expiry_Training.md, README.md — **all 10 updated this session**, current as of this handoff.

---

## MAJOR ARCHITECTURE CHANGE THIS SESSION

Original 4-stage pipeline (OCR → NER → Normalization → Expiry) assumed single-line-per-item OCR format and a fine-tuned DistilBERT NER model tagging FOOD/QTY/UNIT/PRICE within that line.

Testing against real PaddleOCR output on real Pakistani retail receipts broke both assumptions:

1. OCR output is box-per-field, not line-per-item — needed row reconstruction.
2. **No two stores share a receipt header format.** Seen: `Quantity|Price|Discount|Total`, `Qty|Item|Rate|Amount`, merged headers like `Quantity Price|Discount|Total`, `M.R.P|Price|Qty/Wt|Tax(%)|Disc. Amount`. A model trained on one synthetic format can't generalize to this — a **header-driven parser that reads each receipt's own column layout** does, deterministically, no training data needed.

**Decision made and executed: DistilBERT NER retired.** Once row parsing extracts qty/price/discount/total structurally via header position, NER's job shrank to near-nothing. What's left in `item_name` (unit, brand, is_food) is solved with regex + fuzzy lexicon + LLM gate — no training data, no served model, generalizes better to novel item text than a small trained classifier would have.

---

## Current Pipeline (7 stages)

```
Receipt Image
  → Stage 1: OCR (PaddleOCR, pretrained) — text + bounding boxes
  → Stage 1.5: Row Reconstruction — deskew + y-position clustering into logical rows
  → Stage 1.6: Prefilter — drops metadata rows (GST#, Invoice#, Payments, etc.)
  → Stage 1.7: Row Parser — header-driven column mapping → {item_name, quantity, price, discount, total}
  → Stage 2: Item Field Extraction — regex unit + fuzzy brand lexicon + LLM is_food gate → {is_food, brand, unit}
       is_food=false → surfaced to user flagged "excluded", stops here (skips Stage 3+4)
       is_food=true  ↓
  → Stage 3: Normalization — lookup → fuzzy → LLM canonical naming (unchanged mechanism, now gated)
  → Stage 4: Expiry Prediction — rule-based shelf-life lookup (unchanged)
  → Structured Inventory Item
```

---

## Stage-by-stage status

### Stage 1 — OCR: ✅ CLOSED (unchanged this session)

PaddleOCR (pretrained PP-OCRv6), beat fine-tuned TrOCR on accuracy + latency. Full history in OCR_Training.md. No changes this session.

### Stage 1.5 — Row Reconstruction: ✅ DONE (issue #14, closed)

`rec_boxes`/`rec_polys` confirmed present in PaddleOCR output (weren't previously extracted). Built deskew correction (median skew angle from box polygons, corrects y-position drift) + y-position clustering. Validated on 4 real receipts — correctly reconstructs rows even with real photograph skew.
Module: `ml_service/ocr/row_reconstruction.py`

### Stage 1.6 — Prefilter: ✅ DONE, relocated

Moved from `ml_service/ner/prefilter.py` → `ml_service/parsing/prefilter.py` (never NER-specific, just ran before it). Rule-based keyword/regex drop. Runs on reconstructed rows, excludes the header row (Row Parser consumes header before prefilter sees the rest, otherwise header keywords collide with `DROP_KEYWORDS`).
Known gap: doesn't catch every leak (e.g. "Ex1. Amt", "HBL") — low priority, doesn't corrupt item data.

### Stage 1.7 — Row Parser: ✅ DONE (issue #22, ready to close via PR)

Header-driven column mapping. Fuzzy header detection (strict `ratio` first, `partial_ratio` fallback for merged tokens like "Quantity Price" — fixes a real bug where strict-only matching caused single-word headers like "Discount" to false-positive against "Total"). x-position nearest-match for column assignment. OCR noise cleanup in numeric tokens (`"11^9.00"`→119.00, `"Rs7.948.80"`→7948.80) while still rejecting real text. Split-row merge (item name and its numbers sometimes print on separate physical lines, sometimes same line — both handled). Fallback path (magnitude/position heuristics) when no header detected.
**Validated: output matches ground truth exactly on all fields across 1-4.jpg (4 real receipts).**
Module: `ml_service/parsing/row_parser.py`
**Known unsupported format (issue #23, open):** 5.jpg — 2-line header, new `Tax(%)` column, merged numeric sub-values (`"0.00(0) 3.00"`), fused item-code+name. Deferred, needs separate parsing path, not a patch.

### Stage 2 — Item Field Extraction: ⏳ DESIGNED, NOT YET IMPLEMENTED

Replaces retired NER. Output contract: `{is_food: bool, brand: str | None, unit: str | None}`. Design decided, code not written:

- **Unit**: regex extraction from item_name, handles known OCR corruption (`ml`→`M1`, `l`→`1`). No model.
- **Brand**: fuzzy lexicon match (rapidfuzz, reuses Stage 3 infra). No labeled data — lexicon-based, inspectable/extendable. **Open question: where does the brand lexicon come from** — manually curated, or grown from real receipts as seen? Not decided.
- **is_food**: LLM gate, binary classification. Replaces old Phase B "NOT_FOOD gate" concept (which was going to live in Stage 3 Pass 3 — now correctly placed at Stage 2 instead, so Stage 3 doesn't re-ask the same question).
  Empty stub files exist: `ml_service/item_extraction/{unit_extractor,brand_matcher,food_classifier,extractor}.py` — none have implementation yet.

**Distinct from Stage 3 Pass 3's LLM call** — Stage 2 asks "is this food at all", Stage 3 Pass 3 asks "what's the canonical name for this (already-confirmed) food item". Different questions, both needed, not merged (yet — flagged as a possible future optimization, not decided).

### Stage 3 — Normalization: ✅ BUILT, mechanism unchanged, NOW GATED

3-pass (abbreviation map ~800 entries → rapidfuzz → LLM fallback). **Now only runs for items where Stage 2 returns `is_food=true`.** Still untested against real Stage 1.5–2 output (only synthetic test cases in `evaluate.py`).

### Stage 4 — Expiry Prediction: ✅ BUILT, unchanged

Rule-based 3-level lookup + confidence scoring. Still untested against real pipeline output.

---

## DB / API changes made this session (docs updated, code NOT yet migrated)

- `inventory_items` table needs new `brand VARCHAR(255)` column — migration SQL given in DB_Schema.md, **not yet run**.
- `price`, `discount`, `total`, `is_food` confirmed **not persisted** — extracted internally, never written to DB.
- `is_food=false` items: **surfaced to user, not silently dropped** (decision made this session) — confirmation modal shows them flagged "detected but excluded — not a food item." API response for `/receipts/upload` needs `brand`, `is_food` fields added to `extracted_items[]` — documented in API_Spec.md, **Pydantic schemas not yet updated in code**.

---

## Folder structure (already executed by user, confirmed current)

```
ml_service/
├── ocr/
│   ├── model.py                  # empty stub — PaddleOCR loader, needs extracting from notebook
│   ├── row_reconstruction.py     # empty stub — cluster_rows_deskewed logic exists in paddleocr.ipynb, needs extracting into this module
│   ├── paddleocr.ipynb           # working notebook, has validated row reconstruction code
│   └── kaggle_trocr.ipynb        # historical, TrOCR
├── parsing/
│   ├── prefilter.py              # DONE, moved from ner/
│   └── row_parser.py             # DONE, moved from ocr/
├── item_extraction/
│   ├── unit_extractor.py         # empty stub
│   ├── brand_matcher.py          # empty stub
│   ├── food_classifier.py        # empty stub
│   └── extractor.py              # empty stub
├── normalization/                # unchanged, existing files
├── expiry/                       # unchanged, existing files
├── ner/
│   └── archive/
│       └── ner-training.ipynb    # historical record only
├── models/
│   └── trocr-smart-stock/        # retired TrOCR weights — cleanup decision deferred, not urgent
└── pipeline.py                   # EMPTY — orchestrator not yet wired for new stage order
```

**Note:** `paddleocr.ipynb`'s validated logic (deskew + clustering) has NOT yet been extracted into `ml_service/ocr/row_reconstruction.py` as an importable module — it currently only exists as notebook cells. Same applies to `model.py` (PaddleOCR loader) — currently just a touch'd empty file.

---

## GitHub issue status (as of this handoff)

**Open, valid, keep:**

- #16 — Confidence-score garbage-line filter needs real distribution analysis (blocked on row-level text, which now exists — unblocked, ready to pick up)
- #23 — 5.jpg multi-line header + Tax(%) format not supported

**Ready to close via PR (user pushing code + docs this session):**

- #22 — Row parser deliverable — DONE, validated, ready to close

**To close as not-planned/superseded (user doing this via PR):**

- #18 — Phase B NOT_FOOD gate in Stage 3 LLM fallback — SUPERSEDED, is_food is now Stage 2's job
- #19 — End-to-end validation Stage 3+4 — STALE, references retired NER + closed #18 as dependency, needs refiling with corrected deps
- #20 — Export DistilBERT NER to ONNX — DEAD, model retired

**Also i added these new issues to repo. I added these 10 issues, with proper description.**

1. Wire `ml_service/pipeline.py` orchestrator — stitch Stage 1→1.5→1.6→1.7→2→3→4 with is_food gating
2. Implement `unit_extractor.py` — regex unit extraction
3. Implement `brand_matcher.py` — fuzzy lexicon match (needs lexicon data source decided first)
4. Implement `food_classifier.py` — LLM is_food gate
5. Implement `extractor.py` — orchestrates Stage 2's three components
6. DB migration: add `brand` column to `inventory_items`
7. Update Pydantic schemas: add `brand`, `is_food` to `/receipts/upload` response
8. Refile end-to-end validation (replaces #19) with corrected dependency chain
9. Re-measure pipeline latency end-to-end (flagged unmeasured in README/ML_Pipeline.md)
10. Build labeled is_food sample to measure Stage 2's LLM gate accuracy

---

## Immediate Next Steps (in order)

1. I already Filed the 10 new issues above.
2. Pick up implementation — recommended order: #2/#3/#4/#5 (Stage 2 components, since they're pure design-done/code-missing) before #1 (orchestrator, needs Stage 2 working first) before #6/#7 (DB/API wiring) before #8/#9/#10 (validation/measurement, needs the pipeline actually running). Or go through all the issues, and give order for which issues to do in which order.

---

## Key decisions made this session (context for future reference, not to be re-litigated without new evidence)

- DistilBERT NER retired — full reasoning in ML_Pipeline.md §0, preserved historically in NER_Training.md (retirement banner added).
- Row Parser (header-driven, per-receipt column detection) chosen over any fixed-position or single-model approach — real receipts have no standard format, confirmed via 10+ real header samples across different formats.
- Item Field Extraction uses regex/lexicon/LLM, explicitly NOT a trained model — unit and brand are deterministic/lookup problems, is_food has no labeled data and an LLM gate generalizes better than a small classifier would.
- `is_food=false` items are surfaced to the user (transparent), not silently dropped.
- `brand` gets persisted to DB (new column) — decided useful, cheap to store.
- `price`/`discount`/`total` extracted but never persisted — confirmed out of DB_Schema.md scope from the start (predates this session).
