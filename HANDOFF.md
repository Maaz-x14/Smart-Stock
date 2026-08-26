# HANDOFF.md — Context Document for Next Chat Session

## Smart-Stock: AI-Powered Inventory & Waste Reduction System

---

## Writing style for this project (important)
- No over-explanation. Short, direct answers.
- Propose approach before code. Ask one clarifying question at a time, don't assume.
- Surgical edits to existing docs/code — never full rewrites unless explicitly asked.
- Push back on bad ideas. Distinguish facts from speculation.
- Individual per-file commits.

---

## Project Summary

Smart-Stock: portfolio/CV project. Reads grocery receipts, predicts expiry dates, reduces food waste. Full stack: React + TypeScript → FastAPI → PostgreSQL → ML Pipeline (4 stages).

**Docs (update in place, don't regenerate):** PRD.md, Architecture.md, API_Spec.md, DB_Schema.md, ML_Pipeline.md, OCR_Training.md, NER_Training.md, Normalization_Training.md, Expiry_Training.md, README.md

---

## Pipeline Status

Receipt Image
|
v
Stage 1: OCR ✅ CLOSED — PaddleOCR (pretrained, no fine-tuning)
|
v
Stage 2: NER ⏳ IN PROGRESS — DistilBERT F1 0.907, evaluating alternatives
|
v
Stage 3: Normalization ✅ BUILT — 3-pass (map/fuzzy/LLM), not yet tested against real OCR output
|
v
Stage 4: Expiry Prediction ✅ BUILT — rule-based lookup, not yet tested against real pipeline



---

## Stage 1 — OCR: CLOSED

**Winner: PaddleOCR (pretrained PP-OCRv6, no fine-tuning needed).**

TrOCR was fine-tuned for ~2 months (CER 0.0631, val), hit a real plateau (r=16 vs r=32 gave no improvement — 0.0631 vs 0.0633). Full TrOCR history preserved in OCR_Training.md as historical record (dataset, CRAFT line-detection pipeline, and debugging lessons — all reused to evaluate PaddleOCR quickly).

PaddleOCR pretrained beat fine-tuned TrOCR decisively on same 4 real receipt photos:
- Accuracy: correct on item/price/qty lines across all 4 receipts (TrOCR had digit errors, word merges)
- CPU latency: ~5-6s/receipt vs TrOCR's ~480s/receipt

**Production config:**
```python
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device='cpu',
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
)
```

**Environment gotchas (don't re-debug):** requires `paddlepaddle==3.2.0` exact match; PP-OCRv6 uses `small`/`tiny` naming not `mobile`; use `.predict()` not deprecated `.ocr()`; never benchmark PaddleOCR's full pipeline on pre-cropped line images — invalid, gives inflated CER (detection/orientation stages assume full-document input).

Full details, issue resolution comments, decision log: `OCR_Training.md`.

---

## Stage 2 — NER: ✅ LOCKED IN

**Model:** DistilBERT, fine-tuned, F1 0.907 (original test run). Saved on Kaggle: `distilbert-ner-smart-stock` (Run 2 weights confirmed correct via comparison notebook — DistilBERT loaded from this path scored F1 0.9156 on a freshly-drawn test split, consistent with Run 2, not Run 1's 0.835).

**Model comparison — done.** Compared against BERT-base, RoBERTa-base, ModernBERT-base (same `ner_splits`, 15 epochs, seqeval F1, CPU latency). DistilBERT won on both F1 (0.9156) and latency (28.75ms) — see NER_Training.md §23 for full table and an overfitting caveat on the other 3 models (not a training-budget-controlled architecture ranking). spaCy comparison skipped — clear winner already emerged, no need per original plan's own condition.

**Decision:** DistilBERT locked in. No further model comparison.

### Real-OCR pipeline analysis — in progress (new, not in original plan)

Ran actual PaddleOCR output (4 real receipts) through pipeline analysis for the first time. Found a structural blocker not visible from synthetic/CORD data:

- **PaddleOCR emits one line per detected text-box, not one line per logical receipt row.** An item's name, quantity, price, discount, and total are 5+ separate unordered lines (field order even varies receipt-to-receipt — qty-before-price on some, price-before-qty on others). Synthetic training data assumes single-line `"STRWBRY 1 LB 2.99"` format — real OCR output doesn't look like that.
- **Open question:** does Stage 1's PaddleOCR call retain bounding-box coordinates, or only flat text? Unconfirmed — needs checking against actual extraction code/latest PaddleOCR docs. If coordinates exist, row-reconstruction (cluster boxes by y-position into logical rows) is needed as a new Stage 1.5, before Phase A/B filtering makes sense.
- Price token: **confirmed safe to discard** — not in DB_Schema.md `inventory_items` at all.
- Quantity: **confirmed required** — `inventory_items.quantity NUMERIC(10,2) NOT NULL`, not discardable.
- PaddleOCR returns a per-line confidence score (`[0.98]`, etc.) — usable as a garbage-line signal, but no reliable threshold yet (checked against real data: confidence doesn't cleanly separate garbage from real text — a garbage line scored 1.00, another real line scored 0.67). Needs proper distribution analysis on a larger labeled sample before picking a cutoff, not a guessed number.
- Real evidence (not hypothetical) that Phase B's `NOT_FOOD` gate is needed: receipt 1 is a pharmacy receipt with real non-food items (`"Supravit-M Tablet 10's"`).

Full training details, dataset sources, entity schema, comparison table: `NER_Training.md`.

## Stage 3 — Normalization: BUILT, untested against real pipeline

3-pass (abbreviation map ~800 entries → rapidfuzz → Groq LLM fallback), `shelf_life_reference` DB table, `normalization_cache` table. Runs as pure Python in `ml_service/normalization/`, no GPU/training. Full module code, schema, eval harness in `Normalization_Training.md` — looked complete on review, not modified.

**Not yet tested with actual OCR→NER output** (only synthetic test cases in `evaluate.py`). Should validate once Stage 2 is locked.

---

## Stage 4 — Expiry Prediction: BUILT, untested against real pipeline

Rule-based 3-level lookup (exact match → category fallback → hard default) + confidence scoring, reads `shelf_life_reference` from Stage 3. Pure Python, `ml_service/expiry/`. Full code, schema addition (`InventoryItem`), eval harness in `Expiry_Training.md` — looked complete on review, not modified.

Same status as Stage 3: built but not run against real end-to-end pipeline output yet.

---

## Immediate Next Steps (in order)

1. File GitHub issues for the items below (do this first in next session).
2. Check Stage 1 OCR extraction code / latest PaddleOCR docs — does it retain bounding-box coordinates per detected text box, or only flat text?
3. If coordinates available: design + implement Stage 1.5 row-reconstruction (cluster text-boxes into logical receipt rows by y-position) — blocks everything downstream, do this before Phase A/B filtering.
4. Within a reconstructed row: discard price token (confirmed unused in DB_Schema.md), keep item name + quantity (confirmed required field).
5. Determine a real confidence-score threshold for dropping garbage OCR lines — via distribution analysis on a labeled sample, not a guessed number.
6. Implement Phase A rule-based prefilter (`ml_service/ner/prefilter.py`, already drafted in ML_Pipeline.md) — run it on reconstructed rows, not raw boxes.
7. Implement Phase B `NOT_FOOD` gate in Stage 3 Pass 3 (`llm_fallback.py`) — add NOT_FOOD escape hatch to LLM prompt, return `(None, 0.0)` on that response. Decide: silent discard vs. surfaced in confirmation modal as "detected but excluded."
8. Run Stage 3 + Stage 4 against real reconstructed Stage 1+2 output for the first time — validate end-to-end.
9. Export final DistilBERT NER model to ONNX (scaffolded in NER_Training.md).

---

## GitHub

Repo: `github.com/Maaz-x14/Smart-Stock`
Issue #8 (OCR line-segmentation) — closed.
Issue #9 (PaddleOCR evaluation) — closed, PaddleOCR adopted.
No open NER issue yet — consider filing one for the model comparison task.

---

## Docs status (as of this handoff)

| Doc | Status |
|---|---|
| OCR_Training.md | Up to date — PaddleOCR-first, TrOCR history preserved |
| NER_Training.md | Up to date for DistilBERT Run 2 results; needs comparison results once done |
| Normalization_Training.md | Complete, untested against real pipeline |
| Expiry_Training.md | Complete, untested against real pipeline |
| ML_Pipeline.md | Updated for PaddleOCR-first OCR; latency/eval caveats noted |
| Architecture.md | Updated for PaddleOCR in diagrams, hosting, and technical decisions |
| README.md | Updated for PaddleOCR-first status and revised latency target |
| PRD.md | Updated to clarify owned pipeline, pretrained PaddleOCR choice, and in-house NER |
