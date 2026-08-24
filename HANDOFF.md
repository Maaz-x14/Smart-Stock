# HANDOFF.md — Context Document for Next Chat Session

## Smart-Stock: AI-Powered Inventory & Waste Reduction System

---

## Writing style for this project (important)
- No over-explanation. Short, direct answers.
- Propose approach before code. Ask clarifying question at a time, don't assume.
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

## Stage 2 — NER: IN PROGRESS

**Current model:** DistilBERT, fine-tuned, F1 0.907 (test), exceeds all targets (F1≥0.88, P≥0.90, R≥0.86). Saved on Kaggle: `distilbert-ner-smart-stock` (verify: is the dataset's `-best` folder Run 2 output (F1 0.907) — last known status was "not updated yet," needs confirming before using as baseline).

**Not locking in yet** — given the OCR lesson (fine-tuned TrOCR lost to pretrained PaddleOCR), want to verify DistilBERT is actually the best option before committing.

### Plan (in progress, not started)

1. **Model comparison** — test DistilBERT (current) vs BERT-base, RoBERTa-base, ModernBERT-base, spaCy transformer pipeline. Same train/val/test split, seqeval F1/precision/recall, CPU latency per line. Use PaddleOCR output as real-world test input, not just clean synthetic/CORD val set.
2. **OCR→NER data flow decisions (made, not yet implemented):**
   - Rule-based regex/keyword pre-filter before NER (drop price-only lines, header/footer keywords like GST/Invoice/Total/CNIC/Payments) — cheap, reduces NER workload and false positives.
   - NER itself learns to output `O` for non-grocery items (electronics/furniture on mixed-category receipts, e.g. Imtiaz Mart/cash-and-carry stores selling everything) — via training data reflecting this, OR post-NER filter against the Stage 3 `shelf_life_reference` whitelist (leaning toward this — simpler, reuses existing table, no re-labeling of CORD/TASTEset needed).
3. Not yet started: actual comparison notebook, filter implementation.

**Question left open for next session:** confirm Kaggle `distilbert-ner-smart-stock` dataset actually holds Run 2 (F1 0.907) weights before treating that number as trustworthy baseline for comparison.

Full training details, dataset sources, entity schema, known bugs: `NER_Training.md`.

---

## Stage 3 — Normalization: BUILT, untested against real pipeline

3-pass (abbreviation map ~800 entries → rapidfuzz → Groq LLM fallback), `shelf_life_reference` DB table, `normalization_cache` table. Runs as pure Python in `ml_service/normalization/`, no GPU/training. Full module code, schema, eval harness in `Normalization_Training.md` — looked complete on review, not modified.

**Not yet tested with actual OCR→NER output** (only synthetic test cases in `evaluate.py`). Should validate once Stage 2 is locked.

---

## Stage 4 — Expiry Prediction: BUILT, untested against real pipeline

Rule-based 3-level lookup (exact match → category fallback → hard default) + confidence scoring, reads `shelf_life_reference` from Stage 3. Pure Python, `ml_service/expiry/`. Full code, schema addition (`InventoryItem`), eval harness in `Expiry_Training.md` — looked complete on review, not modified.

Same status as Stage 3: built but not run against real end-to-end pipeline output yet.

---

## Immediate Next Steps (in order)

1. Confirm Kaggle `distilbert-ner-smart-stock` dataset version (Run 1 vs Run 2 weights).
2. Build NER model comparison notebook (DistilBERT vs BERT-base/RoBERTa-base/ModernBERT-base vs spaCy) using PaddleOCR real-receipt output as test input.
3. Decide/implement OCR→NER pre-filter (regex/keyword line filter).
4. Decide/implement non-grocery-item filtering (NER `O`-tagging vs post-filter against shelf_life_reference).
5. Lock in final NER model, update NER_Training.md with comparison results and decision.
6. Run Stage 3 + Stage 4 against real Stage 1+2 output for the first time — validate end-to-end.
7. Save/export whichever NER model wins (ONNX already scaffolded in NER_Training.md).

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
