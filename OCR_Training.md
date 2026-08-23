
# OCR_Training.md — OCR Model Guide

## SmartStock: Stage 1 OCR

**Current production model: PaddleOCR (pretrained, no fine-tuning required).**
**Superseded: TrOCR (fine-tuned, CER 0.0631) — full training history retained below for reference.**

**Version:** 16.0 — PaddleOCR adopted as production OCR model. TrOCR Stage 1 training history preserved as historical record. File renamed from TrOCR_Training.md back to OCR_Training.md now that a single doc covers Stage 1 regardless of model.
**Last Updated:** Post PaddleOCR evaluation — pretrained PP-OCRv6 (small det/rec, doc-orientation/unwarping/textline-orientation disabled) beat fine-tuned TrOCR on both accuracy and CPU latency on real receipt photos.

---

## STATUS: Stage 1 (OCR) — CLOSED — PaddleOCR is production model

### Why the switch

TrOCR (fine-tuned, CER 0.0631 val) had two unresolved production blockers even after the CRAFT line-detection fix and generation_config bug fixes:
- CPU inference: ~480s for 65 lines (one receipt) — not viable for a real upload flow
- WER 0.234 on val — roughly 1 in 4 words wrong, risked compounding errors into Stage 2 NER

PaddleOCR (PP-OCRv6, **pretrained, no fine-tuning**) was evaluated per Issue #9 and beat TrOCR on both accuracy and speed on the same real receipt photos — no training needed to reach this result.

### Real-receipt comparison (same 4 photos, both pipelines)

| Aspect | TrOCR (fine-tuned) | PaddleOCR (pretrained) |
|---|---|---|
| 1.jpg text quality | `CoreCSachet`, `1000ng`, `117.45` (digit error) | `Core C Sachet 1's 1000mg`, `1137.45` — correct |
| Item/price line accuracy | Frequent word-merge and digit errors | Consistently correct on items, quantities, prices across all 4 receipts |
| CPU latency (1 receipt) | ~480s (65 lines, unoptimized) | ~5–6s (small det/rec models, doc-orientation/unwarping/textline-orientation disabled) |
| Fine-tuning required | Yes — ~2 months, LoRA on 50K+ line crops | No — pretrained PP-OCRv6 sufficient |

**Verdict: PaddleOCR wins decisively on both axes.** Confirmed on all 4 real test receipts, not just one. Switch adopted — no PaddleOCR fine-tuning planned unless a future accuracy gap appears on a larger real-world sample.

### PaddleOCR — production config

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device='cpu',
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
)

result = ocr.predict(image_path_or_array)
res = result[0]
texts  = res["rec_texts"]
scores = res["rec_scores"]
```

**Environment notes (hard-won, do not re-debug):**
- PaddleOCR 3.x requires PaddlePaddle ≥ 3.0. Install exact versions per official docs, not "latest": `paddlepaddle==3.2.0` (or matching `paddlepaddle-gpu==3.2.0` build) — mismatched versions cause `set_optimization_level`/PIR/oneDNN errors that look like environment bugs but are version mismatches.
- `PP-OCRv6` model family uses `small`/`tiny` naming, **not** `mobile` (mobile naming is v3–v5 only). `PP-OCRv6_mobile_det` does not exist and will raise `UnknownModelError`.
- `ocr.ocr()` is deprecated — use `ocr.predict()`. Return format changed: no more `[[box, (text, score)], ...]` — now a dict with `rec_texts`/`rec_scores` keys.
- GPU on Kaggle repeatedly failed (`libcuda.so.1` missing, driver/environment issue unrelated to package versions) — not worth fighting further; CPU-only with small models already met latency needs (~5s/receipt).
- Running the full PaddleOCR pipeline (doc-orientation + unwarping + detection) on **already-cropped single-line images** (e.g. val set crops) is invalid — those stages are built for full document photos. Produces garbage (CER 0.66+) that looks like a model failure but is actually a pipeline/input-shape mismatch. Always benchmark PaddleOCR at the full-receipt level, not on pre-cropped line images.

**Next steps for Stage 1:** none planned. Move to Stage 2 (NER) using PaddleOCR output.

---

## STAGE 1 HISTORY — TrOCR (superseded, kept for reference)

**Final model: r=16 LoRA, CER 0.0631 (val), saved as `smart-stock-model-data` v4 on Kaggle.**

| Run | Config | Val CER | Verdict |
|---|---|---|---|
| v4 run 1 | r=16, lora_alpha=32 | 0.0687 | superseded |
| v4 run 2 | r=16, resumed | 0.0637 | superseded |
| v4 run 3 (Optuna LR) | r=16, LR=8.84e-5 | 0.0652 | worse, rejected |
| v4 run 5 epochs | r=16, LR=1.4824e-4 | **0.0631** | best TrOCR result |
| r=32 experiment | r=32, lora_alpha=64 | 0.0633 | no improvement, within noise — confirms plateau |

**Conclusion:** LoRA rank increase does not break the plateau. Error analysis (see Per-Source CER Diagnostic below) shows the remaining error is dominated by WildReceipt label noise and multi-line crop mismatches, not model capacity. Target of CER ≤ 0.05 was not reached (final: 0.0631). Superseded by PaddleOCR — see above.

The rest of this document (architecture notes, full 12-hour session budget, training cells, generation_config bug writeup, TrOCR production inference pipeline, GitHub Issue #8 resolution note, and the full notebook appendix) is retained below as the historical training record.

---

## Pipeline Overview

```
Receipt Image → TrOCR (Stage 1) → NER (Stage 2) → Normalization → Expiry Prediction
```

This document covers **Stage 1: TrOCR fine-tuning only.**

**Base model:** `microsoft/trocr-base-printed`
**Adapter:** LoRA on decoder only (encoder frozen)
**Trainable params:** 1,523,712 of 335M total (0.45%)
**Target:** CER ≤ 0.05 | WER ≤ 0.10 (final achieved: CER 0.0631, WER 0.2344 — target not met, declared ceiling)
**Current best saved model:** `smart-stock-model-data` v4 on Kaggle → `trocr-smart-stock-best/` — CER 0.0631
**Rule:** Do NOT overwrite unless a new run beats 0.0631. r=32 tested and did not beat it (0.0633) — no further overwrite expected without a data-cleaning pass.

---

## Architecture Notes

### Why LoRA + Frozen Encoder

Full fine-tuning of 335M params caused plateau and optimizer instability (CER stuck at 0.1758). Current approach freezes the ViT encoder entirely (preserves pretrained visual features) and applies LoRA only to the RoBERTa decoder's attention projections.

**Known ceiling:** The frozen encoder is the current performance ceiling. Once a stable single-GPU baseline is re-established on v3 data, unfreezing the top 2–4 ViT encoder blocks alongside LoRA is the highest-impact next step.

### Why Line Crops

TrOCR was pretrained on single-line text images. Feeding full receipts (20–50 lines) is a domain mismatch. Line crops are the single biggest fix — CER dropped from 0.757 (full images) → 0.133 after switching.

---

## 12-Hour Session Budget (Kaggle T4, Single GPU)

| Phase                                                      | Time                          |
| ---------------------------------------------------------- | ----------------------------- |
| pip installs                                               | ~3 min                        |
| Dataset load from disk (v3 saved, no rebuild)              | ~2 min                        |
| Model setup                                                | ~2 min                        |
| 8 epochs × 8,558 steps × ~0.55s (fp16, LoRA, frozen enc) | ~10.5 hr                      |
| Val eval × 8 epochs (3,193 samples)                       | ~16 min                       |
| Save & Export                                              | ~5 min                        |
| Test eval — manual loop, 9,301 samples, beam=4            | ~60 min                       |
| **Total**                                            | **~12.0 hr** ⚠️ tight |

> **Next run resumes from checkpoint — dataset already on disk.** No rebuild cost. 8 epochs on 68k examples = ~10.5 hr training + ~60 min test eval = ~11.5 hr. Monitor epoch 1 — if it takes > 85 min, reduce to 7 epochs.
>
> **After partial encoder unfreeze:** each step will be slower (~0.7s vs 0.55s). Reduce to 5–6 epochs when unfreezing encoder blocks. Total budget with unfrozen encoder at 6 epochs ≈ 9.5 hr — fits comfortably.

**Use Save & Run All — never draft mode.** Draft mode does not commit `/kaggle/working/` outputs.

---

## Dataset v3 — Composition

| Source                         | Train            | Val             | Test            | Notes                                                                              |
| ------------------------------ | ---------------- | --------------- | --------------- | ---------------------------------------------------------------------------------- |
| CORD                           | 2,105            | 221             | 251             | Indonesian restaurant receipts, line-level by group_id                             |
| SROIE                          | 14,476           | —              | 8,050           | English retail receipts,**2× weighted in train**, line-level by Y-proximity |
| WildReceipt (train.txt → 90%) | 26,741           | —              | —              | Per-annotation crops                                                               |
| WildReceipt (train.txt → 10%) | —               | 2,972           | —              | Val split from train.txt crops                                                     |
| WildReceipt (test.txt, capped) | 10,665→train    | —              | 1,000           | Excess test crops moved to train                                                   |
| **Total v3**             | **68,463** | **3,193** | **9,301** |                                                                                    |

**WildReceipt labels excluded:** 0 (empty/illegible), 25 (catch-all: terminal IDs, legal text, thank-you messages)
**WildReceipt cropping strategy:** per-annotation (not line grouping) — eliminates two-column merging bug
**Why per-annotation for WildReceipt only:** CORD groups by group_id (logical receipt lines, working well). SROIE uses Y-proximity (words well-spaced, no column issues). WildReceipt had two-column layouts causing Y-grouping to merge item names + prices into nonsense crops — corrupted ~33% of training data, caused CER regression 0.088 → 0.339
**Test capped at 1,000 WildReceipt crops** — prevents beam search hanging for hours (previous session lost 6+ hrs to this)

---

## Kaggle Dataset Inputs (Current Session)

| Kaggle slug                 | Purpose                                        | Update frequency                       |
| --------------------------- | ---------------------------------------------- | -------------------------------------- |
| `smart-stock-dataset-v3`  | Combined CORD+SROIE+WildReceipt dataset        | Never — only if new data source added |
| `trocr-smart-stock-model` | Current best model weights + resume checkpoint | After every training run               |
| `wild-receipt`            | Raw WildReceipt images + annotations           | Never                                  |

**`smart-stock-dataset-v3` structure:**

```
smart_stock_dataset_v3/
├── train/
├── validation/
├── test/
└── dataset_dict.json
```

Path: `/kaggle/input/datasets/maazahmad69/smart-stock-dataset-v3/smart_stock_dataset_v3`

**`trocr-smart-stock-model` structure:**

```
trocr-smart-stock-best/     ← current best merged model (CER 0.0687)
trocr-smart-stock/
└── checkpoint-51348/       ← only the best checkpoint, delete rest after each run
```

Paths:

- Model: `/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock-best/trocr-smart-stock-best`
- Checkpoint: `/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock/trocr-smart-stock/checkpoint-51348`

**`wild-receipt` structure:**

```
wildreceipt/
├── image_files/
├── train.txt
├── test.txt
├── class_list.txt
└── dict.txt
```

Path: `/kaggle/input/datasets/maazahmad69/wild-receipt/wildreceipt`

**Best practice — what to re-upload after each run:**

- `trocr-smart-stock-model` → new version: updated `trocr-smart-stock-best/` + latest single checkpoint only
- `smart-stock-dataset-v3` → never touched unless a new data source is added
- `wild-receipt` → never touched

---

## Full Notebook — Every Cell

### Cell 1 — Setup & Paths (pip installs + env)

```python
# ── Install dependencies ─────────────────────────────────────────────────────
!pip install -q transformers==5.0.0 datasets evaluate jiwer albumentations
!pip install -q "peft==0.13.2"
!pip install -q "torchao>=0.16.0"
!pip install -q optuna
```

> **peft==0.13.2** — last version before torchao dependency conflict. Install with `--no-deps` if needed.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # MUST be before ALL imports — especially before torch

from pathlib import Path

INPUT_DIR   = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")

# ── Dataset (smart-stock-dataset-v3 — never changes between runs) ────────────
DATASET_DIR = Path("/kaggle/input/datasets/maazahmad69/smart-stock-dataset-v3/smart_stock_dataset_v3")

# ── WildReceipt raw (wild-receipt — only needed if rebuilding dataset) ────────
WILDRECEIPT_DIR = Path("/kaggle/input/datasets/maazahmad69/wild-receipt/wildreceipt")

# ── Model weights (trocr-smart-stock-model — updated after every run) ─────────
MODEL_INPUT = Path("/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock-best/trocr-smart-stock-best")

# Checkpoint resume source — only checkpoint-51348 kept in trocr-smart-stock-model
INPUT_CHECKPOINT_DIR = Path("/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock/trocr-smart-stock/checkpoint-51348")

# Output dirs — writable
CHECKPOINT_DIR = WORKING_DIR / "trocr-smart-stock"
BEST_MODEL_DIR = WORKING_DIR / "trocr-smart-stock-best"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset dir     : {DATASET_DIR}")
print(f"WildReceipt dir : {WILDRECEIPT_DIR}")
print(f"Model input     : {MODEL_INPUT}")
print(f"Checkpoints     : {CHECKPOINT_DIR}")
print(f"Best model      : {BEST_MODEL_DIR}")

# Verify model files present
for f in ["config.json", "model.safetensors", "generation_config.json"]:
    exists = (MODEL_INPUT / f).exists()
    print(f"  {'✅' if exists else '❌'} {f}")
```

> **Critical:** `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` must be the very first Python line, before any import. If placed after `import torch` (even in a prior cell), Kaggle's 2×T4 silently runs DataParallel, doubling the effective batch size and invalidating the Optuna-tuned LR.

---

### Cell 2 — CORD Crop Extractor

> Skip running — output already in `smart_stock_dataset_v3`. Cell defines `extract_cord_crops()` which is called by the dataset builder.

```python
import json
from collections import defaultdict
from PIL import Image

def extract_cord_crops(image: Image.Image, ground_truth_str: str) -> list:
    """
    Extract line-level crops from a CORD receipt image.

    CORD's ground_truth is a raw JSON string. Bounding boxes are in
    valid_line[].words[].quad — NOT in gt_parse.menu.
    Each group_id in valid_line = one logical receipt line.
    We group words by group_id, merge their quads into one bbox, crop it,
    and use the joined word texts as the OCR target.
    Only menu.* categories are kept (skip total, tax, header lines).

    Returns list of (cropped_PIL_image, text) pairs.
    One CORD receipt → ~8–12 crops.
    """
    try:
        data = json.loads(ground_truth_str)
    except (json.JSONDecodeError, AttributeError):
        return []

    valid_lines = data.get("valid_line", [])
    if not valid_lines:
        return []

    groups = defaultdict(list)
    for line in valid_lines:
        groups[line["group_id"]].append(line)

    w, h = image.size
    crops = []

    for gid, lines in groups.items():
        if not any(l.get("category", "").startswith("menu.") for l in lines):
            continue

        all_words, all_xs, all_ys = [], [], []
        for line in lines:
            for word in line.get("words", []):
                text = word.get("text", "").strip()
                if text:
                    all_words.append(text)
                q = word.get("quad", {})
                if q:
                    all_xs += [q.get("x1",0), q.get("x2",0), q.get("x3",0), q.get("x4",0)]
                    all_ys += [q.get("y1",0), q.get("y2",0), q.get("y3",0), q.get("y4",0)]

        if not all_words or not all_xs:
            continue

        text = " ".join(all_words)
        x1, y1 = max(0, min(all_xs)), max(0, min(all_ys))
        x2, y2 = min(w, max(all_xs)), min(h, max(all_ys))
        if x2 <= x1 or y2 <= y1:
            continue

        crops.append((image.crop((x1, y1, x2, y2)), text))

    return crops
```

---

### Cell 3 — SROIE Crop Extractor

> Skip running — output already in `smart_stock_dataset_v3`. Defines `extract_sroie_crops()`.

```python
def extract_sroie_crops(image: Image.Image, words: list, bboxes: list) -> list:
    """
    Group SROIE words into lines by Y-coordinate proximity (15px tolerance).
    Each line bbox is cropped and paired with its joined text.
    SROIE has bboxes per word — no pre-grouped lines.

    Returns list of (cropped_PIL_image, text) pairs.
    """
    if not words or not bboxes:
        return []

    items = sorted(zip(words, bboxes), key=lambda x: x[1][1])

    lines = []
    current_words, current_boxes = [items[0][0]], [items[0][1]]
    for word, box in items[1:]:
        if abs(box[1] - current_boxes[-1][1]) <= 15:
            current_words.append(word)
            current_boxes.append(box)
        else:
            lines.append((current_words, current_boxes))
            current_words, current_boxes = [word], [box]
    lines.append((current_words, current_boxes))

    w, h = image.size
    crops = []
    for line_words, line_boxes in lines:
        text = " ".join(line_words).strip()
        if not text:
            continue
        x1 = max(0, min(b[0] for b in line_boxes))
        y1 = max(0, min(b[1] for b in line_boxes))
        x2 = min(w, max(b[2] for b in line_boxes))
        y2 = min(h, max(b[3] for b in line_boxes))
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append((image.crop((x1, y1, x2, y2)), text))
    return crops
```

---

### Cell 4 — WildReceipt Extractor

> Skip running — output already in `smart_stock_dataset_v3`. Defines `extract_wildreceipt_crops()`.
> This is a **new standalone cell** added for v3. Insert between SROIE extractor and Dataset Builder.

**Why per-annotation crops (not line grouping):**
WildReceipt receipts have two-column layouts — item names on the left, prices on the right — with nearly identical Y coordinates per row. Y-proximity grouping merged left-column and right-column annotations into nonsense crops (e.g. `"*BtDietCoke £136.50 65.000@£2.10"` as one crop). This corrupted ~33% of training data and caused CER to regress from 0.088 → 0.339. Per-annotation cropping eliminates this entirely — each annotation becomes its own crop, no column detection needed. Also produces more training examples (29,713 vs 12,667 train crops from same images).

```python
import json

# Labels to EXCLUDE from WildReceipt:
# 0  = empty string / illegible text
# 25 = catch-all "other" (terminal IDs, legal footnotes, thank-you messages, promo text)
# All other labels (1=store name, 3=address, 5=phone, 7=date, 9=time,
# 11=item name, 13=quantity, 14=count, 15=item price, 17=subtotal,
# 18=subtotal label, 19=tax amount, 20=tax label, 22=tip, 23=total,
# 24=total label) are included.
WILDRECEIPT_EXCLUDE = {0, 25}

def extract_wildreceipt_crops(annotation_file: Path) -> list:
    """
    Parse a WildReceipt annotation file (one JSON object per line).
    Each annotation becomes its own crop — no line grouping.

    Why no grouping: WildReceipt has two-column layouts (item names left,
    prices right) with nearly identical Y coordinates per row. Y-proximity
    grouping merged columns into nonsense crops, corrupting ~33% of train data
    and causing CER regression from 0.088 → 0.339.

    Per-annotation cropping is also closer to TrOCR's pretraining distribution
    (single word/phrase level crops) than multi-word line crops.

    Each annotation box: [x1,y1, x2,y1, x2,y2, x1,y2] clockwise from top-left.
    Returns list of (PIL.Image, text) pairs.
    """
    crops = []

    with open(annotation_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            img_path = WILDRECEIPT_DIR / record["file_name"]
            if not img_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            img_w, img_h = image.size

            for ann in record["annotations"]:
                # Skip excluded labels and empty text
                if ann["label"] in WILDRECEIPT_EXCLUDE:
                    continue
                text = ann.get("text", "").strip()
                if not text:
                    continue

                # Box: [x1,y1, x2,y1, x2,y2, x1,y2]
                box = ann["box"]
                xs = [box[0], box[2], box[4], box[6]]
                ys = [box[1], box[3], box[5], box[7]]

                x1 = max(0, int(min(xs)))
                y1 = max(0, int(min(ys)))
                x2 = min(img_w, int(max(xs)))
                y2 = min(img_h, int(max(ys)))

                # Skip degenerate crops — beam search hangs on these
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue

                crops.append((image.crop((x1, y1, x2, y2)), text))

    return crops
```

---

### Cell 5 — Dataset Builder

> **First run:** downloads CORD + SROIE from HuggingFace, reads all WildReceipt images, builds crops, saves to `/kaggle/working/smart_stock_dataset_v3`. Takes ~35 min. Download this folder and upload as Kaggle dataset `smart-stock-dataset-v3`.
> **Subsequent runs:** `DATASET_SAVE.exists()` is True → loads instantly from disk, skips all building.

```python
import io
import json
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from PIL import Image

# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_bytes(img: Image.Image) -> bytes:
    """Encode PIL image to PNG bytes immediately — prevents RAM accumulation."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()

def iter_to_dataset(iterator) -> Dataset:
    """
    Convert iterator of (PIL Image, text) tuples into a HuggingFace Dataset.
    Encodes each image to bytes immediately — never holds multiple PIL objects in RAM.
    """
    img_bytes, texts = [], []
    for img, text in iterator:
        img_bytes.append(pil_to_bytes(img))
        texts.append(text)
    return Dataset.from_dict({"image_bytes": img_bytes, "text": texts})

# ── Combined dataset builder ──────────────────────────────────────────────────

DATASET_SAVE = DATASET_DIR  # points to smart-stock-dataset-v3 — loads if exists, builds if not

WR_TEST_KEEP = 1000  # cap WildReceipt test crops; remainder folds into train

def build_and_save_dataset():
    if DATASET_SAVE.exists():
        print(f"Dataset found at {DATASET_SAVE} — loading from disk...")
        return DatasetDict.load_from_disk(str(DATASET_SAVE))

    print("Building v3 line-crop dataset (CORD + SROIE + WildReceipt)...")

    # ── CORD ──────────────────────────────────────────────────────────────────
    print("Loading CORD...")
    cord = load_dataset("naver-clova-ix/cord-v2")

    def cord_iter(split):
        for ex in cord[split]:
            for crop, text in extract_cord_crops(ex["image"], ex["ground_truth"]):
                yield crop, text

    cord_train      = iter_to_dataset(cord_iter("train"))
    cord_validation = iter_to_dataset(cord_iter("validation"))
    cord_test       = iter_to_dataset(cord_iter("test"))
    print(f"  CORD  train: {len(cord_train)} | val: {len(cord_validation)} | test: {len(cord_test)}")
    del cord  # free RAM before loading next dataset

    # ── SROIE ─────────────────────────────────────────────────────────────────
    print("Loading SROIE...")
    sroie = load_dataset("sizhkhy/SROIE")

    def sroie_iter(split):
        for ex in sroie[split]:
            for crop, text in extract_sroie_crops(ex["images"], ex["words"], ex["bboxes"]):
                yield crop, text

    sroie_train = iter_to_dataset(sroie_iter("train"))
    sroie_test  = iter_to_dataset(sroie_iter("test"))
    print(f"  SROIE train: {len(sroie_train)} | test: {len(sroie_test)}")
    del sroie

    # ── WildReceipt ───────────────────────────────────────────────────────────
    print("Loading WildReceipt...")
    wr_train_crops = extract_wildreceipt_crops(WILDRECEIPT_DIR / "train.txt")
    wr_test_crops  = extract_wildreceipt_crops(WILDRECEIPT_DIR / "test.txt")

    wr_train_raw = iter_to_dataset(iter(wr_train_crops))
    wr_test_raw  = iter_to_dataset(iter(wr_test_crops))
    print(f"  WildReceipt raw — train: {len(wr_train_raw)} | test: {len(wr_test_raw)}")

    # 90% train / 10% val split from WildReceipt train.txt crops
    wr_train_split = wr_train_raw.train_test_split(test_size=0.1, seed=42)
    wr_train_final = wr_train_split["train"]
    wr_val         = wr_train_split["test"]
    print(f"  WildReceipt train (after val split): {len(wr_train_final)} | val: {len(wr_val)}")

    # Cap test.txt crops at WR_TEST_KEEP (1000), move remainder to train
    # Reason: 5,103 test crops * beam search = hours of eval time (killed prior session)
    if len(wr_test_raw) > WR_TEST_KEEP:
        wr_test_split    = wr_test_raw.train_test_split(test_size=WR_TEST_KEEP, seed=42)
        wr_test_final    = wr_test_split["test"]
        wr_test_to_train = wr_test_split["train"]
    else:
        wr_test_final    = wr_test_raw
        wr_test_to_train = None

    print(f"  WildReceipt test kept: {len(wr_test_final)} | moved to train: {len(wr_test_to_train) if wr_test_to_train else 0}")

    # ── Combine ───────────────────────────────────────────────────────────────
    # SROIE 2x weighted in train — more English receipt signal vs CORD's Indonesian
    # Excess WildReceipt test crops folded into train (free data)
    train_parts = [cord_train, sroie_train, sroie_train, wr_train_final]
    if wr_test_to_train:
        train_parts.append(wr_test_to_train)

    dataset_dict = DatasetDict({
        "train":      concatenate_datasets(train_parts),
        "validation": concatenate_datasets([cord_validation, wr_val]),
        "test":       concatenate_datasets([cord_test, sroie_test, wr_test_final]),
    })

    # Save to working dir — download and upload as Kaggle dataset smart-stock-dataset-v3
    save_path = WORKING_DIR / "smart_stock_dataset_v3"
    dataset_dict.save_to_disk(str(save_path))
    print(f"\n✅ Saved to {save_path}")
    print(f"   Train      : {len(dataset_dict['train'])}")
    print(f"   Validation : {len(dataset_dict['validation'])}")
    print(f"   Test       : {len(dataset_dict['test'])}")
    return dataset_dict

combined_dataset = build_and_save_dataset()
```

---

### Cell 6 — Augmentation

> Applied only to training images, inline during `preprocess_trocr`. Never on val or test.

```python
import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageOps

receipt_augmentation = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.15), p=0.6),  # thermal fade simulation
    A.GaussNoise(p=0.5),                                                 # scanner noise
    A.Rotate(limit=8, border_mode=cv2.BORDER_REPLICATE, p=0.5),         # crumple/tilt
    A.Perspective(scale=(0.02, 0.08), p=0.4),                           # phone photo angle
    A.MotionBlur(blur_limit=5, p=0.3),                                  # shaky photo
    A.ImageCompression(p=0.5),                                           # JPEG artifact
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),                           # focus blur
    A.RandomShadow(p=0.2),                                               # shadow on receipt
])

def apply_augmentation(pil_image: Image.Image) -> Image.Image:
    """Pad tiny line crops to min 32px height, then augment."""
    pil_image = pil_image.convert("RGB")
    if pil_image.height < 32:
        pad = 32 - pil_image.height
        pil_image = ImageOps.expand(pil_image, border=(0, pad//2, 0, pad - pad//2), fill=255)
    img_np = np.array(pil_image)
    augmented = receipt_augmentation(image=img_np)["image"]
    return Image.fromarray(augmented)
```

---

### Cell 7 — Preprocessing Function + TrOCRDataset

```python
import io
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

def preprocess_trocr(example, augment: bool = False):
    """
    Preprocess a single (image_bytes, text) example at access time.
    Returns dict with pixel_values (tensor) and labels (tensor).
    Padding tokens replaced with -100 so loss ignores them.
    """
    image = Image.open(io.BytesIO(example["image_bytes"])).convert("RGB")

    if augment:
        image = apply_augmentation(image)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    labels = processor.tokenizer(
        example["text"],
        padding="max_length",
        max_length=128,
        truncation=True,
    ).input_ids

    labels = [
        t if t != processor.tokenizer.pad_token_id else -100
        for t in labels
    ]

    return {
        "pixel_values": pixel_values.squeeze(),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class TrOCRDataset(TorchDataset):
    """
    On-the-fly preprocessing — processes each example at access time.
    Replaces .map() which caused either:
      - OOM (keep_in_memory=True on 46k examples)
      - OSError Errno 28 (cache writing to read-only /kaggle/input/)
    Zero disk usage, zero RAM accumulation, full dataset used.
    Compatible with Seq2SeqTrainer.

    NOTE: This is a PyTorch Dataset — it has no .select() method.
    For Optuna subset selection, use combined_dataset["train"].select(...)
    (the HuggingFace dataset) and then wrap in TrOCRDataset().
    """
    def __init__(self, hf_dataset, augment: bool = False):
        self.data    = hf_dataset
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return preprocess_trocr(self.data[idx], augment=self.augment)


train_dataset = TrOCRDataset(combined_dataset["train"],      augment=True)
val_dataset   = TrOCRDataset(combined_dataset["validation"], augment=False)
test_dataset  = TrOCRDataset(combined_dataset["test"],       augment=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
```

---

### Cell 8 — Model Setup

```python
from pathlib import Path
from transformers import VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# INPUT_CHECKPOINT_DIR already defined in Cell 1
# Redeclared here for clarity — safe to run again
INPUT_CHECKPOINT_DIR = Path("/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock/trocr-smart-stock/checkpoint-51348")

# Load base model from stored best weights
base_model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_INPUT))

# Freeze encoder entirely — ViT visual features preserved from pretraining
# This is the current performance ceiling; unfreeze top 2-4 blocks in future session
for param in base_model.encoder.parameters():
    param.requires_grad = False

# ── Checkpoint resume priority ────────────────────────────────────────────────
# 1. /kaggle/working/trocr-smart-stock  (current session checkpoints, most recent)
# 2. /kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock/trocr-smart-stock/checkpoint-51348 (uploaded from prior session — fallback)
# 3. Fresh LoRA (no prior checkpoint found)

def find_lora_checkpoints(directory: Path):
    """Find checkpoints with LoRA adapter weights in either format."""
    return sorted([
        ckpt for ckpt in directory.glob("checkpoint-*")
        if (ckpt / "lora_adapter").exists() or (ckpt / "adapter_config.json").exists()
    ])

working_checkpoints = find_lora_checkpoints(CHECKPOINT_DIR)

if working_checkpoints:
    latest = working_checkpoints[-1]
    adapter_path = latest / "lora_adapter" if (latest / "lora_adapter").exists() else latest
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
    print(f"Resumed from working dir: {adapter_path}")

elif INPUT_CHECKPOINT_DIR.exists() and (
    (INPUT_CHECKPOINT_DIR / "lora_adapter").exists() or
    (INPUT_CHECKPOINT_DIR / "adapter_config.json").exists()
):
    # INPUT_CHECKPOINT_DIR points directly to checkpoint-51348 — no glob needed
    adapter_path = INPUT_CHECKPOINT_DIR / "lora_adapter" if (INPUT_CHECKPOINT_DIR / "lora_adapter").exists() else INPUT_CHECKPOINT_DIR
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
    print(f"Resumed from input dir: {adapter_path}")

else:
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,           # rank — increase to 32 in future for more capacity
        lora_alpha=32,  # scaling factor = lora_alpha / r = 2.0
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # decoder attention projections only
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    print("No adapter checkpoint found — fresh LoRA applied")

# Required decoder config — must be set on model.config, not model.generation_config
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size

# Generation config — must go on model.generation_config, NOT model.config
model.generation_config.eos_token_id   = processor.tokenizer.sep_token_id
model.generation_config.early_stopping = True
model.generation_config.length_penalty = 1.0   # was 2.0 — penalised short outputs, caused hallucination
model.generation_config.num_beams      = 4
# no_repeat_ngram_size deliberately removed — receipt text has legitimate
# repetitions (e.g. "60.000 60.000") that ngram blocking incorrectly suppressed,
# causing test CER > 1.0 in v3 run 3 despite val CER of 0.0687
```

---

### Cell 9 — collate_fn

> Standalone cell — extracted from the Optuna block so it's always available even when Optuna is commented out.

```python
def collate_fn(batch):
    """
    Custom collator for TrOCR.
    pixel_values and labels are tensors returned by TrOCRDataset.__getitem__.
    torch.stack combines them into batches for the Trainer.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels       = torch.stack([item["labels"]       for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}
```

---

### Cell 10 — Training Arguments

**Block A — OLD CONFIG (fully commented out, do not use)**

> Kept for reference only. This was the config from Kaggle run 1 (lr=1.695e-4, cosine scheduler). Superseded by Block B below.

```python
# from transformers import Seq2SeqTrainingArguments

# training_args = Seq2SeqTrainingArguments(
#     output_dir=str(CHECKPOINT_DIR),   # /kaggle/working/trocr-smart-stock

#     num_train_epochs=5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,

#     # Hardcoded from Optuna best (Trial 3 of latest run)
#     learning_rate=1.695e-4,
#     warmup_ratio=0.0866,
#     weight_decay=0.01,
#     lr_scheduler_type="cosine",
#     max_grad_norm=1.0,

#     eval_strategy="epoch",
#     save_strategy="steps",
#     save_steps=500,
#     load_best_model_at_end=False,
#     save_total_limit=5,

#     predict_with_generate=True,
#     generation_max_length=128,

#     fp16=True,
#     dataloader_num_workers=2,

#     logging_dir=str(WORKING_DIR / "logs"),
#     logging_steps=50,
#     log_level="info",
#     report_to="none",
# )
```

**Block B — ACTIVE CONFIG (Technique 1: single GPU + correct LR)**

> This is the block that actually runs. `os.environ` line here is redundant (already set in Cell 1) but harmless.

```python
import os

# Redundant — already set in Cell 1 before all imports.
# Kept here as a safety reminder. Has no effect if torch already imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=str(CHECKPOINT_DIR),

    num_train_epochs=8,                # back to 8 — dataset now saved to disk (no rebuild cost)
                                       # val loss still dropping at epoch 6, model not converged
    per_device_train_batch_size=8,     # effective batch = 8 (no DataParallel)
    per_device_eval_batch_size=8,

    # Optuna best from Kaggle run (Trial 0 — CER 0.088 on subset, single GPU)
    learning_rate=1.4824e-4,
    warmup_ratio=0.02672,
    weight_decay=0.01,

    # Single cosine decay — more stable than restarts for fine-tuning
    # Restarts caused CER oscillation in v3 run 2 (0.092→0.079→0.088→0.077→0.082→0.078)
    # num_cycles=2 caused a restart at epoch ~3 which temporarily spiked CER
    lr_scheduler_type="cosine_with_restarts",
    lr_scheduler_kwargs={"num_cycles": 1},

    max_grad_norm=1.0,

    eval_strategy="epoch",
    save_strategy="epoch",           # must match eval_strategy for load_best_model_at_end
    load_best_model_at_end=True,     # was False — caused epoch 6 to be saved even though epoch 4 was best
    metric_for_best_model="eval_cer",
    greater_is_better=False,
    save_total_limit=3,

    predict_with_generate=True,
    generation_max_length=128,

    fp16=True,                         # mixed precision — required on T4
    dataloader_num_workers=2,

    logging_dir=str(WORKING_DIR / "logs"),
    logging_steps=50,
    log_level="info",
    report_to="none",                  # disable wandb

    # Technique 2 — Gradient Accumulation (COMMENTED OUT)
    # Uncomment if batch 8 causes OOM after encoder unfreezing in a future session.
    # gradient_accumulation_steps=2 gives effective batch 16 without DataParallel.
    # Does NOT invalidate the Optuna LR the way DataParallel does, because
    # accumulation doesn't change the optimizer step rate.
    # gradient_accumulation_steps=2,
)

print(f"Effective batch size: {training_args.per_device_train_batch_size}")
print(f"GPU count visible: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
```

---

### Cell 11 — Metrics

```python
!pip install jiwer -q
```

```python
import numpy as np
from jiwer import cer, wer

def compute_metrics(pred):
    """
    Compute CER and WER for Seq2SeqTrainer.
    Called at end of each eval epoch with generated token ids.
    """
    pred_ids   = pred.predictions
    labels_ids = pred.label_ids

    # pred_ids may be float logits (ndim=3) — argmax to get token ids
    if pred_ids.dtype != np.int64 and pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    # Clip to valid vocab range — prevents OverflowError during decode
    vocab_size = processor.tokenizer.vocab_size
    pred_ids   = np.clip(pred_ids, 0, vocab_size - 1)

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Replace -100 (padding mask) with pad token id before decoding
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    return {
        "cer": round(cer(label_str, pred_str), 4),
        "wer": round(wer(label_str, pred_str), 4),
    }
```

---

### Cell 12 — Optuna Hyperparameter Search (COMMENTED OUT)

> **Why commented:** Current LR (1.4824e-4) was tuned by Optuna on a prior subset run. Running Optuna again costs ~3–4 hours of the 12-hour session before any real training starts.
>
> **When to uncomment:** After a stable epoch 1 completes on v3 data and CER confirms ~0.088–0.092. If the LR feels off on the new dataset distribution, run Optuna in a separate dedicated session.
>
> **Config notes:** `n_trials=4` (not 10) — 4 trials × 1 epoch on 1/8 of 46,500 = ~5,800 examples per trial ≈ 45 min per trial ≈ 3 hr total. Fits in a dedicated session. `predict_with_generate=True` kept (unlike old v12 config) — we need CER not just loss to rank trials meaningfully.

```python
# import gc
# import copy
# from transformers import Seq2SeqTrainer
# from peft import LoraConfig, get_peft_model, TaskType

# def optuna_hp_space(trial):
#     return {
#         "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
#         "warmup_ratio":  trial.suggest_float("warmup_ratio", 0.0, 0.1),
#     }

# def model_init():
#     gc.collect()
#     torch.cuda.empty_cache()

#     m = VisionEncoderDecoderModel.from_pretrained(str(MODEL_INPUT))

#     for param in m.encoder.parameters():
#         param.requires_grad = False

#     lora_config = LoraConfig(
#         task_type=TaskType.SEQ_2_SEQ_LM,
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         target_modules=["q_proj", "v_proj"],
#         bias="none",
#     )
#     m = get_peft_model(m, lora_config)

#     m.config.decoder_start_token_id = processor.tokenizer.cls_token_id
#     m.config.pad_token_id           = processor.tokenizer.pad_token_id
#     m.config.vocab_size             = m.config.decoder.vocab_size
#     m.generation_config.eos_token_id   = processor.tokenizer.sep_token_id
#     m.generation_config.early_stopping = True
#     m.generation_config.length_penalty = 1.0
#     m.generation_config.num_beams      = 4
#     # no_repeat_ngram_size removed — causes CER > 1.0 on receipt text
#     return m

# search_args = copy.deepcopy(training_args)
# search_args.num_train_epochs = 1
# search_args.predict_with_generate = True
# search_args.eval_accumulation_steps = 4
# search_args.dataloader_num_workers = 0
# search_args.per_device_train_batch_size = 4
# search_args.per_device_eval_batch_size = 4
# search_args.output_dir = str(WORKING_DIR / "optuna_search")
# search_args.save_strategy = "no"
# search_args.load_best_model_at_end = False
# search_args.eval_strategy = "epoch"
# search_args.logging_steps = 50

# # Use 1/8 of training data for speed — val capped at 200 samples
# search_val_hf = combined_dataset["validation"].select(range(min(200, len(combined_dataset["validation"]))))
# search_val = TrOCRDataset(search_val_hf, augment=False)

# search_hf = combined_dataset["train"].select(range(len(combined_dataset["train"]) // 8))
# search_train = TrOCRDataset(search_hf, augment=True)

# search_trainer = Seq2SeqTrainer(
#     model_init=model_init,
#     args=search_args,
#     train_dataset=search_train,
#     eval_dataset=search_val,
#     data_collator=collate_fn,
#     compute_metrics=compute_metrics,
# )

# best_run = search_trainer.hyperparameter_search(
#     direction="minimize",
#     backend="optuna",
#     hp_space=optuna_hp_space,
#     n_trials=4,
# )

# del search_trainer
# gc.collect()
# torch.cuda.empty_cache()

# print("Best hyperparameters:", best_run.hyperparameters)
# for k, v in best_run.hyperparameters.items():
#     setattr(training_args, k, v)
# training_args.predict_with_generate = True
# print("Updated training_args:", training_args.learning_rate, training_args.warmup_ratio)
```

---

### Cell 13 — Trainer and Training

```python
from transformers import Seq2SeqTrainer, TrainerCallback
import shutil

class LoRASaveCallback(TrainerCallback):
    """
    Saves LoRA adapter weights alongside each Trainer checkpoint.

    Without this callback, PEFT silently resets adapter weights on checkpoint
    reload — the base model config is restored instead of the adapter state.
    This was Bug #3 in the training history.

    Saves to: checkpoint-{step}/lora_adapter/
    The find_lora_checkpoints() function in Cell 8 looks for this subdir.
    """
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        adapter_dir = checkpoint_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model"].save_pretrained(str(adapter_dir))
        print(f"LoRA adapter saved to {adapter_dir}")
        return control

print(f"LR: {training_args.learning_rate}")
print(f"Warmup ratio: {training_args.warmup_ratio}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")

# Remove stale checkpoints from prior bad runs (no lora_adapter subdir)
# These would be picked up by resume logic but have no usable adapter weights
if CHECKPOINT_DIR.exists():
    for ckpt in CHECKPOINT_DIR.glob("checkpoint-*"):
        has_adapter = (ckpt / "lora_adapter").exists()
        if not has_adapter:
            shutil.rmtree(ckpt)
            print(f"Removed stale checkpoint (no LoRA adapter): {ckpt}")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[LoRASaveCallback()],
)

# Resume from latest valid checkpoint if available
valid_checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint-*"))

if valid_checkpoints:
    # Working dir has checkpoints from a previous run in this session
    resume_from = str(valid_checkpoints[-1])
    print(f"Resuming from working dir: {resume_from}")
elif INPUT_CHECKPOINT_DIR.exists():
    # No working checkpoints — resume from uploaded input checkpoint.
    # FIXED: previously set resume_from = None on AttributeError, which
    # discarded ALL optimizer/scheduler state and restarted the LR schedule
    # from warmup every session (caused val CER to oscillate 0.0695-0.0739
    # instead of trending down in v3 run 3). Correct fix: copy the checkpoint,
    # delete only the incompatible scaler.pt, resume from the copy — this
    # restores optimizer.pt + scheduler.pt (real momentum + LR position).
    resume_copy = WORKING_DIR / "resume_checkpoint"
    if resume_copy.exists():
        shutil.rmtree(resume_copy)
    shutil.copytree(INPUT_CHECKPOINT_DIR, resume_copy)
    scaler_path = resume_copy / "scaler.pt"
    if scaler_path.exists():
        scaler_path.unlink()
    resume_from = str(resume_copy)
    print(f"Resuming from input checkpoint copy (optimizer+scheduler state intact): {resume_from}")
else:
    resume_from = None
    print("Starting fresh training")

trainer.train(resume_from_checkpoint=resume_from)
```

---

### Cell 14 — Training Curves

```python
import pandas as pd
import matplotlib.pyplot as plt

history = pd.DataFrame(trainer.state.log_history)
print(history.columns)
history.head()
```

```python
train_logs = history[history["loss"].notna()]
eval_logs  = history[history["eval_loss"].notna()]

plt.figure(figsize=(12,6))
plt.plot(train_logs["step"], train_logs["loss"], label="Training Loss")
plt.plot(eval_logs["step"], eval_logs["eval_loss"], label="Validation Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("TrOCR Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
```

```python
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].plot(eval_logs["epoch"], eval_logs["eval_cer"], marker="o")
axes[0].set_title("CER")
axes[0].set_xlabel("Epoch")
axes[1].plot(eval_logs["epoch"], eval_logs["eval_wer"], marker="o")
axes[1].set_title("WER")
axes[1].set_xlabel("Epoch")
plt.tight_layout()
plt.show()
```

```python
summary = eval_logs[["epoch", "eval_loss", "eval_cer", "eval_wer"]]
summary
```

---

### Cell 15 — Save & Export

> Only run this if the new CER beats 0.0856. Otherwise skip to avoid overwriting the best model.

```python
from peft import PeftModel

# Merge LoRA weights into base model — produces a standard VisionEncoderDecoderModel
# with no PEFT dependency. Required for inference without peft installed.
merged_model = model.merge_and_unload()

for save_path in [str(BEST_MODEL_DIR)]:
    merged_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    merged_model.generation_config.save_pretrained(save_path)
    print(f"Saved to: {save_path}")

expected_files = [
    "model.safetensors", "config.json", "generation_config.json",
    "tokenizer_config.json", "tokenizer.json", "processor_config.json",
]
print(f"\nVerification ({BEST_MODEL_DIR}):")
for fname in expected_files:
    fpath = BEST_MODEL_DIR / fname
    exists = fpath.exists()
    size = fpath.stat().st_size / 1e6 if exists else 0
    print(f"  {'✅' if exists else '❌'} {fname} ({size:.1f} MB)")
```

> Expected total size: ~1.3 GB (`model.safetensors` ~1.28 GB, rest are small config files).
> After session completes: go to notebook Output tab → three dots next to `trocr-smart-stock-best/` → Create Dataset → upload as `trocr-smart-stock-best`.

---

### Cell 16 — Evaluate on Test Set

> **Never use `trainer.evaluate(test_dataset)`** — hangs on degenerate 1-pixel-wide crops in the test set. The previous session lost 6+ hours to this. Always use the manual loop below.
>
> **Critical:** `model.generate()` must use keyword argument `pixel_values=pixel_values` not positional. `PeftModelForSeq2SeqLM.generate()` does not accept positional args — caused all 9,301 test samples to be skipped silently in v3 run 2 with 0.0000 CER reported.
>
> **Critical (new):** must call `merged_model.generate()`, NOT `model.generate()`. `model.merge_and_unload()` (Cell 15) mutates the PeftModel's module tree in place and removes the LoRA layers it references — after that call, `model` is a stale/inconsistent object. Calling `model.generate()` on it produces corrupted output (token reordering, digit corruption) even though the underlying weights are fine. This caused Test CER 1.4052 in v3 run 3 despite val CER 0.0695.

```python
from jiwer import cer, wer

merged_model.eval()
all_preds, all_labels = [], []
skipped = 0
first_error_printed = False

for idx, sample in enumerate(combined_dataset["test"]):
    try:
        image = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")
        w, h = image.size
        if w < 4 or h < 4:          # skip degenerate crops — beam search hangs on these
            skipped += 1
            continue

        pixel_values = processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(merged_model.device)

        with torch.no_grad():
            # MUST be keyword arg — PeftModelForSeq2SeqLM.generate() does not
            # accept positional pixel_values. Silently skipped all 9301 samples
            # in v3 run 2 when passed positionally.
            # MUST call on merged_model, not model — see critical note above.
            generated_ids = merged_model.generate(
                pixel_values=pixel_values,
                max_new_tokens=128,
            )

        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_preds.append(pred)
        all_labels.append(sample["text"])

    except Exception as e:
        if not first_error_printed:
            print(f"First exception at idx={idx}: {type(e).__name__}: {e}")
            first_error_printed = True
        skipped += 1

if all_preds:
    test_cer = cer(all_labels, all_preds)
    test_wer = wer(all_labels, all_preds)
    print(f"Test CER : {test_cer:.4f}")
    print(f"Test WER : {test_wer:.4f}")
else:
    print("No predictions collected — all samples skipped or errored")

print(f"Skipped  : {skipped} / {len(combined_dataset['test'])}")
print(f"Evaluated: {len(all_preds)} / {len(combined_dataset['test'])}")

# Targets: CER ≤ 0.05, WER ≤ 0.10
```

---

### Cell 17 — Qualitative Evaluation

```python
import io
from PIL import Image

sample = combined_dataset["test"][0]
image = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")

pixel_values = processor(
    image,
    return_tensors="pt"
).pixel_values.to(merged_model.device)   # merged_model, not model — see Cell 16 note

generated_ids = merged_model.generate(pixel_values=pixel_values, max_new_tokens=128)

prediction = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("GROUND TRUTH:")
print(sample["text"])
print("")
print("="*80)
print("PREDICTION:")
print(prediction)

image
```

---

## Training History

| Run                | Setup                                                                                        | Best Val CER     | Best Val WER     | Notes                                                                                                                                                                                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------- | ---------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Full finetune      | All 333M params                                                                              | 0.1758           | —               | Plateau, optimizer instability                                                                                                                                                                                                                                           |
| LoRA first (Colab) | Frozen enc + LoRA dec                                                                        | 0.0894           | —               | 1 epoch only                                                                                                                                                                                                                                                             |
| Optuna search      | 1 epoch, 1/8 data, batch 4                                                                   | 0.0803–0.0886   | —               | Best: lr=1.4824e-4, warmup=0.02672                                                                                                                                                                                                                                       |
| Kaggle run 1       | 5 epochs, DataParallel batch 16                                                              | 0.1325           | —               | DataParallel doubled batch silently                                                                                                                                                                                                                                      |
| Kaggle run 2       | 8 epochs, attempted single GPU                                                               | 0.1332           | —               | DataParallel still fired — env var after torch import                                                                                                                                                                                                                   |
| v3 run 1           | 6 epochs, v3 46.5K train, WR line-grouped                                                    | 0.3396           | 0.5942           | WildReceipt column merging bug corrupted 33% of data                                                                                                                                                                                                                     |
| **v3 run 2** | 6 epochs, v3 68.5K train, WR per-annotation                                                  | **0.0771** | **0.2383** | `load_best_model_at_end=False` saved epoch 6 not epoch 4                                                                                                                                                                                                               |
| **v3 run 3** | 8 epochs, resumed from checkpoint-51348,`load_best_model_at_end=True`                      | **0.0687** | **0.2159** | Epoch 6. Test CER 1.47 traced to a`no_repeat_ngram_size` misdiagnosis at the time — real cause found in run 4 (see below).                                                                                                                                            |
| **v3 run 4** | 8 epochs, resumed from checkpoint-51348 with fresh optimizer state (Bug 5, then undiagnosed) | 0.0695           | 0.2167           | Epoch 7, checkpoint-59906. Flat vs. run 3 — later understood to be caused by Bug 5 (LR schedule restarting each session). Reported Test CER 1.4052 — later found to be Bug 6 (test eval running on stale`model` instead of `merged_model`), not a real regression. |

**Stored best:** `trocr-smart-stock-best` in `trocr-smart-stock-model` — CER 0.0687, WER 0.2159 (v3 run 3, epoch 6). Run 4's 0.0695 did not beat this, so the stored best model is still from run 3 pending a clean re-run with Bug 5 and Bug 6 both fixed.
**Path:** `/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock-best/trocr-smart-stock-best`
**Convention:** Always overwrite `trocr-smart-stock-best/` with the new best model after each run. No versioned names — the Kaggle dataset version number tracks history.

**Key observation:** CER/WER gap (~3×) is expected for receipt OCR — one wrong character fails entire words like `"BCCHOCCUPCAKES"`. WER improves naturally as CER improves, not a separate problem.

**Key observation — test CER > 1.0 (corrected):** The original theory (`no_repeat_ngram_size=3` blocking legitimate repetition) was a plausible-sounding misdiagnosis for run 3. Run 4 reproduced the same test/val gap with `no_repeat_ngram_size` already removed, which ruled that theory out. The actual cause (found in run 4, and retroactively likely explains run 3's number too) is Bug 6 below: test eval was calling `model.generate()` on the stale PeftModel object left behind after `model.merge_and_unload()`, not on `merged_model`. Val CER (teacher-forced, computed before the merge step) was the trustworthy number all along.

---

## Bugs Fixed

### Bug 1 — DataParallel not disabled ✅ FIXED

`os.environ["CUDA_VISIBLE_DEVICES"] = "0"` was placed after PyTorch imports → Kaggle's 2×T4 silently ran DataParallel → effective batch doubled 8→16 → Optuna LR tuned at batch 4 was mismatched 4× over.

**Fix:** `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` is now the absolute first Python line in Cell 1, before even `from pathlib import Path`.

### Bug 2 — Test eval hangs on degenerate crops ✅ FIXED

Test set contains 1-pixel-wide image crops (`torch.Size([3, 42, 1])`). Beam search with `num_beams=4` hangs indefinitely on these. Previous session spent 6+ hours in test eval and never finished.

**Fix:** Manual loop in Cell 16 skips any sample with `w < 4 or h < 4`. Never use `trainer.evaluate()` on the test set.

### Bug 5 — AttributeError: scaler.load_state_dict on Trainer resume ✅ FIXED (was previously only DOCUMENTED)

When resuming from a checkpoint saved with `fp16=True`, the Trainer calls
`self.accelerator.scaler.load_state_dict()` to restore the gradient scaler.
In newer versions of transformers/accelerate, the scaler is `None` when fp16
is handled differently, causing `AttributeError: 'NoneType' has no attribute 'load_state_dict'`.

**Cause:** Checkpoint saved with one transformers version, resumed with a newer one
where fp16 scaler initialization changed.

**Previous "fix" (v3 run 4) — this was the actual problem, not a real fix:** setting
`resume_from = None` in the Trainer cell for the input checkpoint branch discarded
*all* optimizer and scheduler state, not just the incompatible scaler. This meant
the cosine LR schedule restarted from warmup every session instead of continuing
to anneal — the direct cause of val CER oscillating (0.0695–0.0739 across epochs
in run 4) instead of trending down, and of run 4 failing to beat run 3's 0.0687
despite 8 additional epochs of training.

**Corrected fix:** copy `INPUT_CHECKPOINT_DIR` to a writable location
(`WORKING_DIR / "resume_checkpoint"`), delete only `scaler.pt` from the copy, and
pass the copy's path to `resume_from`. This restores `optimizer.pt` and
`scheduler.pt` (real Adam momentum and correct position in the LR curve) while
still avoiding the `AttributeError`. Confirmed `checkpoint-51348` contains both
`optimizer.pt` and `scheduler.pt` alongside `scaler.pt`, so this fix has real
state to restore.

### Bug 4 — Test eval skips all samples silently ✅ FIXED

`model.generate(pixel_values)` — passing `pixel_values` as a positional argument to `PeftModelForSeq2SeqLM.generate()` raises `TypeError: takes 1 positional argument but 2 were given`. The `except Exception` block silently swallowed this on every sample, reporting CER 0.0000 and WER 0.0000 with all 9,301 samples skipped. Affected v3 run 2 entirely — true test CER from that run is unknown.

**Fix:** Use keyword argument: `model.generate(pixel_values=pixel_values, max_new_tokens=128)`. Also added `first_error_printed` flag so the first exception surfaces instead of being swallowed.

### Bug 3 — PEFT checkpoint adapter reset ✅ FIXED

Without `LoRASaveCallback`, PEFT silently restores base model config instead of adapter weights on checkpoint resume. Training effectively restarts from scratch each time.

**Fix:** `LoRASaveCallback` saves `lora_adapter/` subdir alongside every checkpoint. `find_lora_checkpoints()` filters for checkpoints containing this subdir.

### Bug 6 — Test eval running on stale PeftModel after merge_and_unload() ✅ FIXED

`model.merge_and_unload()` (Cell 15) is destructive: it folds LoRA deltas into the base layers and removes the LoRA modules from the underlying module tree that `model` (the PeftModel wrapper) still references. After this call, `model` is a stale/inconsistent object — only the returned `merged_model` is safe for inference. Cell 16 (test eval) and Cell 17 (qualitative check) were both still calling `model.generate()` after the merge, producing corrupted output (token reordering, digit corruption) despite correct underlying weights. This is the real explanation for the val/test CER gap previously attributed to `no_repeat_ngram_size`.

**Fix:** Cells 16 and 17 now call `merged_model.generate(...)` and use `merged_model.device`, not `model`.

---

## Improvement Roadmap

### Tier 1 — Do next

| Technique                              | Status                          | Notes                                                                                                                                                                                                  |
| -------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Fix`load_best_model_at_end=False`    | ✅ Done (v3 run 3)              | Now saves true best epoch.                                                                                                                                                                             |
| Switch to`num_cycles=1`              | ✅ Done (v3 run 3)              | Single cosine decay, stable convergence.                                                                                                                                                               |
| Fix generation config                  | ✅ Done                         | Removed`no_repeat_ngram_size=3`, set `length_penalty=1.0`. Was causing test CER > 1.0.                                                                                                             |
| Continue training from best checkpoint | ⏳ Next run                     | Resume from`trocr-smart-stock-v2` checkpoint. Val loss still declining at epoch 6 (0.2828). More epochs will push CER below 0.06.                                                                    |
| Partial encoder unfreeze               | ⏳ After CER stabilises < 0.065 | Unfreeze top 2 ViT encoder blocks alongside LoRA. Biggest remaining performance lever. Add`gradient_accumulation_steps=2` if OOM. Each step ~0.7s vs 0.55s — reduce to 5–6 epochs when unfreezing. |
| Re-run Optuna                          | ⏳ After encoder unfreeze       | LR tuned on frozen encoder — gradient flow changes after unfreeze. Dedicated session only, not during main training.                                                                                  |

### Tier 2 — Medium impact

| Technique                                                 | Notes                                                                                            |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Label smoothing (`label_smoothing_factor=0.1`)          | Helps with noisy WildReceipt labels. Add to`Seq2SeqTrainingArguments`.                         |
| LoRA rank increase (r=16 → r=32)                         | ~3M extra params. May help with WildReceipt's text diversity.                                    |
| Gradient accumulation (`gradient_accumulation_steps=2`) | Effective batch 16. Use only if OOM after encoder unfreeze. Already in notebook, just uncomment. |

### Tier 3 — Low effort, test after training

| Technique                                    | Notes                                                                          |
| -------------------------------------------- | ------------------------------------------------------------------------------ |
| Beam search tuning (`num_beams` 4 → 6–8) | Inference only, no retraining needed. Change in model setup cell.              |
| WildReceipt train weighting                  | Currently 1×. Monitor if WildReceipt dominates training signal vs CORD/SROIE. |

---

## Troubleshooting

| Issue                                                                       | Fix                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DataParallel fires despite env var                                          | Must be set before any`import torch` — even in prior cells. Restart kernel and run Cell 1 first.                                                                                                                                                                                                                                            |
| Test eval hangs for hours                                                   | Use manual loop in Cell 16 with`w < 4 or h < 4` skip. Never `trainer.evaluate()` on test.                                                                                                                                                                                                                                                  |
| LoRA adapter resets on resume                                               | `LoRASaveCallback` not attached or removed. Re-add to `callbacks=[LoRASaveCallback()]`.                                                                                                                                                                                                                                                    |
| Stale checkpoints without adapter                                           | Trainer cell removes them:`shutil.rmtree(ckpt)` for checkpoints missing `lora_adapter/`.                                                                                                                                                                                                                                                   |
| Dataset rebuilds every session                                              | `DATASET_SAVE.exists()` check skips rebuild — only triggers if v3 Kaggle dataset not mounted.                                                                                                                                                                                                                                               |
| OOM on batch 8                                                              | Uncomment`gradient_accumulation_steps=2` in training_args — keep batch 8 as-is.                                                                                                                                                                                                                                                             |
| WildReceipt images not found                                                | Check`WILDRECEIPT_DIR` — only needed during build, not inference or training on saved dataset.                                                                                                                                                                                                                                              |
| `AttributeError: TrOCRDataset has no .select()`                           | It's a PyTorch Dataset. Use`combined_dataset["train"].select(...)` then wrap in `TrOCRDataset()`.                                                                                                                                                                                                                                          |
| `GaussNoise` var_limit warning                                            | Use`A.GaussNoise(p=0.5)` without named args — albumentations API changed.                                                                                                                                                                                                                                                                   |
| CORD train: 0 crops extracted                                               | `quad` is in `valid_line[].words[].quad` — NOT in `gt_parse.menu`.                                                                                                                                                                                                                                                                      |
| fp16 scaler error on resume                                                 | Ensure`fp16=True` in training_args when resuming — checkpoint contains scaler state.                                                                                                                                                                                                                                                        |
| `Missing keys: decoder.output_projection.weight`                          | Harmless TrOCR architecture warning. Safe to ignore.                                                                                                                                                                                                                                                                                           |
| `ViTImageProcessor` fast processor warning                                | Safe to ignore, or pass`use_fast=False` to `TrOCRProcessor.from_pretrained()`.                                                                                                                                                                                                                                                             |
| Outputs lost after session                                                  | Quick Save = code only. Use**Save & Run All** to commit `/kaggle/working/` permanently.                                                                                                                                                                                                                                                |
| `warmup_ratio is deprecated`                                              | Harmless for now, will be removed in transformers v5.2.                                                                                                                                                                                                                                                                                        |
| CER regresses after adding WildReceipt                                      | Was caused by two-column merging bug in Y-proximity grouping — fixed by switching to per-annotation crops.                                                                                                                                                                                                                                    |
| `AttributeError: 'NoneType' has no attribute 'load_state_dict'` on resume | Scaler state in checkpoint incompatible with current transformers version. Do NOT set`resume_from = None` (discards optimizer+scheduler too). Instead copy the checkpoint to a writable dir, delete only `scaler.pt`, and resume from the copy.                                                                                            |
| Test CER > 1.0 (e.g. 1.47, 1.40)                                            | Check whether test eval is calling`model.generate()` after `model.merge_and_unload()` has run — `model` is stale post-merge. Must call `merged_model.generate()`. (Previously misdiagnosed as `no_repeat_ngram_size=3`/`length_penalty` — those settings are still worth keeping as removed/1.0, but were not the actual cause.) |
| Test CER still elevated after fixing merged_model call                      | Re-check`length_penalty` (should be 1.0, not 2.0) and confirm `no_repeat_ngram_size` is not set — receipt text has legitimate repetition (e.g. `60.000 60.000`) that ngram blocking incorrectly suppresses.                                                                                                                             |
| Model reorders words or hallucinates in test                                | Generation config issue, not model quality. Val CER (teacher-forced) is the reliable metric during training. Fix generation config before judging test output.                                                                                                                                                                                 |
| `model.generate(pixel_values)` TypeError                                  | PeftModelForSeq2SeqLM doesn't accept positional args. Use`model.generate(pixel_values=pixel_values, max_new_tokens=128)`.                                                                                                                                                                                                                    |
| Test eval reports 0.0000 CER with all samples skipped                       | Silent exception swallowing. Check`first_error_printed` output — likely the generate() positional arg bug above.                                                                                                                                                                                                                            |
| Best epoch not saved — final epoch saved instead                           | `load_best_model_at_end=False` in training_args. Set to True with `metric_for_best_model="eval_cer"`, `greater_is_better=False`, and `save_strategy="epoch"` matching `eval_strategy`.                                                                                                                                               |
| CER oscillates between epochs (e.g. 0.092→0.079→0.088)                    | `num_cycles=2` in cosine_with_restarts causes LR spikes at restart points. Switch to `num_cycles=1` for stable decay.                                                                                                                                                                                                                      |

---

## Critical Bug — generation_config Reset on merge_and_unload()

**Symptom:** Test CER of 1.36–1.40 (and per-source diagnostic CER of 0.78) despite val CER of 0.0631 on the same checkpoint.

**Root cause:** `model.merge_and_unload()` (PEFT) returns a model whose `generation_config` resets to HuggingFace defaults, silently dropping the manually-set `eos_token_id`, `decoder_start_token_id`, `pad_token_id` overrides applied during model setup (Cell 23). Without a correct `eos_token_id`, generation never stops cleanly — predictions run on for many extra tokens, producing character-level CER far above 1.0 on a subset of samples, which dominates the aggregate.

**Fix — must be re-applied every time after `merge_and_unload()`, before eval or save:**

```python
merged_model.generation_config.eos_token_id            = processor.tokenizer.sep_token_id
merged_model.generation_config.decoder_start_token_id  = processor.tokenizer.cls_token_id
merged_model.generation_config.pad_token_id            = processor.tokenizer.pad_token_id
merged_model.generation_config.max_length              = 256
merged_model.generation_config.max_new_tokens          = 128
merged_model.generation_config.early_stopping          = True
merged_model.generation_config.num_beams               = 4
merged_model.generation_config.length_penalty          = 1.0
merged_model.generation_config.no_repeat_ngram_size     = 0
```

Confirmed working values for this tokenizer: `eos_token_id=2`, `decoder_start_token_id=2` (cls), `pad_token_id=1`, `bos_token_id=0`.

**Also found stale in downloaded Kaggle artifacts (both `trocr-smart-stock-model` v1 and `smart-stock-model-data` v4):** saved `generation_config.json` had `max_length: 20` (silently truncates all output) and `use_cache: false` (major inference slowdown). Both were fixed post-hoc on the locally saved copy — see Production Inference Pipeline section. **Action item:** re-save the config correctly at export time in any future training run, not just patch after download.

---

## Production Inference Pipeline (Real Receipt Photos)

### Problem: full-image inference hallucinates (GitHub Issue #8, closed)

Feeding a full receipt photo (not a line crop) into TrOCR produces fluent but entirely unrelated hallucinated text (e.g. Malaysian retail chains on Pakistani FBR receipts). Root cause: TrOCR was trained exclusively on single-line crops; a full multi-line image is out-of-distribution for the encoder, and the decoder falls back to memorized training patterns. This is not a model quality issue — val CER (0.0631) was never measuring this scenario.

**Fix: line-segmentation stage required before TrOCR, matching training distribution.**

### Segmentation approach evolution

1. **Attempt 1 — Hough deskew + Otsu binarize + horizontal morphological dilation + contour detection.** Cheap, no model, but fragile on real phone photos (hands, shadows, wrinkles, tilt) — produced 129 boxes on a receipt with ~40 real lines, mostly noise/background/split-word junk. Predictions on these crops were garbage (model hallucinated on malformed inputs — same failure mode as the WildReceipt tiny-crop bug).
2. **Fix attempt — size + blank filter** (`width ≥ 50px, height ≥ 10px, dark_pixel_ratio ≥ 2%`, matching the training-time filter). Reduced box count only marginally (129→112) — confirmed the problem was box *quality*, not just size, so filtering alone was insufficient.
3. **Final approach — EasyOCR (CRAFT) as detector-only, TrOCR remains the recognizer.** CRAFT is a proper trained text-detection model, far more robust to skew/shadows/hands than the contour heuristic. Box quality became "almost perfect" on all 4 real test receipts.

### Confirmed working pipeline

```
Receipt photo
     │
     ▼
CRAFT (EasyOCR detector) → line-level bounding boxes
     │
     ▼
Size + blank filter (width≥50px, height≥10px, dark_ratio≥2%)
     │
     ▼
Batched TrOCR generate() → per-line text
```

### Speed — CPU vs GPU

| Config | CRAFT detect | TrOCR recognize (65 lines) | Notes |
|---|---|---|---|
| CPU, sequential `predict_crop()` loop | ~20s | 653s | Original naive implementation — unusable |
| GPU (T4), fixed generation_config, batched `predict_batch()` | 1.34–1.39s | 3.62–3.64s | `num_beams=1, use_cache=True` in generation_config; batch_size=16 |
| CPU (local dev machine, no GPU), same fixed pipeline | 53.5s | 480s | Confirms GPU is the dominant lever, not just batching — CPU alone is not production-viable even with correct config |

**Root causes of the original 653s/receipt slowdown (all fixed):**
- `num_beams=4` (beam search) → changed to `num_beams=1` (greedy) for inference
- `use_cache=False` in generation_config → changed to `True` (enables KV caching)
- Sequential `model.generate()` per crop → batched into groups of 16 via `predict_batch()`

**Decision: Kaggle GPU is not usable for production serving** (session-based, no persistent endpoint, 30hr/week quota is for experimentation only). **Locked in: CPU-only serving**, planned via CTranslate2 int8 export (same pattern as `m2m100-roman-deploy`) to bring TrOCR's CPU latency down from the current unoptimized baseline. This plan is superseded by the PaddleOCR evaluation — see Decision Log below.

### Local dev environment setup (reference)

- Python venv-based install. **Gotcha:** installing packages with `pip install --break-system-packages` before activating a venv installs to `~/.local` (user site), not the venv — venv remains empty and imports fail (`ModuleNotFoundError: No module named 'cv2'`). Always `source .venv/bin/activate` **before** any `pip install`.
- To clean up an accidental user-site install: `pip list --user` to confirm contents belong only to the mistaken install, then `rm -rf ~/.local/lib/python<version>/site-packages/*`.
- Required packages for local inference: `easyocr transformers torch opencv-python-headless pillow numpy`.

### Local inference script

See `ml_service/scripts/local_inference.py` (or equivalent) — loads the saved model from `ml_service/models/trocr-smart-stock/`, runs CRAFT detection + filtered + batched TrOCR recognition on a single receipt image. Validated output matches Kaggle GPU run output exactly (same predicted lines), confirming the local setup and saved model/config are correct — only speed differs (CPU vs GPU), not accuracy.

---

## Decision Log

| Decision | Date/Context | Reasoning |
|---|---|---|
| Stop LoRA tuning after r=32 test | r=32 gave 0.0633 vs r=16's 0.0631 — no improvement | 4 techniques (LR retuning, resume, encoder unfreeze, rank increase) all failed to break plateau. Error analysis shows WildReceipt label noise dominates remaining error, not model capacity. |
| CPU-only production serving, no GPU hosting | After confirming Kaggle GPU quota can't back a prod endpoint | Kaggle is experimentation-only, no persistent serving. GPU hosting (Modal/RunPod/Lambda) considered but deferred as unnecessary cost for a CV/portfolio project; CTranslate2 int8 CPU export was the agreed path. |
| **Switch to PaddleOCR (pretrained, no fine-tuning)** | Issue #9 — PaddleOCR pretrained PP-OCRv6 benchmarked against TrOCR on the same 4 real receipts | PaddleOCR beat TrOCR decisively on both accuracy and CPU latency (~5s vs ~480s/receipt) **without any fine-tuning** — pretrained PP-OCRv6 small det/rec models (doc-orientation/unwarping/textline-orientation disabled) were sufficient. Original plan assumed fine-tuning would be required (see prior row); it was not. TrOCR's 2-month fine-tuning effort is retained as historical record — it established the dataset, line-detection pipeline (CRAFT), and debugging playbook (generation_config resets, OOD hallucination) that were directly reused to validate PaddleOCR quickly. **Caveat:** benchmarking PaddleOCR's full pipeline on pre-cropped val line images gives an invalid, inflated CER (0.66+) — detection/orientation stages assume full-document input. Real-receipt-level comparison is the only valid benchmark. |
| Proceed with NER+OCR integration using PaddleOCR output | WER concern resolved — PaddleOCR output is accurate on item/price/quantity lines across all 4 real receipts | Original TrOCR WER (0.234) blocked NER integration since errors would compound through Normalization. PaddleOCR output is accurate enough to proceed. **Unblocked** — Stage 2 (NER) can now use PaddleOCR output. |

---

## GitHub Issue #8 — Resolution Note

**Issue:** `[ML][OCR] Full-receipt inference produces hallucinated OOD text — missing line-segmentation stage before TrOCR`

**Status: Closed.** Suggested closing comment, if not already added:

> Resolved. Root cause confirmed as described — full-image input is out-of-distribution for a model trained exclusively on line crops.
>
> Implemented fix: CRAFT (via EasyOCR) as a dedicated line-detection stage before TrOCR, replacing the originally-proposed projection-profiling approach. Projection profiling was tried first as a cheap contour-based heuristic (Hough deskew + Otsu binarize + morphological dilation) per the issue's suggested investigation order, but proved too fragile on real phone photos (hands, shadows, tilt) — produced ~2-3x more boxes than actual lines, with poor localization, and downstream OCR predictions were still garbage on the resulting crops. Switched to CRAFT, which gave near-perfect line boxes on all 4 real test receipts.
>
> Acceptance criteria status:
> - [x] Line segmentation stage implemented and integrated before TrOCR inference (CRAFT + size/blank filter)
> - [x] Re-ran inference on the 4 sample images — output is no longer hallucinated; item-level lines (product names, prices) are now largely correct
> - [ ] Full-image CER/accuracy benchmark on real photos — not formally established as a numeric target; qualitative review used instead. **Deferred** — full-receipt OCR is being re-evaluated with PaddleOCR as a possible TrOCR replacement (see OCR_Training.md Decision Log), so a formal end-to-end benchmark will be defined against whichever model is chosen, not against TrOCR alone.
> - [x] Documented in OCR_Training.md (Production Inference Pipeline section)
>
> Follow-up: current TrOCR WER (0.234) is high enough that even with correct line segmentation, output quality is being reconsidered before Stage 2 (NER) integration — tracked separately, not blocking this issue's closure.

---

## GitHub Issue #9 — Resolution Note

**Issue:** `[ML][OCR] Evaluate PaddleOCR as TrOCR replacement — CPU latency + WER too high for production`

**Status: Closed.** Suggested closing comment:

> Resolved. PaddleOCR (PP-OCRv6, pretrained — no fine-tuning) was benchmarked against fine-tuned TrOCR on the same 4 real receipt photos.
>
> **Result: PaddleOCR wins decisively on both accuracy and CPU latency**, without fine-tuning:
> - Item/price/quantity line text was consistently correct across all 4 receipts (e.g. `Core C Sachet 1's 1000mg`, `1137.45` — TrOCR gave `CoreCSachet`, `1000ng`, `117.45`)
> - CPU latency: ~5–6s/receipt (PP-OCRv6 small det/rec, doc-orientation/unwarping/textline-orientation disabled) vs TrOCR's ~480s/receipt (unoptimized) on the same hardware
>
> This changes the original proposal: fine-tuning was assumed necessary going in, but pretrained PaddleOCR already cleared both success criteria (CER/WER *and* CPU speed) from the issue description, so no fine-tuning was performed.
>
> **Important caveat found during evaluation, documented in OCR_Training.md:** running PaddleOCR's full pipeline (detection + doc-orientation + unwarping) on pre-cropped single-line val images (rather than full receipt photos) gives an invalid, misleadingly high CER (0.66+) — those pipeline stages assume full-document input. The valid comparison is full-receipt-level only, which is what the 4-photo benchmark above used.
>
> Acceptance / success criteria status:
> - [x] PaddleOCR benchmarked against TrOCR on the same 4 real receipt photos
> - [x] Beats TrOCR on accuracy (real-receipt level)
> - [x] Beats TrOCR on CPU inference speed
> - [x] Decision documented in OCR_Training.md Decision Log
>
> **Decision: PaddleOCR (pretrained) is now the production Stage 1 OCR model.** TrOCR fine-tuning history retained in OCR_Training.md as historical record — the dataset, CRAFT line-detection pipeline, and debugging experience it produced were directly reused to evaluate PaddleOCR quickly.
>
> Next: Stage 2 (NER) integration using PaddleOCR output — no longer blocked by OCR output quality.

---

## Full Notebook Appendix — All Cells (Including Commented-Out)

Complete reference copy of `kaggle_trocr.ipynb`, every cell in order, including disabled/commented blocks kept for historical reference (old training arg configs, Optuna search, beam sweep, crop diagnostics). This is the authoritative source; sections above summarize the key decisions and fixes extracted from it.

## Stage 2 — DistilBERT NER (Not Yet Started)

No longer blocked by OCR output quality — PaddleOCR output is accurate enough on item/price/quantity lines to proceed.

CORD's structured `ground_truth` JSON maps directly to food NER schema with no manual annotation:
`nm` → `FOOD_ITEM` | `cnt` → `QUANTITY` | `price` → `PRICE`

SROIE tags (COMPANY, ADDRESS, DATE, TOTAL) have zero overlap with food entities — not used for NER.

Full NER fine-tuning documented here once TrOCR achieves CER ≤ 0.05.
### ### Smart-Stock TrOCR — Kaggle

### ### Setup & Paths (pip installs + env)

**Cell 2:**

```python
# ── Install dependencies ─────────────────────────────────────────────────────
!pip install -q transformers==5.0.0 datasets evaluate jiwer albumentations
!pip install -q "peft==0.13.2"
!pip install -q "torchao>=0.16.0"
!pip install -q optuna
```

**Cell 3:**

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path

INPUT_DIR   = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")

# ── Dataset — v4 filtered clean dataset ──────────────────────────────────────
DATASET_DIR = Path("/kaggle/input/datasets/maazahmad69/smart-stock-dataset-v4/smart_stock_dataset_v4")

# ── WildReceipt raw ───────────────────────────────────────────────────────────
WILDRECEIPT_DIR = Path("/kaggle/input/datasets/maazahmad69/wild-receipt/wildreceipt")

# ── Model weights — 0.0631 CER best model ────────────────────────────────────
# Using trocr-smart-stock-model which has confirmed model.safetensors
MODEL_INPUT = Path("/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock-best/trocr-smart-stock-best")

# ── Resume checkpoint-31665 from smart-stock-model-data (slug unchanged) ─────
# INPUT_CHECKPOINT_DIR = Path("/kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock/checkpoint-31665")
INPUT_CHECKPOINT_DIR = Path("/kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock/checkpoint-31665")

# ── Output dirs ───────────────────────────────────────────────────────────────
CHECKPOINT_DIR = WORKING_DIR / "trocr-smart-stock"
BEST_MODEL_DIR = WORKING_DIR / "trocr-smart-stock-best"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset dir     : {DATASET_DIR}")
print(f"WildReceipt dir : {WILDRECEIPT_DIR}")
print(f"Model input     : {MODEL_INPUT}")
print(f"Checkpoints     : {CHECKPOINT_DIR}")
print(f"Best model      : {BEST_MODEL_DIR}")

for f in ["config.json", "model.safetensors", "generation_config.json"]:
    exists = (MODEL_INPUT / f).exists()
    print(f"  {'✅' if exists else '❌'} {f}")

# Verify checkpoint exists
lora_path = INPUT_CHECKPOINT_DIR / "lora_adapter"
direct_path = INPUT_CHECKPOINT_DIR / "adapter_config.json"
print(f"\n  {'✅' if lora_path.exists() else '❌'} checkpoint lora_adapter")
print(f"  {'✅' if direct_path.exists() else '❌'} checkpoint adapter_config.json")
```

**Cell 4:**

```python
# Paste in a scratch cell to confirm paths before committing
lora = Path("/kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock/checkpoint-31665/lora_adapter")
direct = Path("/kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock/checkpoint-31665/adapter_config.json")
print(f"lora_adapter exists   : {lora.exists()}")
print(f"adapter_config exists : {direct.exists()}")
```

### #### **CORD's ground_truth** field is a raw JSON string. Parse it and construct a flat text target from the menu array.

**Cell 6:**

```python
import json
from collections import defaultdict
from PIL import Image

def extract_cord_crops(image: Image.Image, ground_truth_str: str) -> list:
    """
    Extract line-level crops from a CORD receipt image.

    CORD's ground_truth is a raw JSON string. Bounding boxes are in
    valid_line[].words[].quad — NOT in gt_parse.menu.
    Each group_id in valid_line = one logical receipt line.
    We group words by group_id, merge their quads into one bbox, crop it,
    and use the joined word texts as the OCR target.
    Only menu.* categories are kept (skip total, tax, header lines).

    Returns list of (cropped_PIL_image, text) pairs.
    One CORD receipt → ~8–12 crops.
    """
    try:
        data = json.loads(ground_truth_str)
    except (json.JSONDecodeError, AttributeError):
        return []

    valid_lines = data.get("valid_line", [])
    if not valid_lines:
        return []

    groups = defaultdict(list)
    for line in valid_lines:
        groups[line["group_id"]].append(line)

    w, h = image.size
    crops = []

    for gid, lines in groups.items():
        if not any(l.get("category", "").startswith("menu.") for l in lines):
            continue

        all_words, all_xs, all_ys = [], [], []
        for line in lines:
            for word in line.get("words", []):
                text = word.get("text", "").strip()
                if text:
                    all_words.append(text)
                q = word.get("quad", {})
                if q:
                    all_xs += [q.get("x1",0), q.get("x2",0), q.get("x3",0), q.get("x4",0)]
                    all_ys += [q.get("y1",0), q.get("y2",0), q.get("y3",0), q.get("y4",0)]

        if not all_words or not all_xs:
            continue

        text = " ".join(all_words)
        x1, y1 = max(0, min(all_xs)), max(0, min(all_ys))
        x2, y2 = min(w, max(all_xs)), min(h, max(all_ys))
        if x2 <= x1 or y2 <= y1:
            continue

        crops.append((image.crop((x1, y1, x2, y2)), text))

    return crops
```

### ### **SROIE** Text Reconstruction
#### SROIE has bboxes per word. Group words by Y-coordinate (within 15px = same line), merge bounding boxes per line, crop each line region.

**Cell 8:**

```python
def extract_sroie_crops(image: Image.Image, words: list, bboxes: list) -> list:
    """
    Group SROIE words into lines by Y-coordinate proximity (15px tolerance).
    Each line bbox is cropped and paired with its joined text.
    SROIE has bboxes per word — no pre-grouped lines.

    Returns list of (cropped_PIL_image, text) pairs.
    """
    if not words or not bboxes:
        return []

    items = sorted(zip(words, bboxes), key=lambda x: x[1][1])

    lines = []
    current_words, current_boxes = [items[0][0]], [items[0][1]]
    for word, box in items[1:]:
        if abs(box[1] - current_boxes[-1][1]) <= 15:
            current_words.append(word)
            current_boxes.append(box)
        else:
            lines.append((current_words, current_boxes))
            current_words, current_boxes = [word], [box]
    lines.append((current_words, current_boxes))

    w, h = image.size
    crops = []
    for line_words, line_boxes in lines:
        text = " ".join(line_words).strip()
        if not text:
            continue
        x1 = max(0, min(b[0] for b in line_boxes))
        y1 = max(0, min(b[1] for b in line_boxes))
        x2 = min(w, max(b[2] for b in line_boxes))
        y2 = min(h, max(b[3] for b in line_boxes))
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append((image.crop((x1, y1, x2, y2)), text))
    return crops
```

### ### WildReceipt Extractor

**Cell 10:**

```python
import json

# Labels to EXCLUDE from WildReceipt:
# 0  = empty string / illegible text
# 25 = catch-all "other" (terminal IDs, legal footnotes, thank-you messages, promo text)
# All other labels (1=store name, 3=address, 5=phone, 7=date, 9=time,
# 11=item name, 13=quantity, 14=count, 15=item price, 17=subtotal,
# 18=subtotal label, 19=tax amount, 20=tax label, 22=tip, 23=total,
# 24=total label) are included.
WILDRECEIPT_EXCLUDE = {0, 25}

def extract_wildreceipt_crops(annotation_file: Path) -> list:
    """
    Parse a WildReceipt annotation file (one JSON object per line).
    Each annotation becomes its own crop — no line grouping.

    Why no grouping: WildReceipt has two-column layouts (item names left,
    prices right) with nearly identical Y coordinates per row. Y-proximity
    grouping merged columns into nonsense crops, corrupting ~33% of train data
    and causing CER regression from 0.088 → 0.339.

    Per-annotation cropping is also closer to TrOCR's pretraining distribution
    (single word/phrase level crops) than multi-word line crops.

    Each annotation box: [x1,y1, x2,y1, x2,y2, x1,y2] clockwise from top-left.
    Returns list of (PIL.Image, text) pairs.
    """
    crops = []

    with open(annotation_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            img_path = WILDRECEIPT_DIR / record["file_name"]
            if not img_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            img_w, img_h = image.size

            for ann in record["annotations"]:
                # Skip excluded labels and empty text
                if ann["label"] in WILDRECEIPT_EXCLUDE:
                    continue
                text = ann.get("text", "").strip()
                if not text:
                    continue

                # Box: [x1,y1, x2,y1, x2,y2, x1,y2]
                box = ann["box"]
                xs = [box[0], box[2], box[4], box[6]]
                ys = [box[1], box[3], box[5], box[7]]

                x1 = max(0, int(min(xs)))
                y1 = max(0, int(min(ys)))
                x2 = min(img_w, int(max(xs)))
                y2 = min(img_h, int(max(ys)))

                # Skip degenerate crops — beam search hangs on these
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue

                crops.append((image.crop((x1, y1, x2, y2)), text))

    return crops
```

### ### Combined Dataset Builder
OOM fix: Images encoded to PNG bytes immediately. Line crops: Each receipt yields multiple (crop, text) pairs. SROIE 2x weighting: SROIE train concatenated twice.

**Cell 12:**

```python
import io
import json
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from PIL import Image

# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_bytes(img: Image.Image) -> bytes:
    """Encode PIL image to PNG bytes immediately — prevents RAM accumulation."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()

def iter_to_dataset(iterator) -> Dataset:
    """
    Convert iterator of (PIL Image, text) tuples into a HuggingFace Dataset.
    Encodes each image to bytes immediately — never holds multiple PIL objects in RAM.
    """
    img_bytes, texts = [], []
    for img, text in iterator:
        img_bytes.append(pil_to_bytes(img))
        texts.append(text)
    return Dataset.from_dict({"image_bytes": img_bytes, "text": texts})

# ── Combined dataset builder ──────────────────────────────────────────────────

DATASET_SAVE = DATASET_DIR  # points to smart-stock-dataset-v3 — loads if exists, builds if not

WR_TEST_KEEP = 1000  # cap WildReceipt test crops; remainder folds into train

def build_and_save_dataset():
    if DATASET_SAVE.exists():
        print(f"Dataset found at {DATASET_SAVE} — loading from disk...")
        return DatasetDict.load_from_disk(str(DATASET_SAVE))

    print("Building v3 line-crop dataset (CORD + SROIE + WildReceipt)...")

    # ── CORD ──────────────────────────────────────────────────────────────────
    print("Loading CORD...")
    cord = load_dataset("naver-clova-ix/cord-v2")

    def cord_iter(split):
        for ex in cord[split]:
            for crop, text in extract_cord_crops(ex["image"], ex["ground_truth"]):
                yield crop, text

    cord_train      = iter_to_dataset(cord_iter("train"))
    cord_validation = iter_to_dataset(cord_iter("validation"))
    cord_test       = iter_to_dataset(cord_iter("test"))
    print(f"  CORD  train: {len(cord_train)} | val: {len(cord_validation)} | test: {len(cord_test)}")
    del cord  # free RAM before loading next dataset

    # ── SROIE ─────────────────────────────────────────────────────────────────
    print("Loading SROIE...")
    sroie = load_dataset("sizhkhy/SROIE")

    def sroie_iter(split):
        for ex in sroie[split]:
            for crop, text in extract_sroie_crops(ex["images"], ex["words"], ex["bboxes"]):
                yield crop, text

    sroie_train = iter_to_dataset(sroie_iter("train"))
    sroie_test  = iter_to_dataset(sroie_iter("test"))
    print(f"  SROIE train: {len(sroie_train)} | test: {len(sroie_test)}")
    del sroie

    # ── WildReceipt ───────────────────────────────────────────────────────────
    print("Loading WildReceipt...")
    wr_train_crops = extract_wildreceipt_crops(WILDRECEIPT_DIR / "train.txt")
    wr_test_crops  = extract_wildreceipt_crops(WILDRECEIPT_DIR / "test.txt")

    wr_train_raw = iter_to_dataset(iter(wr_train_crops))
    wr_test_raw  = iter_to_dataset(iter(wr_test_crops))
    print(f"  WildReceipt raw — train: {len(wr_train_raw)} | test: {len(wr_test_raw)}")

    # 90% train / 10% val split from WildReceipt train.txt crops
    wr_train_split = wr_train_raw.train_test_split(test_size=0.1, seed=42)
    wr_train_final = wr_train_split["train"]
    wr_val         = wr_train_split["test"]
    print(f"  WildReceipt train (after val split): {len(wr_train_final)} | val: {len(wr_val)}")

    # Cap test.txt crops at WR_TEST_KEEP (1000), move remainder to train
    # Reason: 5,103 test crops * beam search = hours of eval time (killed prior session)
    if len(wr_test_raw) > WR_TEST_KEEP:
        wr_test_split    = wr_test_raw.train_test_split(test_size=WR_TEST_KEEP, seed=42)
        wr_test_final    = wr_test_split["test"]
        wr_test_to_train = wr_test_split["train"]
    else:
        wr_test_final    = wr_test_raw
        wr_test_to_train = None

    print(f"  WildReceipt test kept: {len(wr_test_final)} | moved to train: {len(wr_test_to_train) if wr_test_to_train else 0}")

    # ── Combine ───────────────────────────────────────────────────────────────
    # SROIE 2x weighted in train — more English receipt signal vs CORD's Indonesian
    # Excess WildReceipt test crops folded into train (free data)
    train_parts = [cord_train, sroie_train, sroie_train, wr_train_final]
    if wr_test_to_train:
        train_parts.append(wr_test_to_train)

    dataset_dict = DatasetDict({
        "train":      concatenate_datasets(train_parts),
        "validation": concatenate_datasets([cord_validation, wr_val]),
        "test":       concatenate_datasets([cord_test, sroie_test, wr_test_final]),
    })

    # Save to working dir — download and upload as Kaggle dataset smart-stock-dataset-v3
    save_path = WORKING_DIR / "smart_stock_dataset_v3"
    dataset_dict.save_to_disk(str(save_path))
    print(f"\n✅ Saved to {save_path}")
    print(f"   Train      : {len(dataset_dict['train'])}")
    print(f"   Validation : {len(dataset_dict['validation'])}")
    print(f"   Test       : {len(dataset_dict['test'])}")
    return dataset_dict

combined_dataset = build_and_save_dataset()
```

### ### Dataset filter

**Cell 14:**

```python
# # ── Dataset Filter → smart-stock-dataset-v4 ───────────────────────────────────
# # Run AFTER inspecting crop_sample_*.png outputs from the diagnostic cell.
# # Set MIN_WIDTH and MIN_HEIGHT based on what you saw.
# # Also applies non-food label blocklist to remove header/footer WildReceipt noise.

# import io, re
# from PIL import Image
# from datasets import DatasetDict

# # ── SET THESE based on diagnostic output ──────────────────────────────────────
# MIN_WIDTH  = 50   # change after inspecting the PNG grids
# MIN_HEIGHT = 10   # change after inspecting the PNG grids
# MIN_LABEL_CHARS = 2  # drop single-char labels ('2', 'P', etc.)
# # ─────────────────────────────────────────────────────────────────────────────

# # Non-food receipt tokens that slipped through WildReceipt annotation.
# # These cause hallucination — the model sees similar crops and generates
# # entire receipt footers instead of the actual text.
# # Conservative list — only unambiguous non-food header/footer tokens.
# NON_FOOD_BLOCKLIST = {
#     "tax", "salestax", "subtotal", "total", "date", "time", "receipt",
#     "invoice", "change", "cash", "card", "payment", "balance", "tip",
#     "thank you", "gst", "vat", "hst", "pst", "discount", "void",
#     "cashier", "terminal", "ref", "auth", "batch",
#     # Short receipt metadata tokens
#     "ti:", "p:", "ht", "wef", "no.", "no:", "tel:", "tel",
# }

# def should_keep(sample) -> bool:
#     # 1. Size filter
#     try:
#         img = Image.open(io.BytesIO(sample["image_bytes"]))
#         w, h = img.size
#         if w < MIN_WIDTH or h < MIN_HEIGHT:
#             return False
#     except Exception:
#         return False  # corrupt image bytes

#     # 2. Label length filter
#     text = sample["text"].strip()
#     if len(text) < MIN_LABEL_CHARS:
#         return False

#     # 3. Non-food blocklist — exact match on lowercased label
#     # Using exact match only (not substring) to avoid false positives
#     # e.g. "TOTAL CORNFLAKES 2 3.99" should NOT be filtered
#     if text.lower() in NON_FOOD_BLOCKLIST:
#         return False

#     return True

# # ── Apply filter to all splits ────────────────────────────────────────────────
# print(f"Filter thresholds: width≥{MIN_WIDTH}, height≥{MIN_HEIGHT}, label≥{MIN_LABEL_CHARS} chars")
# print(f"Blocklist entries: {len(NON_FOOD_BLOCKLIST)}")
# print()

# filtered = {}
# for split in ["train", "validation", "test"]:
#     original = combined_dataset[split]

#     # Build keep indices manually — avoids dataset.filter() cache write to /kaggle/input/
#     keep_indices = []
#     for i in range(len(original)):
#         if should_keep(original[i]):
#             keep_indices.append(i)

#     kept    = original.select(keep_indices)
#     removed = len(original) - len(kept)
#     filtered[split] = kept
#     print(f"  {split:12s} | original={len(original):6d} | kept={len(kept):6d} | removed={removed:5d} ({100*removed/len(original):.1f}%)")

# filtered_dataset = DatasetDict(filtered)

# # ── Save to working dir ───────────────────────────────────────────────────────
# SAVE_PATH = WORKING_DIR / "smart_stock_dataset_v4"
# filtered_dataset.save_to_disk(str(SAVE_PATH))

# print(f"\n✅ Saved to {SAVE_PATH}")
# print(f"   Train      : {len(filtered_dataset['train'])}")
# print(f"   Validation : {len(filtered_dataset['validation'])}")
# print(f"   Test       : {len(filtered_dataset['test'])}")
# print("\nNext: download smart_stock_dataset_v4 from Kaggle output and upload as new dataset version.")
# print("Then update DATASET_DIR in Cell 2 to point to v4.")
```

### ### Size Distribution Diagnostic

**Cell 16:**

```python
# # ── Crop Size Diagnostic ──────────────────────────────────────────────────────
# # Before filtering anything, inspect the size distribution across all splits.
# # Shows: size bucket counts, samples removed at different thresholds, and saves
# # a grid of sample crops at each size range for manual visual inspection.
# # Run this BEFORE the filter cell — decide threshold based on output.

# import io, os
# from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np

# DIAG_SPLIT = "train"  # train has the most samples, most representative
# samples = combined_dataset[DIAG_SPLIT]
# print(f"Inspecting {DIAG_SPLIT} split: {len(samples)} samples")

# # ── Collect all sizes ─────────────────────────────────────────────────────────
# widths, heights, areas, labels_list = [], [], [], []

# for i in range(len(samples)):
#     s = samples[i]
#     img = Image.open(io.BytesIO(s["image_bytes"]))
#     w, h = img.size
#     widths.append(w)
#     heights.append(h)
#     areas.append(w * h)
#     labels_list.append(s["text"])

# widths  = np.array(widths)
# heights = np.array(heights)
# areas   = np.array(areas)

# # ── Size bucket distribution ──────────────────────────────────────────────────
# print("\n── Width distribution ───────────────────────────────────")
# w_buckets = [0, 10, 20, 30, 50, 100, 200, 500, 99999]
# w_labels  = ["<10", "10-20", "20-30", "30-50", "50-100", "100-200", "200-500", ">500"]
# for i in range(len(w_labels)):
#     mask  = (widths >= w_buckets[i]) & (widths < w_buckets[i+1])
#     count = mask.sum()
#     bar   = "█" * int(40 * count / len(widths))
#     print(f"  w {w_labels[i]:8s} | {count:5d} ({100*count/len(widths):4.1f}%) {bar}")

# print("\n── Height distribution ──────────────────────────────────")
# h_buckets = [0, 8, 10, 15, 20, 30, 50, 99999]
# h_labels  = ["<8", "8-10", "10-15", "15-20", "20-30", "30-50", ">50"]
# for i in range(len(h_labels)):
#     mask  = (heights >= h_buckets[i]) & (heights < h_buckets[i+1])
#     count = mask.sum()
#     bar   = "█" * int(40 * count / len(heights))
#     print(f"  h {h_labels[i]:8s} | {count:5d} ({100*count/len(heights):4.1f}%) {bar}")

# # ── How many samples removed at different thresholds ─────────────────────────
# print("\n── Samples removed at different (min_w, min_h) thresholds ──")
# print(f"  {'Threshold':20s} | {'Removed':>8s} | {'Remaining':>10s} | {'% Removed':>10s}")
# thresholds = [(10,5), (20,8), (30,10), (40,12), (50,15), (64,16)]
# for min_w, min_h in thresholds:
#     removed   = ((widths < min_w) | (heights < min_h)).sum()
#     remaining = len(widths) - removed
#     print(f"  w≥{min_w:<4d} h≥{min_h:<4d}          | {removed:>8d} | {remaining:>10d} | {100*removed/len(widths):>9.2f}%")

# # ── Save sample crops from each size range for visual inspection ──────────────
# # Saves a PNG grid of 5 crops from each width bucket to /kaggle/working/
# print("\n── Saving sample crop grids to /kaggle/working/ ─────────────")

# size_ranges = [
#     ("tiny",   widths <  20),
#     ("small",  (widths >= 20) & (widths < 50)),
#     ("medium", (widths >= 50) & (widths < 150)),
#     ("large",  widths >= 150),
# ]

# for range_name, mask in size_ranges:
#     indices = np.where(mask)[0][:8]  # first 8 samples in this range
#     if len(indices) == 0:
#         print(f"  {range_name}: no samples")
#         continue

#     fig, axes = plt.subplots(2, 4, figsize=(16, 5))
#     fig.suptitle(f"Crop size range: {range_name}", fontsize=14)
#     axes = axes.flatten()

#     for plot_i, sample_idx in enumerate(indices):
#         s   = samples[int(sample_idx)]
#         img = Image.open(io.BytesIO(s["image_bytes"])).convert("RGB")
#         w, h = img.size
#         axes[plot_i].imshow(img)
#         axes[plot_i].set_title(
#             f"{w}x{h}px\n{s['text'][:30]!r}",
#             fontsize=7, wrap=True
#         )
#         axes[plot_i].axis("off")

#     for j in range(len(indices), len(axes)):
#         axes[j].axis("off")

#     out_path = f"/kaggle/working/crop_sample_{range_name}.png"
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=100, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {out_path} ({len(indices)} samples shown)")

# print("\nDone. Check the 4 PNG files in /kaggle/working/ to visually inspect crops.")
# print("Decide threshold based on what you see, then run the filter cell.")
```

**Cell 17:**

```python
# import json
# import io
# from PIL import Image

# # Pick 3 WildReceipt images and show crop count + text per crop
# with open(WILDRECEIPT_DIR / "train.txt") as f:
#     lines = [l.strip() for l in f if l.strip()][:3]

# for line in lines:
#     record = json.loads(line)
#     img = Image.open(WILDRECEIPT_DIR / record["file_name"]).convert("RGB")
#     img_w, img_h = img.size
#     y_tol = max(10, int(img_h * 0.02))
    
#     valid = [a for a in record["annotations"]
#              if a["label"] not in {0, 25} and a.get("text","").strip()]
    
#     print(f"\nImage: {record['file_name']} ({img_w}×{img_h}), y_tol={y_tol}")
#     print(f"Valid annotations: {len(valid)}")
#     for a in valid:
#         box = a["box"]
#         ys = [box[1], box[3], box[5], box[7]]
#         cy = (min(ys) + max(ys)) / 2
#         print(f"  cy={cy:.0f}  text='{a['text']}'  label={a['label']}")
```

### #### **Augmentation**
Apply augmentation only to training images, inline during the preprocess_trocr step.

**Cell 19:**

```python
import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageOps

receipt_augmentation = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.15), p=0.6),  # thermal fade simulation
    A.GaussNoise(p=0.5),                                                 # scanner noise
    A.Rotate(limit=8, border_mode=cv2.BORDER_REPLICATE, p=0.5),         # crumple/tilt
    A.Perspective(scale=(0.02, 0.08), p=0.4),                           # phone photo angle
    A.MotionBlur(blur_limit=5, p=0.3),                                  # shaky photo
    A.ImageCompression(p=0.5),                                           # JPEG artifact
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),                           # focus blur
    A.RandomShadow(p=0.2),                                               # shadow on receipt
])

def apply_augmentation(pil_image: Image.Image) -> Image.Image:
    """Pad tiny line crops to min 32px height, then augment."""
    pil_image = pil_image.convert("RGB")
    if pil_image.height < 32:
        pad = 32 - pil_image.height
        pil_image = ImageOps.expand(pil_image, border=(0, pad//2, 0, pad - pad//2), fill=255)
    img_np = np.array(pil_image)
    augmented = receipt_augmentation(image=img_np)["image"]
    return Image.fromarray(augmented)
```

### ### Preprocessing Function

**Cell 21:**

```python
import io
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

def preprocess_trocr(example, augment: bool = False):
    """
    Preprocess a single (image_bytes, text) example at access time.
    Returns dict with pixel_values (tensor) and labels (tensor).
    Padding tokens replaced with -100 so loss ignores them.
    """
    image = Image.open(io.BytesIO(example["image_bytes"])).convert("RGB")

    if augment:
        image = apply_augmentation(image)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    labels = processor.tokenizer(
        example["text"],
        padding="max_length",
        max_length=128,
        truncation=True,
    ).input_ids

    labels = [
        t if t != processor.tokenizer.pad_token_id else -100
        for t in labels
    ]

    return {
        "pixel_values": pixel_values.squeeze(),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class TrOCRDataset(TorchDataset):
    """
    On-the-fly preprocessing — processes each example at access time.
    Replaces .map() which caused either:
      - OOM (keep_in_memory=True on 46k examples)
      - OSError Errno 28 (cache writing to read-only /kaggle/input/)
    Zero disk usage, zero RAM accumulation, full dataset used.
    Compatible with Seq2SeqTrainer.

    NOTE: This is a PyTorch Dataset — it has no .select() method.
    For Optuna subset selection, use combined_dataset["train"].select(...)
    (the HuggingFace dataset) and then wrap in TrOCRDataset().
    """
    def __init__(self, hf_dataset, augment: bool = False):
        self.data    = hf_dataset
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return preprocess_trocr(self.data[idx], augment=self.augment)


train_dataset = TrOCRDataset(combined_dataset["train"],      augment=True)
val_dataset   = TrOCRDataset(combined_dataset["validation"], augment=False)
test_dataset  = TrOCRDataset(combined_dataset["test"],       augment=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
```

### ### Model Setup

**Cell 23:**

```python
from pathlib import Path
from transformers import VisionEncoderDecoderModel
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# INPUT_CHECKPOINT_DIR already defined in Cell 1
INPUT_CHECKPOINT_DIR = Path("/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock/trocr-smart-stock/checkpoint-51348")

# Load base model from stored best weights
base_model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_INPUT))

# Freeze entire encoder first, then unfreeze top 2 ViT blocks (indices 10 and 11)
for param in base_model.encoder.parameters():
    param.requires_grad = False

for block_idx in [10, 11]:
    for param in base_model.encoder.encoder.layer[block_idx].parameters():
        param.requires_grad = True

encoder_trainable = sum(p.numel() for p in base_model.encoder.parameters() if p.requires_grad)
encoder_total     = sum(p.numel() for p in base_model.encoder.parameters())
print(f"Encoder trainable params (blocks 10+11): {encoder_trainable:,} / {encoder_total:,}")

# ── r=32 EXPERIMENT: always fresh LoRA, no resume ──────────────────────────
# Existing checkpoints (working dir / INPUT_CHECKPOINT_DIR) are r=16 adapters.
# They are NOT shape-compatible with r=32, so resume logic is skipped entirely
# for this run.

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=32,           # was 16
    lora_alpha=64,   # scaled to keep alpha/r ratio = 2, same as r=16 config
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
print("Fresh LoRA r=32 applied — no resume, clean run vs r=16 best (CER 0.0631)")

# Print total trainable params (encoder blocks + LoRA)
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable params: {total_trainable:,}")

# Required decoder config
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size

# Generation config
model.generation_config.eos_token_id   = processor.tokenizer.sep_token_id
model.generation_config.early_stopping = True
model.generation_config.length_penalty = 1.0
model.generation_config.num_beams      = 4
# no_repeat_ngram_size deliberately not set — receipt text has legitimate
# repetitions (e.g. "60.000 60.000") that ngram blocking incorrectly suppresses
```

### #### collate_fn

Standalone cell — extracted from the Optuna block so it's always available even when Optuna is commented out.

**Cell 25:**

```python
def collate_fn(batch):
    """
    Custom collator for TrOCR.
    pixel_values and labels are tensors returned by TrOCRDataset.__getitem__.
    torch.stack combines them into batches for the Trainer.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels       = torch.stack([item["labels"]       for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}
```

### ### Training Arguments
Block A — OLD CONFIG (fully commented out, do not use)


Kept for reference only. This was the config from Kaggle run 1 (lr=1.695e-4, cosine scheduler). Superseded by Block B below.

**Cell 27:**

```python
# from transformers import Seq2SeqTrainingArguments

# training_args = Seq2SeqTrainingArguments(
#     output_dir=str(CHECKPOINT_DIR),   # /kaggle/working/trocr-smart-stock

#     num_train_epochs=5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,

#     # Hardcoded from Optuna best (Trial 3 of latest run)
#     learning_rate=1.695e-4,
#     warmup_ratio=0.0866,
#     weight_decay=0.01,
#     lr_scheduler_type="cosine",
#     max_grad_norm=1.0,

#     eval_strategy="epoch",
#     save_strategy="steps",
#     save_steps=500,
#     load_best_model_at_end=False,
#     save_total_limit=5,

#     predict_with_generate=True,
#     generation_max_length=128,

#     fp16=True,
#     dataloader_num_workers=2,

#     logging_dir=str(WORKING_DIR / "logs"),
#     logging_steps=50,
#     log_level="info",
#     report_to="none",
# )
```

### ##### Technique 1 — Force single GPU + fix LR (most important, run first)

DataParallel doubled effective batch size silently. Forcing single GPU makes real training match Optuna's conditions exactly.

##### Block B — ACTIVE CONFIG (Technique 1: single GPU + correct LR)


This is the block that actually runs. os.environ line here is redundant (already set in Cell 1) but harmless.

**Cell 29:**

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=str(CHECKPOINT_DIR),

    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,     # back to 1 — encoder is frozen, no OOM risk

    # Restored Optuna LR — valid for frozen encoder runs
    learning_rate=1.4824e-4,
    warmup_ratio=0.05,
    weight_decay=0.01,

    lr_scheduler_type="cosine_with_restarts",
    lr_scheduler_kwargs={"num_cycles": 1},

    max_grad_norm=1.0,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_cer",
    greater_is_better=False,
    save_total_limit=3,

    predict_with_generate=True,
    generation_max_length=128,

    fp16=True,
    dataloader_num_workers=2,

    logging_dir=str(WORKING_DIR / "logs"),
    logging_steps=50,
    log_level="info",
    report_to="none",
)

print(f"Effective batch size : {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"GPU visible          : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"LR                   : {training_args.learning_rate}")
print(f"Epochs               : {training_args.num_train_epochs}")
```

### ### Metrics

**Cell 31:**

```python
!pip install jiwer -q
```

**Cell 32:**

```python
import numpy as np
from jiwer import cer, wer

def compute_metrics(pred):
    """
    Compute CER and WER for Seq2SeqTrainer.
    Called at end of each eval epoch with generated token ids.
    """
    pred_ids   = pred.predictions
    labels_ids = pred.label_ids

    # pred_ids may be float logits (ndim=3) — argmax to get token ids
    if pred_ids.dtype != np.int64 and pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    # Clip to valid vocab range — prevents OverflowError during decode
    vocab_size = processor.tokenizer.vocab_size
    pred_ids   = np.clip(pred_ids, 0, vocab_size - 1)

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    # Replace -100 (padding mask) with pad token id before decoding
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    return {
        "cer": round(cer(label_str, pred_str), 4),
        "wer": round(wer(label_str, pred_str), 4),
    }
```

### ### Hyperparameter Search with **Optuna**
After the first clean training run with line crops, use Optuna (Bayesian optimization) via HuggingFace built-in hyperparameter_search to find optimal values.

#### Why commented: Current LR (1.4824e-4) was tuned by Optuna on a prior subset run. Running Optuna again costs ~3–4 hours of the 12-hour session before any real training starts.

When to uncomment: After a stable epoch 1 completes on v3 data and CER confirms ~0.088–0.092. If the LR feels off on the new dataset distribution, run Optuna in a separate dedicated session.

Config notes: n_trials=4 (not 10) — 4 trials × 1 epoch on 1/8 of 46,500 = ~5,800 examples per trial ≈ 45 min per trial ≈ 3 hr total. Fits in a dedicated session. predict_with_generate=True kept (unlike old v12 config) — we need CER not just loss to rank trials meaningfully.

**Cell 34:**

```python
# import gc
# import copy
# from transformers import Seq2SeqTrainer
# from peft import LoraConfig, get_peft_model, TaskType

# def optuna_hp_space(trial):
#     return {
#         "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True),
#         "warmup_ratio":  trial.suggest_float("warmup_ratio", 0.0, 0.1),
#     }

# def model_init():
#     gc.collect()
#     torch.cuda.empty_cache()

#     m = VisionEncoderDecoderModel.from_pretrained(str(MODEL_INPUT))

#     # Match actual training config — freeze encoder, then unfreeze blocks 10+11
#     for param in m.encoder.parameters():
#         param.requires_grad = False
#     for block_idx in [10, 11]:
#         for param in m.encoder.encoder.layer[block_idx].parameters():
#             param.requires_grad = True

#     lora_config = LoraConfig(
#         task_type=TaskType.SEQ_2_SEQ_LM,
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         target_modules=["q_proj", "v_proj"],
#         bias="none",
#     )
#     m = get_peft_model(m, lora_config)

#     m.config.decoder_start_token_id = processor.tokenizer.cls_token_id
#     m.config.pad_token_id           = processor.tokenizer.pad_token_id
#     m.config.vocab_size             = m.config.decoder.vocab_size

#     m.generation_config.eos_token_id   = processor.tokenizer.sep_token_id
#     m.generation_config.early_stopping = True
#     m.generation_config.length_penalty = 1.0
#     m.generation_config.num_beams      = 4

#     return m

# search_args = copy.deepcopy(training_args)
# search_args.num_train_epochs                = 1
# search_args.predict_with_generate           = True
# search_args.eval_accumulation_steps         = 4
# search_args.dataloader_num_workers          = 0
# search_args.per_device_train_batch_size     = 4
# search_args.per_device_eval_batch_size      = 4
# search_args.gradient_accumulation_steps     = 1
# search_args.output_dir                      = str(WORKING_DIR / "optuna_search")
# search_args.save_strategy                   = "no"
# search_args.load_best_model_at_end          = False
# search_args.eval_strategy                   = "epoch"
# search_args.logging_steps                   = 100

# # 1/8 of train, 300 val samples — enough signal, fast enough per trial
# search_hf  = combined_dataset["train"].select(range(len(combined_dataset["train"]) // 8))
# search_val_hf = combined_dataset["validation"].select(range(min(300, len(combined_dataset["validation"]))))
# search_train = TrOCRDataset(search_hf, augment=True)
# search_val   = TrOCRDataset(search_val_hf, augment=False)

# print(f"Optuna train subset : {len(search_train)} samples")
# print(f"Optuna val subset   : {len(search_val)} samples")
# print(f"Est. time per trial : ~25–30 min")
# print(f"Running 20 trials   : ~8–10 hr")

# search_trainer = Seq2SeqTrainer(
#     model_init=model_init,
#     args=search_args,
#     train_dataset=search_train,
#     eval_dataset=search_val,
#     data_collator=collate_fn,
#     compute_metrics=compute_metrics,
# )

# best_run = search_trainer.hyperparameter_search(
#     direction="minimize",
#     backend="optuna",
#     hp_space=optuna_hp_space,
#     n_trials=20,
# )

# del search_trainer
# gc.collect()
# torch.cuda.empty_cache()

# print(f"\nBest trial objective (CER): {best_run.objective:.4f}")
# print(f"Best hyperparameters: {best_run.hyperparameters}")

# # NOTE: Write these down before session ends — not persisted to disk automatically
# # Then hardcode into Cell 10: learning_rate=X, warmup_ratio=Y

# # Apply to training_args for immediate use in same session
# for k, v in best_run.hyperparameters.items():
#     setattr(training_args, k, v)
# training_args.predict_with_generate = True
# print(f"\nApplied to training_args:")
# print(f"  learning_rate : {training_args.learning_rate}")
# print(f"  warmup_ratio  : {training_args.warmup_ratio}")
```

### ### Trainer and Training

**Cell 36:**

```python
from transformers import Seq2SeqTrainer, TrainerCallback
import shutil

class LoRASaveCallback(TrainerCallback):
    """
    Saves LoRA adapter weights alongside each Trainer checkpoint.
    Without this, PEFT silently resets adapter weights on checkpoint reload.
    Saves to: checkpoint-{step}/lora_adapter/
    """
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        adapter_dir = checkpoint_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        kwargs["model"].save_pretrained(str(adapter_dir))
        print(f"LoRA adapter saved to {adapter_dir}")
        return control

print(f"LR: {training_args.learning_rate}")
print(f"Warmup ratio: {training_args.warmup_ratio}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")

# Remove stale checkpoints from prior bad runs (no lora_adapter subdir)
if CHECKPOINT_DIR.exists():
    for ckpt in CHECKPOINT_DIR.glob("checkpoint-*"):
        has_adapter = (ckpt / "lora_adapter").exists()
        if not has_adapter:
            shutil.rmtree(ckpt)
            print(f"Removed stale checkpoint (no LoRA adapter): {ckpt}")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[LoRASaveCallback()],
)

# ── Clean-data run — NO checkpoint resume ─────────────────────────────────────
# Starting fresh from the 0.0687 merged model weights (loaded in Cell 8).
# Reason: optimizer state from checkpoint-51348 was computed on the noisy v3
# dataset. Resuming it would apply momentum from noisy-data gradients to a
# clean-data training run — counterproductive.
#
# If working-dir checkpoints exist from THIS session (mid-run restart),
# resume from those — they have correct clean-data optimizer state.
valid_checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint-*"))
if valid_checkpoints:
    resume_from = str(valid_checkpoints[-1])
    print(f"Resuming from working dir (same-session checkpoint): {resume_from}")
else:
    resume_from = None
    print("Starting fresh training from 0.0687 weights on clean v4 dataset")

trainer.train(resume_from_checkpoint=resume_from)
```

### ### Training Curves

**Cell 38:**

```python
import pandas as pd
import matplotlib.pyplot as plt

history = pd.DataFrame(trainer.state.log_history)
print(history.columns)
history.head()
```

**Cell 39:**

```python
train_logs = history[history["loss"].notna()]
eval_logs  = history[history["eval_loss"].notna()]

plt.figure(figsize=(12,6))
plt.plot(train_logs["step"], train_logs["loss"], label="Training Loss")
plt.plot(eval_logs["step"], eval_logs["eval_loss"], label="Validation Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("TrOCR Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
```

**Cell 40:**

```python
fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].plot(eval_logs["epoch"], eval_logs["eval_cer"], marker="o")
axes[0].set_title("CER")
axes[0].set_xlabel("Epoch")
axes[1].plot(eval_logs["epoch"], eval_logs["eval_wer"], marker="o")
axes[1].set_title("WER")
axes[1].set_xlabel("Epoch")
plt.tight_layout()
plt.show()
```

**Cell 41:**

```python
summary = eval_logs[["epoch", "eval_loss", "eval_cer", "eval_wer"]]
summary
```

### ### Save & Export

**Cell 43:**

```python
from peft import PeftModel

# Merge LoRA weights into base model — produces a standard VisionEncoderDecoderModel
# with no PEFT dependency. Required for inference without peft installed.
merged_model = model.merge_and_unload()

# Fix generation config before saving/eval — merge_and_unload() resets generation_config
# to HF defaults, silently dropping eos/decoder_start/pad token overrides. This was the
# root cause of Test CER 1.4004: without eos_token_id, generation never stops properly,
# producing wildly over-long predictions.
merged_model.generation_config.max_length             = 256
merged_model.generation_config.max_new_tokens          = 128
merged_model.generation_config.length_penalty          = 1.0
merged_model.generation_config.no_repeat_ngram_size    = 0
merged_model.generation_config.eos_token_id            = processor.tokenizer.sep_token_id
merged_model.generation_config.decoder_start_token_id  = processor.tokenizer.cls_token_id
merged_model.generation_config.pad_token_id            = processor.tokenizer.pad_token_id
merged_model.generation_config.early_stopping          = True
merged_model.generation_config.num_beams               = 4

for save_path in [str(BEST_MODEL_DIR)]:
    merged_model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    merged_model.generation_config.save_pretrained(save_path)
    print(f"Saved to: {save_path}")

expected_files = [
    "model.safetensors", "config.json", "generation_config.json",
    "tokenizer_config.json", "tokenizer.json", "processor_config.json",
]
print(f"\nVerification ({BEST_MODEL_DIR}):")
for fname in expected_files:
    fpath = BEST_MODEL_DIR / fname
    exists = fpath.exists()
    size = fpath.stat().st_size / 1e6 if exists else 0
    print(f"  {'✅' if exists else '❌'} {fname} ({size:.1f} MB)")
```

### ### Evaluate on Test Set

**Cell 45:**

```python
from jiwer import cer, wer

# CRITICAL: use merged_model (from Cell 38), NOT model.
# model.merge_and_unload() mutates the PeftModel's underlying module tree
# in place (LoRA layers are folded into base weights and removed from the
# module graph). The `model` variable becomes a stale/inconsistent object
# after that call — calling model.generate() on it produces corrupted,
# scrambled output even though the weights were trained correctly.
# This was the root cause of Test CER 1.4052 in v3 run 3 (val CER was 0.0695).
merged_model.eval()

all_preds, all_labels = [], []
skipped = 0
first_error_printed = False

for idx, sample in enumerate(combined_dataset["test"]):
    try:
        image = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")
        w, h = image.size
        if w < 4 or h < 4:          # skip degenerate crops — beam search hangs on these
            skipped += 1
            continue

        pixel_values = processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(merged_model.device)

        with torch.no_grad():
            # keyword arg required — see Bug #4 in OCR_Training.md
            generated_ids = merged_model.generate(
                pixel_values=pixel_values,
                max_new_tokens=128,
            )

        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_preds.append(pred)
        all_labels.append(sample["text"])

    except Exception as e:
        if not first_error_printed:
            print(f"First exception at idx={idx}: {type(e).__name__}: {e}")
            first_error_printed = True
        skipped += 1

if all_preds:
    test_cer = cer(all_labels, all_preds)
    test_wer = wer(all_labels, all_preds)
    print(f"Test CER : {test_cer:.4f}")
    print(f"Test WER : {test_wer:.4f}")
else:
    print("No predictions collected — all samples skipped or errored")

print(f"Skipped  : {skipped} / {len(combined_dataset['test'])}")
print(f"Evaluated: {len(all_preds)} / {len(combined_dataset['test'])}")

# Targets: CER ≤ 0.05, WER ≤ 0.10
```

### ### Beam search tuning

**Cell 47:**

```python
# # ── Beam Search Sweep (inference only — no training) ─────────────────────────
# # Loads merged model from disk. Do NOT run Cell 15 or 16 before this.
# # Sweeps num_beams over a 300-sample slice of the test set (~5-8 min per config).
# # Pick the best num_beams, then run full test eval with that setting.

# import io
# import torch
# from PIL import Image
# from jiwer import cer, wer
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# SWEEP_MODEL_PATH = str(MODEL_INPUT)  # trocr-smart-stock-best (merged, no PEFT needed)
# SWEEP_SAMPLE_N   = 200
# BEAMS_TO_TRY     = [6, 8, 10]

# print(f"Loading merged model from: {SWEEP_MODEL_PATH}")
# sweep_processor = TrOCRProcessor.from_pretrained(SWEEP_MODEL_PATH)
# sweep_model     = VisionEncoderDecoderModel.from_pretrained(SWEEP_MODEL_PATH)
# sweep_model.eval()
# sweep_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sweep_model.to(sweep_device)
# print(f"Model loaded on {sweep_device}")

# # Fixed 300-sample slice — same slice for every beam config
# test_samples = list(combined_dataset["test"].select(range(SWEEP_SAMPLE_N)))

# results = {}

# for num_beams in BEAMS_TO_TRY:
#     preds, labels = [], []
#     skipped = 0
#     first_error_printed = False

#     for idx, sample in enumerate(test_samples):
#         try:
#             image = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")
#             w, h = image.size
#             if w < 4 or h < 4:
#                 skipped += 1
#                 continue

#             pixel_values = sweep_processor(
#                 images=image, return_tensors="pt"
#             ).pixel_values.to(sweep_device)

#             with torch.no_grad():
#                 generated_ids = sweep_model.generate(
#                     pixel_values=pixel_values,
#                     max_new_tokens=128,
#                     num_beams=num_beams,
#                     length_penalty=1.0,
#                     no_repeat_ngram_size=0,    # ← add this line only
#                 )

#             pred  = sweep_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#             preds.append(pred)
#             labels.append(sample["text"])

#         except Exception as e:
#             if not first_error_printed:
#                 print(f"[beams={num_beams}] First exception at idx={idx}: {type(e).__name__}: {e}")
#                 first_error_printed = True
#             skipped += 1

#     sweep_cer = cer(labels, preds) if preds else float("nan")
#     sweep_wer = wer(labels, preds) if preds else float("nan")
#     results[num_beams] = {"cer": sweep_cer, "wer": sweep_wer, "skipped": skipped, "evaluated": len(preds)}
#     print(f"num_beams={num_beams} | CER={sweep_cer:.4f} | WER={sweep_wer:.4f} | skipped={skipped} | evaluated={len(preds)}")

# # Summary
# print("\n── Sweep Summary ───────────────────────────────────────")
# best_beams = min(results, key=lambda b: results[b]["cer"])
# for b, r in results.items():
#     marker = " ← best" if b == best_beams else ""
#     print(f"  num_beams={b} | CER={r['cer']:.4f} | WER={r['wer']:.4f}{marker}")
# print(f"\nRecommended num_beams: {best_beams}")
```

**Cell 48:**

```python
# ── Per-source CER diagnostic ─────────────────────────────────────────────────
# Checks whether aggregate CER is being skewed by one dataset source.
# Requires sweep_model and sweep_processor already loaded.

import io, torch, collections
from PIL import Image
from jiwer import cer, wer

DIAG_N = 400

sweep_model = merged_model
sweep_processor = processor
sweep_device = merged_model.device

diag_samples = list(combined_dataset["test"].select(range(DIAG_N)))

# Detect source from text characteristics — adjust heuristics if needed
def guess_source(text):
    # CORD labels tend to have Indonesian price patterns (e.g. "17500", "46000")
    # SROIE labels are English retail
    # WildReceipt labels are mixed but often have structured price+item pairs
    # Best proxy: dataset stores a 'source' field if we added one, else fall back to index
    return "unknown"

# Check if dataset has a source field
sample0 = diag_samples[0]
has_source = "source" in sample0
print(f"Dataset has 'source' field: {has_source}")
print(f"Sample keys: {list(sample0.keys())}")

# Group by source if available, else just print first 10 GT/PRED pairs
by_source = collections.defaultdict(lambda: {"preds": [], "labels": []})

for idx, sample in enumerate(diag_samples):
    try:
        image = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")
        w, h = image.size
        if w < 4 or h < 4:
            continue

        pixel_values = sweep_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(sweep_device)

        with torch.no_grad():
            generated_ids = sweep_model.generate(
                pixel_values=pixel_values,
                max_new_tokens=128,
                num_beams=4,
                length_penalty=1.0,
            )

        pred = sweep_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        label = sample["text"]

        source = sample["source"] if has_source else "all"
        by_source[source]["preds"].append(pred)
        by_source[source]["labels"].append(label)

    except Exception as e:
        print(f"Error at idx={idx}: {e}")

# Per-source CER
print("\n── Per-source CER ───────────────────────────────────────")
for src, data in by_source.items():
    if data["preds"]:
        src_cer = cer(data["labels"], data["preds"])
        src_wer = wer(data["labels"], data["preds"])
        print(f"  {src:20s} | n={len(data['preds']):4d} | CER={src_cer:.4f} | WER={src_wer:.4f}")

# Print 10 worst predictions by individual CER
print("\n── 10 sample GT vs PRED (first 10) ─────────────────────")
all_preds  = [v for d in by_source.values() for v in d["preds"]]
all_labels = [v for d in by_source.values() for v in d["labels"]]
for i in range(min(10, len(all_preds))):
    sample_cer = cer([all_labels[i]], [all_preds[i]])
    print(f"[{i}] CER={sample_cer:.3f} | GT: {all_labels[i]!r}")
    print(f"      {'':8s}       PRED: {all_preds[i]!r}")
```

### ### Qualitative Evaluation

**Cell 50:**

```python
import io
from PIL import Image

sample = combined_dataset["test"][0]
image = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")

pixel_values = processor(
    image,
    return_tensors="pt"
).pixel_values.to(merged_model.device)   # merged_model, not model

generated_ids = merged_model.generate(pixel_values=pixel_values, max_new_tokens=128)

prediction = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("GROUND TRUTH:")
print(sample["text"])
print("")
print("="*80)
print("PREDICTION:")
print(prediction)

image
```

### ### Val Set Error Audit

**Cell 52:**

```python
# ── Val Set Error Audit ───────────────────────────────────────────────────────
# Runs inference on the full val set (3,193 samples) using the saved 0.0687 model.
# Tags each sample by likely source using text heuristics (no source field in dataset).
# Outputs: per-source CER, worst 30 predictions, image size distribution of failures.
# Cost: ~25 min inference, zero training quota.

import io, re, torch, collections
from pathlib import Path
from PIL import Image
from jiwer import cer, wer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

AUDIT_MODEL_PATH = str(MODEL_INPUT)  # 0.0687 weights

print(f"Loading model from: {AUDIT_MODEL_PATH}")
audit_processor = TrOCRProcessor.from_pretrained(AUDIT_MODEL_PATH)
audit_model     = VisionEncoderDecoderModel.from_pretrained(AUDIT_MODEL_PATH)
audit_model.eval().to("cuda")
print("Model loaded")

# ── Source heuristic ──────────────────────────────────────────────────────────
# Imperfect but good enough to split the three sources.
# CORD:        Indonesian text, prices like 17500 / 46000 (no comma/dot separator)
# SROIE:       English retail, prices like 1.99 / 12.50, short lines
# WildReceipt: Mixed, often has item codes, longer lines, comma-separated prices

def guess_source(text: str) -> str:
    # 5-digit bare integers are almost exclusively CORD (Indonesian prices)
    if re.search(r'\b\d{4,6}\b', text) and not re.search(r'\d+[.,]\d{2,3}', text):
        return "CORD"
    # English words + decimal prices → SROIE
    if re.search(r'\d+\.\d{2}\b', text) and re.search(r'[A-Z]{2,}', text):
        return "SROIE"
    return "WildReceipt"

# ── Run inference on full val set ─────────────────────────────────────────────
val_samples = combined_dataset["validation"]
print(f"Val set size: {len(val_samples)}")

by_source = collections.defaultdict(lambda: {
    "preds": [], "labels": [], "widths": [], "heights": [], "sample_cers": []
})

for idx in range(len(val_samples)):
    sample = val_samples[idx]
    try:
        image  = Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")
        w, h   = image.size
        if w < 4 or h < 4:
            continue

        pixel_values = audit_processor(
            images=image, return_tensors="pt"
        ).pixel_values.to("cuda")

        with torch.no_grad():
            generated_ids = audit_model.generate(
                pixel_values=pixel_values,
                max_new_tokens=128,
                num_beams=4,
                length_penalty=1.0,
                no_repeat_ngram_size=0,
            )

        pred  = audit_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        label = sample["text"]
        src   = guess_source(label)
        sc    = cer([label], [pred])

        by_source[src]["preds"].append(pred)
        by_source[src]["labels"].append(label)
        by_source[src]["widths"].append(w)
        by_source[src]["heights"].append(h)
        by_source[src]["sample_cers"].append((sc, label, pred, w, h))

    except Exception as e:
        print(f"Error at idx={idx}: {e}")

    if idx % 500 == 0:
        print(f"  {idx}/{len(val_samples)} processed...")

# ── Per-source summary ────────────────────────────────────────────────────────
print("\n── Per-source CER (val set) ─────────────────────────────")
all_sample_cers = []
for src in ["CORD", "SROIE", "WildReceipt"]:
    data = by_source[src]
    if not data["preds"]:
        print(f"  {src:12s} | n=   0 | (no samples tagged)")
        continue
    src_cer = cer(data["labels"], data["preds"])
    src_wer = wer(data["labels"], data["preds"])
    avg_w   = sum(data["widths"])  / len(data["widths"])
    avg_h   = sum(data["heights"]) / len(data["heights"])
    print(f"  {src:12s} | n={len(data['preds']):4d} | CER={src_cer:.4f} | WER={src_wer:.4f} | avg_img={avg_w:.0f}x{avg_h:.0f}px")
    all_sample_cers.extend(data["sample_cers"])

# ── Worst 30 predictions ────────────────────────r──────────────────────────────
print("\n── Worst 30 predictions by CER ──────────────────────────")
worst = sorted(all_sample_cers, key=lambda x: x[0], reverse=True)[:30]
for sc, label, pred, w, h in worst:
    src = guess_source(label)
    print(f"[{src:12s}] CER={sc:.3f} img={w}x{h}px")
    print(f"  GT  : {label!r}")
    print(f"  PRED: {pred!r}")

# ── CER distribution ──────────────────────────────────────────────────────────
print("\n── CER bucket distribution (all val samples) ────────────")
buckets = {"0.00":0, "0.01-0.05":0, "0.06-0.10":0, "0.11-0.20":0, "0.21-0.50":0, ">0.50":0}
for sc, *_ in all_sample_cers:
    if sc == 0:           buckets["0.00"] += 1
    elif sc <= 0.05:      buckets["0.01-0.05"] += 1
    elif sc <= 0.10:      buckets["0.06-0.10"] += 1
    elif sc <= 0.20:      buckets["0.11-0.20"] += 1
    elif sc <= 0.50:      buckets["0.21-0.50"] += 1
    else:                 buckets[">0.50"] += 1
total = sum(buckets.values())
for bucket, count in buckets.items():
    bar = "█" * int(30 * count / total)
    print(f"  CER {bucket:10s} | {count:4d} ({100*count/total:4.1f}%) {bar}")
```

### ### Load Model for Inference

**Cell 54:**

```python
# ============================================================
# CELL N: Load Model for Inference (trocr-smart-stock-best, CER 0.0631)
# ============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # must be first line, per known gotcha

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel

MODEL_INPUT = "/kaggle/input/datasets/maazahmad69/trocr-smart-stock-model/trocr-smart-stock-best/trocr-smart-stock-best"

processor = TrOCRProcessor.from_pretrained(MODEL_INPUT)
base_model = VisionEncoderDecoderModel.from_pretrained(MODEL_INPUT)

# If adapter is stored separately, load + merge. If MODEL_INPUT already contains
# a merged model (no adapter_config.json present), skip this block.
if os.path.exists(os.path.join(MODEL_INPUT, "adapter_config.json")):
    peft_model = PeftModel.from_pretrained(base_model, MODEL_INPUT)
    model = peft_model.merge_and_unload()
else:
    model = base_model

# Known bug: saved generation_config.max_length can silently truncate to 20
model.generation_config.max_length = 256
model.generation_config.num_beams = 1
model.generation_config.use_cache = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"Model loaded on {device}, max_length={model.generation_config.max_length}")
```

### ### Load Your Receipt Photos

**Cell 56:**

```python
# ============================================================
# CELL N+1: Load Your Receipt Photos
# ============================================================
from pathlib import Path
from PIL import Image

# ASSUMPTION: your uploaded photos live under this path — change if wrong
IMAGE_DIR = Path("/kaggle/input/datasets/maazahmad69/inference/inference")  # <-- update to actual dataset slug

image_paths = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpeg")))
print(f"Found {len(image_paths)} images")
for p in image_paths:
    print(" -", p.name)
```

### ### Line Segmentation for Real Receipt Photos

**Cell 58:**

```python
# ============================================================
# CELL N+1b: Line Segmentation for Real Receipt Photos
# ============================================================
import cv2
import numpy as np
from PIL import Image

def deskew(gray: np.ndarray) -> np.ndarray:
    """Hough-line based deskew. Falls back to original if no strong lines found."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    if lines is None:
        return gray
    angles = []
    for rho_theta in lines[:50]:
        theta = rho_theta[0][1]
        angle = (theta * 180 / np.pi) - 90
        if -20 < angle < 20:  # ignore near-vertical lines
            angles.append(angle)
    if not angles:
        return gray
    median_angle = float(np.median(angles))
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def segment_lines(image_path, min_width=20, min_height=8, pad=4, dilate_kernel=(25, 3)):
    """
    Returns list of (cropped_PIL_image, bbox) sorted top-to-bottom,
    for a single full receipt photo.
    """
    pil_img = Image.open(image_path).convert("RGB")
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = deskew(gray)

    # Otsu binarize (inverted: text = white, background = black)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge words on the same line into one blob,
    # while keeping separate lines apart (kernel wider than tall)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_width and h >= min_height:
            boxes.append((x, y, w, h))

    # Sort top-to-bottom, then left-to-right for any same-row boxes
    boxes.sort(key=lambda b: (b[1], b[0]))

    h_img, w_img = gray.shape
    crops = []
    for (x, y, w, h) in boxes:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        crop = pil_img.crop((x1, y1, x2, y2))
        crops.append((crop, (x1, y1, x2, y2)))

    return crops, pil_img

# ── Run on all 4 images and visualize the detected line boxes ──────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

all_crops = {}
for path in image_paths:
    crops, orig = segment_lines(path)
    all_crops[path.name] = crops

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(orig)
    for (_, (x1, y1, x2, y2)) in crops:
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         fill=False, edgecolor="red", linewidth=1))
    ax.set_title(f"{path.name} — {len(crops)} lines detected")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
```

**Cell 59:**

```python
# ============================================================
# CELL N+1c: Filter Crops (size + near-blank) Before Inference
# ============================================================
MIN_WIDTH = 50
MIN_HEIGHT = 10
MIN_DARK_PIXEL_RATIO = 0.02  # drop crops that are <2% dark pixels (near-blank)

def is_valid_crop(crop_img: Image.Image) -> bool:
    w, h = crop_img.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        return False

    gray = cv2.cvtColor(np.array(crop_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_ratio = np.count_nonzero(binary) / binary.size
    if dark_ratio < MIN_DARK_PIXEL_RATIO:
        return False

    return True
```

**Cell 60:**

```python
# ============================================================
# CELL N+1d: Install EasyOCR (detector only — we keep our own TrOCR for recognition)
# ============================================================
!pip install -q easyocr
```

**Cell 61:**

```python
# ============================================================
# CELL N+1e: Text Detection with EasyOCR (CRAFT) — replaces contour heuristic
# ============================================================
import easyocr

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def detect_lines_craft(image_path, pad=4):
    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)

    # detect() returns (horizontal_boxes, free_boxes) — we want horizontal_boxes
    horizontal_list, free_list = reader.detect(img_np)
    boxes = horizontal_list[0]  # [x_min, x_max, y_min, y_max] per box

    # sort top-to-bottom, then left-to-right
    boxes = sorted(boxes, key=lambda b: (b[2], b[0]))

    w_img, h_img = pil_img.size
    crops = []
    for (x_min, x_max, y_min, y_max) in boxes:
        x1 = max(0, int(x_min) - pad)
        y1 = max(0, int(y_min) - pad)
        x2 = min(w_img, int(x_max) + pad)
        y2 = min(h_img, int(y_max) + pad)
        crop = pil_img.crop((x1, y1, x2, y2))
        crops.append((crop, (x1, y1, x2, y2)))

    return crops, pil_img

# Run on all 4 images, apply same size + blank filter, visualize
craft_crops = {}
for path in image_paths:
    crops, orig = detect_lines_craft(path)
    kept = [(img, bbox) for (img, bbox) in crops if is_valid_crop(img)]
    craft_crops[path.name] = kept

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(orig)
    for (_, (x1, y1, x2, y2)) in kept:
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         fill=False, edgecolor="red", linewidth=1))
    ax.set_title(f"{path.name} — {len(crops)} detected, {len(kept)} kept")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
```

### ### OCR Model on Each Detected Line

**Cell 63:**

```python
# ============================================================
# CELL N+2c: Batched Inference on CRAFT Crops (also fixes the speed issue)
# ============================================================
def predict_batch(crop_imgs, batch_size=16):
    texts = []
    for i in range(0, len(crop_imgs), batch_size):
        batch = crop_imgs[i:i + batch_size]
        pixel_values = processor(images=[c.convert("RGB") for c in batch],
                                  return_tensors="pt", padding=True).pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        texts.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
    return texts

for name, crops in craft_crops.items():
    imgs = [c for c, _ in crops]
    preds = predict_batch(imgs)
    print(f"\n=== {name} ({len(imgs)} lines) ===")
    for i, text in enumerate(preds):
        print(f"  line {i:2d}: {text}")
```

**Cell 64:**

```python
print("device:", device)
print("num_beams:", model.generation_config.num_beams)
print("use_cache:", model.generation_config.use_cache)
print("early_stopping:", model.generation_config.early_stopping)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
```

**Cell 65:**

```python
import time

t0 = time.time()
crops, orig = detect_lines_craft(image_paths[0])
t1 = time.time()
kept = [(img, bbox) for (img, bbox) in crops if is_valid_crop(img)]
imgs = [c for c, _ in kept]
preds = predict_batch(imgs)
t2 = time.time()

print(f"CRAFT detection: {t1-t0:.2f}s")
print(f"TrOCR batch generate ({len(imgs)} lines): {t2-t1:.2f}s")
```

