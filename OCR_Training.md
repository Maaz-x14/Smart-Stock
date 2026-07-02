# OCR_Training.md — Model Training Guide
## SmartStock: TrOCR Fine-Tuning

**Version:** 14.0 — Complete notebook reference, every code block included  
**Training Environment:** Kaggle (2× T4 GPU, 30 hr/week quota — single GPU enforced)  
**Last Updated:** Post v3 dataset integration (CORD + SROIE + WildReceipt)

---

## Pipeline Overview

```
Receipt Image → TrOCR (Stage 1) → NER (Stage 2) → Normalization → Expiry Prediction
```

This document covers **Stage 1: TrOCR fine-tuning only.**

**Base model:** `microsoft/trocr-base-printed`  
**Adapter:** LoRA on decoder only (encoder frozen)  
**Trainable params:** 1,523,712 of 335M total (0.45%)  
**Target:** CER ≤ 0.05 | WER ≤ 0.10  
**Current best saved model:** `trocr-smart-stock-best` on Kaggle — CER ~0.0856  
**Rule:** Do NOT overwrite `trocr-smart-stock-best` unless new run beats 0.0856

---

## Architecture Notes

### Why LoRA + Frozen Encoder

Full fine-tuning of 335M params caused plateau and optimizer instability (CER stuck at 0.1758). Current approach freezes the ViT encoder entirely (preserves pretrained visual features) and applies LoRA only to the RoBERTa decoder's attention projections.

**Known ceiling:** The frozen encoder is the current performance ceiling. Once a stable single-GPU baseline is re-established on v3 data, unfreezing the top 2–4 ViT encoder blocks alongside LoRA is the highest-impact next step.

### Why Line Crops

TrOCR was pretrained on single-line text images. Feeding full receipts (20–50 lines) is a domain mismatch. Line crops are the single biggest fix — CER dropped from 0.757 (full images) → 0.133 after switching.

---

## 12-Hour Session Budget (Kaggle T4, Single GPU)

| Phase | Time |
|-------|------|
| pip installs | ~3 min |
| Dataset load from disk (v3 saved, no rebuild) | ~2 min |
| Model setup | ~2 min |
| 8 epochs × 8,558 steps × ~0.55s (fp16, LoRA, frozen enc) | ~10.5 hr |
| Val eval × 8 epochs (3,193 samples) | ~16 min |
| Save & Export | ~5 min |
| Test eval — manual loop, 9,301 samples, beam=4 | ~60 min |
| **Total** | **~12.0 hr** ⚠️ tight |

> **If running 8 epochs with 68k dataset, monitor closely.** If epoch 1 takes longer than 80 min, reduce to 6 epochs. With `load_best_model_at_end=True` the best checkpoint is saved automatically so you won't lose the best model even if you have to stop early.
>
> **After partial encoder unfreeze:** each step will be slower (~0.7s vs 0.55s). Reduce to 5–6 epochs when unfreezing encoder blocks.

**Use Save & Run All — never draft mode.** Draft mode does not commit `/kaggle/working/` outputs.

---

## Dataset v3 — Composition

| Source | Train | Val | Test | Notes |
|--------|-------|-----|------|-------|
| CORD | 2,105 | 221 | 251 | Indonesian restaurant receipts, line-level by group_id |
| SROIE | 14,476 | — | 8,050 | English retail receipts, **2× weighted in train**, line-level by Y-proximity |
| WildReceipt (train.txt → 90%) | 26,741 | — | — | Per-annotation crops |
| WildReceipt (train.txt → 10%) | — | 2,972 | — | Val split from train.txt crops |
| WildReceipt (test.txt, capped) | 10,665→train | — | 1,000 | Excess test crops moved to train |
| **Total v3** | **68,463** | **3,193** | **9,301** | |

**WildReceipt labels excluded:** 0 (empty/illegible), 25 (catch-all: terminal IDs, legal text, thank-you messages)  
**WildReceipt cropping strategy:** per-annotation (not line grouping) — eliminates two-column merging bug  
**Why per-annotation for WildReceipt only:** CORD groups by group_id (logical receipt lines, working well). SROIE uses Y-proximity (words well-spaced, no column issues). WildReceipt had two-column layouts causing Y-grouping to merge item names + prices into nonsense crops — corrupted ~33% of training data, caused CER regression 0.088 → 0.339  
**Test capped at 1,000 WildReceipt crops** — prevents beam search hanging for hours (previous session lost 6+ hrs to this)

---

## Kaggle Dataset Inputs (Current Session)

| Kaggle slug | Path | Purpose |
|-------------|------|---------|
| `smart-stock-model-data` | `/kaggle/input/datasets/maazahmad69/smart-stock-model-data/smart_stock_dataset_v3` | Combined v3 dataset |
| `smart-stock-model-data` | `/kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock-best` | Best model weights (CER 0.0771) |
| `smart-stock-model-data` | `/kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock` | Prior session checkpoints (resume fallback) |
| `wild-receipt` | `/kaggle/input/datasets/maazahmad69/wild-receipt/wildreceipt` | Raw images + annotations, needed only during first dataset build |

> All model data lives in one dataset: **smart-stock-model-data**
> URL: https://www.kaggle.com/datasets/maazahmad69/smart-stock-model-data

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

# v3 dataset (CORD + SROIE + WildReceipt) — loads from disk if exists, builds if not
DATASET_DIR = Path("/kaggle/input/datasets/maazahmad69/smart-stock-model-data/smart_stock_dataset_v3")

# WildReceipt raw input — only needed during dataset build cell
WILDRECEIPT_DIR = Path("/kaggle/input/datasets/maazahmad69/wild-receipt/wildreceipt")

# Best model weights — loaded as base for fine-tuning
MODEL_INPUT = Path("/kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock-best")

# Checkpoint dir from previously uploaded dataset (fallback resume source)
INPUT_CHECKPOINT_DIR = Path("/kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock")

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

DATASET_SAVE = DATASET_DIR  # v3 path — loads if exists, builds if not

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
INPUT_CHECKPOINT_DIR = Path("/kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock")

# Load base model from stored best weights
base_model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_INPUT))

# Freeze encoder entirely — ViT visual features preserved from pretraining
# This is the current performance ceiling; unfreeze top 2-4 blocks in future session
for param in base_model.encoder.parameters():
    param.requires_grad = False

# ── Checkpoint resume priority ────────────────────────────────────────────────
# 1. /kaggle/working/trocr-smart-stock  (current session checkpoints, most recent)
# 2. /kaggle/input/datasets/maazahmad69/smart-stock-model-data/trocr-smart-stock (uploaded from prior session — fallback)
# 3. Fresh LoRA (no prior checkpoint found)

def find_lora_checkpoints(directory: Path):
    """Find checkpoints with LoRA adapter weights in either format."""
    return sorted([
        ckpt for ckpt in directory.glob("checkpoint-*")
        if (ckpt / "lora_adapter").exists() or (ckpt / "adapter_config.json").exists()
    ])

working_checkpoints = find_lora_checkpoints(CHECKPOINT_DIR)
input_checkpoints   = find_lora_checkpoints(INPUT_CHECKPOINT_DIR) if INPUT_CHECKPOINT_DIR.exists() else []

if working_checkpoints:
    latest = working_checkpoints[-1]
    adapter_path = latest / "lora_adapter" if (latest / "lora_adapter").exists() else latest
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
    print(f"Resumed from working dir: {adapter_path}")
elif input_checkpoints:
    latest = input_checkpoints[-1]
    adapter_path = latest / "lora_adapter" if (latest / "lora_adapter").exists() else latest
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
model.generation_config.eos_token_id         = processor.tokenizer.sep_token_id
model.generation_config.early_stopping       = True
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty       = 2.0
model.generation_config.num_beams            = 4
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
#     m.generation_config.eos_token_id         = processor.tokenizer.sep_token_id
#     m.generation_config.early_stopping       = True
#     m.generation_config.no_repeat_ngram_size = 3
#     m.generation_config.length_penalty       = 2.0
#     m.generation_config.num_beams            = 4
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
resume_from = str(valid_checkpoints[-1]) if valid_checkpoints else None

if resume_from:
    print(f"Resuming from: {resume_from}")
else:
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

```python
from jiwer import cer, wer

model.eval()
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
        ).pixel_values.to(model.device)

        with torch.no_grad():
            # MUST be keyword arg — PeftModelForSeq2SeqLM.generate() does not
            # accept positional pixel_values. Silently skipped all 9301 samples
            # in v3 run 2 when passed positionally.
            generated_ids = model.generate(
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
).pixel_values.to(model.device)

generated_ids = model.generate(pixel_values=pixel_values, max_new_tokens=128)

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

| Run | Setup | Best Val CER | Best Val WER | Notes |
|-----|-------|-------------|-------------|-------|
| Full finetune | All 333M params | 0.1758 | — | Plateau, optimizer instability |
| LoRA first (Colab) | Frozen enc + LoRA dec | 0.0894 | — | 1 epoch only |
| Optuna search | 1 epoch, 1/8 data, batch 4 | 0.0803–0.0886 | — | Best: lr=1.4824e-4, warmup=0.02672 |
| Kaggle run 1 | 5 epochs, DataParallel batch 16 | 0.1325 | — | DataParallel doubled batch silently |
| Kaggle run 2 | 8 epochs, attempted single GPU | 0.1332 | — | DataParallel still fired — env var after torch import |
| v3 run 1 | 6 epochs, v3 46.5K train, WR line-grouped | 0.3396 | 0.5942 | WildReceipt column merging bug corrupted 33% of data |
| **v3 run 2 (current best)** | 6 epochs, v3 68.5K train, WR per-annotation | **0.0771** | **0.2383** | New best — saved as `trocr-smart-stock-best` |

**Stored best:** `trocr-smart-stock-best` — CER 0.0771, WER 0.2383 (v3 run 2, epoch 4 was best at 0.0771 but `load_best_model_at_end=False` saved epoch 6 at 0.0782).

**Key observation:** CER/WER gap (~3×) is expected for receipt OCR — one wrong character fails entire words like `"BCCHOCCUPCAKES"`. WER improves naturally as CER improves, not a separate problem.

---

## Bugs Fixed

### Bug 1 — DataParallel not disabled ✅ FIXED
`os.environ["CUDA_VISIBLE_DEVICES"] = "0"` was placed after PyTorch imports → Kaggle's 2×T4 silently ran DataParallel → effective batch doubled 8→16 → Optuna LR tuned at batch 4 was mismatched 4× over.

**Fix:** `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` is now the absolute first Python line in Cell 1, before even `from pathlib import Path`.

### Bug 2 — Test eval hangs on degenerate crops ✅ FIXED
Test set contains 1-pixel-wide image crops (`torch.Size([3, 42, 1])`). Beam search with `num_beams=4` hangs indefinitely on these. Previous session spent 6+ hours in test eval and never finished.

**Fix:** Manual loop in Cell 16 skips any sample with `w < 4 or h < 4`. Never use `trainer.evaluate()` on the test set.

### Bug 4 — Test eval skips all samples silently ✅ FIXED
`model.generate(pixel_values)` — passing `pixel_values` as a positional argument to `PeftModelForSeq2SeqLM.generate()` raises `TypeError: takes 1 positional argument but 2 were given`. The `except Exception` block silently swallowed this on every sample, reporting CER 0.0000 and WER 0.0000 with all 9,301 samples skipped. Affected v3 run 2 entirely — true test CER from that run is unknown.

**Fix:** Use keyword argument: `model.generate(pixel_values=pixel_values, max_new_tokens=128)`. Also added `first_error_printed` flag so the first exception surfaces instead of being swallowed.

### Bug 3 — PEFT checkpoint adapter reset ✅ FIXED
Without `LoRASaveCallback`, PEFT silently restores base model config instead of adapter weights on checkpoint resume. Training effectively restarts from scratch each time.

**Fix:** `LoRASaveCallback` saves `lora_adapter/` subdir alongside every checkpoint. `find_lora_checkpoints()` filters for checkpoints containing this subdir.

---

## Improvement Roadmap

### Tier 1 — Do next

| Technique | Status | Notes |
|-----------|--------|-------|
| Fix `load_best_model_at_end=False` | ✅ Fixed in next run config | Was saving epoch 6 even when epoch 4 had best CER. Now saves true best. |
| Switch to `num_cycles=1` | ✅ Fixed in next run config | 2 restarts caused CER oscillation. Single cosine decay more stable. |
| Partial encoder unfreeze | ⏳ Next run | Unfreeze top 2–4 ViT encoder blocks alongside LoRA. Biggest remaining performance lever. Add `gradient_accumulation_steps=2` if OOM after unfreezing. |
| Re-run Optuna | ⏳ After 1 stable epoch with unfrozen encoder | LR tuned on frozen encoder — gradient flow changes after unfreeze. Dedicated session only. |

### Tier 2 — Medium impact

| Technique | Notes |
|-----------|-------|
| Label smoothing (`label_smoothing_factor=0.1`) | Helps with noisy WildReceipt labels. Add to `Seq2SeqTrainingArguments`. |
| LoRA rank increase (r=16 → r=32) | ~3M extra params. May help with WildReceipt's text diversity. |
| Gradient accumulation (`gradient_accumulation_steps=2`) | Effective batch 16. Use only if OOM after encoder unfreeze. Already in notebook, just uncomment. |

### Tier 3 — Low effort, test after training

| Technique | Notes |
|-----------|-------|
| Beam search tuning (`num_beams` 4 → 6–8) | Inference only, no retraining needed. Change in model setup cell. |
| WildReceipt train weighting | Currently 1×. Monitor if WildReceipt dominates training signal vs CORD/SROIE. |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| DataParallel fires despite env var | Must be set before any `import torch` — even in prior cells. Restart kernel and run Cell 1 first. |
| Test eval hangs for hours | Use manual loop in Cell 16 with `w < 4 or h < 4` skip. Never `trainer.evaluate()` on test. |
| LoRA adapter resets on resume | `LoRASaveCallback` not attached or removed. Re-add to `callbacks=[LoRASaveCallback()]`. |
| Stale checkpoints without adapter | Trainer cell removes them: `shutil.rmtree(ckpt)` for checkpoints missing `lora_adapter/`. |
| Dataset rebuilds every session | `DATASET_SAVE.exists()` check skips rebuild — only triggers if v3 Kaggle dataset not mounted. |
| OOM on batch 8 | Uncomment `gradient_accumulation_steps=2` in training_args — keep batch 8 as-is. |
| WildReceipt images not found | Check `WILDRECEIPT_DIR` — only needed during build, not inference or training on saved dataset. |
| `AttributeError: TrOCRDataset has no .select()` | It's a PyTorch Dataset. Use `combined_dataset["train"].select(...)` then wrap in `TrOCRDataset()`. |
| `GaussNoise` var_limit warning | Use `A.GaussNoise(p=0.5)` without named args — albumentations API changed. |
| CORD train: 0 crops extracted | `quad` is in `valid_line[].words[].quad` — NOT in `gt_parse.menu`. |
| fp16 scaler error on resume | Ensure `fp16=True` in training_args when resuming — checkpoint contains scaler state. |
| `Missing keys: decoder.output_projection.weight` | Harmless TrOCR architecture warning. Safe to ignore. |
| `ViTImageProcessor` fast processor warning | Safe to ignore, or pass `use_fast=False` to `TrOCRProcessor.from_pretrained()`. |
| Outputs lost after session | Quick Save = code only. Use **Save & Run All** to commit `/kaggle/working/` permanently. |
| `warmup_ratio is deprecated` | Harmless for now, will be removed in transformers v5.2. |
| CER regresses after adding WildReceipt | Was caused by two-column merging bug in Y-proximity grouping — fixed by switching to per-annotation crops. |
| `model.generate(pixel_values)` TypeError | PeftModelForSeq2SeqLM doesn't accept positional args. Use `model.generate(pixel_values=pixel_values, max_new_tokens=128)`. |
| Test eval reports 0.0000 CER with all samples skipped | Silent exception swallowing. Check `first_error_printed` output — likely the generate() positional arg bug above. |
| Best epoch not saved — final epoch saved instead | `load_best_model_at_end=False` in training_args. Set to True with `metric_for_best_model="eval_cer"`, `greater_is_better=False`, and `save_strategy="epoch"` matching `eval_strategy`. |
| CER oscillates between epochs (e.g. 0.092→0.079→0.088) | `num_cycles=2` in cosine_with_restarts causes LR spikes at restart points. Switch to `num_cycles=1` for stable decay. |

---

## Stage 2 — DistilBERT NER (Not Yet Started)

CORD's structured `ground_truth` JSON maps directly to food NER schema with no manual annotation:  
`nm` → `FOOD_ITEM` | `cnt` → `QUANTITY` | `price` → `PRICE`

SROIE tags (COMPANY, ADDRESS, DATE, TOTAL) have zero overlap with food entities — not used for NER.

Full NER fine-tuning documented here once TrOCR achieves CER ≤ 0.05.
