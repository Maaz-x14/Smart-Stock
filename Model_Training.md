# Model_Training.md — Model Training Guide
## Smart-Stock: TrOCR + DistilBERT Fine-Tuning

**Version:** 3.0 (Updated — OOM fix, SROIE file-based loader, disk persistence)  
**Training Environment:** Kaggle (T4/P100 GPU)  
**Last Updated:** Based on live dataset audit + runtime error fixes

---

## ⚠️ Critical Dataset Reality Check

Before any code: here is what the two datasets actually contain and what role each plays.

| Dataset | Fields | Usable For | NOT Usable For |
|---|---|---|---|
| `naver-clova-ix/cord-v2` | `image` (PIL), `ground_truth` (JSON str) | **TrOCR fine-tuning** (image → line items text) | NER (no token-level tags) |
| `sizhkhy/SROIE` | `words`, `bboxes`, `ner_tags`, `images`, `fields` | **TrOCR fine-tuning** (words → reconstruct text), **NER pre-training** | Food NER (tags are COMPANY/DATE/ADDRESS/TOTAL only) |

The original notebook's `preprocess_trocr` function referenced `example["image_path"]` and `example["text"]` — **neither field exists in either dataset**. The corrected pipeline is below.

---

## Part 1: TrOCR Fine-Tuning

### 1.1 What TrOCR Learns Here

TrOCR (`microsoft/trocr-base-printed`) is a Vision Encoder-Decoder model. The encoder (ViT) reads the receipt image; the decoder (RoBERTa) generates the transcribed text. Fine-tuning on CORD + SROIE teaches it the visual patterns of receipt typography: compressed thermal fonts, faded ink, price columns, item name abbreviations.

**Input:** Receipt image (PIL RGB)  
**Output:** String of line items (e.g., `"Nasi Campur Bali 1 x 75,000 | Bbk Bengil Nasi 1 x 125,000"`)

---

### 1.2 Dataset Preparation

#### CORD Ground Truth Extraction

CORD's `ground_truth` field is a raw JSON string. You must parse it and construct a flat text target from the `menu` array:

```python
import json

def extract_cord_text(ground_truth_str: str) -> str:
    """
    Parse CORD ground_truth JSON and build a flat text string
    suitable as a TrOCR decoding target.
    
    CORD structure:
    {
      "gt_parse": {
        "menu": [
          {"nm": "Nasi Campur Bali", "cnt": "1 x", "price": "75,000"},
          ...
        ],
        "sub_total": {"subtotal_price": "428,000"},
        "total": {"total_price": "428,000"}
      }
    }
    """
    try:
        parsed = json.loads(ground_truth_str)
        gt = parsed.get("gt_parse", {})
        
        parts = []
        for item in gt.get("menu", []):
            # CORD menu items can be dicts OR plain strings — skip non-dicts
            if not isinstance(item, dict):
                continue
            nm    = item.get("nm", "").strip()
            cnt   = item.get("cnt", "").strip()
            price = item.get("price", "").strip()
            if nm:
                parts.append(f"{nm} {cnt} {price}".strip())
        
        # Optionally append total
        total = gt.get("total", {})
        if isinstance(total, dict) and total.get("total_price", ""):
            parts.append(f"TOTAL {total['total_price']}")
        
        return " | ".join(parts)
    
    except (json.JSONDecodeError, KeyError, AttributeError):
        return ""  # Skip malformed examples
```

**Example output:**
```
Nasi Campur Bali 1 x 75,000 | Bbk Bengil Nasi 1 x 125,000 | MilkShake Starwb 1 x 37,000 | TOTAL 428,000
```

#### SROIE Text Reconstruction

SROIE is loaded directly from the Kaggle dataset files (not HF Hub). The folder structure is:
```
SROIE2019/train/img/        ← receipt images (.jpg)
SROIE2019/train/entities/   ← ground truth .txt files (JSON: company/date/address/total)
SROIE2019/test/img/         ← test images (no entities folder)
```

```python
from pathlib import Path

SROIE_ROOT = Path("/kaggle/input/sroie-datasetv2/SROIE2019")

def load_sroie_from_disk(split: str) -> tuple[list, list]:
    """
    Load SROIE directly from Kaggle dataset files.
    Entities files contain JSON: {"company":..., "date":..., "address":..., "total":...}
    Only train split has entities — test split is skipped (no text targets).
    """
    img_dir = SROIE_ROOT / split / "img"
    images, texts = [], []

    for img_path in sorted(img_dir.glob("*.jpg")):
        entities_path = SROIE_ROOT / split / "entities" / f"{img_path.stem}.txt"
        if not entities_path.exists():
            continue
        try:
            with open(entities_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            parts = [v.strip() for v in data.values() if isinstance(v, str) and v.strip()]
            text = " | ".join(parts)
        except (json.JSONDecodeError, Exception):
            text = ""

        if text:
            images.append(Image.open(img_path).convert("RGB"))
            texts.append(text)

    return images, texts
```

#### Combined Dataset Builder

> **OOM fix:** Images are never held as PIL objects in RAM all at once. Each image is immediately encoded to PNG bytes and stored in the HuggingFace Dataset's Arrow format on disk. The dataset is saved to `/kaggle/working/` on first run — subsequent runs load from disk and skip all reprocessing.

```python
import io
import json
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image

SROIE_ROOT = Path("/kaggle/input/sroie-datasetv2/SROIE2019")
SAVE_PATH  = Path("/kaggle/working/smart_stock_dataset")

def build_and_save_dataset():
    """
    Build combined CORD + SROIE dataset and save to disk.
    On subsequent runs, loads directly from disk — no reprocessing.
    """
    if SAVE_PATH.exists():
        print(f"Dataset found at {SAVE_PATH} — loading from disk...")
        return DatasetDict.load_from_disk(str(SAVE_PATH))

    print("Building dataset from scratch...")

    def pil_to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def make_dataset(images: list, texts: list) -> Dataset:
        # Convert PIL → bytes immediately to avoid RAM accumulation
        return Dataset.from_dict({
            "image_bytes": [pil_to_bytes(img) for img in images],
            "text": texts,
        })

    # --- CORD ---
    print("Loading CORD...")
    cord = load_dataset("naver-clova-ix/cord-v2")
    cord_data = {}
    for split in ["train", "validation", "test"]:
        imgs, txts = [], []
        for ex in cord[split]:
            text = extract_cord_text(ex["ground_truth"])
            if text:
                imgs.append(ex["image"])
                txts.append(text)
        cord_data[split] = (imgs, txts)
        print(f"  CORD {split}: {len(imgs)} examples")
    del cord  # Free RAM before loading SROIE

    # --- SROIE from Kaggle files ---
    print("Loading SROIE from disk...")
    sroie_imgs, sroie_txts = load_sroie_from_disk("train")
    print(f"  SROIE train: {len(sroie_imgs)} examples")

    # --- Combine and save ---
    dataset_dict = DatasetDict({
        "train":      make_dataset(cord_data["train"][0]      + sroie_imgs,
                                   cord_data["train"][1]      + sroie_txts),
        "validation": make_dataset(cord_data["validation"][0],
                                   cord_data["validation"][1]),
        "test":       make_dataset(cord_data["test"][0],
                                   cord_data["test"][1]),
    })

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(SAVE_PATH))

    print(f"\n✅ Saved to {SAVE_PATH}")
    print(f"   Train:      {len(dataset_dict['train'])}")
    print(f"   Validation: {len(dataset_dict['validation'])}")
    print(f"   Test:       {len(dataset_dict['test'])}")
    return dataset_dict

combined_dataset = build_and_save_dataset()
```

Expected sizes (pre-augmentation):
- Train: ~1,200+ examples (CORD train + SROIE train entities)
- Validation: ~100 examples (CORD val only)
- Test: ~100 examples (CORD test only — SROIE test has no text targets)

---

### 1.3 Augmentation

Apply augmentation **only to training images**, inline during the `preprocess_trocr` step. The augmentation simulates real-world receipt degradation.

```python
import albumentations as A
import cv2
import numpy as np
from PIL import Image

receipt_augmentation = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.1), p=0.5),  # Thermal fade
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),                       # Scanner noise
    A.Rotate(limit=5, border_mode=cv2.BORDER_REPLICATE, p=0.5),        # Crumple tilt
    A.Perspective(scale=(0.02, 0.05), p=0.3),                          # Photo angle
    A.MotionBlur(blur_limit=3, p=0.2),                                 # Shaky photo
    A.ImageCompression(quality_lower=60, quality_upper=90, p=0.4),     # JPEG artifact
])

def apply_augmentation(pil_image: Image.Image) -> Image.Image:
    """Convert PIL → numpy → augment → PIL."""
    img_np = np.array(pil_image.convert("RGB"))
    augmented = receipt_augmentation(image=img_np)["image"]
    return Image.fromarray(augmented)
```

---

### 1.4 Preprocessing Function (Corrected)

This replaces the broken version in your current notebook:

```python
import io
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

def preprocess_trocr(example, augment: bool = False):
    """
    Preprocess a single example for TrOCR training.
    
    Args:
        example: dict with keys 'image_bytes' (PNG bytes) and 'text' (str)
        augment: apply receipt degradation augmentation (True for training only)
    
    Returns:
        dict with 'pixel_values' and 'labels' ready for Seq2SeqTrainer
    """
    # Decode from stored bytes — avoids holding all PIL images in RAM
    image = Image.open(io.BytesIO(example["image_bytes"])).convert("RGB")
    
    # Apply augmentation during training
    if augment:
        image = apply_augmentation(image)
    
    # Encode image → pixel_values (ViT expects 384x384)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # Encode text → token ids
    labels = processor.tokenizer(
        example["text"],
        padding="max_length",
        max_length=128,
        truncation=True,
    ).input_ids
    
    # Replace pad token id with -100 so it's ignored in cross-entropy loss
    labels = [
        token_id if token_id != processor.tokenizer.pad_token_id else -100
        for token_id in labels
    ]
    
    return {
        "pixel_values": pixel_values.squeeze(),
        "labels": labels,
    }

# Apply to datasets — augment train only
train_dataset = combined_dataset["train"].map(
    lambda ex: preprocess_trocr(ex, augment=True),
    remove_columns=["image_bytes", "text"],
    desc="Preprocessing train set",
)
val_dataset = combined_dataset["validation"].map(
    lambda ex: preprocess_trocr(ex, augment=False),
    remove_columns=["image_bytes", "text"],
    desc="Preprocessing val set",
)
test_dataset = combined_dataset["test"].map(
    lambda ex: preprocess_trocr(ex, augment=False),
    remove_columns=["image_bytes", "text"],
    desc="Preprocessing test set",
)

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")
```

---

### 1.5 Model Setup (Corrected)

```python
from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Required decoder config — without these the model won't generate properly
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size

# Generation config — controls beam search during evaluation
model.config.eos_token_id  = processor.tokenizer.sep_token_id
model.config.max_new_tokens = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
```

---

### 1.6 Training Arguments

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-smart-stock",
    
    # Training schedule
    num_train_epochs=10,
    per_device_train_batch_size=8,    # Safe for Kaggle T4 (16GB VRAM)
    per_device_eval_batch_size=8,
    
    # Optimizer
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    lr_scheduler_type="cosine",       # Better than linear for OCR tasks
    
    # Eval & saving
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,          # Lower CER = better
    save_total_limit=2,               # Keep only 2 checkpoints (Kaggle disk limit)
    
    # Generation
    predict_with_generate=True,
    generation_max_length=128,
    
    # Performance
    fp16=True,                        # Mixed precision — required on T4
    dataloader_num_workers=2,
    
    # Logging
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",                 # Disable wandb on Kaggle
)
```

---

### 1.7 Metrics

```python
from jiwer import cer, wer

def compute_metrics(pred):
    pred_ids   = pred.predictions
    labels_ids = pred.label_ids
    
    # Decode predictions
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    
    # Replace -100 (padding mask) with pad token id before decoding
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    return {
        "cer": cer(label_str, pred_str),
        "wer": wer(label_str, pred_str),
    }
```

---

### 1.8 Trainer and Training

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

### 1.9 Save & Export

```python
import os

# Save best model locally
trainer.save_model("./trocr-smart-stock/best-model")
processor.save_pretrained("./trocr-smart-stock/best-model")

# On Kaggle: save to /kaggle/working/ for persistence
save_path = "/kaggle/working/trocr-smart-stock-best"
trainer.save_model(save_path)
processor.save_pretrained(save_path)
print(f"Model saved to: {save_path}")
```

To download from Kaggle: go to **Output** tab in your notebook → download the folder.

---

### 1.10 Evaluate on Test Set

```python
results = trainer.evaluate(test_dataset)
print(f"Test CER: {results['eval_cer']:.4f}")
print(f"Test WER: {results['eval_wer']:.4f}")

# Target benchmarks:
# CER ≤ 0.05 (5%)
# WER ≤ 0.10 (10%)
```

---

### 1.11 Kaggle Runtime Estimate

| Phase | Duration (T4 GPU) |
|---|---|
| Dataset loading + extraction | ~5 min |
| Augmentation + preprocessing | ~10–15 min |
| Training (10 epochs, ~1,226 examples) | ~2–3 hours |
| Evaluation on test set | ~10 min |
| **Total** | **~3–4 hours** |

Kaggle sessions cap at 12 hours. This fits comfortably in one session.

---

## Part 2: DistilBERT NER Fine-Tuning

> **Approach: Option A — Remap CORD Annotations**  
> CORD's structured `ground_truth` JSON contains `nm` (item name), `cnt` (count/quantity), and `price` fields per menu item. These map directly to the food NER schema: `FOOD_ITEM`, `QUANTITY`, `PRICE`. No manual annotation required.  
>  
> SROIE's NER tags (COMPANY, ADDRESS, DATE, TOTAL) have **zero overlap** with food entities — SROIE is not used for NER training.

Full NER fine-tuning (data remapping, BIO tagging, training, evaluation) will be documented in **Part 2** once TrOCR training is complete and validated.

---

## Appendix: Troubleshooting

| Issue | Fix |
|---|---|
| `AttributeError: 'str' object has no attribute 'get'` | CORD menu items aren't always dicts — fixed by `if not isinstance(item, dict): continue` in `extract_cord_text` |
| Kernel OOM / restart | Was caused by holding all PIL images in RAM at once — fixed by storing images as PNG bytes in the Dataset and using `DatasetDict.save_to_disk()` |
| `KeyError: 'image'` in `preprocess_trocr` | Dataset now stores `image_bytes`, not `image` — use `Image.open(io.BytesIO(example["image_bytes"]))` |
| Dataset rebuilds every session | `build_and_save_dataset()` checks if `SAVE_PATH` exists first — if yes, loads from disk and skips all reprocessing |
| SROIE not loading from `load_dataset("sizhkhy/SROIE")` | SROIE is loaded from Kaggle file paths via `load_sroie_from_disk()`, not HF Hub |
| CUDA OOM during training on batch 8 | Reduce to `per_device_train_batch_size=4`, add `gradient_accumulation_steps=2` |
| `ViTImageProcessor` fast processor warning | Safe to ignore, or pass `use_fast=False` to `TrOCRProcessor.from_pretrained(...)` |
| Kaggle session timeout before training ends | Save checkpoints every epoch (`save_strategy="epoch"`) and resume with `trainer.train(resume_from_checkpoint=True)` |
