# Model_Training.md — Model Training Guide
## Smart-Stock: TrOCR + DistilBERT Fine-Tuning

**Version:** 6.0 (Line-level crops, SROIE 2x weighting, Optuna, training args adjusted)  
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

**Input:** Single cropped text line from a receipt (PIL RGB) — NOT the full receipt image  
**Output:** Text of that one line (e.g., `"Nasi Campur Bali 1 x 75,000"`)

> **Why line crops?** TrOCR was pretrained on single-line text images. Feeding it full receipt images (20–50 lines) is a domain mismatch — the model has never seen multi-line inputs. Line crops align with its pretraining format and are the single biggest fix for CER.

---

### 1.2 Dataset Preparation

#### CORD Ground Truth Extraction

CORD's `ground_truth` field is a raw JSON string. You must parse it and construct a flat text target from the `menu` array:

```python
import json
from PIL import Image

def extract_cord_crops(image: Image.Image, ground_truth_str: str) -> list:
    """
    Extract line-level crops from a CORD receipt image.
    Uses per-item bounding boxes (quad field) from ground_truth JSON.
    Returns list of (cropped_PIL_image, text) pairs — one per menu item.
    """
    try:
        gt = json.loads(ground_truth_str).get("gt_parse", {})
    except (json.JSONDecodeError, AttributeError):
        return []

    crops = []
    w, h = image.size
    for item in gt.get("menu", []):
        if not isinstance(item, dict):
            continue
        nm = item.get("nm", "").strip()
        if not nm:
            continue
        quad = item.get("quad", None)
        if not quad:
            continue
        try:
            xs = [p["x"] for p in quad]
            ys = [p["y"] for p in quad]
        except (KeyError, TypeError):
            continue
        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(w, max(xs)), min(h, max(ys))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        cnt   = item.get("cnt", "").strip()
        price = item.get("price", "").strip()
        text  = f"{nm} {cnt} {price}".strip()
        crops.append((crop, text))
    return crops
```

**Example:** One CORD receipt → ~8 crops, each paired with one menu item line.

#### SROIE Line-Level Crops

SROIE has `bboxes` per word. Group words by Y-coordinate (within 15px = same line), merge bounding boxes per line, crop each line region:

```python
def extract_sroie_crops(image: Image.Image, words: list, bboxes: list) -> list:
    """
    Group SROIE words into lines by Y-coordinate proximity.
    Each line bbox is cropped and paired with its joined text.
    Returns list of (cropped_PIL_image, text) pairs.
    """
    if not words or not bboxes:
        return []

    # Sort by top-Y of each word bbox
    items = sorted(zip(words, bboxes), key=lambda x: x[1][1])

    # Group into lines: words within 15px vertically = same line
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

#### Combined Dataset Builder

> **OOM fix:** Images encoded to PNG bytes immediately — no PIL accumulation in RAM.  
> **Line crops:** Each receipt yields multiple (crop, text) pairs instead of one (full_image, all_text).  
> **SROIE 2x weighting:** SROIE train concatenated twice — gives more weight to English receipt text vs CORD's Indonesian menus.  
> **New dataset path:** Since this is a rebuilt dataset, save to a new path to avoid loading the old full-image version.

```python
import io
import json
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image

# New path — line-crop dataset is incompatible with old full-image dataset
SAVE_PATH = Path("/kaggle/working/smart_stock_dataset_v2")

def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()

def iter_to_dataset(iterator) -> Dataset:
    """
    Convert an iterator of (PIL Image, text) tuples into a Dataset.
    Encodes each image to bytes immediately — never accumulates PIL objects in RAM.
    """
    img_bytes, texts = [], []
    for img, text in iterator:
        img_bytes.append(pil_to_bytes(img))
        texts.append(text)
    return Dataset.from_dict({"image_bytes": img_bytes, "text": texts})

def build_and_save_dataset():
    """
    Build line-crop CORD + SROIE dataset and save to disk.
    On subsequent runs, loads directly from disk — no reprocessing.
    """
    if SAVE_PATH.exists():
        print(f"Dataset found at {SAVE_PATH} — loading from disk...")
        return DatasetDict.load_from_disk(str(SAVE_PATH))

    print("Building line-crop dataset from scratch...")

    # --- CORD — line crops via bounding boxes ---
    print("Loading CORD...")
    cord = load_dataset("naver-clova-ix/cord-v2")

    def cord_iter(split):
        for ex in cord[split]:
            for crop, text in extract_cord_crops(ex["image"], ex["ground_truth"]):
                yield crop, text

    cord_train      = iter_to_dataset(cord_iter("train"))
    cord_validation = iter_to_dataset(cord_iter("validation"))
    cord_test       = iter_to_dataset(cord_iter("test"))
    print(f"  CORD train: {len(cord_train)} | val: {len(cord_validation)} | test: {len(cord_test)}")
    del cord

    # --- SROIE — line crops via word bbox grouping ---
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

    # --- Combine ---
    # SROIE train duplicated for 2x weight (more English receipt signal)
    # validation = CORD only (SROIE has no val split)
    from datasets import concatenate_datasets
    dataset_dict = DatasetDict({
        "train":      concatenate_datasets([cord_train, sroie_train, sroie_train]),
        "validation": cord_validation,
        "test":       concatenate_datasets([cord_test, sroie_test]),
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

Expected sizes (line crops):
- Train: ~24,000+ examples (CORD ~6k + SROIE ~9k × 2)
- Validation: ~800 examples (CORD val crops)
- Test: ~6,000 examples (CORD + SROIE test crops)

---

### 1.3 Augmentation

Apply augmentation **only to training images**, inline during the `preprocess_trocr` step. The augmentation simulates real-world receipt degradation.

```python
import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageOps

receipt_augmentation = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.1), p=0.5),  # Thermal fade
    A.GaussNoise(p=0.4),                                               # Scanner noise
    A.Rotate(limit=5, border_mode=cv2.BORDER_REPLICATE, p=0.5),        # Crumple tilt
    A.Perspective(scale=(0.02, 0.05), p=0.3),                          # Photo angle
    A.MotionBlur(blur_limit=3, p=0.2),                                 # Shaky photo
    A.ImageCompression(p=0.4),                                         # JPEG artifact
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
# cache_file_name must point to /kaggle/working/ — input dir is read-only
train_dataset = combined_dataset["train"].map(
    lambda ex: preprocess_trocr(ex, augment=True),
    remove_columns=["image_bytes", "text"],
    cache_file_name="/kaggle/working/cache_train.arrow",
    desc="Preprocessing train set",
)
val_dataset = combined_dataset["validation"].map(
    lambda ex: preprocess_trocr(ex, augment=False),
    remove_columns=["image_bytes", "text"],
    cache_file_name="/kaggle/working/cache_val.arrow",
    desc="Preprocessing val set",
)
test_dataset = combined_dataset["test"].map(
    lambda ex: preprocess_trocr(ex, augment=False),
    remove_columns=["image_bytes", "text"],
    cache_file_name="/kaggle/working/cache_test.arrow",
    desc="Preprocessing test set",
)

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")
```

---

### 1.5 Model Setup (Corrected)

```python
from transformers import VisionEncoderDecoderModel, GenerationConfig

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Required decoder config
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size

# Generation config — must go on model.generation_config, not model.config
# max_new_tokens omitted here — set via generation_max_length in training_args
model.generation_config.eos_token_id         = processor.tokenizer.sep_token_id
model.generation_config.early_stopping       = True
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty       = 2.0
model.generation_config.num_beams            = 4
```

---

### 1.6 Training Arguments

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-smart-stock",
    
    # Training schedule
    num_train_epochs=10,              # reduced back to 10 -- larger dataset means more steps per epoch — more time to converge
    per_device_train_batch_size=8,    # Safe for Kaggle T4 (16GB VRAM)
    per_device_eval_batch_size=8,
    
    # Optimizer
    learning_rate=2e-5,               # reduced from 5e-5 — more stable fine-tuning
    warmup_ratio=0.06,                # replaces warmup_steps=500 (was 56% of training — way too long)
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,                # clips exploding gradients — fixes high grad_norm seen in run 1
    
    # Eval & saving
    eval_strategy="epoch",            # renamed from evaluation_strategy in transformers 4.46+
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,          # Lower CER = better
    save_total_limit=3,               # 3 checkpoints -- larger dataset, more valuable intermediates
    
    # Generation — only set max_new_tokens, not max_length (conflict warning)
    predict_with_generate=True,
    generation_max_length=128,
    
    # Performance
    fp16=True,                        # Mixed precision — required on T4
    dataloader_num_workers=2,
    
    # Logging — log every epoch so CER/WER appear in the table
    logging_dir="./logs",
    logging_steps=50,
    log_level="info",
    report_to="none",                 # Disable wandb on Kaggle
)
```

---

### 1.7 Metrics

```python
import numpy as np
from jiwer import cer, wer

def compute_metrics(pred):
    pred_ids   = pred.predictions
    labels_ids = pred.label_ids

    # pred_ids may be float logits — argmax to get token ids
    if pred_ids.dtype != np.int64 and pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    # Clip to valid vocab range to prevent OverflowError during decode
    vocab_size = processor.tokenizer.vocab_size
    pred_ids   = np.clip(pred_ids, 0, vocab_size - 1)

    # Decode predictions
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

### 1.8 Trainer and Training

```python
import torch
from transformers import Seq2SeqTrainer

def collate_fn(batch):
    """
    Custom collator for TrOCR.
    pixel_values and labels are already tensors (from set_format("torch")).
    Use torch.stack — torch.tensor() fails on a list of existing tensors.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# Resume from latest checkpoint if one exists (avoids retraining from scratch on reruns)
checkpoint_dir = Path("./trocr-smart-stock")
checkpoints = sorted(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
resume_from = str(checkpoints[-1]) if checkpoints else None
if resume_from:
    print(f"Resuming from checkpoint: {resume_from}")

trainer.train(resume_from_checkpoint=resume_from)
```

---

### 1.9 Save & Export

```python
# Save best model to /kaggle/working/ — this is what gets committed as output
save_path = "/kaggle/working/trocr-smart-stock-best"
trainer.save_model(save_path)
processor.save_pretrained(save_path)

# Verify both outputs exist and print sizes
from pathlib import Path
for folder in ["smart_stock_dataset", "trocr-smart-stock-best"]:
    path = Path(f"/kaggle/working/{folder}")
    if path.exists():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        print(f"✅ {folder}: {size:.1f} MB")
    else:
        print(f"❌ {folder}: NOT FOUND")
```

#### ⚠️ How to permanently save outputs on Kaggle

**Quick Save does NOT save output files — it only saves notebook code.** The `/kaggle/working/` directory is wiped when the session ends unless you use Save & Run All.

**Correct workflow:**
1. Make sure the save cell above is the last cell in your notebook
2. Click **"Save Version"** → **"Save & Run All"**
3. Wait for the full run to complete (~3–4 hours)
4. After completion, go to your notebook page on kaggle.com → **Output** section
5. Click the three dots next to `smart_stock_dataset` → **"Create Dataset"** — repeat for `trocr-smart-stock-best`
6. These become permanent Kaggle datasets — attach them to any future notebook via **Add Input**

**Loading in future sessions once saved as datasets:**
```python
# Dataset
SAVE_PATH = Path("/kaggle/input/smart-stock-dataset/smart_stock_dataset")

# Model
model = VisionEncoderDecoderModel.from_pretrained("/kaggle/input/trocr-smart-stock-best")
processor = TrOCRProcessor.from_pretrained("/kaggle/input/trocr-smart-stock-best")
```

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
| Dataset building (line crops) | ~20-30 min |
| Augmentation + preprocessing | ~30-45 min |
| Training (10 epochs, ~24,000 examples) | ~6–8 hours |
| Evaluation on test set | ~10 min |
| **Total** | **~8-10 hours** |

Kaggle sessions cap at 12 hours. This fits but is tight -- use Save & Run All.

---

## Part 1b: Hyperparameter Search with Optuna

After the first clean training run with line crops, use Optuna (Bayesian optimization) via HuggingFace built-in `hyperparameter_search` to find optimal values.

> Run Optuna **after** confirming line-crop training improves CER baseline. No point searching on a broken dataset.

```python
# !pip install optuna -q

def optuna_hp_space(trial):
    return {
        "learning_rate":               trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "warmup_ratio":                trial.suggest_float("warmup_ratio", 0.03, 0.15),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
    }

def model_init():
    m = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    m.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    m.config.pad_token_id           = processor.tokenizer.pad_token_id
    m.config.vocab_size             = m.config.decoder.vocab_size
    m.generation_config.eos_token_id         = processor.tokenizer.sep_token_id
    m.generation_config.early_stopping       = True
    m.generation_config.no_repeat_ngram_size = 3
    m.generation_config.length_penalty       = 2.0
    m.generation_config.num_beams            = 4
    return m

# Search on 20% of train data for speed
search_train = train_dataset.select(range(len(train_dataset) // 5))

search_trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=search_train,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

best_run = search_trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10,
)

print("Best hyperparameters:", best_run.hyperparameters)
for k, v in best_run.hyperparameters.items():
    setattr(training_args, k, v)
```

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
| `TypeError: unexpected keyword argument 'tokenizer'` / `ValueError: you provided ['pixel_values', 'labels']` | Don't pass `tokenizer` or `processing_class` to `Seq2SeqTrainer` for vision-encoder-decoder models. Use a custom `data_collator` instead (see Trainer cell above) |
| `ValueError: You have modified the pretrained model configuration to control generation` | Generation params (`num_beams`, `early_stopping`, etc.) must go on `model.generation_config`, not `model.config` — see model setup cell |
| `TypeError: unexpected keyword argument 'evaluation_strategy'` | Renamed to `eval_strategy` in transformers 4.46+ — use `eval_strategy="epoch"` |
| `AttributeError: 'str' object has no attribute 'get'` | CORD menu items aren't always dicts — fixed by `if not isinstance(item, dict): continue` in `extract_cord_text` |
| Kernel OOM / restart | Root cause: list comprehension materialized all images as both PIL objects and bytes simultaneously. Fixed by `iter_to_dataset()` which encodes one image at a time, plus processing one split at a time with `del cord` / `del sroie` between them |
| `KeyError: 'image'` in `preprocess_trocr` | Dataset now stores `image_bytes`, not `image` — use `Image.open(io.BytesIO(example["image_bytes"]))` |
| Dataset rebuilds every session | `build_and_save_dataset()` checks if `SAVE_PATH` exists first — if yes, loads from disk and skips all reprocessing |
| CUDA OOM during training on batch 8 | Reduce to `per_device_train_batch_size=4`, add `gradient_accumulation_steps=2` |
| `OSError: [Errno 30] Read-only file system` in `.map()` | `/kaggle/input/` is read-only — HuggingFace tries to write cache there. Fix: add `cache_file_name="/kaggle/working/cache_train.arrow"` to each `.map()` call |
| `ViTImageProcessor` fast processor warning | Safe to ignore, or pass `use_fast=False` to `TrOCRProcessor.from_pretrained(...)` |
| Kaggle session timeout before training ends | Save checkpoints every epoch (`save_strategy="epoch"`) and resume with `trainer.train(resume_from_checkpoint=True)` |
| Output files lost after session ends | Quick Save only saves code, not outputs. Use **Save & Run All** to commit `/kaggle/working/` contents permanently. Then create Kaggle datasets from the Output tab. |
| `UserWarning: Argument 'var_limit' not valid for GaussNoise` | Albumentations API changed — use `A.GaussNoise(p=0.4)` and `A.ImageCompression(p=0.4)` without named quality/variance args |
| `warmup_ratio is deprecated` warning | Harmless for now, will be removed in transformers v5.2 — no action needed yet |
