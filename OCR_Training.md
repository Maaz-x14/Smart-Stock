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
from collections import defaultdict
from PIL import Image

def extract_cord_crops(image: Image.Image, ground_truth_str: str) -> list:
    """
    Extract line-level crops from a CORD receipt image.

    Bounding boxes are in valid_line[].words[].quad, NOT in gt_parse.menu.
    Each group_id in valid_line = one logical receipt line.
    We group words by group_id, merge their quads into one bbox, crop it,
    and use the joined word texts as the OCR target.
    Only menu.* categories are kept (skip total, tax, header lines).

    Returns list of (cropped_PIL_image, text) pairs.
    """
    try:
        data = json.loads(ground_truth_str)
    except (json.JSONDecodeError, AttributeError):
        return []

    valid_lines = data.get("valid_line", [])
    if not valid_lines:
        return []

    # Group lines by group_id
    groups = defaultdict(list)
    for line in valid_lines:
        groups[line["group_id"]].append(line)

    w, h = image.size
    crops = []

    for gid, lines in groups.items():
        # Only keep menu lines (skip total, sub_total, etc.)
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

**Example:** group_id 3 → bbox=(176,552,664,586) → crop → text=`"1 REAL GANACHE 16,500"`  
One CORD receipt → ~8–12 crops, each one logical receipt line.

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
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

def preprocess_trocr(example, augment: bool = False):
    """
    Preprocess a single (image_bytes, text) example.
    Called per-item at access time — no upfront caching.

    Returns dict with 'pixel_values' (tensor) and 'labels' (tensor).
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
      - OOM (keep_in_memory=True on 31k examples)
      - OSError Errno 28 (cache_file_name on full disk)
    Zero disk usage, zero RAM accumulation, full dataset used.
    Compatible with Seq2SeqTrainer — accepts any PyTorch Dataset.
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
    save_total_limit=5,               # keep 5 checkpoints — prevents losing progress if session times out
    
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
    pixel_values and labels are tensors returned directly by TrOCRDataset.__getitem__.
    torch.stack combines them into batches.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels       = torch.stack([item["labels"]       for item in batch])
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
# Run 1 timed out at step 17,478/19,420 — resume from checkpoint-15536
checkpoint_dir = Path("./trocr-smart-stock")
checkpoints = sorted(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
resume_from = str(checkpoints[-1]) if checkpoints else None
if resume_from:
    print(f"Resuming from: {resume_from}")
else:
    print("No checkpoint found — training from scratch")

trainer.train(resume_from_checkpoint=resume_from)
```

---

### 1.9 Save & Export

```python
from pathlib import Path

save_path = "/kaggle/working/trocr-smart-stock-best"

# Save model weights + config
trainer.save_model(save_path)

# Save processor (tokenizer + image processor)
processor.save_pretrained(save_path)

# Save generation config (beam search params)
model.generation_config.save_pretrained(save_path)

# Verify all expected files are present
expected_files = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "processor_config.json",
]
print(f"\nSaved to: {save_path}")
for fname in expected_files:
    fpath = Path(save_path) / fname
    exists = fpath.exists()
    size = fpath.stat().st_size / 1e6 if exists else 0
    print(f"  {'✅' if exists else '❌'} {fname} ({size:.1f} MB)")

# Verify dataset also present
for folder in ["smart_stock_dataset_v2", "trocr-smart-stock-best"]:
    path = Path(f"/kaggle/working/{folder}")
    if path.exists():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        print(f"✅ {folder}/: {size:.1f} MB total")
    else:
        print(f"❌ {folder}/: NOT FOUND")
```

#### ⚠️ How to permanently save outputs on Kaggle

**Quick Save does NOT save output files — it only saves notebook code.** The `/kaggle/working/` directory is wiped when the session ends unless you use Save & Run All.

**Correct workflow:**
1. Make sure the save cell above is the last cell in your notebook
2. Click **"Save Version"** → **"Save & Run All"**
3. Wait for the full run to complete (~12 hours)
4. After completion, go to your notebook page on kaggle.com → **Output** section
5. Click the three dots next to `smart_stock_dataset` → **"Create Dataset"** — repeat for `trocr-smart-stock-best`
6. These become permanent Kaggle datasets — attach them to any future notebook via **Add Input**

**Loading in future sessions once saved as datasets:**
```python
# Dataset
SAVE_PATH = Path("/kaggle/input/datasets/maazahmad69/smart-stock-dataset/smart_stock_dataset_v2")

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
| Training (10 epochs, ~31,000 examples) | ~11–12 hours |
| Evaluation on test set | ~10 min |
| **Total** | **~12 hours** |

Kaggle sessions cap at 12 hours. Observed: ~10:47 for 10 epochs on T4 with dual GPU. Fits within limit — use Save & Run All.

**Run 1 result:** Session timed out at step 17,478/19,420 (90% complete). Only 2 checkpoints saved due to `save_total_limit=3` deleting earlier ones. Increased to `save_total_limit=5`. Resume from checkpoint-15536 for next run.

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
| `OSError: [Errno 30] Read-only file system` in `.map()` | `/kaggle/input/` is read-only — HuggingFace tries to write cache there. Fix: use `keep_in_memory=True` in each `.map()` call instead of `cache_file_name` |
| `OSError: [Errno 28] No space left on device` in `.map()` | Cache writing to a read-only path (e.g. `/kaggle/input/`). Fix: always point `cache_file_name` to `/kaggle/working/` which has ~20GB free |
| RAM OOM with `keep_in_memory=True` / `OSError Errno 28` with `cache_file_name` | 31k preprocessed examples exceed both Kaggle RAM and disk. Final fix: use `TrOCRDataset(TorchDataset)` which preprocesses on-the-fly per item — zero disk, zero RAM accumulation, full dataset used |
| `ViTImageProcessor` fast processor warning | Safe to ignore, or pass `use_fast=False` to `TrOCRProcessor.from_pretrained(...)` |
| Kaggle session timeout before training ends | Save checkpoints every epoch (`save_strategy="epoch"`) and resume with `trainer.train(resume_from_checkpoint=True)` |
| Output files lost after session ends | Quick Save only saves code, not outputs. Use **Save & Run All** to commit `/kaggle/working/` contents permanently. Then create Kaggle datasets from the Output tab. |
| `UserWarning: Argument 'var_limit' not valid for GaussNoise` | Albumentations API changed — use `A.GaussNoise(p=0.4)` and `A.ImageCompression(p=0.4)` without named quality/variance args |
| `warmup_ratio is deprecated` warning | Harmless for now, will be removed in transformers v5.2 — no action needed yet |
| CORD train: 0 crops extracted | `quad` is NOT in `gt_parse.menu` — it's in `valid_line[].words[].quad`. Use `valid_line` grouped by `group_id` to extract crops, not `menu` items |
